"""Training using 8-mile API

"""
import logging
import time
import numpy as np
from typing import Tuple, List, Optional, Dict
import os
from argparse import ArgumentParser
import torch.nn as nn
import random
from audio8.data import AudioTextLetterDataset
from audio8.text import TextVectorizer, read_vocab_file
from audio8.wav2vec2 import create_acoustic_model, load_fairseq_bin
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, Average, get_num_gpus_multiworker, Offsets, revlut
from eight_mile.optz import *
from eight_mile.pytorch.layers import save_checkpoint, init_distributed, sequence_mask, find_latest_checkpoint
from eight_mile.pytorch.optz import *
from audio8.ctc import CTCLoss, ctc_metrics, prefix_beam_search

logger = logging.getLogger(__file__)
Offsets.GO = 0
Offsets.PAD = 1
Offsets.VALUES[Offsets.GO] = '<s>'
Offsets.VALUES[Offsets.PAD] = '<pad>'
Offsets.VALUES[Offsets.EOS] = '</s>'
Offsets.VALUES[Offsets.UNK] = '<unk>'


def run_step(index2vocab, model, batch, loss_function, device, verbose, training=True):
    inputs, input_lengths, targets, target_lengths, _ = batch
    pad_mask = sequence_mask(input_lengths, inputs.shape[1]).to(device=device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits, pad_mask = model(inputs, pad_mask)
    output_lengths = pad_mask.sum(-1)
    loss = loss_function(logits.transpose(1, 0), output_lengths, targets, target_lengths)
    logits = logits.detach().cpu()
    metrics = {}
    metrics['batch_size'] = inputs.shape[0]
    if not training:

        metrics = ctc_metrics(logits, targets, input_lengths, index2vocab)
        if verbose:
            input_lengths_batch = pad_mask.sum(-1)
            logits_batch = logits
            for logits, input_lengths in zip(logits_batch, input_lengths_batch):
                input_lengths = input_lengths.item()
                probs = logits.exp().cpu().numpy()
                transcription = prefix_beam_search(probs[:input_lengths, :], index2vocab, k=1)[0]
                print(transcription)
    return loss, metrics


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--root_dir")
    parser.add_argument("--train_dataset", type=str, help='Dataset (by name), e.g. train-clean-360')
    parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
    parser.add_argument("--dict_file", type=str, help="Dictionary file", default='dict.ltr.txt')
    parser.add_argument("--dataset_key", default="LibriSpeech",
                        help="dataset key for basedir")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--sr", type=int, choices=[8, 16], default=16)
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--max_sample_len", type=int, default=325_000, help="Max sample length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, default=0., help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=2.0e-5, help="Learning rate")
    parser.add_argument("--clip", type=float, default=25.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints", choices=['step', 'ignore'])
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--saves_per_epoch", type=int, default=10, help="The number of saves per epoch")
    parser.add_argument("--model_type", default="wav2vec2")
    parser.add_argument("--unfreeze_enc_after_step", default=10_000)
    parser.add_argument("--train_steps", type=int, default=400_000, help="Num training steps")
    parser.add_argument("--valid_steps", type=int, default=1000, help="Num valid steps to evaluate each time")
    parser.add_argument("--steps_per_update", type=int, default=100)
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000, help="The number of steps per checkpoint")
    parser.add_argument("--verbose", type=str2bool, help="Verbose", default=False)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--distributed",
                        type=str2bool,
                        default=False,
                        help="Are we doing distributed training?")
    parser.add_argument("--vocab_file", help="Vocab for output decoding")
    parser.add_argument("--target_tokens_per_batch", type=int, default=700_000)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local rank for distributed training (-1 means use the environment variables to find)")

    args = parser.parse_args()

    # Get the basedir to save results and checkpoints
    if args.basedir is None:
        args.basedir = f'{args.model_type}-{args.dataset_key}-{os.getpid()}'
    os.makedirs(args.basedir, exist_ok=True)

    # Setup logger
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    num_gpus = get_num_gpus_multiworker()
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    if args.distributed:
        args.device, updated_local_rank = init_distributed(args.local_rank)
        args.local_rank = updated_local_rank

    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.root_dir, args.dict_file)
    vocab = read_vocab_file(vocab_file)
    vec = TextVectorizer(vocab)
    index2vocab = revlut(vocab)
    train_dataset = os.path.join(args.root_dir, args.train_dataset)
    valid_dataset = os.path.join(args.root_dir, args.valid_dataset)

    train_set = AudioTextLetterDataset(train_dataset, vec, args.target_tokens_per_batch, args.max_sample_len, shuffle=True, distribute=args.distributed)
    valid_set = AudioTextLetterDataset(valid_dataset, vec, args.target_tokens_per_batch, args.max_sample_len, distribute=False, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=None)  # , num_workers=args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=None)

    logger.info("Loaded datasets")

    num_labels = len(vocab)
    model = create_acoustic_model(num_labels, args.sr, args.d_model, args.num_heads, args.num_layers,
                                  args.dropout, args.d_ff).to(args.device)

    loss_function = CTCLoss().to(args.device)
    logger.info("Loaded model and loss")

    update_on = args.steps_per_checkpoint
    validate_on = min(args.train_steps//2, update_on * 10)
    report_on = max(10, update_on) // 10
    lr_decay = CosineDecaySchedulerPyTorch(decay_steps=args.train_steps, alpha=args.lr_alpha, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)

    global_step = 0
    if args.restart_from:

        if args.restart_from.endswith('.pt'):
            # The pretrained fairseq checkpoints differ depending on whether they are pretrained wav2vec2
            # or wav2vec2-ctc.  First, we try loading as pretrained wav2vec2, then back off to ctc
            try:
                unmapped = load_fairseq_bin(model.encoder, args.restart_from)
            except:
                unmapped = load_fairseq_bin(model, args.restart_from, ctc=True)
            print(unmapped)
            args.tick_type = None
        else:
            if os.path.isdir(args.restart_from):
                args.restart_from, _ = find_latest_checkpoint(args.restart_from)
            try:
                model.load_state_dict(torch.load(args.restart_from))
            except:
                print('Trying to load a8 checkpoint from pretrained wav2vec2 w/o CTC')
                unmapped = model.encoder.load_state_dict(torch.load(args.restart_from), strict=False)
                print(unmapped)
            if args.restart_tt:
                tick_type = args.restart_tt
            else:
                vec = args.restart_from.split("-")
                tick_type = vec[-2]

            if tick_type == 'step':
                vec = args.restart_from.split("-")
                step_num = int(vec[-1].split(".")[0])
                global_step = step_num
            else:
                logger.warning(f"Setting step number to 0")

            logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d",
                        args.restart_from, global_step)

    optimizer = OptimizerManager(model, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Prepare model for distributed training if needed
    if args.distributed:
        # This program assume pure data parallelism, each model is on a single gpu
        # If we wanted to support model and data parallelism we would need to update
        # the selection of gpus based on rank, it would need to select multiple ids
        # based on rank, here we select only a single gpu and use it for input and
        # output.
        model = DistributedDataParallel(model, device_ids=[args.device], output_device=args.device, find_unused_parameters=True)
        _model = model.module
        logger.info("Model located on %s", args.device)
    else:
        _model = model
    model_base = os.path.join(args.basedir, 'checkpoint')
    steps = global_step

    train_itr = iter(train_loader)
    avg_loss = Average('average_train_loss')
    step_time = Average('average_step_time')
    batch_sizes = Average('batch_size')
    model.train()

    for i in range(steps, args.train_steps):

        if steps > args.unfreeze_enc_after_step:
            _model.freeze = False
        metrics = {}
        optimizer.zero_grad()
        start = time.time()
        # This loader will iterate for ever
        batch = next(train_itr)

        loss, step_metrics = run_step(index2vocab, model, batch, loss_function, args.device, args.verbose)
        batch_sizes.update(step_metrics['batch_size'])
        steps += 1

        try:
            avg_loss.update(loss.item())
            loss.backward()
            if steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()
            elapsed = time.time() - start
            step_time.update(elapsed)

            if (steps + 1) % report_on == 0:
                steps_per_sec = 1.0 / step_time.avg
                logging.info('%s, steps/min %f, LR %.6f, avg batch size %.2f', avg_loss, steps_per_sec*60, optimizer.current_lr, batch_sizes.avg)

            if (steps + 1) % update_on == 0 and args.local_rank < 1:
                save_checkpoint(model, model_base, steps, tick_type='step')
            if (steps + 1) % validate_on == 0 and args.local_rank < 1:
                train_token_loss = avg_loss.avg
                metrics['average_train_loss'] = train_token_loss
                avg_valid_loss = Average('average_valid_loss')

                model.eval()
                valid_start = time.time()
                valid_itr = iter(valid_loader)
                c_errors = 0
                c_total = 0
                w_errors = 0
                w_total = 0

                valid_metrics = {}
                for j, batch in enumerate(valid_itr):
                    if j > args.valid_steps:
                        break

                    try:
                        with torch.no_grad():
                            loss, valid_step_metrics = run_step(index2vocab, model, batch, loss_function, args.device, verbose=args.verbose, training=False)
                        c_errors += valid_step_metrics['c_errors']
                        w_errors += valid_step_metrics['w_errors']
                        c_total += valid_step_metrics['c_total']
                        w_total += valid_step_metrics['w_total']
                        avg_valid_loss.update(loss.item())
                        elapsed = time.time() - valid_start
                        valid_token_loss = avg_valid_loss.avg
                        valid_metrics['average_valid_loss'] = valid_token_loss
                        valid_metrics['valid_elapsed_epoch'] = elapsed
                        valid_metrics['cer'] = (c_errors / c_total) * 100
                        valid_metrics['wer'] = (w_errors / w_total) * 100
                        if j % args.steps_per_update == 0:
                            logger.info(valid_metrics)
                    except Exception as e:
                        logger.error(e)

                logger.info(metrics)
                model.train()
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    train()
