"""Pretraining using 8-mile API

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
from audio8.text import BPEVectorizer, TextTransformerPooledEncoder, TextBoWPooledEncoder
from audio8.wav2vec2 import create_paired_model
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.pytorch.serialize import load_tlm_npz
from eight_mile.utils import str2bool, Average, get_num_gpus_multiworker, revlut
from baseline.pytorch.embeddings import *
import baseline.embeddings
from eight_mile.pytorch.optz import OptimizerManager
from eight_mile.utils import Offsets
from eight_mile.pytorch.layers import save_checkpoint, init_distributed, find_latest_checkpoint
from eight_mile.pytorch.optz import *
from audio8.utils import create_lrs

logger = logging.getLogger(__file__)


def is_raw_checkpoint(checkpoint):
    if 'mask_emb' in checkpoint:
        return True
    return False


def run_step(batch, loss_function, device):
    inputs, input_lengths, targets, target_lengths, _ = batch
    pad_mask = sequence_mask(input_lengths, inputs.shape[1]).to(device=device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    loss = loss_function((inputs, pad_mask), (targets, target_lengths))
    num_tokens = target_lengths.sum().item()
    metrics = {}
    metrics['batch_size'] = inputs.shape[0]
    metrics['num_tokens'] = num_tokens
    return loss, metrics


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--root_dir")
    parser.add_argument("--train_dataset", type=str, help='Dataset (by name), e.g. train-clean-360')
    parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--input_sample_rate", type=int, default=16_000)
    parser.add_argument("--target_sample_rate", type=int, default=16_000)
    parser.add_argument("--dataset_key", default="LibriSpeech", help="dataset key for basedir")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--audio_d_model", type=int, default=768, help="Audio model dimension (and embedding dsz)")
    parser.add_argument("--audio_d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--audio_d_k", type=int, default=None, help="Reduction for audio pooling")
    parser.add_argument("--audio_num_heads", type=int, default=12, help="Number of audio heads")
    parser.add_argument("--audio_num_layers", type=int, default=12, help="Number of audio layers")
    parser.add_argument(
        "--audio_reduction_type",
        type=str,
        choices=['2ha', 'sha', '2ha_mean', 'sha_mean', '2ha_max', 'sha_max', 'max'],
        default='max',
    )
    parser.add_argument("--stacking_layers", type=int, nargs="+", default=[])
    parser.add_argument("--text_encoder_type", type=str, default="transformer", choices=["transformer", "bow"])
    parser.add_argument("--text_d_model", type=int, default=512, help="Audio model dimension (and embedding dsz)")
    parser.add_argument("--text_d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--text_d_k", type=int, default=None, help="Reduction for text pooling")
    parser.add_argument("--text_num_heads", type=int, default=8, help="Number of text heads")
    parser.add_argument("--text_num_layers", type=int, default=8, help="Number of text layers")
    parser.add_argument(
        "--text_reduction_type",
        type=str,
        choices=['2ha', 'sha', '2ha_mean', 'sha_mean', '2ha_max', 'sha_max', 'mean', 'max'],
        default='max',
    )
    parser.add_argument("--text_begin_tok", type=str, default="<GO>")
    parser.add_argument("--text_end_tok", type=str, default="<EOS>")
    parser.add_argument("--text_rpr_k", type=int, default=8, help="Relative Attention Representation length")
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--nctx", type=int, default=256)
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--max_sample_len", type=int, default=325_000, help="Max sample length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--layer_drop", type=float, default=0.0, help="Layer Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_alpha", type=float, default=0.0, help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=2.0e-5, help="Learning rate")
    parser.add_argument("--clip", type=float, default=25.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument(
        "--restart_tt", type=str, help="Optional param for legacy checkpoints", choices=['step', 'ignore']
    )
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--plateau_steps", type=int, default=0, help="Num plateau steps")
    parser.add_argument("--saves_per_epoch", type=int, default=10, help="The number of saves per epoch")
    parser.add_argument("--model_type", default="wav2vec2")
    parser.add_argument("--audio_unfreeze_after_step", default=100_000, type=int)
    parser.add_argument("--text_unfreeze_after_step", default=100_000, type=int)
    parser.add_argument("--train_steps", type=int, default=400_000, help="Num training steps")
    parser.add_argument("--valid_steps", type=int, default=1000, help="Num valid steps to evaluate each time")
    parser.add_argument("--steps_per_update", type=int, default=100)
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000, help="The number of steps per checkpoint")
    parser.add_argument("--verbose", type=str2bool, help="Verbose", default=False)
    parser.add_argument("--learn_temp", type=str2bool, default=True, help="Should we learn the temperature scaling")
    parser.add_argument("--init_temp", type=float, default=1.0, help="Initialize the temperature")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    parser.add_argument("--distributed", type=str2bool, default=False, help="Are we doing distributed training?")
    parser.add_argument("--vocab_file", help="Vocab for output decoding")
    parser.add_argument("--target_tokens_per_batch", type=int, default=700_000)
    parser.add_argument("--warmstart_text", help="Restore text encoder from an existing checkpoint")
    parser.add_argument(
        "--pretok", help="Is the text data already pre-tokenized into sub-words?", type=str2bool, default=False
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1 means use the environment variables to find)",
    )

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

    vec = BPEVectorizer(
        model_file=args.subword_model_file,
        vocab_file=args.subword_vocab_file,
        emit_begin_tok=args.text_begin_tok,
        emit_end_tok=args.text_end_tok,
    )

    train_dataset = os.path.join(args.root_dir, args.train_dataset)
    valid_dataset = os.path.join(args.root_dir, args.valid_dataset)

    tgt_type = AudioTextLetterDataset.TGT_BPE if args.pretok else AudioTextLetterDataset.TGT_WRD
    train_set = AudioTextLetterDataset(
        train_dataset,
        vec,
        args.target_tokens_per_batch,
        args.max_sample_len,
        input_sample_rate=args.input_sample_rate,
        target_sample_rate=args.target_sample_rate,
        shuffle=True,
        distribute=args.distributed,
        tgt_type=tgt_type,
    )
    valid_set = AudioTextLetterDataset(
        valid_dataset,
        vec,
        args.target_tokens_per_batch,
        args.max_sample_len,
        input_sample_rate=args.input_sample_rate,
        target_sample_rate=args.target_sample_rate,
        distribute=False,
        shuffle=False,
        tgt_type=tgt_type,
    )
    train_loader = DataLoader(train_set, batch_size=None)  # , num_workers=args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=None)

    logger.info("Loaded datasets")

    preproc_data = baseline.embeddings.load_embeddings(
        'x',
        dsz=args.text_d_model,
        known_vocab=vec.vocab,
        preserve_vocab_indices=True,
        embed_type='default',
        # This is ugly, but basically if its word embeddings, load them upfront
        embed_file=args.warmstart_text if args.text_encoder_type == 'bow' else None,
    )

    embeddings = preproc_data['embeddings']
    model = create_paired_model(embeddings, **vars(args)).to(args.device)
    logger.info(f"init temperature: {args.init_temp}, learnable: {args.learn_temp}")
    loss_function = model.create_loss('symmetric', init_temp=args.init_temp, learn_temp=args.learn_temp).to(args.device)
    logger.info("Loaded model and loss")

    update_on = args.steps_per_checkpoint
    validate_on = min(args.train_steps // 2, update_on * 10)
    report_on = max(10, update_on) // 10
    lr_sched = create_lrs(
        args.lr,
        args.train_steps,
        args.lr_scheduler,
        alpha=args.lr_alpha,
        warmup_steps=args.warmup_steps,
        plateau_steps=args.plateau_steps,
    )

    global_step = 0
    if args.restart_from:

        if os.path.isdir(args.restart_from):
            args.restart_from, _ = find_latest_checkpoint(args.restart_from)
        try:
            model.load_state_dict(torch.load(args.restart_from))
        except:
            print('Trying to load a8 checkpoint from pretrained wav2vec')
            checkpoint = torch.load(args.restart_from)
            if is_raw_checkpoint(checkpoint):
                unmapped = model.encoder_1.encoder.load_state_dict(checkpoint, strict=False)
            else:
                unmapped = model.encoder_1.load_state_dict(checkpoint, strict=False)
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

        logger.info(
            "Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d", args.restart_from, global_step
        )

    # For learned temperature, we need to pass the loss function in so that param is learnable.
    # Since the loss_function owns the model, we can only pass it in
    optimizer = OptimizerManager(
        loss_function, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay
    )
    logger.info(
        "Model has {:,} parameters".format(sum(p.numel() for p in loss_function.parameters() if p.requires_grad))
    )

    # Prepare model for distributed training if needed
    if args.distributed:
        # This program assume pure data parallelism, each model is on a single gpu
        # If we wanted to support model and data parallelism we would need to update
        # the selection of gpus based on rank, it would need to select multiple ids
        # based on rank, here we select only a single gpu and use it for input and
        # output.
        model = DistributedDataParallel(
            model, device_ids=[args.device], output_device=args.device, find_unused_parameters=True
        )
        _model = model.module
        logger.info("Model located on %s", args.device)
    else:
        _model = model
    model_base = os.path.join(args.basedir, 'checkpoint')

    train_itr = iter(train_loader)
    avg_loss = Average('average_train_loss')
    step_time = Average('average_step_time')
    batch_size_sent = Average('batch_size')
    batch_size_toks = Average('batch_toks')
    batch_size = torch.tensor(0, device=args.device, dtype=int)
    num_tokens_this_batch = torch.tensor(0, device=args.device, dtype=int)

    model.train()
    iters = 0
    start = time.time()

    optimizer.zero_grad()

    while optimizer.global_step < args.train_steps:

        if optimizer.global_step > args.audio_unfreeze_after_step:
            _model.encoder_1.freeze = False

        if optimizer.global_step > args.text_unfreeze_after_step:
            _model.encoder_2.freeze = False
        metrics = {}
        iters += 1
        is_dist_step = iters % args.grad_accum == 0

        try:
            with model.no_sync() if (args.distributed and not is_dist_step) else contextlib.ExitStack():
                # This loader will iterate for ever
                batch = next(train_itr)
                loss, step_metrics = run_step(batch, loss_function, args.device)
                num_tokens_this_batch += step_metrics['num_tokens']
                batch_size += step_metrics['batch_size']

                avg_loss.update(loss.item())
                loss.backward()

                if is_dist_step:
                    # This will allreduce the gradients which will be scaled by 1/num_gpus
                    if args.distributed:
                        torch.distributed.all_reduce(batch_size)
                        torch.distributed.all_reduce(num_tokens_this_batch)

                    optimizer.scale_grads(num_gpus / batch_size)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_size_sent.update(batch_size.cpu().item())
                    batch_size_toks.update(num_tokens_this_batch.cpu().item())
                    num_tokens_this_batch *= 0
                    batch_size *= 0
                    elapsed = time.time() - start
                    step_time.update(elapsed)
                    start = time.time()
                    if optimizer.global_step % report_on == 0:
                        if step_time.avg != 0:
                            steps_per_sec = 1.0 / step_time.avg
                            logging.info(
                                '%s, steps/min %f, LR %.6f, batch (samples %.2f, toks %.2f, toks/min %.2f)',
                                avg_loss,
                                steps_per_sec * 60,
                                optimizer.current_lr,
                                batch_size_sent.avg,
                                batch_size_toks.avg,
                                batch_size_toks.avg * steps_per_sec * 60,
                            )

                    if optimizer.global_step % update_on == 0 and args.local_rank < 1:
                        save_checkpoint(model, model_base, optimizer.global_step, tick_type='step')
                    if optimizer.global_step % validate_on == 0 and args.local_rank < 1:
                        train_token_loss = avg_loss.avg
                        metrics['average_train_loss'] = train_token_loss
                        avg_valid_loss = Average('average_valid_loss')

                        model.eval()
                        valid_start = time.time()
                        valid_itr = iter(valid_loader)

                        valid_metrics = {}
                        for j, batch in enumerate(valid_itr):
                            if j > args.valid_steps:
                                break

                            try:
                                with torch.no_grad():
                                    loss, _ = run_step(batch, loss_function, args.device)

                                avg_valid_loss.update(loss.item())
                                elapsed = time.time() - valid_start
                                valid_token_loss = avg_valid_loss.avg
                                valid_metrics['average_valid_loss'] = valid_token_loss
                                valid_metrics['valid_elapsed_epoch'] = elapsed

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
