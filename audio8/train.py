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
from audio8.data import LibriSpeechDataset, collate_fn
from audio8.wav2vec2 import create_acoustic_model, load_fairseq_bin
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, Average, get_num_gpus_multiworker, Offsets, revlut
from eight_mile.optz import *
from eight_mile.pytorch.layers import save_checkpoint, init_distributed, sequence_mask
from eight_mile.pytorch.optz import *
import torch.nn.functional as F


logger = logging.getLogger(__file__)


class CTCLoss(nn.Module):
    def __init__(self, zero_infinity=False):
        super().__init__()
        self.zero_infinity = zero_infinity

    # The input length signal seems to be irrelevant
    # unless we can apply the filter length transformation to figure it out
    def forward(self, log_prob, input_lengths, targets, target_lengths):
        pad_mask = (targets != Offsets.PAD) & (targets != Offsets.EOS)
        targets_flat = targets.masked_select(pad_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                log_prob,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=Offsets.GO,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            return loss


def read_vocab_file(vocab_file: str):
    vocab = []
    for v in Offsets.VALUES:
        vocab.append(v)
    with open(vocab_file) as rf:
        for i, line in enumerate(rf):
            v = line.split()[0]
            vocab.append(v)
        return {v: i for i, v in enumerate(vocab)}


def run_step(model, batch, loss_function, device):
    inputs, input_lengths, targets, target_lengths = batch
    inputs = inputs.to(device)
    pad_mask = sequence_mask(input_lengths).to(device)
    targets = targets.to(device)
    logits, pad_mask = model(inputs, pad_mask)
    input_lengths = pad_mask.sum(-1)
    loss = loss_function(logits.transpose(1, 0), input_lengths, targets, target_lengths)
    return loss


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--root_dir")
    parser.add_argument("--train_dataset", type=str, help='Dataset (by name), e.g. train-clean-360')
    parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
    parser.add_argument("--dataset_key", default="LibriSpeech",
                        help="dataset key for basedir")
    parser.add_argument("--num_vq_vars", type=int, default=320)
    parser.add_argument("--num_vq_groups", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--sr", type=int, choices=[8, 16], default=16)
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--tokens_per_batch", type=int, default=1_400_000, help="Number of tokens per batch")
    parser.add_argument("--max_sample_len", type=int, default=250_000, help="Max sample length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, default=0., help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--train_steps", type=int, default=400_000, help="Num training steps")
    parser.add_argument("--valid_steps", type=int, default=10_000, help="Num valid steps to evaluate each time")

    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000, help="The number of steps per checkpoint")
    parser.add_argument("--preprocessed", type=str2bool, default=True, help="Has the data already been preprocessed?")
    parser.add_argument("--model_type", default="wav2vec2")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--distributed",
                        type=str2bool,
                        default=False,
                        help="Are we doing distributed training?")
    parser.add_argument("--vocab_file", help="Vocab for output decoding")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int)

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local rank for distributed training (-1 means use the environment variables to find)")

    args = parser.parse_args()

    if args.basedir is None:
        args.basedir = f'{args.model_type}-{args.dataset_key}-{os.getpid()}'
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    os.makedirs(args.basedir, exist_ok=True)
    num_gpus = get_num_gpus_multiworker()
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    if args.distributed:
        args.device, updated_local_rank = init_distributed(args.local_rank)
        args.local_rank = updated_local_rank
    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.root_dir, args.dataset_key, 'dict.ltr.txt')
    vocab = read_vocab_file(vocab_file)

    train_set = LibriSpeechDataset(vocab, args.root_dir, url=args.train_dataset, folder_in_archive=args.dataset_key)
    valid_set = LibriSpeechDataset(vocab, args.root_dir, url=args.valid_dataset, folder_in_archive=args.dataset_key)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_train_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch_size, collate_fn=collate_fn)
    logger.info("Loaded datasets")

    num_labels = len(vocab)
    model = create_acoustic_model(num_labels, args.sr, args.d_model, args.num_heads, args.num_layers,
                                  args.dropout, args.d_ff).to(args.device)

    loss_function = CTCLoss().to(args.device)
    logger.info("Loaded model and loss")

    # according to pytorch, len(train_loader) will return len(train_set) when train_set is IterableDataset, so manually
    # correct it here
    valid_steps = args.valid_steps
    update_on = args.steps_per_checkpoint
    validate_on = update_on * 10
    report_on = max(10, update_on) // 10
    lr_decay = CosineDecaySchedulerPyTorch(decay_steps=args.train_steps, alpha=args.lr_alpha, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)

    global_step = 0
    if args.restart_from:

        if args.restart_from.endswith('.pt'):
            print(load_fairseq_bin(model.encoder, args.restart_from))
        else:
            model.load_state_dict(torch.load(args.restart_from))
            vec = args.restart_from.split("-")
            global_step = int(vec[-1].split(".")[0])
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
        model = DistributedDataParallel(model, device_ids=[args.device], output_device=args.device)
        logger.info("Model located on %s", args.device)

    model_base = os.path.join(args.basedir, 'checkpoint')
    steps = global_step

    train_itr = iter(train_loader)
    start_of_run = 0
    avg_loss = Average('average_train_loss')
    step_time = Average('average_step_time')
    for i in range(steps, args.train_steps):

        metrics = {}
        optimizer.zero_grad()
        start = time.time()
        model.train()
        # This loader will iterate for ever
        batch = next(train_itr)
        loss = run_step(model, batch, loss_function, args.device)

        steps += 1
        loss.backward()
        avg_loss.update(loss.item())

        if steps % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
        elapsed = time.time() - start
        step_time.update(elapsed)

        if (steps + 1) % report_on == 0:
            steps_per_sec = 1.0 / step_time.avg
            logging.info('%s, steps/min %f, LR %.6f', avg_loss, steps_per_sec*60, optimizer.current_lr)

        if (steps + 1) % update_on == 0 and args.local_rank < 1:
            save_checkpoint(model, model_base, steps, tick_type='step')
        if (steps + 1) % validate_on == 0 and args.local_rank < 1:
            # How much time elapsed in minutes
            elapsed = (time.time() - start_of_run) / 60
            metrics['train_elapsed_min'] = elapsed

            train_token_loss = avg_loss.avg
            metrics['average_train_loss'] = train_token_loss
            avg_valid_loss = Average('average_valid_loss')

            model.eval()
            valid_start = time.time()
            valid_itr = iter(valid_loader)
            for j in range(valid_steps):
                batch = next(valid_itr)

                with torch.no_grad():
                    run_step(model, batch, loss_function, args.device)
                    avg_valid_loss.update(loss.item())
            valid_token_loss = avg_valid_loss.avg
            metrics['average_valid_loss'] = valid_token_loss
            elapsed = (time.time() - valid_start) / 60
            metrics['valid_elapsed_epoch'] = elapsed
            logger.info(metrics)


if __name__ == "__main__":
    train()
