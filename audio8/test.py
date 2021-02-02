"""Training using 8-mile API

"""
from pynvml import *

nvmlInit()
import logging
import time
import numpy as np
from typing import Tuple, List, Optional, Dict
import os
from argparse import ArgumentParser
import torch.nn as nn
import random
from audio8.data import AudioTextLetterDataset
from audio8.wav2vec2 import create_acoustic_model, load_fairseq_bin, W2V_CTC_MAP
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, Average, get_num_gpus_multiworker, Offsets, revlut
from eight_mile.optz import *
from eight_mile.pytorch.layers import save_checkpoint, init_distributed, sequence_mask, find_latest_checkpoint
from eight_mile.pytorch.optz import *
import torch.nn.functional as F


logger = logging.getLogger(__file__)

Offsets.GO = 0
Offsets.PAD = 1
Offsets.VALUES[Offsets.GO] = '<s>'
Offsets.VALUES[Offsets.PAD] = '<pad>'
Offsets.VALUES[Offsets.EOS] = '</s>'
Offsets.VALUES[Offsets.UNK] = '<unk>'

def postproc_letters(sentence):

    sentence = sentence.replace(" ", "").replace("|", " ").strip()
    return sentence


def ctc_errors(lprobs_t, target, input_lengths, index2vocab):
    logging_output = {}
    import editdistance
    BLANK_IDX = Offsets.GO
    with torch.no_grad():
        #lprobs_t = lprobs.transpose(0, 1).float().cpu()

        c_err = 0
        c_len = 0
        w_errs = 0
        w_len = 0
        wv_errs = 0
        for lp, t, inp_l in zip(
                lprobs_t,
                target,
                input_lengths,
        ):
            lp = lp[:inp_l].unsqueeze(0)
            p = (t != Offsets.PAD) & (
                    t != Offsets.EOS
            )
            targ = t[p]
            targ_units = [index2vocab[x.item()] for x in targ]
            targ_units_arr = targ.tolist()

            toks = lp.argmax(dim=-1).unique_consecutive()
            pred_units_arr = toks[toks != BLANK_IDX].tolist()

            c_err += editdistance.eval(pred_units_arr, targ_units_arr)
            c_len += len(targ_units_arr)

            targ_words = postproc_letters(''.join(targ_units)).split()

            pred_units = [index2vocab[x] for x in pred_units_arr]
            pred_words_raw = postproc_letters(''.join(pred_units)).split()

            dist = editdistance.eval(pred_words_raw, targ_words)
            w_errs += dist
            wv_errs += dist

            w_len += len(targ_words)

        logging_output["wv_errors"] = wv_errs
        logging_output["w_errors"] = w_errs
        logging_output["w_total"] = w_len
        logging_output["c_errors"] = c_err
        logging_output["c_total"] = c_len
    return logging_output

def read_vocab_file(vocab_file: str):
    vocab = []
    for v in Offsets.VALUES:
        vocab.append(v)
    with open(vocab_file) as rf:
        for i, line in enumerate(rf):
            v = line.split()[0]
            vocab.append(v)
        return {v: i for i, v in enumerate(vocab)}


def run_step(model, batch, device, index2vocab):
    inputs, input_lengths, targets, target_lengths = batch
    inputs = inputs.to(device)
    pad_mask = sequence_mask(input_lengths, inputs.shape[1]).to(device)
    #targets = targets.to(device)
    logits, output_lengths = model(inputs, pad_mask)
    #input_lengths = pad_mask.sum(-1)

    print(ctc_errors(logits, targets, input_lengths, index2vocab))
    logits = logits.detach().cpu()
    #input_lengths = input_lengths.detach().cpu()
    return logits, output_lengths #input_lengths


def evaluate():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--root_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
    parser.add_argument("--dict_file", type=str, help="Dictionary file")
    parser.add_argument("--dataset_key", default="LibriSpeech",
                        help="dataset key for basedir")
    parser.add_argument("--sr", type=int, choices=[8, 16], default=16)
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--max_sample_len", type=int, default=250_000, help="Max sample length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, default=0., help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints", choices=['step', 'epoch', 'ignore'])
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--preprocessed", type=str2bool, default=True, help="Has the data already been preprocessed?")
    parser.add_argument("--model_type", default="wav2vec2")

    parser.add_argument("--valid_steps", type=int, default=1000, help="Num valid steps to evaluate each time")
    parser.add_argument("--buckets", type=int, nargs="+",
                        help="Bucket sizes if bucketing",
                        default=[11111, 35714, 38461, 41666, 45454, 50000, 55555, 62500, 71428, 83333, 100000, 125000, 166666,
                                 250000, 275000, 300000, 325000])#, 350000, 400000, 425000])
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000, help="The number of steps per checkpoint")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--vocab_file", help="Vocab for output decoding")
    parser.add_argument("--target_tokens_per_batch", type=int, default=700_000)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local rank for distributed training (-1 means use the environment variables to find)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.root_dir, 'dict.ltr.txt')
    vocab = read_vocab_file(vocab_file)
    index2vocab = revlut(vocab)
    valid_dataset = os.path.join(args.root_dir, args.valid_dataset)

    valid_set = AudioTextLetterDataset(valid_dataset, vocab, args.target_tokens_per_batch, 325000, distribute=False, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=None)
    logger.info("Loaded datasets")

    num_labels = len(vocab)
    model = create_acoustic_model(num_labels, args.sr, args.d_model, args.num_heads, args.num_layers,
                                  args.dropout, args.d_ff).to(args.device)

    if not args.checkpoint:
        args.checkpoint = find_latest_checkpoint(args.basedir)
    if args.checkpoint.endswith('.pt'):
        print(load_fairseq_bin(model, args.checkpoint, ctc=True))

    model.eval()
    valid_itr = iter(valid_loader)
    for i in range(args.valid_steps):


        batch = next(valid_itr)
        with torch.no_grad():
            logits, input_lengths = run_step(model, batch, args.device, index2vocab)


            logits = torch.argmax(logits[0], -1).tolist()
            input_lengths = input_lengths[0].item()
            print([index2vocab[k] for k in logits[:input_lengths] if k not in [0, 1, 2]])
            #print([index2vocab[b.item()] for b in batch[-2][0]])





if __name__ == "__main__":
    evaluate()
