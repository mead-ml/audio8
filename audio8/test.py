"""Training using 8-mile API

"""

import os
from argparse import ArgumentParser
from audio8.data import AudioTextLetterDataset
from audio8.wav2vec2 import create_acoustic_model, load_fairseq_bin, W2V_CTC_MAP
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, Offsets, revlut
from eight_mile.pytorch.layers import sequence_mask, find_latest_checkpoint
from eight_mile.pytorch.optz import *
from ctc import ctc_metrics

logger = logging.getLogger(__file__)

Offsets.GO = 0
Offsets.PAD = 1
Offsets.VALUES[Offsets.GO] = '<s>'
Offsets.VALUES[Offsets.PAD] = '<pad>'
Offsets.VALUES[Offsets.EOS] = '</s>'
Offsets.VALUES[Offsets.UNK] = '<unk>'


def read_vocab_file(vocab_file: str):
    vocab = []
    for v in Offsets.VALUES:
        vocab.append(v)
    with open(vocab_file) as rf:
        for i, line in enumerate(rf):
            v = line.split()[0]
            vocab.append(v)
        return {v: i for i, v in enumerate(vocab)}


def run_step(index2vocab, model, batch, device, verbose=False):
    with torch.no_grad():
        inputs, input_lengths, targets, target_lengths = batch
        inputs = inputs.to(device)
        pad_mask = sequence_mask(input_lengths, inputs.shape[1]).to(device)
        logits, output_lengths = model(inputs, pad_mask)

        if verbose:
            logits = torch.argmax(logits[0], -1).tolist()
            input_lengths = input_lengths[0].item()
            print([index2vocab[k] for k in logits[:input_lengths] if k not in [0, 1, 2]])

        metrics = ctc_metrics(logits, targets, input_lengths, index2vocab)
    return metrics


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
    parser.add_argument("--max_sample_len", type=int, default=325_000, help="Max sample length")
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
    parser.add_argument("--model_type", default="wav2vec2")
    parser.add_argument("--verbose", type=str2bool, help="Verbose", default=False)
    parser.add_argument("--valid_steps", type=int, help="Num valid steps to evaluate", default=40_000)
    parser.add_argument("--steps_per_update", type=int, default=100)
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

    valid_set = AudioTextLetterDataset(valid_dataset, vocab, args.target_tokens_per_batch, args.max_sample_len, distribute=False, shuffle=False)
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
    metrics = {}
    c_errors = 0
    c_total = 0
    w_errors = 0
    w_total = 0

    for j, batch in enumerate(valid_itr):
        if j > args.valid_steps:
            break

        try:
            step_metrics = run_step(index2vocab, model, batch, args.device, args.verbose)
            c_errors += step_metrics['c_errors']
            w_errors += step_metrics['w_errors']

            c_total += step_metrics['c_total']
            w_total += step_metrics['w_total']
            metrics['cer'] = (c_errors / c_total) * 100
            metrics['wer'] = (w_errors / w_total) * 100
            if j % args.steps_per_update == 0:
                logger.info(metrics)
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    evaluate()
