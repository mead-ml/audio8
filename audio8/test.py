"""Training using 8-mile API

"""

import os
from argparse import ArgumentParser
from audio8.data import AudioTextLetterDataset
from audio8.text import TextVectorizer, read_vocab_file
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


def run_step(index2vocab, model, batch, device, verbose=False):
    with torch.no_grad():
        inputs, input_lengths, targets, target_lengths, _ = batch
        inputs = inputs.to(device)
        pad_mask = sequence_mask(input_lengths, inputs.shape[1]).to(device)
        logits, _ = model(inputs, pad_mask)
        #output_lengths = pad_mask.sum(-1)

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
    parser.add_argument("--dict_file", type=str, help="Dictionary file", default='dict.ltr.txt')
    parser.add_argument("--dataset_key", default="LibriSpeech",
                        help="dataset key for basedir")
    parser.add_argument("--sr", type=int, choices=[8, 16], default=16)
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--max_sample_len", type=int, default=325_000, help="Max sample length")
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.root_dir, args.dict_file)
    vocab = read_vocab_file(vocab_file)
    vec = TextVectorizer(vocab)
    index2vocab = revlut(vocab)
    valid_dataset = os.path.join(args.root_dir, args.valid_dataset)

    valid_set = AudioTextLetterDataset(valid_dataset, vec, args.target_tokens_per_batch, args.max_sample_len, distribute=False, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=None)
    logger.info("Loaded datasets")

    num_labels = len(vocab)
    model = create_acoustic_model(num_labels, args.sr, args.d_model, args.num_heads, args.num_layers,
                                  0.0, args.d_ff).to(args.device)

    if not args.checkpoint:
        args.checkpoint = find_latest_checkpoint(args.basedir)
    if args.checkpoint.endswith('.pt'):
        print(load_fairseq_bin(model, args.checkpoint, ctc=True))
    else:
        model.load_state_dict(torch.load(args.checkpoint))

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
