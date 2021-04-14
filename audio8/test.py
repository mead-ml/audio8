"""Training using 8-mile API

"""

import os
from argparse import ArgumentParser
from audio8.data import AudioTextLetterDataset
from audio8.text import TextVectorizer, read_vocab_file
from audio8.wav2vec2 import create_acoustic_model, load_fairseq_bin
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, Offsets, revlut
from eight_mile.pytorch.layers import sequence_mask, find_latest_checkpoint
from eight_mile.pytorch.optz import *
from ctc import ctc_metrics, decode_text_wer
logger = logging.getLogger(__file__)

Offsets.GO = 0
Offsets.PAD = 1
Offsets.VALUES[Offsets.GO] = '<s>'
Offsets.VALUES[Offsets.PAD] = '<pad>'
Offsets.VALUES[Offsets.EOS] = '</s>'
Offsets.VALUES[Offsets.UNK] = '<unk>'


def run_step(index2vocab, model, batch, device, verbose=False, ctc_decoder=None):
    with torch.no_grad():
        inputs, input_lengths, targets, target_lengths, _ = batch
        inputs = inputs.to(device)
        pad_mask = sequence_mask(input_lengths, inputs.shape[1]).to(device)
        logits_batch, _ = model(inputs, pad_mask)
        metrics = ctc_metrics(logits_batch, targets, input_lengths, index2vocab)
        metrics['wbeam_errors'] = 0

        if ctc_decoder:
            B = inputs.shape[0]
            beam_results, beam_scores, timesteps, out_lens = ctc_decoder.decode(logits_batch)

            for b in range(B):
                transcription_ids = beam_results[b][0][:out_lens[b][0]]
                transcription = ''.join([index2vocab[t.item()] for t in transcription_ids])
                if verbose:
                   print(transcription)

                werr, _ = decode_text_wer(transcription, targets[b], index2vocab)
                metrics['wbeam_errors'] += werr

    return metrics


def read_vocab_list(vocab_file: str):
    vocab = []
    for v in Offsets.VALUES:
        vocab.append(v)
    with open(vocab_file) as rf:
        for i, line in enumerate(rf):
            v = line.split()[0]
            vocab.append(v)
        return vocab

def evaluate():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--root_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
    parser.add_argument("--dict_file", type=str, help="Dictionary file", default='dict.ltr.txt')
    parser.add_argument("--dataset_key", default="LibriSpeech", help="dataset key for basedir")
    parser.add_argument("--input_sample_rate", type=int, default=16_000)
    parser.add_argument("--target_sample_rate", type=int, default=16_000)
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--max_sample_len", type=int, default=325_000, help="Max sample length")
    parser.add_argument(
        "--restart_tt", type=str, help="Optional param for legacy checkpoints", choices=['step', 'epoch', 'ignore']
    )
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--model_type", default="wav2vec2")
    parser.add_argument("--verbose", type=str2bool, help="Verbose", default=False)
    parser.add_argument("--valid_steps", type=int, help="Num valid steps to evaluate", default=40_000)
    parser.add_argument("--steps_per_update", type=int, default=100)
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000, help="The number of steps per checkpoint")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    parser.add_argument("--vocab_file", help="Vocab for output decoding")
    parser.add_argument("--target_tokens_per_batch", type=int, default=700_000)
    parser.add_argument("--lm")
    parser.add_argument("--beam", type=int, default=1, help="Beam size")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.root_dir, args.dict_file)
    vocab_list = read_vocab_file(vocab_file)

    beam_lm_key = None
    ctc_decoder = None
    # Prefix beam search with optional LM
    if args.beam > 1 or args.lm:
        from ctcdecode import CTCBeamDecoder
        ctc_decoder = CTCBeamDecoder(
            labels=vocab_list,
            model_path=args.lm,
            alpha=args.alpha,
            beta=args.beta,
            beam_width=args.beam,
            blank_id=Offsets.GO,
            log_probs_input=True,
        )
        beam_lm_key = f'werr_lm_{args.beam}' if args.lm else f'werr_{args.beam}'

    vocab = {v: i for i, v in enumerate(vocab_list)}
    vec = TextVectorizer(vocab)
    index2vocab = revlut(vocab)
    valid_dataset = os.path.join(args.root_dir, args.valid_dataset)

    valid_set = AudioTextLetterDataset(
        valid_dataset,
        vec,
        args.target_tokens_per_batch,
        args.max_sample_len,
        input_sample_rate=args.input_sample_rate,
        target_sample_rate=args.target_sample_rate,
        distribute=False,
        shuffle=False,
        is_infinite=False,
    )
    valid_loader = DataLoader(valid_set, batch_size=None)
    logger.info("Loaded datasets")

    num_labels = len(vocab)
    model = create_acoustic_model(num_labels, args.target_sample_rate // 1000, **vars(args)).to(args.device)

    if not args.checkpoint:
        args.checkpoint = find_latest_checkpoint(args.basedir)
    if args.checkpoint.endswith('.pt'):
        print(load_fairseq_bin(model, args.checkpoint, ctc=True, sr=args.target_sample_rate // 1000))
    else:
        model.load_state_dict(torch.load(args.checkpoint))

    model.eval()
    valid_itr = iter(valid_loader)
    metrics = {}
    c_errors = 0
    c_total = 0
    w_errors = 0
    w_total = 0
    wlm_errors = 0


    for j, batch in enumerate(valid_itr):
        if j > args.valid_steps:
            break

        try:
            step_metrics = run_step(index2vocab, model, batch, args.device, args.verbose, ctc_decoder)

            c_errors += step_metrics['c_errors']
            w_errors += step_metrics['w_errors']
            if 'wbeam_errors' in step_metrics:
                wlm_errors += step_metrics['wbeam_errors']
            c_total += step_metrics['c_total']
            w_total += step_metrics['w_total']
            metrics['cer'] = (c_errors / c_total) * 100
            metrics['wer'] = (w_errors / w_total) * 100
            if beam_lm_key:
                metrics[beam_lm_key] = (wlm_errors / w_total) * 100
            metrics['step'] = j + 1
            if (j + 1) % args.steps_per_update == 0:
                logger.info(metrics)
        except Exception as e:
            logger.error(e)
    logger.info("Final results")
    logger.info(metrics)

if __name__ == "__main__":
    evaluate()
