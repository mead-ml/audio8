import torch
import torch.nn
import torch.nn.functional
from eight_mile.utils import Offsets
from typing import Dict, Optional, Callable
import numpy as np
from collections import defaultdict, Counter

def logits2text(logits, vocab):
    chars = ''
    last_ltr = ''
    eow = '|'
    for l in logits:
        if l in [Offsets.PAD, Offsets.EOS]:
            continue

        lower = vocab[l].lower()
        if lower == eow:
            lower = ' '
        if lower != last_ltr:
            last_ltr = lower
            if last_ltr != '<s>':
               chars += last_ltr

    return chars


def prefix_beam_search(logits: np.ndarray, vocab: Dict[int, str],
                       k: int = 10,
                       min_thresh: float = 0.0001,
                       decoder_blank: str = '<s>',
                       decoder_eow: str = '|',
                       decoder_eos: str = '</s>'):
    """Use a prefix beam search (https://arxiv.org/pdf/1408.2873.pdf) to decode

    The implementation here is "Algorithm 1" from the paper, and also partly based on the excellent article here:
    https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306

    :param logits: The output of a single utterance, of shape ``[T, C]``
    :param vocab: A mapping from the integer indices of ``C`` to graphemes
    :param p_lm: An optional language model, defaults to ``None``
    :param k:
    :param alpha:
    :param beta:
    :param min_thresh:
    :return:
    """
    #p_lm = p_lm if p_lm is not None else lambda x: 1
    p_non_blank = defaultdict(Counter)
    p_blank = defaultdict(Counter)
    eos = '.'
    eow = ' '
    A_prev = ['']
    A_next = ['']
    T = logits.shape[0]
    blank_idx = 0
    p_blank[0][''] = 1
    p_non_blank[0][''] = 0

    for t in range(1, T):
        chars_above_thresh = np.where(logits[t] > min_thresh)[0]
        for hyp in A_prev:
            # If we hit the end of a sentence, already we need to propagate the probability through
            if len(hyp) > 0 and hyp[-1] == eos:
                p_blank[t][hyp] += p_blank[t-1][hyp]
                p_non_blank[t][hyp] += p_non_blank[t-1][hyp]
                A_next.append(hyp)
                continue

            p_at_t = logits[t]

            for c in chars_above_thresh:

                v = vocab[c]

                if v == decoder_blank:
                    p_blank[t][hyp] += p_at_t[blank_idx] * (p_blank[t-1][hyp] + p_non_blank[t-1][hyp])
                    A_next.append(hyp)

                else:
                    v = v.replace(decoder_eos, eos).replace(decoder_eow, eow)
                    hyp_next = hyp + v

                    if v == eos:
                        # rewrite eos to our internal repr
                        hyp_next = hyp + eos
                        p_non_blank[t][hyp_next] += p_at_t[c] * p_blank[t - 1][hyp]
                        p_non_blank[t][hyp] += p_at_t[c] * p_blank[t-1][hyp]
                    elif len(hyp) > 0 and v == hyp[-1]:
                        p_non_blank[t][hyp_next] += p_at_t[c] * p_blank[t - 1][hyp]
                        p_non_blank[t][hyp] += p_at_t[c] * p_non_blank[t-1][hyp]
                    elif v == eow:
                        # TODO: add LM here
                        p_non_blank[t][hyp_next] += p_at_t[c] * (p_blank[t - 1][hyp] + p_non_blank[t - 1][hyp])
                    else:
                        p_non_blank[t][hyp_next] += p_at_t[c] * (p_blank[t - 1][hyp] + p_non_blank[t - 1][hyp])
                    if hyp_next not in A_prev:
                        p_blank[t][hyp_next] += p_at_t[blank_idx] * (p_blank[t - 1][hyp_next] + p_non_blank[t - 1][hyp_next])
                        p_non_blank[t][hyp_next] += p_at_t[c] * p_non_blank[t - 1][hyp_next]
                    A_next.append(hyp_next)

        A_next = sorted(A_next, key=lambda s: (p_non_blank[t][s] + p_blank[t][s]), reverse=True)
        A_prev = A_next[:k]
    return A_prev[0]

def postproc_letters(sentence):
    sentence = sentence.replace(" ", "").replace("|", " ").strip()
    return sentence


def ctc_metrics(lprobs_t, target, input_lengths, index2vocab):
    metrics = {}
    import editdistance
    BLANK_IDX = Offsets.GO
    with torch.no_grad():

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

        metrics["wv_errors"] = wv_errs
        metrics["w_errors"] = w_errs
        metrics["w_total"] = w_len
        metrics["c_errors"] = c_err
        metrics["c_total"] = c_len
    return metrics


class CTCLoss(torch.nn.Module):
    def __init__(self, zero_infinity=True):
        super().__init__()
        self.zero_infinity = zero_infinity

    def forward(self, log_prob, input_lengths, targets, target_lengths):
        pad_mask = (targets != Offsets.PAD) & (targets != Offsets.EOS)
        targets_flat = targets.masked_select(pad_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = torch.nn.functional.ctc_loss(
                log_prob,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=Offsets.GO,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
        return loss
