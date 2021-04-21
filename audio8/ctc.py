import torch
import torch.nn
import re
import torch.nn.functional
from eight_mile.utils import Offsets
from typing import Dict, Optional, Callable
import numpy as np
from collections import defaultdict, Counter


class PrefixBeamSearch:
    def __init__(self, vocab_list, alpha=0.2, beta=5.0, beam: int = 100, lm_file: Optional[str] = None):
        from ctcdecode import CTCBeamDecoder

        self.vocab_list = [v for v in vocab_list]
        self.use_bar = False
        self.bar_off = self.vocab_list.index('|')
        if self.bar_off >= 0:
            self.use_bar = True
            self.vocab_list[self.bar_off] = ' '
        self.beam = beam
        self.ctc_decoder = CTCBeamDecoder(
            labels=self.vocab_list,
            model_path=lm_file,
            alpha=alpha,
            beta=beta,
            beam_width=beam,
            blank_id=Offsets.GO,
            log_probs_input=True,
        )

    def run(self, log_probs: np.ndarray, n_best=None, return_ids=False):
        """Return n_best results from prefix beam decode.  If the n_best=1, then we will collapse the singleton dim

        :param log_probs: The log probabilities
        :param n_best: The number of results to return, defaults to beam size
        :param return_ids: Whether to return the raw ids or the characters
        :return: A list of transcriptions if 1 best, otherwise A list of n-bests of transcriptions
        """
        B = log_probs.shape[0]
        if n_best == None:
            n_best = self.beam
        beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(log_probs)

        def transform_ids(t):
            return t if return_ids else (self.vocab_list[t] if t != self.bar_off else '|')

        transcriptions = []
        if n_best == 1:
            for b in range(B):
                transcription = [transform_ids(t) for t in beam_results[b][0][: out_lens[b][0]]]
                transcriptions.append(transcription)
        else:
            for b in range(B):
                n_bests = []
                for n in range(n_best):
                    n_best = [transform_ids(t) for t in beam_results[b][n][: out_lens[b][n]]]
                    n_bests.append(n_best)
                transcriptions.append(n_bests)
        return transcriptions


def postproc_letters(sentence):
    sentence = ''.join(sentence)
    sentence = sentence.replace(" ", "").replace("|", " ").strip()
    return sentence


def postproc_bpe(sentence):
    sentence = ' '.join(sentence)
    sentence = sentence.replace("@@ ", "").strip()
    return sentence


def decode_text_wer(pred_units, t, index2vocab, postproc_fn=postproc_letters):
    import editdistance

    with torch.no_grad():
        w_errs = 0
        w_len = 0
        p = (t != Offsets.PAD) & (t != Offsets.EOS)
        targ = t[p]
        targ_units = [index2vocab[x.item()] for x in targ]
        targ_words = postproc_fn(targ_units).split()
        pred_words_raw = postproc_fn(pred_units).split()
        dist = editdistance.eval(pred_words_raw, targ_words)
        w_errs += dist
        w_len += len(targ_words)
    return w_errs, w_len


def decode_metrics(decoded, target, input_lengths, index2vocab, postproc_fn=postproc_letters):
    metrics = {}
    import editdistance

    BLANK_IDX = Offsets.GO
    with torch.no_grad():

        c_err = 0
        c_len = 0
        w_errs = 0
        w_len = 0
        wv_errs = 0
        for dp, t, inp_l in zip(
            decoded,
            target,
            input_lengths,
        ):
            dp = dp[:inp_l].unsqueeze(0)
            p = (t != Offsets.PAD) & (t != Offsets.EOS)
            targ = t[p]
            targ_units_arr = targ.tolist()
            toks = dp.unique_consecutive()
            pred_units_arr = toks[toks != BLANK_IDX].tolist()

            c_err += editdistance.eval(pred_units_arr, targ_units_arr)
            c_len += len(targ_units_arr)

            targ_units = [index2vocab[x.item()] for x in targ]
            targ_words = postproc_fn(targ_units).split()

            pred_units = [index2vocab[x] for x in pred_units_arr]
            pred_words_raw = postproc_fn(pred_units).split()

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


def ctc_metrics(lprobs_t, target, input_lengths, index2vocab, postproc_fn=postproc_letters):
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
            p = (t != Offsets.PAD) & (t != Offsets.EOS)
            targ = t[p]
            targ_units_arr = targ.tolist()

            toks = lp.argmax(dim=-1).unique_consecutive()
            pred_units_arr = toks[toks != BLANK_IDX].tolist()

            c_err += editdistance.eval(pred_units_arr, targ_units_arr)
            c_len += len(targ_units_arr)
            targ_units = [index2vocab[x.item()] for x in targ]
            targ_words = postproc_fn(targ_units).split()

            pred_units = [index2vocab[x] for x in pred_units_arr]
            pred_words_raw = postproc_fn(pred_units).split()

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
    def __init__(self, zero_infinity=True, reduction_type="sum"):
        super().__init__()
        self.zero_infinity = zero_infinity
        self.reduction_type = reduction_type

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
                reduction=self.reduction_type,
                zero_infinity=self.zero_infinity,
            )
        return loss
