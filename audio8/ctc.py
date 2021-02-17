import torch
import torch.nn
import torch.nn.functional
from eight_mile.utils import Offsets


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
