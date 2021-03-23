import numpy as np
from typing import Dict
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import Offsets
import torch
from eight_mile.pytorch.layers import (
    EmbeddingsStack,
    TransformerEncoderStack,
    TransformerDecoderStack,
    SingleHeadReduction,
    TwoHeadConcat,
    MeanPool1D,
    MaxPool1D,
    WeightTieDense,
    subsequent_mask,
    sequence_mask_mxlen,
)
import torch.nn as nn
import contextlib


def read_vocab_file(vocab_file: str):
    vocab = []
    for v in Offsets.VALUES:
        vocab.append(v)
    with open(vocab_file) as rf:
        for i, line in enumerate(rf):
            v = line.split()[0]
            vocab.append(v)
        return {v: i for i, v in enumerate(vocab)}


class TextVectorizer:
    def __init__(self, vocab: Dict[str, int], emit_begin_tok=[], emit_end_tok=[]):
        self.vocab = vocab
        self._emit_begin_tok = emit_begin_tok
        self._emit_end_tok = emit_end_tok

    def run(self, text) -> np.ndarray:
        """

        :param text:
        :return:
        """
        return np.array([self.vocab[w] for w in text], dtype=np.int)

    @property
    def emit_begin_tok(self):
        return self._emit_begin_tok

    @property
    def emit_end_tok(self):
        return self._emit_end_tok


class BPEVectorizer:
    def __init__(self, model_file, vocab_file, emit_begin_tok=[], emit_end_tok=[]):
        self.internal = BPEVectorizer1D(
            model_file=model_file,
            vocab_file=vocab_file,
            emit_begin_tok=emit_begin_tok,
            emit_end_tok=emit_end_tok,
            # transform_fn=str.lower,
        )

    @property
    def emit_begin_tok(self):
        return self.internal.emit_begin_tok

    @property
    def emit_end_tok(self):
        return self.internal.emit_begin_tok

    @property
    def vocab(self):
        return self.internal.vocab

    def run(self, text) -> np.ndarray:
        z = [x for x in self.internal._next_element(text, self.vocab)]
        return np.array(z, dtype=np.int)


class TextBoWPooledEncoder(nn.Module):
    def __init__(self, embeddings, reduction_type='mean'):
        super().__init__()
        self.embeddings = EmbeddingsStack({'x': embeddings})
        self.output_dim = self.embeddings.output_dim
        self.freeze = True
        self.pooler = MaxPool1D(self.output_dim) if reduction_type == 'max' else MeanPool1D(self.output_dim)

    def forward(self, query):
        (query, query_length) = query
        with torch.no_grad() if self.freeze else contextlib.ExitStack():
            embedded = self.embeddings({'x': query})
        return self.pooler((embedded, query_length))


class TextTransformerPooledEncoder(nn.Module):
    def __init__(
        self,
        embeddings,
        d_model,
        d_ff,
        dropout,
        num_heads,
        num_layers,
        d_k=None,
        rpr_k=None,
        reduction_d_k=64,
        reduction_type='SHA',
        ffn_pdrop=0.1,
        windowed_ra=False,
        rpr_value_on=False,
    ):
        super().__init__()
        self.embeddings = EmbeddingsStack({'x': embeddings})
        self.transformer = TransformerEncoderStack(
            num_heads=num_heads,
            d_model=d_model,
            pdrop=dropout,
            layers=num_layers,
            activation='gelu',
            d_ff=d_ff,
            ffn_pdrop=ffn_pdrop,
            d_k=d_k,
            rpr_k=rpr_k,
            windowed_ra=windowed_ra,
            rpr_value_on=rpr_value_on,
        )
        self.output_dim = d_model
        reduction_type = reduction_type.lower()
        if reduction_type == "2ha":
            self.reduction_layer = nn.Sequential(
                TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k), nn.Linear(2 * d_model, d_model)
            )
        elif reduction_type == "2ha_mean":
            self.reduction_layer = nn.Sequential(
                TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k, pooling='mean'),
                nn.Linear(2 * d_model, d_model),
            )
        elif reduction_type == "2ha_max":
            self.reduction_layer = nn.Sequential(
                TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k, pooling='max'),
                nn.Linear(2 * d_model, d_model),
            )
        elif reduction_type == "sha":
            self.reduction_layer = SingleHeadReduction(d_model, dropout, scale=False, d_k=reduction_d_k)
        elif reduction_type == "sha_mean":
            self.reduction_layer = SingleHeadReduction(d_model, dropout, scale=False, d_k=reduction_d_k, pooling='mean')
        elif reduction_type == "sha_max":
            self.reduction_layer = SingleHeadReduction(d_model, dropout, scale=False, d_k=reduction_d_k, pooling='max')
        elif reduction_type == 'max':
            self.reduction_layer = MaxPool1D(d_model)
        elif reduction_type == 'mean':
            self.reduction_layer = MeanPool1D(d_model)
        else:
            raise Exception("Unknown exception type")
        self.freeze = True

    def forward(self, query):
        (query, query_lengths) = query
        att_mask = sequence_mask_mxlen(query_lengths, query.shape[1]).to(query.device)
        with torch.no_grad() if self.freeze else contextlib.ExitStack():
            embedded = self.embeddings({'x': query})
            encoded = self.transformer((embedded, att_mask.unsqueeze(1).unsqueeze(1)))

        if isinstance(self.reduction_layer, MaxPool1D) or isinstance(self.reduction_layer, MeanPool1D):
            lengths = att_mask.sum(-1)
            encoded_reduced = self.reduction_layer((encoded, lengths))
        else:
            encoded_reduced = self.reduction_layer((encoded, encoded, encoded, att_mask.unsqueeze(1).unsqueeze(1)))
        return encoded_reduced


class TextTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        tgt_embeddings,
        dropout=0.1,
        num_layers=2,
        hsz=768,
        num_heads=4,
        scale=True,
        layer_drop=0.0,
        activation='gelu',
        **kwargs,
    ):
        super().__init__()
        self.tgt_embeddings = tgt_embeddings
        dsz = self.tgt_embeddings.get_dsz()
        if hsz is None:
            hsz = dsz

        d_ff = int(kwargs.get('d_ff', 4 * hsz))

        self.transformer = TransformerDecoderStack(
            num_heads,
            d_model=hsz,
            d_ff=d_ff,
            pdrop=dropout,
            scale=scale,
            layers=num_layers,
            layer_drop=layer_drop,
            activation_type=activation,
        )

        self.preds = WeightTieDense(self.tgt_embeddings)

    def forward(self, encoder_output, src_mask, dst, dst_mask):
        embed_out_bth = self.tgt_embeddings(dst)
        context_bth = encoder_output
        T = embed_out_bth.shape[1]
        dst_attn_mask = subsequent_mask(T)
        dst_attn_mask = (dst_attn_mask & dst_mask.unsqueeze(1).unsqueeze(1).type_as(dst_attn_mask)).type_as(
            embed_out_bth
        )
        src_mask = src_mask.unsqueeze(1).unsqueeze(1)
        output = self.transformer((embed_out_bth, context_bth, src_mask, dst_attn_mask))
        prob = self.output(output)
        return prob

    def output(self, x):
        pred = torch.log_softmax(self.preds(x.view(x.size(0) * x.size(1), -1)), dim=-1)
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred
