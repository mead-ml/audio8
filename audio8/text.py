import numpy as np
from typing import Dict
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import Offsets
from eight_mile.pytorch.layers import EmbeddingsStack, TransformerEncoderStack, SingleHeadReduction, TwoHeadConcat, sequence_mask_mxlen
import torch.nn as nn

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
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab

    def run(self, text) -> np.ndarray:
        """

        :param text:
        :return:
        """
        return np.array([self.vocab[w] for w in text], dtype=np.int)

class BPEVectorizer:
    def __init__(self, model_file, vocab_file, emit_begin_tok=[], emit_end_tok=[]):
        self.internal = BPEVectorizer1D(model_file=model_file, vocab_file=vocab_file,
                              emit_begin_tok=emit_begin_tok, emit_end_tok=emit_end_tok, transform_fn=str.lower)

    @property
    def vocab(self):
        return self.internal.vocab

    def run(self, text) -> np.ndarray:
        z = [x for x in self.internal._next_element(text, self.vocab)]
        return np.array(z, dtype=np.int)


class TextTransformerPooledEncoder(nn.Module):
    def __init__(self,
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
                 rpr_value_on=False):
        super().__init__()
        self.embeddings = EmbeddingsStack({'x': embeddings})
        self.transformer = TransformerEncoderStack(num_heads=num_heads, d_model=d_model,
                                                   pdrop=dropout, layers=num_layers, activation='gelu', d_ff=d_ff,
                                                   ffn_pdrop=ffn_pdrop,
                                                   d_k=d_k, rpr_k=rpr_k, windowed_ra=windowed_ra, rpr_value_on=rpr_value_on)
        self.output_dim = d_model
        if reduction_type == "2HA":
            self.reduction_layer = nn.Sequential(TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k),
                                                   nn.Linear(2*d_model, d_model))
        elif reduction_type == "SHA":
            self.reduction_layer = SingleHeadReduction(d_model, dropout, scale=False, d_k=reduction_d_k)
        else:
            raise Exception("Unknown exception type")

    def forward(self, query):
        (query, query_lengths) = query
        #query_mask = (query != Offsets.PAD)
        att_mask = sequence_mask_mxlen(query_lengths, query.shape[1]).to(query.device)
        embedded = self.embeddings({'x': query})
        att_mask = att_mask.unsqueeze(1).unsqueeze(1)
        encoded_query = self.transformer((embedded, att_mask))

        encoded_query = self.reduction_layer((encoded_query, encoded_query, encoded_query, att_mask))
        return encoded_query
