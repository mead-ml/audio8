from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from eight_mile.pytorch.layers import pytorch_conv1d, pytorch_linear, Conv1DSame, TransformerEncoderStack, Dense
import contextlib
from collections import namedtuple
CONV_FEATURES = {16: [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
                 8: [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]}

START_TEMP = 2
END_TEMP = 0.5
TEMP_DECAY_FACTOR = 0.999995
XE_WGT = 0.1
DIVERSITY_WGT = 10





W2V_CTC_NESTED_MAP = {
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.k_proj.weight':     'encoder.encoder.transformer.encoders.{}.self_attn.w_K.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.k_proj.bias':       'encoder.encoder.transformer.encoders.{}.self_attn.w_K.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.v_proj.weight':     'encoder.encoder.transformer.encoders.{}.self_attn.w_V.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.v_proj.bias':       'encoder.encoder.transformer.encoders.{}.self_attn.w_V.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.q_proj.weight':     'encoder.encoder.transformer.encoders.{}.self_attn.w_Q.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.q_proj.bias':       'encoder.encoder.transformer.encoders.{}.self_attn.w_Q.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.out_proj.weight':   'encoder.encoder.transformer.encoders.{}.self_attn.w_O.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.out_proj.bias':     'encoder.encoder.transformer.encoders.{}.self_attn.w_O.layer.bias',
    # Wav2vec2 ref impl is run with LN first
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn_layer_norm.weight': 'encoder.encoder.transformer.encoders.{}.ln2.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn_layer_norm.bias':   'encoder.encoder.transformer.encoders.{}.ln2.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc1.weight': 'encoder.encoder.transformer.encoders.{}.ffn.0.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc1.bias':   'encoder.encoder.transformer.encoders.{}.ffn.0.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc2.weight': 'encoder.encoder.transformer.encoders.{}.ffn.3.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc2.bias':   'encoder.encoder.transformer.encoders.{}.ffn.3.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.final_layer_norm.weight':  'encoder.encoder.transformer.encoders.{}.ln1.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.final_layer_norm.bias':   'encoder.encoder.transformer.encoders.{}.ln1.bias'

}

# We use a primitive from 8mi called Dense which owns the linear as a sub-layer, so convert those
W2V2_CTC_FLAT_MAP = {
    'w2v_encoder.w2v_model.post_extract_proj.weight': 'encoder.proj_to_input.layer.weight',
    'w2v_encoder.w2v_model.post_extract_proj.bias': 'encoder.proj_to_input.layer.bias',
    #'w2v_encoder.w2v_model.project_q.weight': 'project_q.layer.weight',
    #'w2v_encoder.w2v_model.project_q.bias': 'project_q.layer.bias',
    #'w2v_encoder.w2v_model.final_proj.weight': 'final_proj.layer.weight',
    #'w2v_encoder.w2v_model.final_proj.bias': 'final_proj.layer.bias',
    'w2v_encoder.w2v_model.encoder.layer_norm.weight': 'encoder.encoder.ln.weight',
    'w2v_encoder.w2v_model.encoder.layer_norm.bias': 'encoder.encoder.ln.bias',
    'w2v_encoder.w2v_model.encoder.pos_conv.0.bias': 'encoder.encoder.pos_conv.conv.1.bias',
    'w2v_encoder.w2v_model.encoder.pos_conv.0.weight_g': 'encoder.encoder.pos_conv.conv.1.weight_g',
    'w2v_encoder.w2v_model.encoder.pos_conv.0.weight_v': 'encoder.encoder.pos_conv.conv.1.weight_v',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.0.0.weight': 'encoder.feature_extractor.conv_layers.0.0.weight',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.0.2.weight': 'encoder.feature_extractor.conv_layers.0.2.weight',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.0.2.bias': 'encoder.feature_extractor.conv_layers.0.2.bias',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.1.0.weight': 'encoder.feature_extractor.conv_layers.1.0.weight',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.2.0.weight': 'encoder.feature_extractor.conv_layers.2.0.weight',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.3.0.weight': 'encoder.feature_extractor.conv_layers.3.0.weight',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.4.0.weight': 'encoder.feature_extractor.conv_layers.4.0.weight',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.5.0.weight': 'encoder.feature_extractor.conv_layers.5.0.weight',
    'w2v_encoder.w2v_model.feature_extractor.conv_layers.6.0.weight': 'encoder.feature_extractor.conv_layers.6.0.weight',
    'w2v_encoder.w2v_model.mask_emb': 'encoder.mask_emb',
    'w2v_encoder.w2v_model.layer_norm.weight': 'encoder.layer_norm.weight',
    'w2v_encoder.w2v_model.layer_norm.bias': 'encoder.layer_norm.bias',
    'w2v_encoder.proj.weight': 'proj.weight',
    'w2v_encoder.proj.bias': 'proj.bias'
    #'layer_norm.weight': 'encoder.ln.weight',
    #'layer_norm.bias': 'encoder.ln.bias'
}


CheckpointMapping = namedtuple('CheckpointMapping', ['nested', 'flat'])

W2V_NESTED_MAP = {
        'encoder.layers.{}.self_attn.k_proj.weight':     'encoder.transformer.encoders.{}.self_attn.w_K.layer.weight',
        'encoder.layers.{}.self_attn.k_proj.bias':       'encoder.transformer.encoders.{}.self_attn.w_K.layer.bias',
        'encoder.layers.{}.self_attn.v_proj.weight':     'encoder.transformer.encoders.{}.self_attn.w_V.layer.weight',
        'encoder.layers.{}.self_attn.v_proj.bias':       'encoder.transformer.encoders.{}.self_attn.w_V.layer.bias',
        'encoder.layers.{}.self_attn.q_proj.weight':     'encoder.transformer.encoders.{}.self_attn.w_Q.layer.weight',
        'encoder.layers.{}.self_attn.q_proj.bias':       'encoder.transformer.encoders.{}.self_attn.w_Q.layer.bias',
        'encoder.layers.{}.self_attn.out_proj.weight':   'encoder.transformer.encoders.{}.self_attn.w_O.layer.weight',
        'encoder.layers.{}.self_attn.out_proj.bias':     'encoder.transformer.encoders.{}.self_attn.w_O.layer.bias',
        # Wav2vec2 ref impl is run with LN first
        'encoder.layers.{}.self_attn_layer_norm.weight': 'encoder.transformer.encoders.{}.ln2.weight',
        'encoder.layers.{}.self_attn_layer_norm.bias':   'encoder.transformer.encoders.{}.ln2.bias',
        'encoder.layers.{}.fc1.weight': 'encoder.transformer.encoders.{}.ffn.0.layer.weight',
        'encoder.layers.{}.fc1.bias':   'encoder.transformer.encoders.{}.ffn.0.layer.bias',
        'encoder.layers.{}.fc2.weight': 'encoder.transformer.encoders.{}.ffn.3.layer.weight',
        'encoder.layers.{}.fc2.bias':   'encoder.transformer.encoders.{}.ffn.3.layer.bias',
        'encoder.layers.{}.final_layer_norm.weight':  'encoder.transformer.encoders.{}.ln1.weight',
        'encoder.layers.{}.final_layer_norm.bias':   'encoder.transformer.encoders.{}.ln1.bias'

    }


# We use a primitive from 8mi called Dense which owns the linear as a sub-layer, so convert those
W2V2_FLAT_MAP = {
    'post_extract_proj.weight': 'proj_to_input.layer.weight',
    'post_extract_proj.bias': 'proj_to_input.layer.bias',
    'project_q.weight': 'project_q.layer.weight',
    'project_q.bias': 'project_q.layer.bias',
    'final_proj.weight': 'final_proj.layer.weight',
    'final_proj.bias': 'final_proj.layer.bias',
    'encoder.layer_norm.weight': 'encoder.ln.weight',
    'encoder.layer_norm.bias': 'encoder.ln.bias',
    'encoder.pos_conv.0.bias': 'encoder.pos_conv.conv.1.bias',
    'encoder.pos_conv.0.weight_g': 'encoder.pos_conv.conv.1.weight_g',
    'encoder.pos_conv.0.weight_v': 'encoder.pos_conv.conv.1.weight_v',
    #'layer_norm.weight': 'encoder.ln.weight',
    #'layer_norm.bias': 'encoder.ln.bias'
}

W2V_MAP = CheckpointMapping(nested=W2V_NESTED_MAP, flat=W2V2_FLAT_MAP)
W2V_CTC_MAP = CheckpointMapping(nested=W2V_CTC_NESTED_MAP, flat=W2V2_CTC_FLAT_MAP)



def convert_keys(num_layers: int, d: Dict, nested_layer_map: Dict = W2V_MAP, flat_map: Dict = W2V2_FLAT_MAP) -> Dict:

    m = {}
    for i in range(num_layers):
        for k, v in nested_layer_map.items():
            key = k.format(i)
            m[v.format(i)] = d.pop(key)

    for k, v in flat_map.items():
        m[v] = d.pop(k)

    for k, v in d.items():
        m[k] = v

    return m

def load_fairseq_bin(w2v: nn.Module, bin_file: str, ctc: bool=False):

    if ctc:
        checkpoint_mapping = W2V_CTC_MAP
        transformer = w2v.encoder.encoder.transformer
    else:
        checkpoint_mapping = W2V_MAP
        transformer = w2v.encoder.transformer

    d = torch.load(bin_file)["model"]

    num_layers = len(transformer.encoders)
    mapped_keys = convert_keys(num_layers, d, checkpoint_mapping.nested, checkpoint_mapping.flat)
    #for k in mapped_keys.keys():
    #    if 'attn' in k:
    #        t = mapped_keys[k].T
    #        mapped_keys[k] = t
    unknown_keys = w2v.load_state_dict(mapped_keys, strict=False)
    missing_keys = [key for key in unknown_keys.missing_keys]
    return {'missing': missing_keys, 'unexpected': unknown_keys.unexpected_keys}


def timestep_masking(
        shape: Tuple[int, int],
        p_start: float = 0.65,
        mask_length: int = 10
) -> np.ndarray:
    bsz, input_length = shape
    mask = np.full((bsz, input_length), False)
    num_mask = int(p_start * input_length / float(mask_length) + np.random.rand())
    mask_idcs = []
    for i in range(bsz):
        sz = input_length
        lengths = np.full(num_mask, mask_length)
        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

        mask_idc = np.asarray(
            [
                mask_idc[j] + offset
                for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ]
        )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    ls = [len(m) for m in mask_idcs]
    min_len = min(ls)
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


def create_model(sample_rate: int, num_vq_vars, num_vq_groups, d_model, num_heads, num_layers, dropout, d_ff):
    model = Wav2Vec2Model(CONV_FEATURES[sample_rate], num_vq_vars,
                          START_TEMP, END_TEMP, TEMP_DECAY_FACTOR, num_vq_groups, d_model,
                          num_heads, num_layers,
                          dropout, d_ff)
    return model


def create_acoustic_model(num_labels, sample_rate, d_model, num_heads, num_layers, dropout, d_ff):
    model = Wav2Vec2AcousticModel(num_labels, CONV_FEATURES[sample_rate],
                                  d_model,
                                  num_heads, num_layers,
                                  dropout, d_ff)
    return model


class Wav2Vec2Loss(nn.Module):
    def __init__(self, n_vars, n_negatives=100):
        super().__init__()
        self.n_vars = n_vars
        self.sample = Sampler(n_negatives)

    def __call__(self, model, features):
        outputs, latents, gs_probs, time_mask = model(features)
        y = latents.unsqueeze(0)
        outputs_shape = outputs.shape
        outputs = outputs[time_mask.unsqueeze(-1).expand_as(outputs)].view(outputs_shape[0], -1, outputs_shape[-1])
        outputs = outputs.unsqueeze(0)
        neg, _ = self.sample.negatives(latents)
        targets = torch.cat([y, neg], dim=0)
        logits = torch.cosine_similarity(outputs, targets, dim=-1)
        logits = logits.transpose(2, 0)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        diversity = DIVERSITY_WGT * (self.n_vars - gs_probs) / self.n_vars
        xe_loss = F.cross_entropy(logits, targets)
        cross_entropy = XE_WGT * xe_loss
        return cross_entropy + diversity

def create_loss(n_vars, n_negatives):
    return Wav2Vec2Loss(n_vars, n_negatives)

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            conv_bias: bool = False,
    ):
        super().__init__()

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_group_norm=False,
                conv_bias=False,
        ):

            if is_group_norm:
                return nn.Sequential(
                    pytorch_conv1d(n_in, n_out, k, initializer="kaiming", stride=stride, bias=conv_bias),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(
                    pytorch_conv1d(n_in, n_out, k, initializer="kaiming", stride=stride, bias=conv_bias),
                    nn.Dropout(p=dropout),
                    nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_group_norm=i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        # BxCxT -> BxTxC
        #x = x.transpose(1, 2)
        return x

class GumbelVectorQuantizer(nn.Module):
    def __init__(
            self,
            dim,
            num_vars,
            min_temperature,
            max_temperature,
            temperature_decay,
            num_groups,
            vq_dim
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temperature: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            vq_dim: dimensionality of the resulting quantized vector
        """
        super().__init__()

        self.num_groups = num_groups
        self.input_dim = dim
        self.num_vars = num_vars

        assert (
                vq_dim % num_groups == 0
        ), f"dim {vq_dim} must be divisible by groups {num_groups} for concatenation"

        # per var
        var_dim = vq_dim // num_groups
        # vars count is the groups by the number of vars per group
        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        # projection
        self.weight_proj = nn.Linear(self.input_dim, num_groups * num_vars)
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.curr_temperature = self.max_temperature
        # Why dont they init this, I guess because its not necessarily used in training
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temperature = max(
            self.max_temperature * self.temperature_decay ** num_updates, self.min_temperature
        )

    # Create codebook on the fly
    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.num_groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            self.codebook_indices = self.codebook_indices.view(
                self.num_vars ** self.num_groups, -1
            )
            for b in range(1, self.num_groups):
                self.codebook_indices[:, b] += self.num_vars * b
            self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
                .index_select(0, indices)
                .view(self.num_vars ** self.num_groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.num_groups)
        cb_size = indices.size(0)
        assert (
                n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.num_groups):
            exponent = self.num_groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def targets_for(self, x):
        """Get the output of the gumbel softmax or hard estimator and convert to one-hots

        :param x: [B, T, GxV]
        :return: y [B, T, G]
        """
        bsz = x.shape[0]
        tsz = x.shape[1]
        x = x.view(bsz * tsz, -1)
        targets = x.view(bsz * tsz * self.num_groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
        return targets

    def forward(self, x):

        bsz, tsz, fsz = x.shape
        # This should NOT be required, PyTorch folds under the hood
        x = self.weight_proj(x)
        # The output back out is BxTx(GxV)
        x = x.view(bsz * tsz * self.num_groups, -1)
        avg_probs = torch.softmax(
            x.float(), dim=-1
        ).mean(dim=0)

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temperature, hard=True).type_as(x)
        else:
            # Max over vars
            _, k = x.max(-1)
            hard_x = (
                x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.num_groups, -1)
            )
            x = hard_x

        x = x.view(bsz * tsz, self.num_groups, -1)
        prob_ppl = torch.sum(torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), -1)
        ))

        # broadcast the quantization table
        # [B, T, (GxV), 1] *. [1, (GxV), qsz] = [B, T, (GxV), qsz]
        x = x.view(bsz * tsz, -1, 1)
        x = x * self.vars

        x = x.view(bsz * tsz, self.num_groups, self.num_vars, -1)
        # This collapses over the variables
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        return x, prob_ppl


class AudioTransformerEncoder(nn.Module):
    def __init__(
            self,
            num_heads: int,
            d_model: int,
            pdrop: float,
            layers: int = 1,
            activation: str = "gelu",
            d_ff: Optional[int] = None,
            conv_pos_kernel: int = 128,
            conv_groups: int = 16,
            **kwargs):
        super().__init__()
        self.d_model = d_model
        self.conv_pos_kernel = conv_pos_kernel
        self.conv_groups = conv_groups
        self.dropout = nn.Dropout(pdrop)

        std = math.sqrt((4 * (1.0 - pdrop)) / (self.conv_pos_kernel * self.d_model))
        self.pos_conv = Conv1DSame(d_model, d_model, self.conv_pos_kernel, activation="gelu",
                                   groups=self.conv_groups, unif=std, initializer="normal")
        self.pos_conv.conv[1] = nn.utils.weight_norm(self.pos_conv.conv[1], name="weight", dim=2)
        if not d_ff:
            d_ff = 4 * d_model

        self.transformer = TransformerEncoderStack(num_heads=num_heads,
                                                   d_model=d_model,
                                                   pdrop=pdrop,
                                                   layers=layers,
                                                   activation=activation,
                                                   layer_norms_after=True,
                                                   d_ff=d_ff)
        self.ln = nn.LayerNorm(self.d_model)

    def forward(self, x, pad_mask=None):
        x = self.extract_features(x, pad_mask)
        return x

    def extract_features(self, x, pad_mask=None):

        if pad_mask is not None:
            x[~pad_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)

        # x_conv = self.pos_conv(x)
        x += x_conv
        x = self.ln(x)
        x = self.dropout(x)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        x = self.transformer((x, pad_mask))

        return x


class Wav2Vec2Encoder(nn.Module):
    """This model is for use of wav2vec2 embeddings in downstream application

    Care must be taken to ensure that padding is handled correctly, but most of the pretraining complexities are not
    required here.  When this model is reloaded from a checkpoint, we do expect the VQ and projection keys to be missing




    """
    def __init__(self, conv_features, d_model, num_heads, num_layers, dropout=0.1, d_ff=None, dropout_input=0.1, dropout_features=0.1, do_timestep_masking=False):
        super().__init__()
        fx_dsz = conv_features[-1][0]
        self.layer_norm = torch.nn.LayerNorm(fx_dsz)
        self.dropout_input = torch.nn.Dropout(dropout_input)
        self.dropout_features = torch.nn.Dropout(dropout_features)

        self.feature_extractor = ConvFeatureExtractionModel(conv_features)
        self.proj_to_input = Dense(fx_dsz, d_model)
        self.encoder = AudioTransformerEncoder(num_heads, d_model, dropout, num_layers, d_ff=d_ff)
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(d_model).uniform_()
        )
        self.do_timestep_masking = do_timestep_masking

    def forward(self, x, pad_mask=None):
        fx = self.feature_extractor(x)
        fx = fx.transpose(1, 2)

        features = self.layer_norm(fx)

        if pad_mask is not None:
            extra = pad_mask.size(1) % features.size(1)
            if extra > 0:
                pad_mask = pad_mask[:, :-extra]
            pad_mask = pad_mask.view(pad_mask.size(0), features.size(1), -1)
            pad_mask = pad_mask.all(-1)

        B, T, _ = features.shape
        features = self.proj_to_input(features)

        if self.do_timestep_masking:
            features = self.dropout_input(features)
            time_mask = timestep_masking((B, T))
            time_mask = torch.from_numpy(time_mask).to(x.device)
            features[time_mask] = self.mask_emb
        x = self.encoder(features, pad_mask)
        return x, pad_mask.sum(-1)


class Wav2Vec2AcousticModel(nn.Module):
    def __init__(self, num_labels, conv_features, d_model, num_heads, num_layers, dropout=0.1, d_ff=None, dropout_input=0.1,
                 dropout_features=0.1):
        super().__init__()
        self.encoder = Wav2Vec2Encoder(conv_features, d_model, num_heads, num_layers, dropout, d_ff, dropout_input, dropout_features)
        self.proj = pytorch_linear(d_model, num_labels)
        self.freeze = True

    def forward(self, x, pad_mask=None):

        with torch.no_grad() if self.freeze else contextlib.ExitStack():
            encoded, valid_lengths = self.encoder(x, pad_mask)
        encoded = self.proj(encoded)
        return F.log_softmax(encoded, dim=-1), valid_lengths


class Wav2Vec2Model(nn.Module):
    """Raw pretraining model for wav2vec.  Assumes that we are not doing any padding

    The batches are picked by trimming the signals, which means that any complex code required to deal
    with padding is not required here, whereas in downstream, any of the code to do with VQ quantiziation
    is not required, keeping both models simple and focused on a single task

    """
    def __init__(self, conv_features, num_vq_vars, start_temp, end_temp, temp_decay_factor,
                 num_vq_groups, d_model, num_heads, num_layers, dropout=0.1, d_ff=None, final_dim=256,
                 dropout_input=0.1, dropout_features=0.1):
        super().__init__()
        fx_dsz = conv_features[-1][0]
        self.layer_norm = torch.nn.LayerNorm(fx_dsz)
        self.dropout_input = torch.nn.Dropout(dropout_input)
        self.dropout_features = torch.nn.Dropout(dropout_features)

        self.feature_extractor = ConvFeatureExtractionModel(conv_features)
        self.proj_to_input = Dense(fx_dsz, d_model)
        self.quantizer = GumbelVectorQuantizer(fx_dsz, num_vq_vars, start_temp, end_temp, temp_decay_factor,
                                               num_vq_groups, final_dim)
        self.encoder = AudioTransformerEncoder(num_heads, d_model, dropout, num_layers, d_ff=d_ff)
        self.project_q = Dense(final_dim, final_dim)
        self.final_proj = Dense(d_model, final_dim)
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(d_model).uniform_()
        )

    def set_num_updates(self, s):
        self.quantizer.set_num_updates(s)

    def forward(self, x):
        fx = self.feature_extractor(x)
        features = self.layer_norm(fx)
        unmasked_features = features.clone()
        features = self.proj_to_input(features)
        B, T, _ = unmasked_features.shape
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        time_mask = timestep_masking((B, T))
        time_mask = torch.from_numpy(time_mask).to(x.device)
        features[time_mask] = self.mask_emb

        y = unmasked_features[time_mask].view(
            unmasked_features.size(0), -1, unmasked_features.size(-1)
        )
        x = self.encoder(features)
        y, vq_probs = self.quantizer(y)

        y = self.project_q(y)
        x = self.final_proj(x)
        return x, y, vq_probs, time_mask


class Sampler:

    def __init__(self, n_negatives=100):
        self.n_negatives = n_negatives

    def negatives(self, y):
        B, T, C = y.shape
        y = y.view(-1, C)  # BTC => (BxT)C

        with torch.no_grad():
            Ts = torch.arange(T).unsqueeze(-1)
            Ts = Ts.expand(-1, self.n_negatives)
            Ts = Ts.reshape(-1)
            neg_idxs = np.random.randint(0, T - 1, (B, self.n_negatives * T))
            neg_idxs = torch.from_numpy(neg_idxs)
            neg_idxs[neg_idxs >= Ts] += 1
            stride = torch.arange(B) * T
            stride = stride.unsqueeze(-1)

        neg_idxs = neg_idxs + stride
        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            B, T, self.n_negatives, C
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs
