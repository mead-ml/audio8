from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from eight_mile.utils import Offsets
from eight_mile.pytorch.serialize import load_tlm_npz
from eight_mile.pytorch.layers import (
    pytorch_conv1d,
    pytorch_linear,
    PassThru,
    Conv1DSame,
    TransformerEncoderStack,
    Dense,
    MaxPool1D,
    TwoHeadConcat,
    SingleHeadReduction,
    BasicDualEncoderModel,
    sequence_mask,
)
from audio8.text import TextBoWPooledEncoder, TextTransformerPooledEncoder
import contextlib
from collections import namedtuple

CONV_FEATURES = {
    16: [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
    8: [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
}

START_TEMP = 2
END_TEMP = 0.5
TEMP_DECAY_FACTOR = 0.999995
XE_WGT = 0.1
DIVERSITY_WGT = 10


# Transfer fairseq keys to audio8
W2V_CTC_NESTED_MAP = {
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.k_proj.weight': 'encoder.encoder.transformer.encoders.{}.self_attn.w_K.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.k_proj.bias': 'encoder.encoder.transformer.encoders.{}.self_attn.w_K.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.v_proj.weight': 'encoder.encoder.transformer.encoders.{}.self_attn.w_V.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.v_proj.bias': 'encoder.encoder.transformer.encoders.{}.self_attn.w_V.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.q_proj.weight': 'encoder.encoder.transformer.encoders.{}.self_attn.w_Q.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.q_proj.bias': 'encoder.encoder.transformer.encoders.{}.self_attn.w_Q.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.out_proj.weight': 'encoder.encoder.transformer.encoders.{}.self_attn.w_O.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn.out_proj.bias': 'encoder.encoder.transformer.encoders.{}.self_attn.w_O.layer.bias',
    # Wav2vec2 ref impl is run with LN first
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn_layer_norm.weight': 'encoder.encoder.transformer.encoders.{}.ln2.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.self_attn_layer_norm.bias': 'encoder.encoder.transformer.encoders.{}.ln2.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc1.weight': 'encoder.encoder.transformer.encoders.{}.ffn.0.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc1.bias': 'encoder.encoder.transformer.encoders.{}.ffn.0.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc2.weight': 'encoder.encoder.transformer.encoders.{}.ffn.3.layer.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.fc2.bias': 'encoder.encoder.transformer.encoders.{}.ffn.3.layer.bias',
    'w2v_encoder.w2v_model.encoder.layers.{}.final_layer_norm.weight': 'encoder.encoder.transformer.encoders.{}.ln1.weight',
    'w2v_encoder.w2v_model.encoder.layers.{}.final_layer_norm.bias': 'encoder.encoder.transformer.encoders.{}.ln1.bias',
}

# We use a primitive from 8mi called Dense which owns the linear as a sub-layer, so convert those
W2V2_CTC_FLAT_MAP_16 = {
    'w2v_encoder.w2v_model.post_extract_proj.weight': 'encoder.proj_to_input.layer.weight',
    'w2v_encoder.w2v_model.post_extract_proj.bias': 'encoder.proj_to_input.layer.bias',
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
    'w2v_encoder.proj.bias': 'proj.bias',
}
W2V2_CTC_FLAT_MAP_8 = {
    'w2v_encoder.w2v_model.post_extract_proj.weight': 'encoder.proj_to_input.layer.weight',
    'w2v_encoder.w2v_model.post_extract_proj.bias': 'encoder.proj_to_input.layer.bias',
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
    'w2v_encoder.w2v_model.mask_emb': 'encoder.mask_emb',
    'w2v_encoder.w2v_model.layer_norm.weight': 'encoder.layer_norm.weight',
    'w2v_encoder.w2v_model.layer_norm.bias': 'encoder.layer_norm.bias',
    'w2v_encoder.proj.weight': 'proj.weight',
    'w2v_encoder.proj.bias': 'proj.bias',
}


CheckpointMapping = namedtuple('CheckpointMapping', ['nested', 'flat'])

W2V_NESTED_MAP = {
    'encoder.layers.{}.self_attn.k_proj.weight': 'encoder.transformer.encoders.{}.self_attn.w_K.layer.weight',
    'encoder.layers.{}.self_attn.k_proj.bias': 'encoder.transformer.encoders.{}.self_attn.w_K.layer.bias',
    'encoder.layers.{}.self_attn.v_proj.weight': 'encoder.transformer.encoders.{}.self_attn.w_V.layer.weight',
    'encoder.layers.{}.self_attn.v_proj.bias': 'encoder.transformer.encoders.{}.self_attn.w_V.layer.bias',
    'encoder.layers.{}.self_attn.q_proj.weight': 'encoder.transformer.encoders.{}.self_attn.w_Q.layer.weight',
    'encoder.layers.{}.self_attn.q_proj.bias': 'encoder.transformer.encoders.{}.self_attn.w_Q.layer.bias',
    'encoder.layers.{}.self_attn.out_proj.weight': 'encoder.transformer.encoders.{}.self_attn.w_O.layer.weight',
    'encoder.layers.{}.self_attn.out_proj.bias': 'encoder.transformer.encoders.{}.self_attn.w_O.layer.bias',
    # Wav2vec2 ref impl is run with LN first
    'encoder.layers.{}.self_attn_layer_norm.weight': 'encoder.transformer.encoders.{}.ln2.weight',
    'encoder.layers.{}.self_attn_layer_norm.bias': 'encoder.transformer.encoders.{}.ln2.bias',
    'encoder.layers.{}.fc1.weight': 'encoder.transformer.encoders.{}.ffn.0.layer.weight',
    'encoder.layers.{}.fc1.bias': 'encoder.transformer.encoders.{}.ffn.0.layer.bias',
    'encoder.layers.{}.fc2.weight': 'encoder.transformer.encoders.{}.ffn.3.layer.weight',
    'encoder.layers.{}.fc2.bias': 'encoder.transformer.encoders.{}.ffn.3.layer.bias',
    'encoder.layers.{}.final_layer_norm.weight': 'encoder.transformer.encoders.{}.ln1.weight',
    'encoder.layers.{}.final_layer_norm.bias': 'encoder.transformer.encoders.{}.ln1.bias',
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
W2V_CTC_MAP_16 = CheckpointMapping(nested=W2V_CTC_NESTED_MAP, flat=W2V2_CTC_FLAT_MAP_16)
W2V_CTC_MAP_8 = CheckpointMapping(nested=W2V_CTC_NESTED_MAP, flat=W2V2_CTC_FLAT_MAP_8)

W2V_CTC_MAP = {8: W2V_CTC_MAP_8, 16: W2V_CTC_MAP_16}


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


def load_fairseq_bin(w2v: nn.Module, bin_file: str, ctc: bool = False, sr: int = 16):

    if ctc:
        checkpoint_mapping = W2V_CTC_MAP[sr]
        transformer = w2v.encoder.encoder.transformer
    else:
        checkpoint_mapping = W2V_MAP
        transformer = w2v.encoder.transformer

    d = torch.load(bin_file)["model"]

    num_layers = len(transformer.encoders)
    mapped_keys = convert_keys(num_layers, d, checkpoint_mapping.nested, checkpoint_mapping.flat)
    unknown_keys = w2v.load_state_dict(mapped_keys, strict=False)
    missing_keys = [key for key in unknown_keys.missing_keys]
    return {'missing': missing_keys, 'unexpected': unknown_keys.unexpected_keys}


def create_mask(shape: Tuple[int, int], p_start: float = 0.65, mask_length: int = 10) -> np.ndarray:
    bsz, input_length = shape
    mask = np.full((bsz, input_length), False)
    num_mask = int(p_start * input_length / float(mask_length) + np.random.rand())
    if num_mask == 0:
        return mask
    mask_idcs = []
    for i in range(bsz):
        sz = input_length
        lengths = np.full(num_mask, mask_length)
        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    ls = [len(m) for m in mask_idcs]
    min_len = min(ls)
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


def create_model(
    sample_rate=16,
    num_vq_vars=320,
    num_vq_groups=2,
    d_model=768,
    num_heads=12,
    num_layers=12,
    dropout=0.1,
    d_ff=None,
    final_dim=256,
    dropout_input=0.1,
    dropout_features=0.1,
    timestep_masking=0.65,
    channel_masking=0.0,
    timestep_mask_len=10,
    channel_mask_len=64,
    layer_drop=0.0,
    **kwargs,
):
    model = Wav2Vec2Model(
        CONV_FEATURES[sample_rate],
        num_vq_vars,
        START_TEMP,
        END_TEMP,
        TEMP_DECAY_FACTOR,
        num_vq_groups,
        d_model,
        num_heads,
        num_layers,
        dropout,
        d_ff,
        final_dim,
        dropout_input,
        dropout_features,
        timestep_masking,
        channel_masking,
        timestep_mask_len,
        channel_mask_len,
        layer_drop,
    )
    return model


def create_acoustic_model(
    num_labels,
    sample_rate=16,
    d_model=768,
    num_heads=12,
    num_layers=12,
    dropout=0.1,
    d_ff=None,
    dropout_input=0.0,
    timestep_masking=0.5,
    channel_masking=0.1,
    timestep_mask_len=10,
    channel_mask_len=64,
    layer_drop=0.0,
    freeze_fx=True,
    **kwargs,
):
    model = Wav2Vec2AcousticModel(
        num_labels,
        CONV_FEATURES[sample_rate],
        d_model,
        num_heads,
        num_layers,
        dropout,
        d_ff,
        dropout_input,
        0.0,
        timestep_masking,
        channel_masking,
        timestep_mask_len,
        channel_mask_len,
        layer_drop,
        freeze_fx,
    )
    return model


def create_paired_model(
    embeddings,
    target_sample_rate=16,
    audio_d_model=768,
    audio_num_heads=12,
    audio_num_layers=12,
    audio_dropout=0.1,
    audio_d_ff=3072,
    audio_reduction_type='max',
    audio_d_k=64,
    audio_dropout_input=0.0,
    audio_timestep_masking=0.5,
    audio_channel_masking=0.1,
    audio_timestep_mask_len=10,
    audio_channel_mask_len=64,
    audio_layer_drop=0.0,
    text_d_model=512,
    text_num_heads=8,
    text_num_layers=8,
    text_dropout=0.1,
    text_d_ff=2048,
    text_rpr_k=8,
    text_reduction_type='max',
    text_d_k=64,
    stacking_layers=[],
    output_dim=256,
    text_encoder_type='transformer',
    warmstart_text=None,
    **kwargs,
):
    audio_sr = target_sample_rate // 1000
    audio_encoder = Wav2Vec2PooledEncoder(
        conv_features=CONV_FEATURES[audio_sr],
        d_model=audio_d_model,
        num_heads=audio_num_heads,
        num_layers=audio_num_layers,
        dropout=audio_dropout,
        d_ff=audio_d_ff,
        reduction_type=audio_reduction_type,
        reduction_d_k=audio_d_k,
        dropout_input=audio_dropout_input,
        timestep_masking=audio_timestep_masking,
        channel_masking=audio_channel_masking,
        timestep_mask_len=audio_timestep_mask_len,
        channel_mask_len=audio_channel_mask_len,
        layer_drop=audio_layer_drop,
    )

    if text_encoder_type == 'transformer':

        text_encoder = TextTransformerPooledEncoder(
            embeddings,
            d_model=text_d_model,
            d_ff=text_d_ff,
            dropout=text_dropout,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            reduction_d_k=text_d_k,
            rpr_k=text_rpr_k,
            rpr_value_on=False,
            reduction_type=text_reduction_type,
        )

        if warmstart_text:
            # Assume for now that its going to be an NPZ file
            load_tlm_npz(text_encoder, warmstart_text)
    else:
        text_encoder = TextBoWPooledEncoder(embeddings, reduction_type=text_reduction_type)
    de = BasicDualEncoderModel(audio_encoder, text_encoder, stacking_layers, output_dim)
    return de


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
                    nn.GELU(),
                )

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

        return x


class GumbelVectorQuantizer(nn.Module):
    def __init__(self, dim, num_vars, min_temperature, max_temperature, temperature_decay, num_groups, vq_dim):
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

        assert vq_dim % num_groups == 0, f"dim {vq_dim} must be divisible by groups {num_groups} for concatenation"

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
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temperature = max(self.max_temperature * self.temperature_decay ** num_updates, self.min_temperature)

    # Create codebook on the fly
    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.num_groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(inds, dtype=torch.long, device=self.vars.device).flatten()

            self.codebook_indices = self.codebook_indices.view(self.num_vars ** self.num_groups, -1)
            for b in range(1, self.num_groups):
                self.codebook_indices[:, b] += self.num_vars * b
            self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return self.vars.squeeze(0).index_select(0, indices).view(self.num_vars ** self.num_groups, -1)

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.num_groups)
        cb_size = indices.size(0)
        assert n < cb_size, f"sample size {n} is greater than size of codebook {cb_size}"
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
        avg_probs = torch.softmax(x.float(), dim=-1).mean(dim=0)

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temperature, hard=True).type_as(x)
        else:
            # Max over vars
            _, k = x.max(-1)
            hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.num_groups, -1)
            x = hard_x

        x = x.view(bsz * tsz, self.num_groups, -1)
        prob_ppl = torch.sum(torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), -1)))

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
        layer_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.conv_pos_kernel = conv_pos_kernel
        self.conv_groups = conv_groups
        self.dropout = nn.Dropout(pdrop)

        std = math.sqrt((4 * (1.0 - pdrop)) / (self.conv_pos_kernel * self.d_model))
        self.pos_conv = Conv1DSame(
            d_model,
            d_model,
            self.conv_pos_kernel,
            activation="gelu",
            groups=self.conv_groups,
            unif=std,
            initializer="normal",
        )
        self.pos_conv.conv[1] = nn.utils.weight_norm(self.pos_conv.conv[1], name="weight", dim=2)
        if not d_ff:
            d_ff = 4 * d_model

        self.transformer = TransformerEncoderStack(
            num_heads=num_heads,
            d_model=d_model,
            pdrop=pdrop,
            layers=layers,
            activation=activation,
            layer_norms_after=True,
            d_ff=d_ff,
            layer_drop=layer_drop,
        )
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
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        x = self.transformer((x, pad_mask))

        return x


class Wav2Vec2Encoder(nn.Module):
    """This model is for use of wav2vec2 embeddings in downstream application

    Care must be taken to ensure that padding is handled correctly, but most of the pretraining complexities are not
    required here.  When this model is reloaded from a checkpoint, we do expect the VQ and projection keys to be missing




    """

    def __init__(
        self,
        conv_features=CONV_FEATURES[16],
        d_model=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        d_ff=None,
        dropout_input=0.1,
        dropout_features=0.0,
        timestep_masking=0.5,
        channel_masking=0.1,
        timestep_mask_len=10,
        channel_mask_len=64,
        layer_drop=0.0,
        freeze_fx=True,
    ):
        super().__init__()
        fx_dsz = conv_features[-1][0]
        self.layer_norm = torch.nn.LayerNorm(fx_dsz)
        self.dropout_input = torch.nn.Dropout(dropout_input)
        self.dropout_features = torch.nn.Dropout(dropout_features)

        self.feature_extractor = ConvFeatureExtractionModel(conv_features)
        self.proj_to_input = Dense(fx_dsz, d_model)
        self.encoder = AudioTransformerEncoder(
            num_heads, d_model, dropout, num_layers, d_ff=d_ff, layer_drop=layer_drop
        )
        self.mask_emb = nn.Parameter(torch.FloatTensor(d_model).uniform_())
        self.timestep_masking = timestep_masking
        self.channel_masking = channel_masking
        self.timestep_mask_len = timestep_mask_len
        self.channel_mask_len = channel_mask_len
        self.output_dim = d_model
        self.freeze_fx = freeze_fx

    def forward(self, x, pad_mask=None):
        with torch.no_grad() if self.freeze_fx else contextlib.ExitStack():
            fx = self.feature_extractor(x)
        fx = fx.transpose(1, 2)

        features = self.layer_norm(fx)

        if pad_mask is not None:
            extra = pad_mask.size(1) % features.size(1)
            if extra > 0:
                pad_mask = pad_mask[:, :-extra]
            pad_mask = pad_mask.view(pad_mask.size(0), features.size(1), -1)
            pad_mask = pad_mask.all(-1)

        features = self.proj_to_input(features)
        B, T, C = features.shape

        features = self.dropout_input(features)
        if self.training and self.timestep_masking > 0.0:
            time_mask = create_mask((B, T), p_start=self.timestep_masking, mask_length=self.timestep_mask_len)
            time_mask = torch.from_numpy(time_mask).to(x.device)
            features[time_mask] = self.mask_emb
        if self.training and self.channel_masking > 0.0:
            channel_mask = create_mask((B, C), p_start=self.channel_masking, mask_length=self.channel_mask_len)
            channel_mask = torch.from_numpy(channel_mask).to(x.device).unsqueeze(1).expand(-1, T, -1)
            features[channel_mask] = 0
        x = self.encoder(features, pad_mask)
        return x, pad_mask


class Wav2Vec2AcousticModel(nn.Module):
    def __init__(
        self,
        num_labels,
        conv_features=CONV_FEATURES[16],
        d_model=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        d_ff=None,
        dropout_input=0.0,
        dropout_features=0.0,
        timestep_masking=0.5,
        channel_masking=0.1,
        timestep_mask_len=10,
        channel_mask_len=64,
        layer_drop=0.0,
        freeze_fx=True,
    ):
        super().__init__()
        self.encoder = Wav2Vec2Encoder(
            conv_features,
            d_model,
            num_heads,
            num_layers,
            dropout,
            d_ff,
            dropout_input,
            dropout_features,
            timestep_masking,
            channel_masking,
            timestep_mask_len,
            channel_mask_len,
            layer_drop,
            freeze_fx=freeze_fx,
        )
        self.proj = pytorch_linear(d_model, num_labels)
        self.freeze = True

    def forward(self, x, pad_mask=None):

        with torch.no_grad() if self.freeze else contextlib.ExitStack():
            encoded, pad_mask = self.encoder(x, pad_mask)
        encoded = self.proj(encoded)
        return F.log_softmax(encoded, dim=-1), pad_mask


class Wav2Vec2PooledEncoder(nn.Module):
    def __init__(
        self,
        conv_features=CONV_FEATURES[16],
        d_model=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        d_ff=None,
        dropout_input=0.0,
        dropout_features=0.0,
        timestep_masking=0.5,
        channel_masking=0.1,
        timestep_mask_len=10,
        channel_mask_len=64,
        layer_drop=0.0,
        reduction_type='SHA',
        reduction_d_k=64,
        final_output_dim=None,
    ):
        super().__init__()
        self.encoder = Wav2Vec2Encoder(
            conv_features,
            d_model,
            num_heads,
            num_layers,
            dropout,
            d_ff,
            dropout_input,
            dropout_features,
            timestep_masking,
            channel_masking,
            timestep_mask_len,
            channel_mask_len,
            layer_drop,
        )

        if final_output_dim:
            self.output_dim = final_output_dim
            self.proj_layer = pytorch_linear(self.encoder.output_dim, final_output_dim)
        else:
            self.output_dim = self.encoder.output_dim
            self.proj_layer = PassThru(self.output_dim)

        reduction_type = reduction_type.lower()
        self.reduction_fn = self._reduction_3
        if reduction_type == "2ha":
            self.reduction_layer = nn.Sequential(
                TwoHeadConcat(self.output_dim, dropout, scale=False, d_k=reduction_d_k), nn.Linear(2 * self.output_dim, self.output_dim)
            )
        elif reduction_type == "2ha_max":
            self.reduction_layer = nn.Sequential(
                TwoHeadConcat(self.output_dim, dropout, scale=False, d_k=reduction_d_k, pooling='max'),
                nn.Linear(2 * self.output_dim, self.output_dim),
            )
        elif reduction_type == "2ha_mean":
            self.reduction_layer = nn.Sequential(
                TwoHeadConcat(self.output_dim, dropout, scale=False, d_k=reduction_d_k, pooling='mean'),
                nn.Linear(2 * self.output_dim, self.output_dim),
            )
        elif reduction_type == "sha":
            self.reduction_layer = SingleHeadReduction(self.output_dim, dropout, scale=False, d_k=reduction_d_k)
        elif reduction_type == "sha_max":
            self.reduction_layer = SingleHeadReduction(self.output_dim, dropout, scale=False, d_k=reduction_d_k, pooling='max')
        elif reduction_type == "sha_mean":
            self.reduction_layer = SingleHeadReduction(self.output_dim, dropout, scale=False, d_k=reduction_d_k, pooling='mean')
        elif reduction_type == "max":
            self.reduction_layer = MaxPool1D(self.output_dim)
            self.reduction_fn = self._reduction_1
        elif reduction_type == "none":
            self.reduction_fn = self._no_reduction_mask
        else:
            raise Exception("Unknown exception type")
        self.freeze = True

    def _reduction_1(self, encoded, pad_mask):
        """Do a reduction using just the lengths and input"""
        lengths = pad_mask.sum(-1)
        return self.reduction_layer((encoded, lengths))

    def _reduction_3(self, encoded, pad_mask):
        """Do a reduction using an attention layer with encoder as KQ and V"""
        encoded_query = self.reduction_layer((encoded, encoded, encoded, pad_mask.unsqueeze(1).unsqueeze(1)))
        return encoded_query

    def _no_reduction_mask(self, encoded, pad_mask):
        """Do no reduction and return the tensor and the pad vector"""
        return (encoded, pad_mask,)
    
    def forward(self, x):

        (x, pad_mask) = x
        with torch.no_grad() if self.freeze else contextlib.ExitStack():
            encoded, pad_mask = self.encoder(x, pad_mask)
        return self.reduction_fn(encoded, pad_mask)


class Wav2Vec2Model(nn.Module):
    """Raw pretraining model for wav2vec.  Assumes that we are not doing any padding

    The batches are picked by trimming the signals, which means that any complex code required to deal
    with padding is not required here, whereas in downstream, any of the code to do with VQ quantiziation
    is not required, keeping both models simple and focused on a single task

    """

    def __init__(
        self,
        conv_features=CONV_FEATURES[16],
        num_vq_vars=320,
        start_temp=START_TEMP,
        end_temp=END_TEMP,
        temp_decay_factor=TEMP_DECAY_FACTOR,
        num_vq_groups=2,
        d_model=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        d_ff=None,
        final_dim=256,
        dropout_input=0.1,
        dropout_features=0.1,
        timestep_masking=0.65,
        channel_masking=0.0,
        timestep_mask_len=10,
        channel_mask_len=64,
        layer_drop=0.0,
    ):
        super().__init__()
        fx_dsz = conv_features[-1][0]
        self.layer_norm = torch.nn.LayerNorm(fx_dsz)
        self.dropout_input = torch.nn.Dropout(dropout_input)
        self.dropout_features = torch.nn.Dropout(dropout_features)

        self.feature_extractor = ConvFeatureExtractionModel(conv_features)
        self.proj_to_input = Dense(fx_dsz, d_model)
        self.quantizer = GumbelVectorQuantizer(
            fx_dsz, num_vq_vars, start_temp, end_temp, temp_decay_factor, num_vq_groups, final_dim
        )
        self.encoder = AudioTransformerEncoder(
            num_heads, d_model, dropout, num_layers, d_ff=d_ff, layer_drop=layer_drop
        )
        self.project_q = Dense(final_dim, final_dim)
        self.final_proj = Dense(d_model, final_dim)
        self.timestep_masking = timestep_masking
        self.channel_masking = channel_masking
        self.timestep_mask_len = timestep_mask_len
        self.channel_mask_len = channel_mask_len
        self.mask_emb = nn.Parameter(torch.FloatTensor(d_model).uniform_())

    def set_num_updates(self, s):
        self.quantizer.set_num_updates(s)

    def forward(self, x):
        fx = self.feature_extractor(x)
        fx = fx.transpose(1, 2)
        features = self.layer_norm(fx)
        unmasked_features = features.clone()
        features = self.proj_to_input(features)
        B, T, C = unmasked_features.shape
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        time_mask = create_mask((B, T), p_start=self.timestep_masking, mask_length=self.timestep_mask_len)
        time_mask = torch.from_numpy(time_mask).to(x.device)
        features[time_mask] = self.mask_emb

        if self.channel_masking > 0.0:
            channel_mask = create_mask((B, C), p_start=self.channel_masking, mask_length=self.channel_mask_len)
            channel_mask = torch.from_numpy(channel_mask).to(x.device).unsqueeze(1).view(-1, T, -1)
            features[channel_mask] = 0

        y = unmasked_features[time_mask].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
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
        negs = negs.view(B, T, self.n_negatives, C).permute(2, 0, 1, 3)  # to NxBxTxC
        return negs, neg_idxs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, pad_mask, dst, dst_lengths):
        dst_mask = sequence_mask(dst_lengths)
        encoded_input, pad_mask = self.encoder(input, pad_mask)
        output = self.decoder(encoded_input, pad_mask, dst, dst_mask)
        return output

    def decode(self, input, pad_mask, max_output_len=100):
        with torch.no_grad():
            encoded_input, pad_mask = self.encoder(input, pad_mask)

            dst = torch.full((input.shape[0], 1), Offsets.GO, dtype=torch.long).to(input.device)
            dst_mask = torch.ones_like(dst).bool()
            for i in range(max_output_len):

                outputs = self.decoder(encoded_input, pad_mask, dst, dst_mask)
                best = torch.argmax(outputs[:, i], -1)
                end_mask = (best != Offsets.EOS).unsqueeze(1)
                if all(~end_mask):
                    break
                dst_mask = torch.cat([dst_mask, end_mask], 1)
                dst = torch.cat([dst, best.unsqueeze(1)], 1)
            return dst[:, 1:]
