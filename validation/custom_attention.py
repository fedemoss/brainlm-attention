"""Build custom attention matrices from the "attention is FC + noise" hypothesis.

The [seq_len, seq_len] attention matrix is treated as a (num_patches x num_patches) grid of
(num_regions x num_regions) submatrices. Blocks on the grid diagonal -- the same-time-patch
blocks -- hold the subject's FC matrix. Optionally, blocks displaced from the diagonal by
`block_offset` also hold FC. Everything else is Gaussian noise. Rows are then normalized so the
result is a valid, non-negative attention matrix.

BrainLM's token order is region-major: token i = region * num_patches + time_patch.

Shared by custom_attention_construction.ipynb (builds and plots them) and
custom_attention_zero_shot.ipynb (feeds them to the model).
"""
import numpy as np
import torch


def compute_fc(window):
    """Pearson correlation matrix of a [num_regions, window_size] time series."""
    return np.corrcoef(window)


def token_region_patch(num_regions, num_patches):
    """Region and time-patch index of every token, in BrainLM's region-major order."""
    idx = np.arange(num_regions * num_patches)
    return idx // num_patches, idx % num_patches


def fc_block_mask(patches, block_offset=None):
    """Which (i, j) token pairs get FC rather than noise.

    Always: pairs sharing a time patch (the grid-diagonal blocks).
    With block_offset: also pairs whose time patches differ by exactly that many blocks
    (a set of submatrices displaced from the grid diagonal).
    """
    time_gap = np.abs(patches[:, None] - patches[None, :])
    mask = time_gap == 0
    if block_offset is not None:
        mask = mask | (time_gap == block_offset)
    return mask


def noise_schedule(mean_abs_fc, num_layers, start_fraction=1 / 3):
    """Noise std per layer: a fraction of mean|FC| at layer 0, falling to exactly 0 at the last
    layer. Keeps FC the dominant signal everywhere, and makes the deepest layer noise-free."""
    return np.linspace(mean_abs_fc * start_fraction, 0.0, num_layers)


def build_custom_attention(fc_matrix, regions, patches, noise_scale, rng, block_offset=None):
    """The custom [n, n] attention matrix for n tokens given by their region/patch indices.

    Works both for the full unmasked sequence and for the subset of tokens the model keeps
    after masking -- just pass the region/patch indices of whichever tokens are present.

    Normalization divides each row by its sum of absolute values: always strictly positive, so
    unlike a signed row-sum it can never flip signs or blow up on a near-zero denominator.
    """
    mask = fc_block_mask(patches, block_offset)
    fc_values = fc_matrix[regions[:, None], regions[None, :]].astype(np.float32)
    noise = (rng.standard_normal((len(regions), len(regions))) * noise_scale).astype(np.float32)

    weights = np.abs(np.where(mask, fc_values, noise))
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return (weights / row_sums).astype(np.float32)


def region_lag_profile(matrix, num_patches, max_lag):
    """Mean |value| at each region-lag L, i.e. along the diagonal at token offset L * num_patches.

    In region-major order a fixed region-lag sits at a constant token offset, so this is the
    natural way to see the near-diagonal band and any displaced (e.g. homotopic) bands.
    """
    n = matrix.shape[0]
    rows = np.arange(n)
    profile = np.empty(max_lag)
    for lag in range(max_lag):
        cols = rows + lag * num_patches
        valid = cols < n
        profile[lag] = np.abs(matrix[rows[valid], cols[valid]]).mean()
    return profile


def make_custom_attention_hook(custom_attn):
    """Forward hook replacing one encoder layer's attention with `custom_attn` [n, n].

    n is the number of kept (unmasked) tokens; the CLS row/column at index 0 is left untouched.
    The layer's output is then recomputed from the substituted attention.
    """
    attn_tensor = torch.as_tensor(custom_attn)

    def hook(module, inputs, output):
        hidden_states = inputs[0]
        _, attn_probs = output  # [batch, heads, 1 + n, 1 + n]
        modified = attn_probs.clone()
        modified[:, :, 1:, 1:] = attn_tensor.to(device=modified.device, dtype=modified.dtype)

        value_layer = module.transpose_for_scores(module.value(hidden_states))
        context = torch.matmul(modified, value_layer)
        if module.conv_kernel_size is not None:
            context = context + module.conv(value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*(context.size()[:-2] + (module.all_head_size,)))
        return (context, modified)

    return hook


def _decoder_forward(self_attention, custom_attn, original_forward):
    """Forward that swaps in `custom_attn` for one decoder layer's attention."""
    tensor = torch.as_tensor(custom_attn)

    def wrapped(hidden_states, attention_mask=None, output_attentions=False):
        _, probs = original_forward(hidden_states, attention_mask, True)
        modified = probs.clone()
        modified[:, :, 1:, 1:] = tensor.to(device=modified.device, dtype=modified.dtype)

        value = self_attention.transpose_for_scores(self_attention.value(hidden_states))
        context = torch.matmul(modified, value)
        if self_attention.conv_kernel_size is not None:
            context = context + self_attention.conv(value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*(context.size()[:-2] + (self_attention.all_head_size,)))
        return (context, modified) if output_attentions else (context,)

    return wrapped


def install_decoder_attention(model, matrices):
    """Replace every decoder layer's attention with the supplied [n, n] matrices.

    The decoder calls its layers with output_attentions=False, so a forward hook never sees the
    attention probabilities and cannot replace them -- the layer's forward has to be wrapped
    instead. Note the decoder runs on the FULL token sequence (mask tokens reinserted), so the
    matrices here cover all seq_len tokens, unlike the encoder's kept-tokens-only ones.

    Returns handles for restore_decoder_attention; the caller must always restore.
    """
    handles = []
    for layer, custom in zip(model.decoder.decoder_layers, matrices):
        self_attention = layer.attention.self
        original = self_attention.forward
        handles.append((self_attention, original))
        self_attention.forward = _decoder_forward(self_attention, custom, original)
    return handles


def restore_decoder_attention(handles):
    for self_attention, original in handles:
        self_attention.forward = original
