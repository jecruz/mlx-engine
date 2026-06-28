"""
Qwen3.5 MRoPE patch using derive and override pattern.

The mlx-lm Qwen3.5 model uses standard RoPE (via Qwen3NextAttention), which works
for text-only generation. However, Qwen3.5 requires Multimodal RoPE (MRoPE) with
3D position IDs for vision tasks.

This patch adds a dual code path to each decoder layer, selected by position_ids:

- Text-only (position_ids=None): uses the original mlx-lm modules (Qwen3NextAttention
  with nn.RoPE, GatedDeltaNet with _precise_swiglu) — bit-identical to unpatched mlx-lm.
- Vision (position_ids provided): mirrors mlx-vlm's computation (MRoPE attention,
  GatedDeltaNet with the shared norm/output path) — bit-identical to native mlx-vlm.

Both paths read weights from the same modules; no weight duplication.

Reference implementations:
  https://github.com/ml-explore/mlx-lm/blob/aa4f880/mlx_lm/models/qwen3_5.py#L86-L206
  https://github.com/Blaizzy/mlx-vlm/blob/58e2435/mlx_vlm/models/qwen3_5/language.py#L92-L356
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_5 import (
    DecoderLayer,
    Model as Qwen3_5Model,
    Qwen3_5TextModel,
    TextModel as Qwen3_5LanguageModel,
)
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.qwen3_next import Qwen3NextAttention
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.qwen3_5 import language as vlm_qwen3_5_language
from mlx_vlm.models.qwen3_5.language import (
    LanguageModel as VlmQwen3_5LanguageModel,
    Qwen3_5Attention as VlmQwen3_5Attention,
    Qwen3_5GatedDeltaNet as VlmQwen3_5GatedDeltaNet,
    Qwen3_5Model as VlmQwen3_5Model,
    Qwen3_5RotaryEmbedding,
)

# Stable aliases to the pristine mlx-lm classes captured before apply_patches()
# mutates mlx_lm.models.qwen3_5 in place.
OriginalDecoderLayer = DecoderLayer
OriginalQwen3_5ModelCall = Qwen3_5Model.__call__
OriginalQwen3_5LanguageModelCall = Qwen3_5LanguageModel.__call__
OriginalQwen3_5TextModel = Qwen3_5TextModel
OriginalQwen3_5TextModelCall = Qwen3_5TextModel.__call__
OriginalVlmQwen3_5AttentionInit = VlmQwen3_5Attention.__init__
OriginalVlmQwen3_5AttentionCall = VlmQwen3_5Attention.__call__
OriginalVlmQwen3_5GatedDeltaNetCall = VlmQwen3_5GatedDeltaNet.__call__
OriginalVlmQwen3_5ModelCall = VlmQwen3_5Model.__call__
OriginalVlmQwen3_5LanguageModelCall = VlmQwen3_5LanguageModel.__call__
OriginalVlmQwen3_5GetRopeIndex = VlmQwen3_5LanguageModel.get_rope_index
OriginalVlmQwen3_5IsSingleRowBatchCache = (
    vlm_qwen3_5_language._is_single_row_batch_cache
)
OriginalVlmQwen3_5RaggedDecodeAttention = (
    vlm_qwen3_5_language._qwen3_5_ragged_decode_attention
)


def _patched_vlm_qwen3_5_ragged_decode_attention(*args, **kwargs):
    return None


def _patched_vlm_qwen3_5_get_rope_index(
    self,
    input_ids: mx.array,
    image_grid_thw: Optional[mx.array] = None,
    video_grid_thw: Optional[mx.array] = None,
    attention_mask: Optional[mx.array] = None,
):
    """Handle fully padded vision rows before deferring to upstream rope logic."""
    if (
        input_ids is None
        or attention_mask is None
        or attention_mask.ndim != 2
        or not isinstance(attention_mask, mx.array)
        or (image_grid_thw is None and video_grid_thw is None)
    ):
        return OriginalVlmQwen3_5GetRopeIndex(
            self,
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    keep_rows = [index for index, row in enumerate(attention_mask.tolist()) if any(row)]
    if len(keep_rows) == input_ids.shape[0]:
        return OriginalVlmQwen3_5GetRopeIndex(
            self,
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    if not keep_rows:
        position_ids = mx.ones(
            (3, input_ids.shape[0], input_ids.shape[1]),
            dtype=input_ids.dtype,
        )
        rope_deltas = mx.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)
        return position_ids, rope_deltas

    active_input_ids = input_ids[keep_rows, :]
    active_attention_mask = attention_mask[keep_rows, :]
    active_position_ids, active_rope_deltas = OriginalVlmQwen3_5GetRopeIndex(
        self,
        active_input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=active_attention_mask,
    )

    inactive_position = mx.ones((3, 1, input_ids.shape[1]), dtype=active_position_ids.dtype)
    position_rows = []
    delta_rows = []
    active_row = 0
    for row_index in range(input_ids.shape[0]):
        if row_index in keep_rows:
            position_rows.append(active_position_ids[:, active_row : active_row + 1, :])
            delta_rows.append(active_rope_deltas[active_row : active_row + 1])
            active_row += 1
        else:
            position_rows.append(inactive_position)
            delta_rows.append(mx.zeros((1, 1), dtype=active_rope_deltas.dtype))

    return mx.concatenate(position_rows, axis=1), mx.concatenate(delta_rows, axis=0)


def _vlm_qwen3_5_gated_delta_net_fast_path(
    linear,
    inputs: mx.array,
    mask: Optional[mx.array],
    cache: Optional[Any],
    use_kernel: bool | None = None,
) -> mx.array:
    """Pre-target-verify Qwen3.5 GDN path for ordinary decode.

    Upstream mlx-vlm added target-verification and ragged-batch helpers to this
    layer. Those are needed for MTP/ragged server decode, but the decode-only
    conv/projection helpers are slower for mlx-engine's common batch-size-one
    path. This preserves the older plain path when no target-verify state is in
    play.
    """
    B, S, _ = inputs.shape

    mixed_qkv = linear.in_proj_qkv(inputs)

    z = linear.in_proj_z(inputs)
    z = z.reshape(B, S, -1, linear.head_v_dim)

    b = linear.in_proj_b(inputs)
    a = linear.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
        if conv_state.shape[0] != B:
            conv_state = mx.zeros(
                (B, linear.conv_kernel_size - 1, linear.conv_dim),
                dtype=inputs.dtype,
            )
    else:
        conv_state = mx.zeros(
            (B, linear.conv_kernel_size - 1, linear.conv_dim),
            dtype=inputs.dtype,
        )

    if mask is not None:
        if mask.shape[0] != B:
            mask = None
        else:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    if cache is not None:
        cache[0] = mx.contiguous(conv_input[:, -(linear.conv_kernel_size - 1) :])
    conv_out = nn.silu(linear.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    if state is not None and state.shape[0] != B:
        state = None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        linear.A_log,
        linear.dt_bias,
        state,
        mask,
        use_kernel=not getattr(linear, "training", False)
        if use_kernel is None
        else use_kernel,
    )

    if cache is not None:
        cache[1] = state
        if hasattr(cache, "advance"):
            cache.advance(S)

    out = linear.norm(out, z)
    return linear.out_proj(out.reshape(B, S, -1))


def _has_vlm_qwen3_5_ragged_cache_state(cache) -> bool:
    return (
        getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
    )


def _patched_vlm_qwen3_5_is_single_row_batch_cache(cache_entry) -> bool:
    left_padding = getattr(cache_entry, "left_padding", None)
    if not (
        isinstance(left_padding, mx.array)
        and left_padding.ndim > 0
        and left_padding.size == 1
    ):
        return False

    cached = getattr(cache_entry, "_mlx_engine_qwen3_5_single_row_batch_cache", None)
    if cached is None or cached[0] is not left_padding:
        cached = (left_padding, bool(int(left_padding.item()) > 0))
        cache_entry._mlx_engine_qwen3_5_single_row_batch_cache = cached
    return cached[1]


def _restore_batch_padding_metadata(cache_entry, offsets, steps: int):
    """Restore merged batch-cache offsets and left-padding after row fallback."""
    if offsets is None:
        return cache_entry
    if not (
        hasattr(cache_entry, "offset")
        and hasattr(cache_entry, "left_padding")
        and hasattr(cache_entry, "_idx")
    ):
        return cache_entry
    cache_entry.offset = offsets + steps
    cache_entry.left_padding = cache_entry._idx - cache_entry.offset
    return cache_entry


def _patched_vlm_qwen3_5_model_call(
    self,
    inputs: mx.array,
    inputs_embeds: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    cache=None,
    position_ids: Optional[mx.array] = None,
    capture_layer_ids: Optional[list[int]] = None,
    hidden_sink: Optional[list] = None,
    gdn_sink: Optional[list] = None,
):
    """Guard mixed-padding row fallback from fully padded rows.

    Upstream mlx-vlm's row-wise fallback can recurse into empty per-row inputs
    when a batch contains fully padded query rows. Skip those rows and restore
    merged batch-cache offset/left-padding metadata after the per-row merge.
    """
    batch_size = int(
        inputs_embeds.shape[0] if inputs_embeds is not None else inputs.shape[0]
    )
    seq_length = int(
        inputs_embeds.shape[1] if inputs_embeds is not None else inputs.shape[1]
    )
    if (
        cache is None
        or batch_size <= 1
        or seq_length <= 1
        or hidden_sink is not None
        or gdn_sink is not None
    ):
        return OriginalVlmQwen3_5ModelCall(
            self,
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            gdn_sink=gdn_sink,
        )

    fa_cache = cache[self.fa_idx]
    if (
        fa_cache is not None
        and hasattr(fa_cache, "extract")
        and hasattr(fa_cache.__class__, "merge")
        and isinstance(getattr(fa_cache, "offset", None), mx.array)
        and fa_cache.offset.ndim > 0
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds
        query_left_padding = mx.minimum(mx.maximum(-fa_cache.offset, 0), h.shape[1])
        cache_left_padding = getattr(fa_cache, "left_padding", None)
        has_left_padding = (
            isinstance(cache_left_padding, mx.array)
            and cache_left_padding.ndim > 0
            and int(cache_left_padding.max().item()) > 0
        )
        if has_left_padding or int(query_left_padding.max().item()) > 0:
            row_outputs = []
            row_caches = [[] for _ in cache]
            batch_offsets = []
            for cache_entry in cache:
                offsets = getattr(cache_entry, "offset", None)
                if (
                    isinstance(offsets, mx.array)
                    and offsets.ndim > 0
                    and offsets.size >= h.shape[0]
                ):
                    batch_offsets.append(offsets[: h.shape[0]])
                else:
                    batch_offsets.append(None)

            for row, pad in enumerate(query_left_padding.tolist()):
                pad = min(max(int(pad), 0), h.shape[1])
                current_cache = []
                for cache_entry in cache:
                    if cache_entry is None:
                        current_cache.append(None)
                    else:
                        current_cache.append(
                            vlm_qwen3_5_language._extract_row_cache(
                                cache_entry, row
                            )
                        )

                if pad == h.shape[1]:
                    row_outputs.append(mx.zeros_like(h[row : row + 1]))
                    for i, cache_entry in enumerate(current_cache):
                        row_caches[i].append(cache_entry)
                    continue

                row_inputs = inputs[row : row + 1, pad:]
                row_embeds = h[row : row + 1, pad:]
                row_position_ids = None
                if position_ids is not None:
                    if position_ids.ndim == 2:
                        row_position_ids = position_ids[row : row + 1, pad:]
                    else:
                        row_position_ids = position_ids[:, row : row + 1, pad:]

                row_out = OriginalVlmQwen3_5ModelCall(
                    self,
                    row_inputs,
                    inputs_embeds=row_embeds,
                    cache=current_cache,
                    position_ids=row_position_ids,
                )
                if pad > 0:
                    row_out = vlm_qwen3_5_language._pad_row_time(
                        row_out, pad, h.shape[1]
                    )
                row_outputs.append(row_out)
                for i, cache_entry in enumerate(current_cache):
                    row_caches[i].append(cache_entry)

            for i, entries in enumerate(row_caches):
                if cache[i] is None:
                    continue
                if hasattr(cache[i].__class__, "merge"):
                    cache[i] = _restore_batch_padding_metadata(
                        cache[i].__class__.merge(entries),
                        batch_offsets[i],
                        h.shape[1],
                    )
            return mx.concatenate(row_outputs, axis=0)

    return OriginalVlmQwen3_5ModelCall(
        self,
        inputs,
        inputs_embeds=inputs_embeds,
        mask=mask,
        cache=cache,
        position_ids=position_ids,
        capture_layer_ids=capture_layer_ids,
        hidden_sink=hidden_sink,
        gdn_sink=gdn_sink,
    )


class PatchedDecoderLayer(DecoderLayer):
    """
    DecoderLayer that accepts position_ids and uses MRoPE when they are provided.

    For text-only calls (position_ids=None), delegates to the original
    Qwen3NextAttention and GatedDeltaNet — bit-identical to the unpatched model.

    For vision calls (position_ids provided), uses mlx-vlm's MRoPE attention
    and GatedDeltaNet — bit-identical to the native mlx-vlm model.
    """

    def __init__(self, args, layer_idx):
        super().__init__(args, layer_idx)
        self._mrope = None
        if not self.is_linear:
            rope_params = args.rope_parameters
            mrope_section = rope_params.get("mrope_section")
            if mrope_section is not None:
                self._mrope = Qwen3_5RotaryEmbedding(
                    int(
                        self.self_attn.head_dim
                        * rope_params["partial_rotary_factor"]
                    ),
                    max_position_embeddings=args.max_position_embeddings,
                    base=rope_params["rope_theta"],
                    mrope_section=mrope_section,
                )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if self.is_linear:
            if position_ids is None:
                # Text-only: use original mlx-lm GatedDeltaNet
                r = self.linear_attn(self.input_layernorm(x), mask, cache)
            else:
                # Vision: use mlx-vlm GatedDeltaNet computation path
                r = self._vlm_gated_delta_net(self.input_layernorm(x), mask, cache)
        elif position_ids is None:
            # Text-only: use original Qwen3NextAttention with nn.RoPE
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        else:
            # Vision: apply MRoPE using the original attention module's weights
            r = self._mrope_attention(
                self.input_layernorm(x), mask, cache, position_ids
            )
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

    def _mrope_attention(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        position_ids: mx.array,
    ) -> mx.array:
        """
        MRoPE attention path, reusing the original attention module's weights.

        Mirrors Qwen3_5Attention.__call__ from mlx-vlm but operates on
        self.self_attn's projections and norms directly.
        """
        if self._mrope is None:
            raise ValueError("Qwen3.5 MRoPE config is required for vision requests")

        attn = self.self_attn
        B, L, D = x.shape

        q_proj_output = attn.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = attn.k_proj(x), attn.v_proj(x)

        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        # Use the rotary module's apply_rotary, which selects the compiled fused
        # MRoPE kernel when available (Metal). This matches native mlx-vlm
        # Qwen3_5Attention (which calls self.rotary_emb.apply_rotary) and is
        # ~20% faster than the two-step cos/sin + apply path. See Redmine #1123.
        queries, keys = self._mrope.apply_rotary(
            queries, keys, position_ids, unsqueeze_dim=1
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=attn.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return attn.o_proj(output * mx.sigmoid(gate))

    def _vlm_gated_delta_net(
        self,
        inputs: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> mx.array:
        """
        mlx-vlm GatedDeltaNet computation path, reusing self.linear_attn's weights.

        Mirrors Qwen3_5GatedDeltaNet.__call__ from mlx-vlm to produce
        bit-identical output with the native mlx-vlm model.
        """
        linear = self.linear_attn
        B, S, _ = inputs.shape

        mixed_qkv = linear.in_proj_qkv(inputs)

        z = linear.in_proj_z(inputs)
        z = z.reshape(B, S, -1, linear.head_v_dim)

        b = linear.in_proj_b(inputs)
        a = linear.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = mx.zeros(
                    (B, linear.conv_kernel_size - 1, linear.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (B, linear.conv_kernel_size - 1, linear.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(linear.conv_kernel_size - 1) :])
        conv_out = nn.silu(linear.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
                [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
                [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        if state is not None and state.shape[0] != B:
            state = None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            linear.A_log,
            linear.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            # Follow mlx-vlm cache advance logic (conditional cache.advance call)
            # ref: https://github.com/Blaizzy/mlx-vlm/blob/58e2435/mlx_vlm/models/qwen3_5/language.py#L350-L353
            # mlx-lm is not conditional
            # ref: https://github.com/ml-explore/mlx-lm/blob/aa4f880/mlx_lm/models/qwen3_5.py#L196-L198
            if hasattr(cache, "advance"):
                cache.advance(S)

        out = linear.norm(out, z)
        return linear.out_proj(out.reshape(B, S, -1))


def _qwen3_5_dflash_rollback_base_history_len(gdn_state) -> int:
    """Return the pre-verify history length for a single gdn_state entry.

    ``gdn_state`` is one entry from the per-layer ``gdn_states`` list the
    DFlash runtime captured during target_verify. Supported shapes:

    * ``None``: caller has no per-layer state available.
    * mlx-vlm GDN sink tuple (12-tuple): no ``base_history_len`` attribute,
      so return 0 (treat the pre-verify history length as unknown).
    * Simple namespace / object exposing ``base_history_len``: return it.
    """
    if gdn_state is None:
        return 0
    if isinstance(gdn_state, tuple):
        return 0
    return int(getattr(gdn_state, "base_history_len", 0) or 0)


def _qwen3_5_arrays_cache_is_sequential_single_sequence(cache_layer) -> bool:
    """True iff ``cache_layer`` is a real mlx-lm ``ArraysCache`` in single-sequence mode.

    Real single-sequence Qwen3.5/Qwen3.6 GDN layers use ``ArraysCache``
    with ``lengths`` and ``left_padding`` both set to ``None`` and the
    GDN state stored in the ``cache`` list (``cache[0]`` is the conv
    window, ``cache[1]`` is the running gated-delta state). Ragged /
    multi-row batched variants set ``lengths`` or ``left_padding`` to
    non-``None`` arrays and must NOT be treated as the proven rollback
    surface.
    """
    if cache_layer is None:
        return False
    cache_list = getattr(cache_layer, "cache", None)
    if not isinstance(cache_list, list) or len(cache_list) < 2:
        return False
    if getattr(cache_layer, "lengths", None) is not None:
        return False
    if getattr(cache_layer, "left_padding", None) is not None:
        return False
    return True


def _qwen3_5_dflash_arrays_cache_rollback(
    cache_layer,
    gdn_state,
    accepted: int,
    block_size: int,
) -> bool:
    """Roll back a real ``ArraysCache`` layer's GDN state for partial DFlash rejection.

    ``cache_layer`` must be the sequential single-sequence ArraysCache
    shape (see :func:`_qwen3_5_arrays_cache_is_sequential_single_sequence`).
    ``gdn_state`` is the mlx-vlm GDN sink tuple captured during the
    verify call:

        (q, k, v, a, b, A_log, dt_bias,
         initial_state, mask, conv_input, conv_kernel_size, intermediate_states)

    ``initial_state`` (index 7) is the GDN state BEFORE the verify call
    (= ``cache[1]`` at the time the verify started). ``conv_input`` (index
    9) is the extended conv input (``[cache[0] prepended] + new tokens``).
    ``intermediate_states`` (index 11) holds the per-step GDN state
    snapshots.

    Returns ``True`` if the rollback was applied. Returns ``False`` if
    the layer is not the proven sequential shape, the ``gdn_state`` is
    missing or malformed, or the rollback was a no-op (full acceptance).
    """

    if gdn_state is None or not _qwen3_5_arrays_cache_is_sequential_single_sequence(
        cache_layer
    ):
        return False
    if not isinstance(gdn_state, tuple) or len(gdn_state) < 12:
        return False

    # Full acceptance: rollback is a no-op. Keep the live GDN state
    # unchanged so the next verify round starts from the post-verify
    # state.
    if accepted >= block_size - 1:
        return True

    initial_state = gdn_state[7]
    conv_input = gdn_state[9]
    conv_kernel_size = int(gdn_state[10])
    intermediate_states = gdn_state[11]

    if (
        initial_state is None
        or conv_input is None
        or intermediate_states is None
        or conv_kernel_size <= 0
    ):
        return False

    cache_list = cache_layer.cache
    conv_window = conv_kernel_size - 1

    # accepted == 0 (only bonus target kept) and accepted == -1 (defensive)
    # both mean "no drafts accepted": restore the live cache to the
    # pre-verify state from the gdn_sink tuple.
    if accepted < 0:
        cache_list[1] = initial_state
        if conv_input.ndim == 3 and conv_window > 0 and conv_input.shape[1] >= conv_window:
            cache_list[0] = mx.contiguous(conv_input[:, :conv_window, :])
        return True

    # Partial acceptance: use mlx-vlm's gated_delta_accept_states helper
    # to compute the cache[1] state and cache[0] conv window at the
    # accepted boundary. For sequential single-sequence use the
    # ``accepted`` array has a single element.
    try:
        from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_accept_states
    except Exception:
        return False

    live_state = cache_list[1]
    live_conv = cache_list[0]

    if live_state is None or not hasattr(live_state, "shape"):
        # Defensive: nothing to restore against.
        cache_list[1] = initial_state
        if conv_input.ndim == 3 and conv_window > 0 and conv_input.shape[1] >= conv_window:
            cache_list[0] = mx.contiguous(conv_input[:, :conv_window, :])
        return True

    if live_conv is None or not hasattr(live_conv, "shape"):
        if conv_input.ndim == 3 and conv_window > 0 and conv_input.shape[1] >= conv_window:
            live_conv = mx.zeros(
                (conv_input.shape[0], conv_window, conv_input.shape[2]),
                dtype=conv_input.dtype,
            )
        else:
            return False

    accepted_array = mx.array([int(accepted)], dtype=mx.int32)
    try:
        new_state, new_conv = gated_delta_accept_states(
            intermediate_states,
            conv_input,
            live_state,
            live_conv,
            accepted_array,
            conv_kernel_size,
            use_kernel=False,
        )
    except Exception:
        return False

    if new_state is None:
        cache_list[1] = initial_state
    else:
        cache_list[1] = new_state
    if new_conv is not None:
        cache_list[0] = mx.contiguous(new_conv)
    return True


def _qwen3_5_dflash_rollback_rewind_layer(
    cache_layer,
    keep: int,
    gdn_state=None,
    accepted: int = -1,
    block_size: int = 0,
) -> None:
    """Truncate a single cache layer back to ``keep`` tokens.

    The DFlash rollback is default-off and only invoked from the runtime
    path. It must never widen the supported cache surface; the goal is the
    narrow rollback that lets ``dflash_stream_generate`` recover from
    partial DFlash rejection without leaking unverified tokens into the
    live cache state.

    For the real sequential single-sequence ``ArraysCache`` shape (Qwen3.5
    / Qwen3.6 GDN linear-attention layers), the rollback uses the
    per-layer GDN sink tuple captured during target_verify and mlx-vlm's
    ``gated_delta_accept_states`` helper to restore both ``cache[0]``
    (the conv window) and ``cache[1]`` (the running gated-delta state)
    to the boundary between accepted and rejected proposal tokens. The
    truncation is performed in-place on the live cache list and matches
    the mlx-vlm GDN state machine exactly so a subsequent verify round
    continues from the right GDN state without re-using rejected
    proposal tokens.

    Any other ``ArraysCache`` variant (non-``None`` ``lengths`` /
    ``left_padding`` arrays) is left untouched here; the validator keeps
    those shapes fail-closed so this branch is never reached for
    unproven ragged variants.
    """
    if cache_layer is None:
        return

    if _qwen3_5_arrays_cache_is_sequential_single_sequence(cache_layer):
        _qwen3_5_dflash_arrays_cache_rollback(
            cache_layer, gdn_state, accepted, block_size
        )
        return

    history = getattr(cache_layer, "history", None)
    if isinstance(history, list) and keep >= 0:
        del history[keep:]

    original_offset = getattr(cache_layer, "offset", None)

    keys = getattr(cache_layer, "keys", None)
    values = getattr(cache_layer, "values", None)
    if (
        keys is not None
        and values is not None
        and hasattr(keys, "shape")
        and keep >= 0
    ):
        seq_axis = -2 if keys.ndim >= 3 else 1
        current_offset = (
            original_offset if isinstance(original_offset, int) else None
        )
        if current_offset is not None and current_offset > keep:
            if seq_axis == -2:
                cache_layer.keys = keys[..., :keep, :]
                cache_layer.values = values[..., :keep, :]
            else:
                cache_layer.keys = keys[:, :keep, ...]
                cache_layer.values = values[:, :keep, ...]

    if isinstance(original_offset, int) and keep >= 0:
        cache_layer.offset = keep

    idx_attr = getattr(cache_layer, "_idx", None)
    if isinstance(idx_attr, int) and keep >= 0:
        cache_layer._idx = keep

    lengths = getattr(cache_layer, "lengths", None)
    if lengths is not None and hasattr(lengths, "shape") and keep >= 0:
        try:
            broadcast_shape = tuple(
                1 if axis == 0 else lengths.shape[axis]
                for axis in range(lengths.ndim)
            )
            floor = mx.full(broadcast_shape, keep, dtype=lengths.dtype)
            cache_layer.lengths = mx.minimum(lengths, floor)
        except Exception:
            try:
                cache_layer.lengths = mx.full(
                    lengths.shape, keep, dtype=lengths.dtype
                )
            except Exception:
                pass


def _qwen3_5_dflash_rollback(
    prompt_cache,
    gdn_states,
    accepted: int,
    block_size: int,
) -> None:
    """Roll back partial DFlash rejection in the sequential text path.

    See :meth:`PatchedQwen3_5TextModel.rollback_speculative_cache` for the
    full contract. Module-level helper so focused tests and the DFlash
    runtime can call the same rollback semantics without constructing a
    real Qwen3.5 model.
    """
    if prompt_cache is None or accepted < 0:
        return
    if accepted >= block_size - 1:
        # Full acceptance; nothing to roll back.
        return

    keep_extra = accepted + 1  # bonus target token + accepted drafts
    for layer_index, cache_layer in enumerate(prompt_cache):
        if cache_layer is None:
            continue
        layer_gdn_state = (
            gdn_states[layer_index]
            if gdn_states is not None and layer_index < len(gdn_states)
            else None
        )
        if _qwen3_5_arrays_cache_is_sequential_single_sequence(cache_layer):
            # ArraysCache rollback uses the per-layer GDN sink tuple and
            # is independent of the simple ``keep`` arithmetic.
            _qwen3_5_dflash_arrays_cache_rollback(
                cache_layer, layer_gdn_state, accepted, block_size
            )
            continue
        base_history_len = _qwen3_5_dflash_rollback_base_history_len(
            layer_gdn_state
        )
        keep = base_history_len + keep_extra
        _qwen3_5_dflash_rollback_rewind_layer(
            cache_layer, keep, gdn_state=layer_gdn_state,
            accepted=accepted, block_size=block_size,
        )


class PatchedQwen3_5TextModel(Qwen3_5TextModel):
    """
    Qwen3_5TextModel with MRoPE position state management.

    Adds position_ids and rope_deltas attributes that can be set externally
    (by the vision add-on) before generation. During forward passes, computes
    the appropriate position_ids from this state and threads them to decoder layers.

    Position state logic (ported from mlx_vlm LanguageModel.__call__):
    - Both None: text-only path, computes sequential 3D positions from cache state
    - position_ids set: prefill, slices stored positions by cache offset
    - Only rope_deltas set: autoregressive generation, computes from delta
    """

    def __init__(self, args):
        super().__init__(args)
        self.position_ids = None
        self.rope_deltas = None

    def reset_mrope_state(self):
        """
        Reset MRoPE position state.

        Called by the vision add-on's clear_prediction_state before every
        prediction. For vision requests, compute_embeddings sets fresh
        state immediately after.
        """
        self.position_ids = None
        self.rope_deltas = None

    def rollback_speculative_cache(
        self,
        prompt_cache,
        gdn_states,
        accepted: int,
        block_size: int,
    ) -> None:
        """Roll back partial DFlash rejection in the sequential text path.

        ``dflash_stream_generate`` invokes this hook when a draft block was
        partially rejected by the target (``accepted < block_size - 1``). The
        implementation must restore the live prompt_cache so that:

        * only the bonus target token plus the ``accepted`` draft tokens
          remain in cache history after the rollback,
        * the previously verified target tokens (those already in cache before
          this verify call) are preserved,
        * the rejected draft tokens never appear in the live cache state.

        ``gdn_states`` is the optional per-layer GDN sink list captured during
        target_verify. Each entry may be a SimpleNamespace exposing
        ``base_history_len`` or an mlx-vlm GDN sink tuple. When empty or
        ``None``, the method falls back to a base length of 0 per layer.

        The method is default-off in practice: it is only called by the DFlash
        runtime path (``dflash_stream_generate``) and never wired into
        ordinary text generation. It does not widen DFlash beyond the
        sequential text surface; batched, VLM, adapter, SpecPrefill, and
        loaded-draft-model combinations remain fail-closed by the boundary
        check.
        """
        _qwen3_5_dflash_rollback(prompt_cache, gdn_states, accepted, block_size)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[list[int]] = None,
        hidden_sink: Optional[list] = None,
        gdn_sink: Optional[list] = None,
    ) -> mx.array:
        capture_requested = (
            capture_layer_ids is not None or hidden_sink is not None or gdn_sink is not None
        )
        if capture_requested and hidden_sink is None:
            hidden_sink = []

        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        position_ids = self._compute_position_ids(inputs, cache)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        capture_set = set(capture_layer_ids) if capture_layer_ids is not None else set()

        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=layer_cache, position_ids=position_ids
            )
            if hidden_sink is not None and capture_layer_ids is not None and i in capture_set:
                hidden_sink.append(hidden_states)

        return self.norm(hidden_states)

    def _compute_position_ids(self, inputs: mx.array, cache) -> Optional[mx.array]:
        """
        Compute position_ids for the current forward pass from stored state.

        Inspired by mlx_vlm.models.qwen3_5.language.LanguageModel.__call__ .
        """
        cache_offset = 0
        cache_offset_scalar = 0
        if cache is not None and cache[self.fa_idx] is not None:
            offset = cache[self.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
                cache_offset_scalar = offset
            elif isinstance(offset, mx.array) and offset.ndim == 0:
                cache_offset = offset.item()
                cache_offset_scalar = cache_offset
            elif isinstance(offset, mx.array):
                cache_offset = offset
                cache_offset_scalar = offset[0].item()

        batch_size, seq_length = inputs.shape

        # Text-only path; no MRoPE state was injected for this call.
        # Return None so PatchedDecoderLayer uses the original nn.RoPE path.
        if self.position_ids is None and self.rope_deltas is None:
            return None
        if self.position_ids is not None and self.rope_deltas is None:
            raise ValueError(
                "MRoPE state is inconsistent: position_ids are set but rope_deltas "
                "are missing."
            )

        # This branch is taken for vision requests while the current call is still
        # consuming prompt tokens that were part of the original multimodal prompt.
        # That includes chunked prompt prefill where cache_offset > 0 but we are still
        # inside the image span, and it also covers callers whose chunk begins inside
        # the stored prompt positions but extends past them by stitching a stored
        # prefix to a sequential tail.
        if self.position_ids is not None:
            stored_seq_length = self.position_ids.shape[2]
            if cache_offset_scalar < stored_seq_length:
                stored_end = min(cache_offset_scalar + seq_length, stored_seq_length)
                stored_positions = self.position_ids[
                    :, :, cache_offset_scalar:stored_end
                ]
                if stored_end - cache_offset_scalar == seq_length:
                    return stored_positions

                tail_seq_length = seq_length - (stored_end - cache_offset_scalar)
                tail_positions = self._compute_sequential_position_ids(
                    batch_size=batch_size,
                    seq_length=tail_seq_length,
                    start_offset=stored_seq_length,
                    rope_deltas=self.rope_deltas,
                )
                return mx.concatenate([stored_positions, tail_positions], axis=2)

        return self._compute_sequential_position_ids(
            batch_size=batch_size,
            seq_length=seq_length,
            start_offset=cache_offset,
            rope_deltas=self.rope_deltas,
        )

    def _compute_sequential_position_ids(
        self,
        *,
        batch_size: int,
        seq_length: int,
        start_offset: int | mx.array,
        rope_deltas: Optional[mx.array] = None,
    ) -> mx.array:
        """Build sequential 3D positions from cache offsets and optional
        rope_deltas once prompt positions have been exhausted."""
        delta = mx.array(start_offset)
        if rope_deltas is not None:
            delta = delta + rope_deltas

        position_ids = mx.arange(seq_length).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

        if delta.ndim == 0:
            delta = mx.broadcast_to(delta.reshape(1, 1), (batch_size, 1))
        elif delta.ndim == 1:
            delta = delta[:batch_size].reshape(-1, 1)
            if delta.shape[0] == 1 and batch_size > 1:
                delta = mx.broadcast_to(delta, (batch_size, 1))
        else:
            delta = delta[:batch_size]
            if delta.shape[0] == 1 and batch_size > 1:
                delta = mx.broadcast_to(delta, (batch_size, delta.shape[1]))

        position_ids = mx.add(position_ids, delta)[None, ...]
        return mx.broadcast_to(position_ids, (3, batch_size, seq_length))


def _patched_qwen3_5_language_model_call(
    self,
    inputs: mx.array,
    cache: Optional[Any] = None,
    input_embeddings: Optional[mx.array] = None,
    capture_layer_ids: Optional[list[int]] = None,
    hidden_sink: Optional[list] = None,
    gdn_sink: Optional[list] = None,
):
    capture_requested = (
        capture_layer_ids is not None or hidden_sink is not None or gdn_sink is not None
    )
    if capture_requested and hidden_sink is None:
        hidden_sink = []

    hidden_states = self.model(
        inputs,
        cache=cache,
        input_embeddings=input_embeddings,
        capture_layer_ids=capture_layer_ids,
        hidden_sink=hidden_sink,
        gdn_sink=gdn_sink,
    )

    if self.args.tie_word_embeddings:
        logits = self.model.embed_tokens.as_linear(hidden_states)
    else:
        logits = self.lm_head(hidden_states)

    if not capture_requested:
        return logits

    return LanguageModelOutput(
        logits=logits,
        hidden_states=hidden_sink,
        gdn_states=gdn_sink,
    )


def _patched_qwen3_5_model_call(
    self,
    inputs: mx.array,
    cache=None,
    input_embeddings: Optional[mx.array] = None,
    **kwargs,
):
    return self.language_model(
        inputs,
        cache=cache,
        input_embeddings=input_embeddings,
        **kwargs,
    )


def _patched_vlm_qwen3_5_attention_init(self, args):
    OriginalVlmQwen3_5AttentionInit(self, args)
    rope_params = args.rope_parameters
    self.rope = initialize_rope(
        int(self.head_dim * rope_params["partial_rotary_factor"]),
        base=rope_params["rope_theta"],
        traditional=False,
        scaling_config=rope_params,
        max_position_embeddings=args.max_position_embeddings,
    )


def _patched_vlm_qwen3_5_attention_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
    position_ids: Optional[mx.array] = None,
    position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    target_verify: bool = False,
) -> mx.array:
    if (
        position_ids is not None
        or position_embeddings is not None
        or target_verify
        or (isinstance(mask, str) and mask == "left_padded_decode")
    ):
        return OriginalVlmQwen3_5AttentionCall(
            self,
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            target_verify=target_verify,
        )

    return Qwen3NextAttention.__call__(self, x, mask, cache)


def _patched_vlm_qwen3_5_gated_delta_net_call(
    self,
    inputs: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
    gdn_sink: Optional[list] = None,
    target_verify: bool = False,
) -> mx.array:
    if (
        target_verify
        or gdn_sink is not None
        or inputs.shape[1] != 1
        or cache is None
        or _has_vlm_qwen3_5_ragged_cache_state(cache)
    ):
        return OriginalVlmQwen3_5GatedDeltaNetCall(
            self,
            inputs,
            mask=mask,
            cache=cache,
            gdn_sink=gdn_sink,
            target_verify=target_verify,
        )

    return _vlm_qwen3_5_gated_delta_net_fast_path(
        self,
        inputs,
        mask,
        cache,
        use_kernel=not self.training,
    )


def _is_vlm_qwen3_5_text_only(
    *,
    mask,
    position_ids,
    pixel_values,
    image_grid_thw,
    video_grid_thw,
    capture_layer_ids,
    rope_deltas_kw,
    stored_position_ids,
    stored_rope_deltas,
) -> bool:
    return (
        mask is None
        and position_ids is None
        and pixel_values is None
        and image_grid_thw is None
        and video_grid_thw is None
        and capture_layer_ids is None
        and rope_deltas_kw is None
        and stored_position_ids is None
        and stored_rope_deltas is None
    )


def _vlm_qwen3_5_batched_left_padding_position_ids(
    cache,
    fa_idx: int,
    batch_size: int,
    seq_length: int,
    dtype,
) -> mx.array | None:
    if cache is None or fa_idx >= len(cache) or seq_length != 1:
        return None
    fa_cache = cache[fa_idx]
    left_padding = getattr(fa_cache, "left_padding", None)
    offset = getattr(fa_cache, "offset", None)
    if (
        not isinstance(left_padding, mx.array)
        or left_padding.ndim == 0
        or int(left_padding.max().item()) <= 0
        or not isinstance(offset, mx.array)
        or offset.ndim == 0
    ):
        return None

    offsets = mx.maximum(offset[:batch_size], 0).reshape(-1, 1)
    position_ids = mx.arange(seq_length, dtype=dtype).reshape(1, -1)
    position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
    return mx.add(position_ids, offsets)


def _patched_vlm_qwen3_5_language_model_call(
    self,
    inputs: mx.array,
    inputs_embeds: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    cache=None,
    **kwargs,
):
    position_ids = kwargs.get("position_ids")
    pixel_values = kwargs.get("pixel_values")
    image_grid_thw = kwargs.get("image_grid_thw")
    video_grid_thw = kwargs.get("video_grid_thw")
    capture_layer_ids = kwargs.get("capture_layer_ids")
    rope_deltas_kw = kwargs.get("rope_deltas")
    text_only = _is_vlm_qwen3_5_text_only(
        mask=mask,
        position_ids=position_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        capture_layer_ids=capture_layer_ids,
        rope_deltas_kw=rope_deltas_kw,
        stored_position_ids=getattr(self, "_position_ids", None),
        stored_rope_deltas=getattr(self, "_rope_deltas", None),
    )

    if text_only:
        position_ids = _vlm_qwen3_5_batched_left_padding_position_ids(
            cache,
            self.model.fa_idx,
            inputs.shape[0],
            inputs.shape[1],
            inputs.dtype,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

    if text_only and position_ids is None:
        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=None,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(
            logits=out,
            hidden_states=None,
            gdn_states=None,
        )

    if rope_deltas_kw is not None:
        # mlx-vlm gates kwarg rope_deltas on this side state during decode.
        self._rope_deltas = rope_deltas_kw

    return OriginalVlmQwen3_5LanguageModelCall(
        self,
        inputs,
        inputs_embeds=inputs_embeds,
        mask=mask,
        cache=cache,
        **kwargs,
    )


def apply_patches():
    """
    Apply Qwen3.5 MRoPE patches.
    """
    import mlx_lm.models.qwen3_5

    mlx_lm.models.qwen3_5.DecoderLayer = PatchedDecoderLayer
    mlx_lm.models.qwen3_5.TextModel.__call__ = _patched_qwen3_5_language_model_call
    mlx_lm.models.qwen3_5.Model.__call__ = _patched_qwen3_5_model_call
    mlx_lm.models.qwen3_5.Qwen3_5TextModel = PatchedQwen3_5TextModel

    VlmQwen3_5Attention.__init__ = _patched_vlm_qwen3_5_attention_init
    VlmQwen3_5Attention.__call__ = _patched_vlm_qwen3_5_attention_call
    VlmQwen3_5GatedDeltaNet.__call__ = _patched_vlm_qwen3_5_gated_delta_net_call
    VlmQwen3_5Model.__call__ = _patched_vlm_qwen3_5_model_call
    VlmQwen3_5LanguageModel.get_rope_index = _patched_vlm_qwen3_5_get_rope_index
    VlmQwen3_5LanguageModel.__call__ = _patched_vlm_qwen3_5_language_model_call
    vlm_qwen3_5_language._is_single_row_batch_cache = (
        _patched_vlm_qwen3_5_is_single_row_batch_cache
    )
    vlm_qwen3_5_language._qwen3_5_ragged_decode_attention = (
        _patched_vlm_qwen3_5_ragged_decode_attention
    )
