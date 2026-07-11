"""Opt-in SpecPrefill routing primitives for sequential prompt processing."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
import math
from typing import Any, Callable, Sequence

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlx_engine.utils.sampling import create_sampler


DEFAULT_SPECPREFILL_KEEP_PCT = 0.2
DEFAULT_SPECPREFILL_THRESHOLD = 1024
SPECPREFILL_SELECTION_CHUNK_SIZE = 32

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecPrefillOptions:
    """Validated options for the guarded SpecPrefill prompt-processing path."""

    enabled: bool
    keep_pct: float = DEFAULT_SPECPREFILL_KEEP_PCT
    threshold: int = DEFAULT_SPECPREFILL_THRESHOLD
    system_tokens: int = 0

    def __post_init__(self) -> None:
        """Validate option values at construction time."""
        if not isinstance(self.enabled, bool):
            raise ValueError("specprefill enabled must be a boolean")
        if (
            isinstance(self.keep_pct, bool)
            or not isinstance(self.keep_pct, int | float)
            or not 0 < self.keep_pct <= 1.0
        ):
            raise ValueError("specprefill_keep_pct must be in the interval (0, 1]")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, int)
            or self.threshold < 1
        ):
            raise ValueError("specprefill_threshold must be a positive integer")
        if (
            isinstance(self.system_tokens, bool)
            or not isinstance(self.system_tokens, int)
            or self.system_tokens < 0
        ):
            raise ValueError("specprefill_system_tokens must be a non-negative integer")


@dataclass(frozen=True)
class SpecPrefillResult:
    """Result from a sparse-prefill attempt."""

    cache: list[Any]
    seed_tokens: mx.array
    live_tokens: Sequence[int]


class _AttentionCapture:
    """Wrap an attention module while capturing post-RoPE query vectors."""

    def __init__(self, original, buf_idx, query_buffer, query_extractor):
        self._original = original
        self._buf_idx = buf_idx
        self._query_buffer = query_buffer
        self._query_extractor = query_extractor

    def __call__(self, x, mask=None, cache=None, **kwargs):
        """Capture queries, then delegate the attention call."""
        if kwargs and _accepts_extractor_kwargs(self._query_extractor, kwargs):
            queries = self._query_extractor(self._original, x, cache, **kwargs)
        else:
            queries = self._query_extractor(self._original, x, cache)
        self._query_buffer[self._buf_idx].append(queries)
        return self._original(x, mask=mask, cache=cache, **kwargs)

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped attention module."""
        return getattr(self._original, name)


class _PositionMappedRoPE:
    """Apply RoPE at non-contiguous original prompt positions."""

    def __init__(self, original_rope, all_positions, cache_start=0):
        self._original = original_rope
        self._all_positions = all_positions
        self._cache_start = _scalar_offset(cache_start)
        self._has_custom_freqs = hasattr(original_rope, "_freqs")

        if self._has_custom_freqs:
            self._freqs = original_rope._freqs
            self._dims = _get_dims(original_rope)
            self._pre_scale = _get_pre_scale(original_rope)
        else:
            self._dims = original_rope.dims
            self._base = original_rope.base
            self._scale = original_rope.scale

    def __call__(self, x, offset=0):
        """Apply the original RoPE using positions selected during sparse prefill."""
        length = x.shape[2]
        idx = _scalar_offset(offset) - self._cache_start
        positions = self._all_positions[idx : idx + length]
        if self._has_custom_freqs:
            return manual_rope_with_freqs(
                x, positions, self._dims, self._freqs, pre_scale=self._pre_scale
            )
        return manual_rope(x, positions, self._dims, base=self._base, scale=self._scale)


class _OffsetAdjustedRoPE:
    """Add a constant offset to decode RoPE after sparse prefill."""

    def __init__(self, original_rope, adjustment):
        self._original = original_rope
        self._adjustment = adjustment

    def __call__(self, x, offset=0):
        """Apply the wrapped RoPE at the adjusted decode offset."""
        return self._original(x, offset=offset + self._adjustment)


def resolve_specprefill_options(
    *,
    specprefill_toggle: bool | None,
    specprefill_keep_pct: float | None,
    specprefill_threshold: int | None,
    specprefill_system_tokens: int | None,
    draft_model: Any | None,
) -> SpecPrefillOptions | None:
    """Resolve public generation kwargs into internal SpecPrefill options."""
    tuning_supplied = any(
        value is not None
        for value in (
            specprefill_keep_pct,
            specprefill_threshold,
            specprefill_system_tokens,
        )
    )
    if specprefill_toggle is not True:
        if tuning_supplied:
            raise ValueError(
                "SpecPrefill tuning options require specprefill_toggle=True"
            )
        return None
    if draft_model is None:
        raise ValueError("SpecPrefill requires a loaded compatible draft model")
    return SpecPrefillOptions(
        enabled=True,
        keep_pct=(
            DEFAULT_SPECPREFILL_KEEP_PCT
            if specprefill_keep_pct is None
            else specprefill_keep_pct
        ),
        threshold=(
            DEFAULT_SPECPREFILL_THRESHOLD
            if specprefill_threshold is None
            else specprefill_threshold
        ),
        system_tokens=0
        if specprefill_system_tokens is None
        else specprefill_system_tokens,
    )


def _accepts_extractor_kwargs(extractor, kwargs) -> bool:
    """Return whether a query extractor accepts the supplied keyword arguments."""
    try:
        params = inspect.signature(extractor).parameters.values()
    except (TypeError, ValueError):
        return True

    for param in params:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    accepted = {
        param.name
        for param in params
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return set(kwargs).issubset(accepted)


def _qwen35_extract_queries(attn, x, cache=None, **kwargs):
    """Extract Qwen3.5 gated attention queries after q_norm and RoPE."""
    batch, length, _ = x.shape
    q_out = attn.q_proj(x)
    queries, _gate = mx.split(
        q_out.reshape(batch, length, attn.num_attention_heads, -1), 2, axis=-1
    )
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        return attn.rope(queries, offset=cache.offset)
    return attn.rope(queries)


def _qwen36_extract_queries(attn, x, cache=None, **kwargs):
    """Extract Qwen3.6 non-gated q_norm attention queries."""
    batch, length, _ = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x).reshape(batch, length, n_heads, -1)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        return attn.rope(queries, offset=cache.offset)
    return attn.rope(queries)


def _llama_extract_queries(attn, x, cache=None, **kwargs):
    """Extract standard transformer q_proj queries after RoPE."""
    batch, length, _ = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x)
    queries = queries.reshape(batch, length, n_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        return attn.rope(queries, offset=cache.offset)
    return attn.rope(queries)


def _gemma4_extract_queries(attn, x, cache=None, offset=None, **kwargs):
    """Extract Gemma 4 queries with shared-KV offset support."""
    batch, length, _ = x.shape
    n_heads = getattr(attn, "n_heads", getattr(attn, "num_attention_heads", None))
    queries = attn.q_proj(x).reshape(batch, length, n_heads, -1)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    rope_offset = (
        offset if offset is not None else (cache.offset if cache is not None else 0)
    )
    return attn.rope(queries, offset=rope_offset)


def _nemotron_h_extract_queries(attn, x, cache=None, **kwargs):
    """Extract Nemotron-H queries for content-only attention layers."""
    batch, length, _ = x.shape
    return (
        attn.q_proj(x).reshape(batch, length, attn.num_heads, -1).transpose(0, 2, 1, 3)
    )


def _find_attention_layers(model) -> list[tuple[int, Any]]:
    """Find full-attention layers across supported model topologies."""
    results = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            results.append((idx, layer))
        elif getattr(layer, "block_type", None) == "*":
            results.append((idx, layer))
    return results


def _get_attn_module(layer):
    """Return the attention module from a model layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if getattr(layer, "block_type", None) == "*":
        return layer.mixer
    return None


def _set_attn_module(layer, module) -> None:
    """Set the attention module on a model layer."""
    if hasattr(layer, "self_attn"):
        layer.self_attn = module
    elif getattr(layer, "block_type", None) == "*":
        layer.mixer = module


def _build_layer_to_cache_map(model) -> dict[int, int]:
    """Build a layer-index to cache-index map for sparse cache access."""
    for root in (
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "model", None),
        model,
    ):
        previous_kvs = getattr(root, "previous_kvs", None)
        if previous_kvs is not None:
            return dict(enumerate(previous_kvs))

    has_block_type = any(hasattr(layer, "block_type") for layer in model.layers)
    if not has_block_type:
        return {i: i for i in range(len(model.layers))}

    layer_to_cache = {}
    cache_idx = 0
    for layer_idx, layer in enumerate(model.layers):
        if getattr(layer, "block_type", None) in ("M", "*"):
            layer_to_cache[layer_idx] = cache_idx
            cache_idx += 1
    return layer_to_cache


def _linear_output_dims(linear) -> int | None:
    """Best-effort output dimension lookup for linear layer variants."""
    if linear is None:
        return None
    if hasattr(linear, "out_features"):
        return getattr(linear, "out_features")
    if hasattr(linear, "output_dims"):
        return getattr(linear, "output_dims")
    weight = getattr(linear, "weight", None)
    if weight is not None and getattr(weight, "shape", None):
        return weight.shape[0]
    return None


def _uses_gated_q_proj(attn_obj) -> bool:
    """Detect Qwen-style gated q_proj layout from projection dimensions."""
    n_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    head_dim = getattr(attn_obj, "head_dim", None)
    q_out = _linear_output_dims(getattr(attn_obj, "q_proj", None))
    if None in (n_heads, head_dim, q_out):
        return False
    return q_out == 2 * n_heads * head_dim


def _uses_non_gated_q_norm(attn_obj) -> bool:
    """Detect Qwen3.6-style per-head q_norm attention without gate split."""
    q_norm = getattr(attn_obj, "q_norm", None)
    if q_norm is None:
        return False
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_norm_weight is None:
        return False
    shape = getattr(q_norm_weight, "shape", None)
    if not shape:
        return False
    q_norm_dim = shape[0]

    n_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    q_out = _linear_output_dims(getattr(attn_obj, "q_proj", None))
    if not n_heads or q_out is None:
        return False
    return q_out == n_heads * q_norm_dim


def _is_gemma4_attention(attn_obj) -> bool:
    """Detect Gemma 4 attention by its shared-KV call contract."""
    if not hasattr(attn_obj, "q_norm") or not hasattr(attn_obj, "rope"):
        return False
    call = getattr(attn_obj, "__call__", None)
    if call is None:
        return False
    try:
        params = inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False
    return "shared_kv" in params and "offset" in params


def _detect_query_extractor(attn_obj) -> Callable:
    """Auto-detect the appropriate query extractor for a model architecture."""
    if _uses_gated_q_proj(attn_obj):
        return _qwen35_extract_queries
    if _is_gemma4_attention(attn_obj):
        return _gemma4_extract_queries
    if not hasattr(attn_obj, "rope"):
        return _nemotron_h_extract_queries
    if _uses_non_gated_q_norm(attn_obj):
        return _qwen36_extract_queries
    return _llama_extract_queries


def _patch_attention_for_capture(model, query_buffer, query_extractor):
    """Replace attention modules with query-capture wrappers."""
    originals = []
    attn_indices = []
    for layer_idx, layer in _find_attention_layers(model):
        buf_idx = len(attn_indices)
        attn_indices.append(layer_idx)
        original = _get_attn_module(layer)
        _set_attn_module(
            layer,
            _AttentionCapture(original, buf_idx, query_buffer, query_extractor),
        )
        originals.append((layer_idx, original))
    return originals, attn_indices


def _unpatch_attention_capture(model, originals) -> None:
    """Restore attention modules after query capture."""
    for layer_idx, original in originals:
        _set_attn_module(model.layers[layer_idx], original)


def _sync_and_clear_cache() -> None:
    """Synchronize MLX work before returning buffers to the Metal cache."""
    mx.synchronize()
    mx.clear_cache()


def _prefill_draft(model, tokens, cache, step_size=2048, progress_callback=None):
    """Prefill a draft model and return logits for the final prompt token."""
    prompt = mx.array(tokens) if not isinstance(tokens, mx.array) else tokens
    n_tokens = len(tokens)
    processed = 0
    while n_tokens - processed > 1:
        chunk = min(step_size, n_tokens - processed - 1)
        if progress_callback is not None:
            progress_callback(processed, n_tokens)
        model(prompt[processed : processed + chunk][None], cache=cache)
        mx.eval([c.state for c in cache])
        processed += chunk
        if progress_callback is not None:
            progress_callback(processed, n_tokens)
        _sync_and_clear_cache()
    logits = model(prompt[processed:][None], cache=cache)
    mx.eval(logits)
    if progress_callback is not None:
        progress_callback(n_tokens, n_tokens)
    return logits


def _lookahead_decode(model, first_logits, cache, n_steps, temp=0.6, top_p=0.95):
    """Run autoregressive lookahead decode while capture wrappers record queries."""
    sampler = create_sampler(
        temp=temp,
        top_p=top_p,
        min_p=0.0,
        min_tokens_to_keep=1,
        top_k=0,
    )
    token = sampler(first_logits[:, -1, :])
    mx.eval(token)
    generated = [token.item()]
    for _ in range(n_steps):
        logits = model(token.reshape(1, -1), cache=cache)
        token = sampler(logits[:, -1, :])
        mx.eval(token)
        generated.append(token.item())
    return generated


def _avg_pool1d(x, kernel_size):
    """Apply 1D average pooling along the last axis using a prefix sum."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    zeros = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    prefix = mx.concatenate([zeros, mx.cumsum(padded, axis=-1)], axis=-1)
    return (prefix[..., kernel_size:] - prefix[..., :-kernel_size]) / kernel_size


def _compute_importance(
    query_buffer, attn_caches, n_prompt, n_attn_heads, n_kv_heads, pool_kernel=13
):
    """Compute per-token importance from captured lookahead queries and KV keys."""
    if n_attn_heads is None:
        raise RuntimeError(
            "Cannot compute SpecPrefill importance without attention heads"
        )
    n_kv_heads = n_attn_heads if n_kv_heads is None else n_kv_heads
    heads_per_group = max(1, n_attn_heads // n_kv_heads)
    all_scores = []

    for layer_i, captures in enumerate(query_buffer):
        if not captures:
            continue
        cache = attn_caches[layer_i]
        if not hasattr(cache, "keys") or cache.keys is None:
            continue
        prompt_keys = cache.keys[..., :n_prompt, :]
        if prompt_keys.shape[-2] < n_prompt:
            continue
        head_dim = prompt_keys.shape[-1]
        q_stack = mx.concatenate(captures, axis=2)
        expanded_keys = (
            mx.repeat(prompt_keys, heads_per_group, axis=1)
            if heads_per_group > 1
            else prompt_keys
        )
        scores = (q_stack @ expanded_keys.transpose(0, 1, 3, 2)) * (head_dim**-0.5)
        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
        all_scores.append(weights.squeeze(0))

    if not all_scores:
        raise RuntimeError("No attention scores captured; check model/patching")

    combined = mx.concatenate(all_scores, axis=0)
    if pool_kernel and pool_kernel > 1:
        combined = _avg_pool1d(combined, pool_kernel)
    max_scores = mx.max(combined, axis=0)
    return mx.mean(max_scores, axis=0)


def score_tokens(
    model,
    tokens,
    n_lookahead: int = 8,
    pool_kernel: int = 13,
    temp: float = 0.6,
    top_p: float = 0.95,
    prefill_step_size: int = 2048,
    query_extractor: Callable | None = None,
) -> mx.array:
    """Score prompt-token importance with draft-model lookahead attention."""
    if isinstance(tokens, mx.array):
        tokens = tokens.tolist()
    n_prompt = len(tokens)
    if n_prompt <= 1:
        raise ValueError("SpecPrefill scoring requires at least two tokens")

    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        raise RuntimeError("SpecPrefill could not find attention layers")

    n_attn_layers = len(attn_layers)
    attn_obj = _get_attn_module(attn_layers[0][1])
    n_attn_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    n_kv_heads = getattr(
        attn_obj, "num_key_value_heads", getattr(attn_obj, "n_kv_heads", None)
    )
    if query_extractor is None:
        query_extractor = _detect_query_extractor(attn_obj)

    cache = make_prompt_cache(model)
    logits = _prefill_draft(model, tokens, cache, step_size=prefill_step_size)
    pre_lookahead_offset = cache[0].offset if hasattr(cache[0], "offset") else n_prompt

    query_buffer = [[] for _ in range(n_attn_layers)]
    patches, attn_indices = _patch_attention_for_capture(
        model, query_buffer, query_extractor
    )
    try:
        _lookahead_decode(model, logits, cache, n_lookahead, temp=temp, top_p=top_p)
        mx.eval(query_buffer)
    finally:
        _unpatch_attention_capture(model, patches)

    layer_to_cache = _build_layer_to_cache_map(model)
    attn_caches = [cache[layer_to_cache[i]] for i in attn_indices]
    importance = _compute_importance(
        query_buffer,
        attn_caches,
        n_prompt,
        n_attn_heads,
        n_kv_heads,
        pool_kernel=pool_kernel if pool_kernel > 0 else None,
    )
    mx.eval(importance)

    for cache_entry in cache:
        if hasattr(cache_entry, "offset") and cache_entry.offset > pre_lookahead_offset:
            if hasattr(cache_entry, "keys") and cache_entry.keys is not None:
                cache_entry.keys = cache_entry.keys[..., :pre_lookahead_offset, :]
                cache_entry.values = cache_entry.values[..., :pre_lookahead_offset, :]
            cache_entry.offset = pre_lookahead_offset

    del logits, query_buffer, attn_caches
    _sync_and_clear_cache()
    return importance


def select_chunks(
    importance: mx.array,
    keep_pct: float = DEFAULT_SPECPREFILL_KEEP_PCT,
    chunk_size: int = SPECPREFILL_SELECTION_CHUNK_SIZE,
) -> mx.array:
    """Select top-K percent token chunks by average importance."""
    n_tokens = importance.shape[0]
    if keep_pct >= 1.0:
        return mx.arange(n_tokens)

    n_chunks = math.ceil(n_tokens / chunk_size)
    keep_n = max(1, math.ceil(n_chunks * keep_pct))

    chunk_scores = []
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_tokens)
        chunk_scores.append(mx.mean(importance[start:end]).item())

    top_chunks = sorted(range(n_chunks), key=lambda i: chunk_scores[i], reverse=True)[
        :keep_n
    ]
    top_chunks.sort()

    indices = []
    for chunk_idx in top_chunks:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_tokens)
        indices.extend(range(start, end))

    return mx.array(indices)


def manual_rope(x, positions, dims, base=10000.0, scale=1.0):
    """Apply RoPE at arbitrary non-contiguous positions."""
    half = dims // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    scaled_pos = positions.astype(mx.float32) / scale
    angles = scaled_pos[:, None] * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


def manual_rope_with_freqs(x, positions, dims, freqs, pre_scale=1.0):
    """Apply RoPE at arbitrary positions using pre-computed frequencies."""
    half = dims // 2
    inv_freq = (1.0 / freqs).astype(mx.float32)
    angles = positions[:, None].astype(mx.float32) * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    if pre_scale != 1.0:
        x_rot = pre_scale * x_rot
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


def _scalar_offset(offset) -> int:
    """Coerce scalar RoPE offsets to Python integers."""
    if isinstance(offset, mx.array):
        return int(offset.item())
    return int(offset)


def _get_dims(rope_module):
    """Extract rotary dimensions from supported RoPE variants."""
    for attr in ("_dims", "dim", "dims"):
        if hasattr(rope_module, attr):
            return getattr(rope_module, attr)
    raise ValueError(f"Cannot determine dims from {type(rope_module)}")


def _get_pre_scale(rope_module):
    """Extract pre-scale factor from supported custom RoPE variants."""
    if hasattr(rope_module, "mscale"):
        return rope_module.mscale
    if hasattr(rope_module, "_scale") and hasattr(rope_module, "dim"):
        return rope_module._scale
    return 1.0


def sparse_prefill(
    model,
    tokens,
    selected_indices,
    cache,
    step_size: int = 2048,
    position_offset: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> mx.array:
    """Prefill target KV cache with selected tokens at original positions."""
    if not isinstance(tokens, mx.array):
        tokens = mx.array(tokens)
    if not isinstance(selected_indices, mx.array):
        selected_indices = mx.array(selected_indices)

    n_prompt = tokens.shape[0]
    if selected_indices.shape[0] == 0:
        raise ValueError("SpecPrefill requires at least one selected token")

    selected_positions = selected_indices.astype(mx.int32) + position_offset
    selected_tokens = tokens[selected_indices]
    n_selected = selected_tokens.shape[0]

    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        raise RuntimeError("SpecPrefill could not find attention layers")
    layer_to_cache = _build_layer_to_cache_map(model)
    first_attn_layer_idx = attn_layers[0][0]
    first_attn_cache_idx = layer_to_cache[first_attn_layer_idx]
    cache_start = (
        cache[first_attn_cache_idx].offset
        if hasattr(cache[first_attn_cache_idx], "offset")
        else 0
    )

    first_attn = _get_attn_module(attn_layers[0][1])
    has_rope = hasattr(first_attn, "rope")
    original_ropes = {}
    if has_rope:
        for layer_idx, layer in attn_layers:
            attn = _get_attn_module(layer)
            original_ropes[layer_idx] = attn.rope
            attn.rope = _PositionMappedRoPE(
                attn.rope, selected_positions, cache_start=cache_start
            )

    try:
        processed = 0
        while int(n_selected) - processed > 1:
            chunk = min(step_size, int(n_selected) - processed - 1)
            if progress_callback is not None:
                progress_callback(processed, int(n_selected))
            model(selected_tokens[processed : processed + chunk][None], cache=cache)
            mx.eval([c.state for c in cache])
            processed += chunk
            if progress_callback is not None:
                progress_callback(processed, int(n_selected))
            _sync_and_clear_cache()

        logits = model(selected_tokens[processed:][None], cache=cache)
        mx.eval(logits)
        if progress_callback is not None:
            progress_callback(int(n_selected), int(n_selected))
    finally:
        if has_rope:
            total_prompt_len = position_offset + n_prompt
            final_cache_offset = cache_start + n_selected
            adjustment = int(total_prompt_len) - int(final_cache_offset)
            for layer_idx, layer in attn_layers:
                attn = _get_attn_module(layer)
                original = original_ropes[layer_idx]
                if adjustment > 0:
                    attn.rope = _OffsetAdjustedRoPE(original, adjustment)
                else:
                    attn.rope = original

    return logits


def cleanup_rope(model) -> None:
    """Restore original RoPE wrappers after a SpecPrefill request."""
    for _, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if attn is None or not hasattr(attn, "rope"):
            continue
        rope = attn.rope
        if isinstance(rope, (_OffsetAdjustedRoPE, _PositionMappedRoPE)):
            attn.rope = rope._original


def _prefill_system_tokens(
    *,
    model: Any,
    tokens: mx.array,
    cache: list[Any],
    chunk_size: int,
    reporter: Any,
) -> None:
    """Fully prefill protected system tokens before sparse conversation prefill."""
    processed = 0
    total = int(tokens.shape[0])
    while total - processed > 0:
        chunk = min(chunk_size, total - processed)
        reporter.update(False, processed)
        model(tokens[processed : processed + chunk][None], cache=cache)
        mx.eval([c.state for c in cache])
        processed += chunk
        reporter.update(False, processed)
        _sync_and_clear_cache()


def _decrement_decode_rope_adjustment(model) -> None:
    """Account for the final seed token that stream_generate processes next."""
    for _, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if (
            attn is not None
            and hasattr(attn, "rope")
            and isinstance(attn.rope, _OffsetAdjustedRoPE)
        ):
            attn.rope._adjustment -= 1


def try_sparse_prefill(
    *,
    model: Any,
    draft_model: Any,
    prompt_tokens: mx.array,
    uncached_tokens: mx.array,
    cache: list[Any],
    cached_tokens: int,
    options: SpecPrefillOptions,
    chunk_size: int,
    reporter: Any,
) -> SpecPrefillResult | None:
    """Attempt guarded SpecPrefill and return a sparse cache plus seed token."""
    if len(uncached_tokens) <= 1:
        return None

    system_tokens = options.system_tokens
    conversation_tokens = uncached_tokens[system_tokens:]
    if len(conversation_tokens) <= options.threshold:
        return None

    try:
        importance = score_tokens(
            draft_model,
            conversation_tokens,
            prefill_step_size=chunk_size,
        )
        selected = select_chunks(importance, keep_pct=options.keep_pct)
        last_idx = len(conversation_tokens) - 1
        selected_list = selected.tolist()
        if last_idx in selected_list:
            selected_list.remove(last_idx)
        if len(selected_list) == 0:
            return None
        selected = mx.array(sorted(selected_list), dtype=mx.int32)

        if system_tokens > 0:
            _prefill_system_tokens(
                model=model,
                tokens=uncached_tokens[:system_tokens],
                cache=cache,
                chunk_size=chunk_size,
                reporter=reporter,
            )

        sparse_prefill(
            model,
            conversation_tokens,
            selected,
            cache,
            step_size=chunk_size,
            position_offset=cached_tokens + system_tokens,
            progress_callback=lambda processed, total: reporter.update(
                False, system_tokens + processed
            ),
        )
        _decrement_decode_rope_adjustment(model)
    except Exception:
        cleanup_rope(model)
        raise

    logger.info(
        "SpecPrefill selected %d/%d uncached conversation tokens",
        int(selected.shape[0]),
        len(conversation_tokens),
    )
    return SpecPrefillResult(
        cache=cache,
        seed_tokens=uncached_tokens[-1:],
        live_tokens=prompt_tokens.tolist(),
    )
