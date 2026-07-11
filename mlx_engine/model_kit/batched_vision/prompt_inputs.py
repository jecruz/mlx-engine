import hashlib
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx_vlm
from mlx.utils import tree_flatten

from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PreparedPromptMetadata,
    PromptImageSpan,
)
from mlx_engine.model_kit.batched_vision.qwen_mrope import (
    apply_qwen_image_mrope_state,
    build_qwen_image_mrope_state,
)
from mlx_engine.model_kit.batched_vision.vision_feature_memoizer import (
    VisionFeatureMemoizer,
)
from mlx_engine.utils.image_utils import convert_to_pil


@dataclass
class PreparedPrompt:
    """Prompt tokens plus optional mlx-vlm multimodal processor inputs."""

    prompt_input_ids: list[int]
    raw_inputs: dict[str, Any] | None
    image_spans: list[PromptImageSpan]
    vision_cache_key: str | None = None


def build_prepared_prompt_request_key(
    prompt_tokens: list[int],
    images_b64: list[str] | None,
) -> str | None:
    """Return an exact request key for image-bearing prompt preparation."""
    if not images_b64:
        return None
    digest = hashlib.sha256()
    digest.update(b"mlx-engine-vlm-prepared-prompt-v1\0")
    for token in prompt_tokens:
        digest.update(int(token).to_bytes(8, "little", signed=True))
    digest.update(b"\0images\0")
    for image_b64 in images_b64:
        image_digest = hashlib.sha256(image_b64.encode("utf-8")).digest()
        digest.update(len(image_b64).to_bytes(8, "little"))
        digest.update(image_digest)
    return digest.hexdigest()


def metadata_from_prepared_prompt(
    request_key: str,
    prepared_prompt: PreparedPrompt,
) -> PreparedPromptMetadata:
    """Build persistent metadata for an exact prepared VLM prompt."""
    image_grid_thw = None
    if prepared_prompt.raw_inputs is not None:
        raw_grid = prepared_prompt.raw_inputs.get("image_grid_thw")
        if raw_grid is not None:
            image_grid_thw = raw_grid.tolist()
    return PreparedPromptMetadata(
        request_key=request_key,
        prompt_input_ids=list(prepared_prompt.prompt_input_ids),
        image_spans=list(prepared_prompt.image_spans),
        vision_cache_key=prepared_prompt.vision_cache_key,
        image_grid_thw=image_grid_thw,
    )


def prepared_prompt_from_metadata(metadata: PreparedPromptMetadata) -> PreparedPrompt:
    """Rebuild minimal prompt inputs from exact persistent metadata."""
    raw_inputs: dict[str, Any] | None = None
    if metadata.image_spans:
        raw_inputs = {
            "input_ids": mx.array(
                [metadata.prompt_input_ids],
                dtype=mx.int32,
            ),
        }
        if metadata.image_grid_thw is not None:
            raw_inputs["image_grid_thw"] = mx.array(
                metadata.image_grid_thw,
                dtype=mx.int32,
            )
    return PreparedPrompt(
        prompt_input_ids=list(metadata.prompt_input_ids),
        raw_inputs=raw_inputs,
        image_spans=list(metadata.image_spans),
        vision_cache_key=metadata.vision_cache_key,
    )


def prepare_prompt_inputs(
    *,
    prompt_tokens: list[int],
    images_b64: list[str] | None,
    tokenizer,
    processor,
    config: dict,
) -> PreparedPrompt:
    """Prepare one request for local batched VLM prompt processing."""
    if len(prompt_tokens) == 0:
        prompt_tokens = _tokenize(tokenizer, " ")

    if not images_b64:
        return PreparedPrompt(
            prompt_input_ids=list(prompt_tokens),
            raw_inputs=None,
            image_spans=[],
            vision_cache_key=None,
        )

    # Request prep runs on the cache I/O thread before generation insertion.
    prompt = tokenizer.decode(prompt_tokens) or " "
    images = convert_to_pil(images_b64)
    image_token_index = get_image_token_index(config)
    raw_inputs = mlx_vlm.prepare_inputs(
        processor=processor,
        images=images,
        prompts=prompt,
        image_token_index=image_token_index,
        resize_shape=None,
    )
    _eval_mlx_arrays(raw_inputs)
    prompt_input_ids = raw_inputs["input_ids"].squeeze(0).tolist()
    image_hashes = [_hash_prompt_image(image) for image in images]
    return PreparedPrompt(
        prompt_input_ids=prompt_input_ids,
        raw_inputs=raw_inputs,
        image_spans=_get_image_spans(
            prompt_input_ids,
            image_hashes,
            image_token_index,
        ),
        vision_cache_key=_build_vision_cache_key(image_hashes),
    )


def _eval_mlx_arrays(value: Any) -> None:
    """Materialize processor tensors before handing them to another thread."""
    arrays = [leaf for _, leaf in tree_flatten(value) if isinstance(leaf, mx.array)]
    if arrays:
        mx.eval(arrays)


def build_prompt_kwargs(
    model,
    prepared_prompt: PreparedPrompt,
    vision_feature_memoizer: VisionFeatureMemoizer | None = None,
) -> dict:
    """Build model kwargs for a full prompt prefill."""
    if prepared_prompt.raw_inputs is None:
        input_ids = mx.array(prepared_prompt.prompt_input_ids, dtype=mx.int32)[None, :]
        _clear_qwen3_5_text_rope_state(model)
        embedding_output = model.get_input_embeddings(input_ids)
        _clear_qwen3_5_text_rope_state(model)
        return {
            key: value
            for key, value in embedding_output.to_dict().items()
            if value is not None
        }

    raw_inputs = prepared_prompt.raw_inputs
    input_ids = raw_inputs["input_ids"]
    pixel_values = raw_inputs.get("pixel_values")
    attention_mask = raw_inputs.get("attention_mask")
    data_kwargs = {
        key: value
        for key, value in raw_inputs.items()
        if key not in {"input_ids", "pixel_values", "attention_mask"}
    }
    embedding_kwargs = {
        **data_kwargs,
        **_build_vision_feature_cache_kwargs(
            model,
            prepared_prompt,
            pixel_values,
            vision_feature_memoizer,
        ),
    }
    embedding_output = model.get_input_embeddings(
        input_ids,
        pixel_values,
        mask=attention_mask,
        **embedding_kwargs,
    )
    apply_qwen_image_mrope_state(
        model,
        input_ids=input_ids,
        image_grid_thw=raw_inputs.get("image_grid_thw"),
    )
    prompt_kwargs = {
        **data_kwargs,
        **{
            key: value
            for key, value in embedding_output.to_dict().items()
            if value is not None
        },
    }
    _route_attention_mask_4d(model, prompt_kwargs)
    _add_language_model_rope_state(model, prompt_kwargs)
    return prompt_kwargs


def build_cached_prompt_kwargs(
    model,
    prepared_prompt: PreparedPrompt,
    cached_prefix_len: int,
    rope_deltas: Any | None,
    vision_feature_memoizer: VisionFeatureMemoizer | None = None,
) -> dict:
    """Build model kwargs for the uncached suffix after a prefix restore."""
    prompt_input_ids = prepared_prompt.prompt_input_ids[cached_prefix_len:]
    if prepared_prompt.raw_inputs is not None:
        fast_kwargs = _build_qwen_cached_text_suffix_prompt_kwargs(
            model,
            prepared_prompt,
            cached_prefix_len,
        )
        if fast_kwargs is None:
            fast_kwargs = _build_lfm2_cached_text_suffix_prompt_kwargs(
                model,
                prepared_prompt,
                cached_prefix_len,
            )
        if fast_kwargs is not None:
            return fast_kwargs
        if vision_feature_memoizer is None:
            prompt_kwargs = build_prompt_kwargs(model, prepared_prompt)
        else:
            prompt_kwargs = build_prompt_kwargs(
                model,
                prepared_prompt,
                vision_feature_memoizer,
            )
        return slice_prompt_kwargs(
            prompt_kwargs,
            cached_prefix_len,
            len(prepared_prompt.prompt_input_ids),
        )

    input_ids = mx.array(prompt_input_ids, dtype=mx.int32)[None, :]
    _clear_qwen3_5_text_rope_state(model)
    embedding_output = model.get_input_embeddings(input_ids)
    _clear_qwen3_5_text_rope_state(model)
    prompt_kwargs = {
        key: value
        for key, value in embedding_output.to_dict().items()
        if value is not None
    }

    if cached_prefix_len > 0 and _add_qwen_text_restore_rope_state(
        model,
        prompt_kwargs,
        cached_prefix_len,
        len(prompt_input_ids),
        input_ids.dtype,
        rope_deltas,
    ):
        return prompt_kwargs

    # Non-Qwen prefix restores carry the tiny RoPE delta side state in memory.
    if rope_deltas is not None and prompt_kwargs.get("rope_deltas") is None:
        prompt_kwargs["rope_deltas"] = rope_deltas

    return prompt_kwargs


def _build_lfm2_cached_text_suffix_prompt_kwargs(
    model,
    prepared_prompt: PreparedPrompt,
    cached_prefix_len: int,
) -> dict | None:
    """Build an LFM2-VL text suffix without recomputing image features."""
    raw_inputs = prepared_prompt.raw_inputs
    config = getattr(model, "config", None)
    if raw_inputs is None or str(getattr(config, "model_type", "")) != "lfm2_vl":
        return None

    image_token_id = getattr(config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is None:
        return None

    suffix_ids = prepared_prompt.prompt_input_ids[cached_prefix_len:]
    if not suffix_ids or image_token_id in suffix_ids:
        return None

    input_ids = raw_inputs.get("input_ids")
    dtype = input_ids.dtype if input_ids is not None else mx.int32
    suffix_input_ids = mx.array(suffix_ids, dtype=dtype)[None, :]
    embedding_output = model.get_input_embeddings(suffix_input_ids)
    return {
        key: value
        for key, value in embedding_output.to_dict().items()
        if value is not None
    }


def _build_qwen_cached_text_suffix_prompt_kwargs(
    model,
    prepared_prompt: PreparedPrompt,
    cached_prefix_len: int,
) -> dict | None:
    """Build a Qwen image-restore suffix without recomputing vision embeddings."""
    raw_inputs = prepared_prompt.raw_inputs
    if raw_inputs is None:
        return None

    input_ids = raw_inputs.get("input_ids")
    image_grid_thw = raw_inputs.get("image_grid_thw")
    if input_ids is None or image_grid_thw is None:
        return None

    config = getattr(model, "config", None)
    image_token_id = getattr(config, "image_token_id", None)
    vision_config = getattr(config, "vision_config", None)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
    if image_token_id is None or spatial_merge_size is None:
        return None

    suffix_ids = prepared_prompt.prompt_input_ids[cached_prefix_len:]
    if not suffix_ids or image_token_id in suffix_ids:
        return None

    mrope_state = build_qwen_image_mrope_state(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
    )
    suffix_input_ids = mx.array(suffix_ids, dtype=input_ids.dtype)[None, :]
    _clear_qwen3_5_text_rope_state(model)
    embedding_output = model.get_input_embeddings(suffix_input_ids)
    _clear_qwen3_5_text_rope_state(model)
    prompt_kwargs = {
        key: value
        for key, value in embedding_output.to_dict().items()
        if value is not None
    }
    prompt_kwargs["position_ids"] = mrope_state.position_ids[:, :, cached_prefix_len:]
    prompt_kwargs["rope_deltas"] = mrope_state.rope_deltas
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        language_model._position_ids = prompt_kwargs["position_ids"]
        language_model._rope_deltas = prompt_kwargs["rope_deltas"]
    return prompt_kwargs


def slice_prompt_kwargs(
    prompt_kwargs: dict,
    start: int,
    end: int,
    *,
    mask_key_end: int | None = None,
) -> dict:
    """Return prompt kwargs for token range `[start, end)`."""
    sliced = dict(prompt_kwargs)

    if "inputs_embeds" in sliced:
        sliced["inputs_embeds"] = sliced["inputs_embeds"][:, start:end]
    if "position_ids" in sliced:
        # Qwen MRoPE positions are token-local and use shape (3, B, S).
        sliced["position_ids"] = sliced["position_ids"][:, :, start:end]
    if "mask" in sliced:
        # Multimodal masks are query-local; chunked prefill clips future keys.
        sliced["mask"] = sliced["mask"][:, :, start:end, :mask_key_end]
    if "per_layer_inputs" in sliced:
        sliced["per_layer_inputs"] = sliced["per_layer_inputs"][:, start:end]
    if "mm_token_type_ids" in sliced:
        sliced["mm_token_type_ids"] = sliced["mm_token_type_ids"][:, start:end]
    if "token_type_ids" in sliced:
        sliced["token_type_ids"] = sliced["token_type_ids"][:, start:end]

    visual_pos_masks = prompt_kwargs.get("visual_pos_masks")
    if visual_pos_masks is not None:
        sliced["visual_pos_masks"] = visual_pos_masks[:, start:end]
        if "deepstack_visual_embeds" in sliced:
            sliced["deepstack_visual_embeds"] = _slice_deepstack_visual_embeds(
                sliced["deepstack_visual_embeds"],
                visual_pos_masks,
                start,
                end,
            )

    return sliced


def drop_prompt_kwargs_prefix(prompt_kwargs: dict, length: int) -> dict:
    """Drop `length` already-prefilled tokens from prompt-local kwargs."""
    total_len = _prompt_kwargs_token_len(prompt_kwargs)
    if total_len is None:
        return prompt_kwargs
    return slice_prompt_kwargs(prompt_kwargs, length, total_len)


def _route_attention_mask_4d(model, prompt_kwargs: dict) -> None:
    # Batched prefill calls the language model directly, so generic 4D masks
    # must be passed as `mask` where the language model will consume them.
    if getattr(model, "model_type", None) == "gemma3":
        # Gemma3's dense padding mask is not the causal attention mask.
        prompt_kwargs.pop("attention_mask_4d", None)
        return

    attention_mask_4d = prompt_kwargs.pop("attention_mask_4d", None)
    if attention_mask_4d is not None:
        prompt_kwargs["mask"] = attention_mask_4d


def _build_vision_feature_cache_kwargs(
    model,
    prepared_prompt: PreparedPrompt,
    pixel_values: mx.array | None,
    vision_feature_memoizer: VisionFeatureMemoizer | None,
) -> dict:
    cache_key = prepared_prompt.vision_cache_key
    if vision_feature_memoizer is None or cache_key is None or pixel_values is None:
        return {}

    # Match mlx-vlm's batched server path: pass the image-key cache kwargs
    # broadly, then keep them out of the language-model prefill kwargs.
    return {
        "vision_cache": vision_feature_memoizer.cache,
        "_image_key": cache_key,
    }


def _add_language_model_rope_state(model, prompt_kwargs: dict) -> None:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return

    position_ids = getattr(language_model, "_position_ids", None)
    if position_ids is not None:
        prompt_kwargs["position_ids"] = position_ids

    rope_deltas = getattr(language_model, "_rope_deltas", None)
    if rope_deltas is not None:
        prompt_kwargs["rope_deltas"] = rope_deltas


def _clear_qwen3_5_text_rope_state(model) -> bool:
    language_model = getattr(model, "language_model", None)
    model_type = str(getattr(language_model, "model_type", ""))
    if language_model is None or not model_type.startswith("qwen3_5"):
        return False

    language_model._position_ids = None
    language_model._rope_deltas = None
    return True


def _add_qwen_text_restore_rope_state(
    model,
    prompt_kwargs: dict,
    cached_prefix_len: int,
    suffix_len: int,
    dtype,
    rope_deltas: Any | None,
) -> bool:
    language_model = getattr(model, "language_model", None)
    model_type = str(getattr(language_model, "model_type", ""))
    if language_model is None or not model_type.startswith("qwen"):
        return False

    if _clear_qwen3_5_text_rope_state(model):
        return True

    # Text restores start from a nonzero KV offset, so pass explicit text MRoPE
    # positions. Reset model-side deltas so later decode capture is not stale.
    if rope_deltas is None:
        rope_deltas = mx.zeros((1, 1), dtype=dtype)
    language_model._rope_deltas = rope_deltas
    prompt_kwargs["position_ids"] = mx.broadcast_to(
        mx.arange(
            cached_prefix_len,
            cached_prefix_len + suffix_len,
            dtype=dtype,
        ).reshape(1, 1, suffix_len),
        (3, 1, suffix_len),
    )
    return True


def _slice_deepstack_visual_embeds(
    deepstack_visual_embeds,
    visual_pos_masks: mx.array,
    start: int,
    end: int,
):
    # Some models, including Granite4, return DeepStack embeds aligned to the
    # full prompt sequence instead of packed by visual-token order.
    if (
        isinstance(deepstack_visual_embeds, mx.array)
        and deepstack_visual_embeds.ndim >= 3
        and deepstack_visual_embeds.shape[1] == visual_pos_masks.shape[1]
    ):
        return deepstack_visual_embeds[:, start:end]

    # DeepStack embeds are packed by visual-token order, not sequence position.
    visual_start = int(mx.sum(visual_pos_masks[:, :start]).item())
    visual_count = int(mx.sum(visual_pos_masks[:, start:end]).item())
    visual_end = visual_start + visual_count

    if isinstance(deepstack_visual_embeds, tuple):
        return tuple(
            embed[visual_start:visual_end] for embed in deepstack_visual_embeds
        )
    if isinstance(deepstack_visual_embeds, list):
        return [embed[visual_start:visual_end] for embed in deepstack_visual_embeds]
    return deepstack_visual_embeds[visual_start:visual_end]


def _prompt_kwargs_token_len(prompt_kwargs: dict) -> int | None:
    if "position_ids" in prompt_kwargs:
        return prompt_kwargs["position_ids"].shape[-1]
    if "visual_pos_masks" in prompt_kwargs:
        return prompt_kwargs["visual_pos_masks"].shape[1]
    if "per_layer_inputs" in prompt_kwargs:
        return prompt_kwargs["per_layer_inputs"].shape[1]
    if "mm_token_type_ids" in prompt_kwargs:
        return prompt_kwargs["mm_token_type_ids"].shape[1]
    if "token_type_ids" in prompt_kwargs:
        return prompt_kwargs["token_type_ids"].shape[1]
    if "inputs_embeds" in prompt_kwargs:
        return prompt_kwargs["inputs_embeds"].shape[1]
    if "mask" in prompt_kwargs:
        return prompt_kwargs["mask"].shape[-2]
    return None


def get_image_token_index(config: dict) -> int | None:
    for value in (
        config.get("image_token_index"),
        config.get("image_token_id"),
        config.get("media_placeholder_token_id"),
        config.get("vision_config", {}).get("image_token_id"),
    ):
        if value is not None:
            return value
    return None


def _tokenize(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    if isinstance(ids, int):
        return [ids]
    return ids


def _hash_prompt_image(image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode())
    digest.update(f"{image.size[0]}x{image.size[1]}".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _build_vision_cache_key(image_hashes: list[str]) -> str:
    return f"prepared-images:{'|'.join(image_hashes)}"


def _get_image_spans(
    prompt_input_ids: list[int],
    image_hashes: list[str],
    image_token_index: int | None,
) -> list[PromptImageSpan]:
    if not image_hashes:
        return []

    if image_token_index is None:
        # Some processors do not expose a stable image sentinel; keep the
        # cache correct by making the whole prompt image-dependent.
        return [PromptImageSpan(0, len(prompt_input_ids), "|".join(image_hashes))]

    token_spans = []
    span_start = None
    for i, token_id in enumerate(prompt_input_ids):
        if token_id == image_token_index:
            if span_start is None:
                span_start = i
        elif span_start is not None:
            token_spans.append((span_start, i))
            span_start = None
    if span_start is not None:
        token_spans.append((span_start, len(prompt_input_ids)))

    if len(token_spans) != len(image_hashes):
        # Mismatched processor output is rare, but wrong image reuse is worse
        # than missing a cache hit.
        return [PromptImageSpan(0, len(prompt_input_ids), "|".join(image_hashes))]

    return [
        PromptImageSpan(start, end, image_hash)
        for (start, end), image_hash in zip(token_spans, image_hashes)
    ]
