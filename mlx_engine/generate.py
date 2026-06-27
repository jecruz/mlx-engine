from contextlib import contextmanager
from queue import Queue
import uuid
import importlib
from functools import lru_cache
from mlx_engine.model_kit.batched_model_kit import (
    BatchedGenerationResponse,
    BatchedModelKit,
)
from mlx_engine.model_kit.batched_model_kit_types import RequestCancelled
from typing import Any, Iterator, List, Optional, TypeAlias
import json
import logging
from pathlib import Path
import sys
import threading

from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params
from mlx_lm.generate import stream_generate
from mlx_lm.utils import load as mlx_lm_load
from mlx_lm.models.cache import make_prompt_cache

from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.model_kit.distributed_model_kit import DistributedModelKit
from mlx_engine.utils.token import Token
from mlx_engine.utils.eot_tokens import sanitize_eos_tokens
from mlx_engine.utils.top_logprobs import summarize_top_logprobs
from mlx_engine.stop_string_processor import (
    StopStringProcessorResult,
)
from mlx_engine.utils.generation_result import (
    GenerationStopCondition,
    GenerationResult,
    construct_user_cancelled_result,
)
from mlx_engine.utils.request_state import (
    model_kit_has_active_requests,
    request_id_is_empty,
)
from mlx_engine.utils.set_seed import set_seed
from mlx_engine.utils.speculative_decoding import (
    determine_draft_model_for_generation,
    configure_num_draft_tokens_in_generate_args,
    is_speculative_decoding_supported,
    SpeculativeDecodingNotSupportedError,
)
from mlx_engine.utils.suffix_decoding_runtime import (
    resolve_suffix_decoding_options,
    validate_suffix_decoding_compatibility,
    suffix_stream_generate,
)
from mlx_engine.utils.specprefill import (
    DEFAULT_SPECPREFILL_KEEP_PCT,
    DEFAULT_SPECPREFILL_THRESHOLD,
    SpecPrefillOptions,
    resolve_specprefill_options,
)
from outlines.processors.structured import JSONLogitsProcessor
from mlx_engine.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from mlx_engine.cache_wrapper import validate_prefill_step_size
from mlx_engine.utils.prompt_progress_reporter import (
    BatchedMlxLmReporterAdapter,
    LoggerReporter,
    PromptProgressReporter,
    DefaultPromptProgressReporter,
    MlxLmReporterAdapter,
    StopPromptProcessing,
)
from mlx_engine.utils.generation_helpers import (
    setup_repetition_logits_processors,
    validate_top_logprobs,
    create_stop_string_processor,
    process_stop_string_check,
    should_yield_token,
)
from mlx_engine.utils.sampling import create_sampler
from mlx_engine.utils.mlx_lm_stream import (
    describe_stream_configuration,
    emit_stream_configuration_probe,
    log_mlx_generation_exception,
    log_mlx_stream_state,
    prepare_mlx_lm_generation_stream,
)

MAX_TOP_LOGPROBS = 10
DEFAULT_BATCHED_PREFILL_STEP_SIZE = 4096
DEFAULT_SEQUENTIAL_TEXT_PREFILL_STEP_SIZE = 4096


logger = logging.getLogger(__name__)

SequentialGenerationKit: TypeAlias = ModelKit
BatchedGenerationKit: TypeAlias = Any
LoadedModelKit: TypeAlias = (
    SequentialGenerationKit | BatchedGenerationKit | DistributedModelKit
)


def resolve_batched_prefill_step_size(
    prefill_step_size: int,
    prefill_step_size_was_unspecified: bool,
    use_batched_kit: bool,
    model_type: str | None = None,
) -> int:
    """Return the effective prefill chunk size for batched text inference."""
    if (
        prefill_step_size_was_unspecified
        and use_batched_kit
        and model_type is not None
        and model_type == "qwen3_5_text"
    ):
        return DEFAULT_BATCHED_PREFILL_STEP_SIZE
    return prefill_step_size


def resolve_sequential_text_prefill_step_size(
    prefill_step_size: int,
    prefill_step_size_was_unspecified: bool,
    model_type: str | None = None,
) -> int:
    """Return the effective prefill chunk size for sequential text inference."""
    if (
        prefill_step_size_was_unspecified
        and model_type is not None
        and model_type in {"qwen2", "qwen3_5_text"}
    ):
        return DEFAULT_SEQUENTIAL_TEXT_PREFILL_STEP_SIZE
    return prefill_step_size


@lru_cache(maxsize=1)
def _load_batched_vision_model_kit():
    """Load the VLM batcher lazily so text-only imports stay lightweight."""
    from mlx_engine.model_kit.batched_vision import BatchedVisionModelKit

    return BatchedVisionModelKit


class _SequentialModelKitGenerator(Iterator[GenerationResult]):
    def __init__(self, model_kit: ModelKit, prompt_tokens: List[int], kwargs: dict):
        self._model_kit = model_kit
        self._prompt_tokens = prompt_tokens
        self._kwargs = kwargs
        self._request_id = kwargs.get("request_id")
        self._results: Queue[tuple[str, object]] = Queue()
        self._closed = threading.Event()
        self._started = False
        self._exhausted = False
        if self._request_id is not None and self._request_id != "":
            self._model_kit.pending_requests[self._request_id] = threading.Event()

    def __iter__(self) -> "_SequentialModelKitGenerator":
        return self

    def __next__(self) -> GenerationResult:
        if self._closed.is_set() or self._exhausted:
            raise StopIteration
        if not self._started:
            self._started = True
            self._model_kit._executor.submit(self._generate)

        kind, value = self._results.get()
        if kind == "item":
            return value  # type: ignore[return-value]
        if kind == "error":
            self._exhausted = True
            raise value  # type: ignore[misc]
        self._exhausted = True
        raise StopIteration

    def close(self) -> None:
        self._closed.set()
        if self._request_id is not None and self._request_id != "":
            self._model_kit.cancel_request(self._request_id)
            if not self._started:
                self._model_kit.pending_requests.pop(self._request_id, None)

    def _generate(self) -> None:
        stream = None
        try:
            stream = _sequential_generation(
                self._model_kit,
                self._prompt_tokens,
                **self._kwargs,
            )
            for item in stream:
                if self._closed.is_set():
                    break
                self._results.put(("item", item))
                if self._closed.is_set():
                    break
        except BaseException as exc:
            self._results.put(("error", exc))
        finally:
            if stream is not None:
                stream.close()
            self._results.put(("done", None))


def _handle_stop_string_detected(
    tokenizer,
    stop_string_processor_result: StopStringProcessorResult,
    text: str,
    token_buffer: List[Token],
    top_logprobs_buffer: List[List[Token]],
) -> GenerationResult:
    """
    Helper method to Handle completion of text generation when a stop string is
    encountered.

    Args:
        tokenizer: The tokenizer instance
        stop_string_processor_result: Result from stop string processor
        text: Current generated text
        token_buffer: Buffer of generated tokens
        top_logprobs_buffer: Buffer of token probabilities

    Returns:
        GenerationResult: Final generation result including stop condition
    """
    # Finalize detokenizer to get remaining text
    detokenizer = tokenizer.detokenizer
    detokenizer.finalize()
    text += detokenizer.last_segment

    # Process stop string by trimming text segment where it begins
    stop_string = stop_string_processor_result.stop_string
    stop_string_start_pos = text.find(stop_string)

    if stop_string_start_pos != -1:
        text = text[:stop_string_start_pos]
    else:
        # this is known to happen when the eos token is a stop string
        sys.stderr.write(
            f"[mlx-engine] Stop string '{stop_string}' not found in final text segment, "
            "even though a full stop was detected. Not trimming final segment."
        )

    stop_condition = GenerationStopCondition(
        stop_reason="stop_string",
        stop_string=stop_string,
        stop_tokens=stop_string_processor_result.stop_tokens,
    )

    return GenerationResult(
        text=text,
        tokens=token_buffer,
        stop_condition=stop_condition,
        top_logprobs=top_logprobs_buffer,
    )



def _is_known_vlm_model_type(model_type: str) -> bool:
    """Check if model_type is a known mlx-vlm vision architecture.

    Some text-only models (e.g. qwen3.6) inherit a ``vision_config`` key from
    a shared architecture config, so the presence of ``vision_config`` alone
    is not sufficient to determine whether a model should be routed to the
    mlx-vlm batched vision backend.
    """
    try:
        importlib.import_module(f"mlx_vlm.models.{model_type}")
        return True
    except (ImportError, ValueError):
        return False


def load_model(
    model_path: str | Path,
    *,
    vocab_only: bool = False,
    max_kv_size: int | None = 4096,
    max_seq_nums: int | None = None,
    seed: int | None = None,
    trust_remote_code: bool = False,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
    prefill_step_size: Optional[int] = None,
    distributed: bool = False,
    distributed_group: Any = None,
    vlm_prompt_cache_storage_root: str | Path | None = None,
    vlm_prompt_cache_namespace: str | None = None,
    vlm_prompt_cache_min_save_tokens: int | None = None,
) -> LoadedModelKit:
    """
    Load a language model or vision-language model from the specified path.

    This function determines the model type based on the config.json file in the model directory
    and initializes either a standard language model or a vision-language model accordingly.

    Args:
        model_path (str | Path): Path to the model directory containing model files and config.json.
        vocab_only (bool): Only load vocabulary/tokenizer, not the full model.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        max_seq_nums (int | None): The maximum number of parallel generation requests that can be worked on.
            When omitted, text-only loads default to the low-latency sequential path. Pass a value
            greater than 1 to enable batched text inference.
        seed (Optional[int]): Random seed for reproducible generation. If provided, sets the
            random seed for all subsequent generation operations with this model.
        trust_remote_code (bool): Whether to allow loading of remote code during model initialization.
        kv_bits (Optional[int]): Number of bits for KV cache quantization.
        kv_group_size (Optional[int]): Group size for KV cache quantization.
        quantized_kv_start (Optional[int]): Step to begin KV cache quantization when enabled.
        prefill_step_size (Optional[int]): Number of tokens to process per prefill chunk.
            Defaults to PROMPT_PROCESSING_CHUNK_SIZE when None.
        distributed (bool): Load the model through DistributedModelKit for MLX distributed
            tensor parallel inference.
        distributed_group (Any): Optional initialized MLX distributed group. If omitted,
            DistributedModelKit initializes the group.
        vlm_prompt_cache_storage_root (str | Path | None): Optional persistent VLM prompt-cache
            directory. When omitted, VLM prompt cache remains model-load-lifetime temporary storage.
        vlm_prompt_cache_namespace (str | None): Optional namespace used to isolate persistent VLM
            prompt-cache records. Defaults to the resolved model path.
        vlm_prompt_cache_min_save_tokens (int | None): Minimum image-expanded reusable
            prompt tokens needed before VLM prompt-cache records are saved. Persistent
            stores default to 512; temporary stores default to 0.

    Returns:
        LoadedModelKit: An initialized model instance:
            - ModelKit: for sequential text-only models and vocab-only loads
            - BatchedModelKit: for text-only continuous batching
            - BatchedVisionModelKit: for mlx-vlm continuous batching
            - DistributedModelKit: for distributed tensor parallel text-only models

    Raises:
        FileNotFoundError: If config.json is not found in the specified model path
        json.JSONDecodeError: If config.json exists but contains invalid JSON
        ValueError: If the model configuration is invalid or unsupported
    """
    prefill_step_size_was_unspecified = prefill_step_size is None
    set_seed(seed)
    prefill_step_size = validate_prefill_step_size(prefill_step_size)
    vlm_prompt_cache_storage_root = (
        None
        if vlm_prompt_cache_storage_root is None
        else Path(vlm_prompt_cache_storage_root)
    )

    if distributed:
        if vlm_prompt_cache_storage_root is not None:
            raise ValueError(
                "VLM prompt cache persistence is only supported for BatchedVisionModelKit"
            )
        if vlm_prompt_cache_min_save_tokens is not None:
            raise ValueError(
                "VLM prompt cache save admission is only supported for BatchedVisionModelKit"
            )
        if vocab_only:
            raise ValueError("Distributed loading does not support vocab_only")
        if any([kv_bits, kv_group_size, quantized_kv_start]):
            raise ValueError(
                "Distributed loading does not currently support KV cache quantization"
            )
        logger.info(
            "Creating DistributedModelKit model_path=%s max_kv_size=%s max_seq_nums=%s prefill_step_size=%s distributed_group_provided=%s",
            model_path,
            max_kv_size,
            max_seq_nums,
            prefill_step_size,
            distributed_group is not None,
        )
        model_kit = DistributedModelKit(
            model_path,
            prefill_step_size=prefill_step_size,
            max_kv_size=max_kv_size,
            max_seq_nums=max_seq_nums,
            trust_remote_code=trust_remote_code,
            distributed_group=distributed_group,
        )
        logger.info("Sanitizing EOS tokens for DistributedModelKit")
        sanitize_eos_tokens(model_kit)
        logger.info("Starting DistributedModelKit")
        model_kit.start()
        logger.info("DistributedModelKit start completed")
        return model_kit

    model_path = Path(model_path)
    config_json = json.loads((model_path / "config.json").read_text())
    parallel_requested = max_seq_nums is not None and max_seq_nums > 1

    def warn_if_parallel(reason: str) -> None:
        """Helper to warn about batching not being supported, only if parallel was requested."""
        if parallel_requested:
            logger.warning(
                f"max_concurrent_predictions={max_seq_nums} was specified, but {reason}. "
                f"The model will process requests sequentially."
            )

    # Determine which model kit to use based on model capabilities and configuration.
    # The decision tree is:
    # 1. BatchedVisionModelKit: mlx-vlm continuous batching for VLMs
    # 2. BatchedModelKit: continuous batching for text-only models
    # 3. ModelKit: fallback for sequential text-only processing
    # Guard: only route to BatchedVisionModelKit if the model type is actually
    # a known mlx-vlm vision architecture. Some text-only models inherit
    # vision_config from a shared architecture config (e.g. qwen35 arch used
    # by qwen3.6 text models), so "vision_config" alone is not sufficient.
    model_type = config_json.get("model_type", "").lower().replace("-", "_").replace(".", "_")
    is_vlm = "vision_config" in config_json and _is_known_vlm_model_type(model_type)
    if is_vlm and vocab_only:
        if vlm_prompt_cache_storage_root is not None:
            raise ValueError(
                "VLM prompt cache persistence requires loading the VLM backend"
            )
        if vlm_prompt_cache_min_save_tokens is not None:
            raise ValueError(
                "VLM prompt cache save admission requires loading the VLM backend"
            )
        model_kit = ModelKit(
            model_path,
            prefill_step_size=prefill_step_size,
            vocab_only=True,
            seed=seed,
        )
    elif is_vlm:
        BatchedVisionModelKit = _load_batched_vision_model_kit()
        if any([kv_bits, kv_group_size, quantized_kv_start]):
            raise ValueError(
                "The mlx-vlm batched vision path does not support KV cache quantization yet"
            )
        model_kit = BatchedVisionModelKit(
            model_path,
            prefill_step_size=prefill_step_size,
            max_kv_size=max_kv_size,
            max_seq_nums=max_seq_nums,
            trust_remote_code=trust_remote_code,
            seed=seed,
            prompt_cache_storage_root=vlm_prompt_cache_storage_root,
            prompt_cache_namespace=vlm_prompt_cache_namespace,
            prompt_cache_min_save_tokens=vlm_prompt_cache_min_save_tokens,
        )
    else:
        if vlm_prompt_cache_storage_root is not None:
            raise ValueError(
                "VLM prompt cache persistence is only supported for VLM models"
            )
        if vlm_prompt_cache_min_save_tokens is not None:
            raise ValueError(
                "VLM prompt cache save admission is only supported for VLM models"
            )
        # For non-vision models, choose between BatchedModelKit
        # (continuous batching) and ModelKit (sequential).
        kv_bits, kv_group_size, quantized_kv_start = get_kv_cache_quantization_params(
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )

        def is_batchable() -> bool:
            # 0. Ensure the load isn't vocab only
            if vocab_only:
                return False
            # 1. All cache layers must support merge
            model, _ = mlx_lm_load(model_path, lazy=True)
            cache_has_merge_attr = all(
                hasattr(c, "merge") for c in make_prompt_cache(model)
            )
            del model
            if not cache_has_merge_attr:
                warn_if_parallel(
                    "this model architecture does not support continuous batching"
                )
                return False
            # 2. KV cache quantization is not compatible with batching yet
            if kv_bits is not None:
                warn_if_parallel(
                    "concurrency is not supported with KV Cache Quantization"
                )
                return False
            return True

        # If max_seq_nums is set to 1, use ModelKit instead of BatchedModelKit. This gives users an escape hatch,
        # which they could use to enable spec decoding. We can remove this additional restriction once we add
        # spec decoding support to the batched backend
        use_batched_kit = max_seq_nums is not None and max_seq_nums > 1 and is_batchable()
        if not use_batched_kit:
            prefill_step_size = resolve_sequential_text_prefill_step_size(
                prefill_step_size,
                prefill_step_size_was_unspecified,
                model_type=model_type,
            )
        batched_prefill_step_size = resolve_batched_prefill_step_size(
            prefill_step_size,
            prefill_step_size_was_unspecified,
            use_batched_kit,
            model_type=model_type,
        )

        if use_batched_kit:
            emit_stream_configuration_probe(
                reason="load-model-batched-text",
                use_default_stream=False,
            )
            logger.info(
                "Text load resolved to BatchedModelKit model_path=%s model_type=%s "
                "max_seq_nums=%s prefill_step_size=%s stream_config=%s",
                model_path,
                model_type,
                max_seq_nums,
                batched_prefill_step_size,
                describe_stream_configuration(False),
            )
            model_kit = BatchedModelKit(
                model_path,
                max_kv_size=max_kv_size,
                max_seq_nums=max_seq_nums,
                prefill_step_size=batched_prefill_step_size,
                seed=seed,
            )
        else:
            emit_stream_configuration_probe(
                reason="load-model-sequential-text",
                use_default_stream=False,
            )
            logger.info(
                "Text load resolved to ModelKit model_path=%s model_type=%s "
                "max_seq_nums=%s prefill_step_size=%s stream_config=%s",
                model_path,
                model_type,
                max_seq_nums,
                prefill_step_size,
                describe_stream_configuration(False),
            )
            model_kit = ModelKit(
                model_path,
                prefill_step_size=prefill_step_size,
                vocab_only=vocab_only,
                max_kv_size=max_kv_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                seed=seed,
            )
    if type(model_kit) is not ModelKit:
        sanitize_eos_tokens(model_kit)
    model_kit.start()
    return model_kit


def load_draft_model(
    model_kit: LoadedModelKit,
    path: str | Path,
) -> None:
    if not is_speculative_decoding_supported(model_kit):
        raise SpeculativeDecodingNotSupportedError(
            "Speculative decoding is not supported for batched MLX models."
        )
    model_kit.load_draft_model(path)


def is_draft_model_compatible(
    model_kit: LoadedModelKit,
    path: str | Path,
) -> bool:
    if not is_speculative_decoding_supported(model_kit):
        return False
    return model_kit.is_draft_model_compatible(path)


def unload_draft_model(
    model_kit: LoadedModelKit,
) -> None:
    if not is_speculative_decoding_supported(model_kit):
        return
    model_kit.unload_draft_model()


def create_generator(
    model_kit: LoadedModelKit,
    prompt_tokens: List[int],
    **kwargs,
) -> Iterator[GenerationResult]:
    """
    Create a generator that streams text generation results from the model.

    This function sets up and manages the text generation process, handling various generation
    parameters, processing callbacks, and managing generation constraints. It supports both
    standard language models and vision-language models.

    Args:
        model_kit (LoadedModelKit): The initialized model to use for generation
        prompt_tokens (List[int]): List of token IDs representing the input prompt
        prompt_progress_reporter (Optional[PromptProgressReporter]): Reporter for receiving prompt
            processing progress updates. Reporter methods should return True to continue processing,
            or False to stop generation
        images_b64 (Optional[List[str]]): List of base64-encoded images for vision-language models
        max_image_size (Optional[tuple[int, int]]): Maximum dimensions (width, height) for images.
            Images will be resized to fit within these dimensions while maintaining aspect ratio if
            they exceed this size. If None, no resizing.
        stop_strings (Optional[List[str]]): List of strings that will trigger generation to stop
            when encountered
        top_logprobs (Optional[int]): Number of top token probabilities to return per token
            Must be <= MAX_TOP_LOGPROBS
        repetition_penalty (Optional[float]): Penalty factor for repeated tokens. Higher values
            discourage repetition
        repetition_context_size (Optional[int]): Number of previous tokens to consider for
            repetition penalty. Defaults to 20
        temp (Optional[float]): Temperature for sampling. Higher values increase randomness
        top_p (Optional[float]): Top-p (nucleus) sampling parameter
        top_k (Optional[int]): Top-k sampling parameter
        min_p (Optional[float]): Minimum probability threshold for token sampling
        min_tokens_to_keep (Optional[int]): Minimum number of tokens to keep during sampling
        seed (Optional[int]): Random seed for reproducible generation
        json_schema (Optional[str]): JSON schema for structured output generation
        max_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to 10000000
        speculative_decoding_toggle (Optional[bool]): If not set, use speculative decoding
            if a draft model is loaded. If set to true, draft model must be loaded or else error.
            If set to false, speculative decoding is disabled even if a draft model is loaded.
        num_draft_tokens (Optional[int]): Number of tokens to draft when using speculative decoding
        specprefill_toggle (Optional[bool]): Enable the guarded SpecPrefill prompt-processing path.
        specprefill_keep_pct (Optional[float]): Fraction of selected prompt chunks when SpecPrefill is enabled.
        specprefill_threshold (Optional[int]): Minimum uncached prompt tokens needed to try SpecPrefill.
        specprefill_system_tokens (Optional[int]): Protected system-token prefix for future sparse-prefill helpers.
        request_id (Optional[int]): Id associated with the request

    Yields:
        GenerationResult: A named tuple containing:
            - text (str): Generated text segment
            - tokens (List[TokenLogprob]): List of generated tokens with their probabilities
            - top_logprobs (List[List[TokenLogprob]]): Token probability information if requested
            - stop_condition (Optional[GenerationStopCondition]): Information about why
              generation stopped, if applicable

    Raises:
        ValueError: If top_logprobs exceeds MAX_TOP_LOGPROBS or if any parameters are invalid
    """
    BatchedVisionModelKit = _load_batched_vision_model_kit()
    suffix_decoding_options = resolve_suffix_decoding_options(
        kwargs.get("suffix_decoding_toggle"),
        kwargs.get("suffix_decoding_max_draft_tokens"),
    )
    if isinstance(model_kit, (BatchedModelKit, BatchedVisionModelKit)) or (
        isinstance(model_kit, DistributedModelKit)
        and model_kit.uses_distributed_batching()
    ):
        if suffix_decoding_options.enabled:
            raise ValueError(
                "SuffixDecoding is only supported for sequential text generation"
            )
        specprefill_keys = (
            "specprefill_toggle",
            "specprefill_keep_pct",
            "specprefill_threshold",
            "specprefill_system_tokens",
        )
        if kwargs.get("specprefill_toggle") is True or any(
            kwargs.get(key) is not None for key in specprefill_keys[1:]
        ):
            raise ValueError("SpecPrefill is only supported for sequential generation")
        batched_kwargs = dict(kwargs)
        batched_kwargs.pop("suffix_decoding_toggle", None)
        batched_kwargs.pop("suffix_decoding_max_draft_tokens", None)
        for key in specprefill_keys:
            batched_kwargs.pop(key, None)
        return _batched_generation(model_kit, prompt_tokens, **batched_kwargs)
    if isinstance(model_kit, DistributedModelKit):
        request_id = kwargs.get("request_id")
        logger.info(
            "Routing sequential distributed generation request_id=%s prompt_tokens=%s "
            "through distributed model thread",
            request_id,
            len(prompt_tokens),
        )
        return model_kit.run_generator_on_model_thread(
            description=f"sequential-generation request_id={request_id}",
            callback=lambda: _sequential_generation(
                model_kit,
                prompt_tokens,
                **kwargs,
            ),
        )
    return _SequentialModelKitGenerator(model_kit, prompt_tokens, kwargs)


@contextmanager
def _sequential_gen_abort_handler(
    model_kit: SequentialGenerationKit | DistributedModelKit,
    request_id: Optional[str],
):
    """
    Acquires the generation lock for sequential generation, with support for cancellation.

    Creates a per-request cancellation event that can be signaled while waiting for the lock
    or during generation.
    """

    should_track_request = True
    if request_id is None or request_id == "":
        cancel_event = threading.Event()
        logger.warning(
            "request_id missing for sequential generation; cancellation by id is disabled"
        )
        should_track_request = False
    else:
        cancel_event = model_kit.pending_requests.setdefault(
            request_id, threading.Event()
        )

    try:
        # Try to acquire lock, checking for cancellation while waiting
        while True:
            if cancel_event.is_set() or model_kit.is_shutdown():
                # The request is cancelled. Bypass acquiring the lock and let the generator yield a "user cancelled" result
                yield cancel_event
                return

            if model_kit.generation_lock.acquire(timeout=0.1):
                break

        try:
            yield cancel_event
        finally:
            model_kit.generation_lock.release()
    finally:
        if should_track_request:
            model_kit.pending_requests.pop(request_id, None)


def _sequential_generation(
    model_kit: SequentialGenerationKit | DistributedModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_reporter: Optional[PromptProgressReporter] = None,
    images_b64: Optional[List[str]] = None,
    max_image_size: Optional[tuple[int, int]] = None,
    stop_strings: Optional[List[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
    speculative_decoding_toggle: Optional[bool] = None,
    num_draft_tokens: Optional[int] = None,
    suffix_decoding_toggle: Optional[bool] = None,
    suffix_decoding_max_draft_tokens: Optional[int] = None,
    specprefill_toggle: Optional[bool] = None,
    specprefill_keep_pct: Optional[float] = None,
    specprefill_threshold: Optional[int] = None,
    specprefill_system_tokens: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Iterator[GenerationResult]:
    with _sequential_gen_abort_handler(model_kit, request_id) as cancel_event:
        if cancel_event.is_set() or model_kit.is_shutdown():
            yield construct_user_cancelled_result()
            return

        set_seed(seed)

        generate_args = {}
        if prompt_progress_reporter is None:
            prompt_progress_reporter = LoggerReporter()

        # Set up kv cache
        for attr in [
            "max_kv_size",
            "kv_bits",
            "kv_group_size",
            "quantized_kv_start",
        ]:
            value = getattr(model_kit, attr, None)
            if value is not None:
                generate_args[attr] = value

        suffix_decoding_options = resolve_suffix_decoding_options(
            suffix_decoding_toggle,
            suffix_decoding_max_draft_tokens,
        )
        validate_suffix_decoding_compatibility(
            suffix_decoding_enabled=suffix_decoding_options.enabled,
            model_kit=model_kit,
            images_b64=images_b64,
            speculative_decoding_toggle=speculative_decoding_toggle,
            num_draft_tokens=num_draft_tokens,
            specprefill_toggle=specprefill_toggle,
        )

        # Set up speculative decoding and SpecPrefill.  SpecPrefill uses the
        # loaded draft model for prefill scoring only; it deliberately disables
        # decode speculation until that combined mode is tested.
        if specprefill_toggle is True and (
            speculative_decoding_toggle is True or num_draft_tokens is not None
        ):
            raise ValueError(
                "SpecPrefill cannot be combined with decode speculative decoding yet"
            )

        specprefill_options = None
        specprefill_scoring_model = None
        if specprefill_toggle is True:
            threshold = (
                DEFAULT_SPECPREFILL_THRESHOLD
                if specprefill_threshold is None
                else specprefill_threshold
            )
            # Validate tuning even when this prompt is too short to use the
            # sparse path, but do not route ineligible prompts through the
            # SpecPrefill prompt-processing branch.
            SpecPrefillOptions(
                enabled=True,
                keep_pct=(
                    DEFAULT_SPECPREFILL_KEEP_PCT
                    if specprefill_keep_pct is None
                    else specprefill_keep_pct
                ),
                threshold=threshold,
                system_tokens=(
                    0
                    if specprefill_system_tokens is None
                    else specprefill_system_tokens
                ),
            )
            if len(prompt_tokens) > threshold:
                specprefill_scoring_model = getattr(model_kit, "draft_model", None)
                specprefill_options = resolve_specprefill_options(
                    specprefill_toggle=specprefill_toggle,
                    specprefill_keep_pct=specprefill_keep_pct,
                    specprefill_threshold=specprefill_threshold,
                    specprefill_system_tokens=specprefill_system_tokens,
                    draft_model=specprefill_scoring_model,
                )
            draft_model = None
        else:
            draft_model = determine_draft_model_for_generation(
                model_kit, speculative_decoding_toggle
            )
        decode_draft_model = draft_model
        configure_num_draft_tokens_in_generate_args(
            model_kit, decode_draft_model, num_draft_tokens, generate_args
        )

        # Process prompt
        try:
            input_tokens, input_embeddings = model_kit.process_prompt(
                prompt_tokens,
                images_b64,
                prompt_progress_reporter,
                generate_args,
                max_image_size,
                speculative_decoding_toggle,
                draft_model_override=specprefill_scoring_model,
                specprefill_options=specprefill_options,
            )
        except StopPromptProcessing:
            yield construct_user_cancelled_result()
            return
        if decode_draft_model is None and not suffix_decoding_options.enabled:
            # input embeddings not yet supported for speculative decoding in mlx-lm
            generate_args["input_embeddings"] = input_embeddings

        # Setup logits processors
        logits_processors = setup_repetition_logits_processors(
            repetition_penalty,
            repetition_context_size,
            prompt_tokens,
            input_tokens,
        )

        # Set up sampler
        generate_args["sampler"] = create_sampler(
            temp, top_p, min_p, min_tokens_to_keep, top_k
        )

        # Validate top_logprobs
        top_logprobs = validate_top_logprobs(top_logprobs)

        # Keep track of tokens buffered by detokenizer to yield accurate generation results
        token_buffer: List[Token] = []
        top_logprobs_buffer: List[List[Token]] = []

        tokenizer = model_kit.tokenizer

        # Add outlines logits processor if json_schema is provided
        if json_schema is not None:
            logits_processors.append(
                JSONLogitsProcessor(
                    json_schema,
                    OutlinesTransformerTokenizer(model_kit.tokenizer._tokenizer),
                    tensor_library_name="mlx",
                )
            )

        # Set up stop string processor if non-empty stop_strings are provided
        stop_string_processor = create_stop_string_processor(stop_strings, tokenizer)
        text = ""

        # Determine callback for mlx-lm based on processing mode
        # When cache is NOT active (vision prompts), stream_generate handles prompt processing
        # When cache IS active (text-only), cache_wrapper already handled it
        if not model_kit.is_cross_prompt_cache_active():
            mlx_lm_callback = MlxLmReporterAdapter(
                prompt_progress_reporter, emit_begin=True
            )
        else:
            mlx_lm_callback = None

        is_distributed_model = isinstance(model_kit, DistributedModelKit)
        distributed_group = model_kit.group if is_distributed_model else None
        generation_details = (
            f"mode=sequential distributed={is_distributed_model} "
            f"prompt_tokens={len(prompt_tokens)} input_tokens={len(input_tokens)} "
            f"max_tokens={max_tokens} prefill_step_size={model_kit.prefill_step_size} "
            f"max_kv_size={getattr(model_kit, 'max_kv_size', None)} "
            f"cross_prompt_cache={model_kit.is_cross_prompt_cache_active()}"
        )
        try:
            prepare_mlx_lm_generation_stream(
                reason="sequential-generation",
                request_id=request_id,
                distributed_group=distributed_group,
                use_default_stream=is_distributed_model,
            )
            log_mlx_stream_state(
                reason="before-stream-generate",
                request_id=request_id,
                distributed_group=distributed_group,
                details=generation_details,
            )

            if suffix_decoding_options.enabled:
                stream = suffix_stream_generate(
                    model=model_kit.model,
                    tokenizer=tokenizer,
                    prompt=input_tokens,
                    max_tokens=max_tokens,
                    logits_processors=logits_processors,
                    prompt_progress_callback=mlx_lm_callback,
                    prefill_step_size=model_kit.prefill_step_size,
                    max_draft_tokens=suffix_decoding_options.max_draft_tokens,
                    **generate_args,
                )
            else:
                stream = stream_generate(
                    model=model_kit.model,
                    tokenizer=tokenizer,
                    draft_model=decode_draft_model,
                    prompt=input_tokens,
                    max_tokens=max_tokens,
                    logits_processors=logits_processors,
                    prompt_progress_callback=mlx_lm_callback,
                    prefill_step_size=model_kit.prefill_step_size,
                    **generate_args,
                )
            log_mlx_stream_state(
                reason="after-stream-generate-created",
                request_id=request_id,
                distributed_group=distributed_group,
                details=generation_details,
            )

            received_first_generation_result = False
            while not model_kit.is_shutdown() and not cancel_event.is_set():
                try:
                    if not received_first_generation_result:
                        log_mlx_stream_state(
                            reason="before-first-generation-next",
                            request_id=request_id,
                            distributed_group=distributed_group,
                            details=generation_details,
                        )
                    generation_result = next(stream)
                    if not received_first_generation_result:
                        received_first_generation_result = True
                        log_mlx_stream_state(
                            reason="after-first-generation-next",
                            request_id=request_id,
                            distributed_group=distributed_group,
                            details=(
                                f"{generation_details} token={generation_result.token} "
                                f"text_len={len(generation_result.text)}"
                            ),
                        )
                except StopIteration:
                    break
                except StopPromptProcessing:
                    yield construct_user_cancelled_result()
                    return
                except Exception:
                    log_mlx_generation_exception(
                        reason="sequential-generation",
                        request_id=request_id,
                        distributed_group=distributed_group,
                    )
                    raise

                # Token processor
                token = generation_result.token
                text += generation_result.text
                # record generated token to cache, if cache is active
                if model_kit.is_cross_prompt_cache_active():
                    model_kit.record_token_to_cache(token)

                logprobs = generation_result.logprobs
                token_buffer.append(
                    Token(
                        token,
                        tokenizer.decode(token),
                        float(logprobs[token]),
                        from_draft=generation_result.from_draft,
                    )
                )
                if top_logprobs:
                    top_logprobs_buffer.append(
                        summarize_top_logprobs(tokenizer, logprobs, top_logprobs)
                    )

                # Stop processor
                should_stop, should_buffer, stop_result = process_stop_string_check(
                    stop_string_processor, token
                )
                if should_stop:
                    yield _handle_stop_string_detected(
                        tokenizer,
                        stop_result,
                        text,
                        token_buffer,
                        top_logprobs_buffer,
                    )
                    break  # stop generation

                # If we currently have generated a partial match with a stop sequence, or detected an
                # in-progress multi-byte string, generate new tokens until we know if the stop sequence
                # is hit or not (i.e., make sure not to yield yet)
                if should_buffer:
                    continue

                # Standard yield - yield when a non-empty text segment is available or eos token is hit
                should_yield, stop_condition = should_yield_token(text, token, tokenizer)
                if (
                    stop_condition is None
                    and generation_result.finish_reason == "length"
                ):
                    should_yield = True
                    stop_condition = GenerationStopCondition(
                        stop_reason="token_limit",
                        stop_string="",
                        stop_tokens=[],
                    )
                if should_yield:
                    yield GenerationResult(
                        text=text,
                        tokens=token_buffer,
                        stop_condition=stop_condition,
                        top_logprobs=top_logprobs_buffer,
                    )
                    token_buffer = []
                    top_logprobs_buffer = []
                    text = ""
            if cancel_event.is_set() or model_kit.is_shutdown():
                yield construct_user_cancelled_result()
            return
        finally:
            if specprefill_options is not None and hasattr(
                model_kit, "cleanup_specprefill"
            ):
                model_kit.cleanup_specprefill()


def _batched_generation(
    model_kit: BatchedGenerationKit | DistributedModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_reporter: Optional[PromptProgressReporter] = None,
    images_b64: Optional[List[str]] = None,
    max_image_size: Optional[tuple[int, int]] = None,
    stop_strings: Optional[List[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,  # Seed arg is ignored for batched gen
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
    speculative_decoding_toggle: Optional[bool] = None,
    num_draft_tokens: Optional[int] = None,
    request_id: str | None = None,
) -> Iterator[GenerationResult]:
    is_distributed_batched = isinstance(model_kit, DistributedModelKit)
    if is_distributed_batched:
        if images_b64 is not None and len(images_b64) > 0:
            raise ValueError("Distributed batched generation does not support images yet")
        if speculative_decoding_toggle is True or num_draft_tokens is not None:
            raise ValueError(
                "Distributed batched generation does not support speculative decoding yet"
            )
        if json_schema is not None:
            raise ValueError(
                "Distributed batched generation does not support structured JSON output yet"
            )
        if seed is not None:
            raise ValueError(
                "Distributed batched generation does not support request-level seeds yet"
            )

    # We need a request_id so that we can communicate with the batched backend
    if request_id is None or request_id == "":
        logger.warning(
            "Received a generation request without a request_id! Please send a request_id"
        )
        request_id = str(uuid.uuid4())

    input_tokens = prompt_tokens
    if prompt_progress_reporter is None:
        prompt_progress_reporter = DefaultPromptProgressReporter()

    tokenizer = model_kit.tokenizer
    # Validate top_logprobs
    top_logprobs = validate_top_logprobs(top_logprobs)

    # Keep track of tokens buffered by detokenizer to yield accurate generation results
    token_buffer: List[Token] = []
    top_logprobs_buffer: List[List[Token]] = []

    # Set up stop string processor if non-empty stop_strings are provided
    stop_string_processor = create_stop_string_processor(stop_strings, tokenizer)
    text = ""

    logits_processors = setup_repetition_logits_processors(
        repetition_penalty,
        repetition_context_size,
        prompt_tokens,
        input_tokens,
    )
    sampler = create_sampler(temp, top_p, min_p, min_tokens_to_keep, top_k)

    if is_distributed_batched:
        prompt_progress_callback = BatchedMlxLmReporterAdapter(
            prompt_progress_reporter, emit_begin=True
        )
        stream = model_kit.generate(
            prompt_tokens=input_tokens,
            request_id=request_id,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_progress_callback=prompt_progress_callback,
            top_logprobs=top_logprobs,
            sampling={
                "temperature": temp,
                "topP": top_p,
                "topK": top_k,
                "minP": min_p,
                "seed": seed,
            },
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            min_tokens_to_keep=min_tokens_to_keep,
        )
    elif isinstance(model_kit, _load_batched_vision_model_kit()):
        # `max_image_size` is legacy-only; batched VLM lets mlx-vlm processors resize.
        if json_schema is not None:
            from mlx_vlm.structured import build_json_schema_logits_processor

            logits_processors.append(
                build_json_schema_logits_processor(
                    model_kit.tokenizer._tokenizer,
                    json_schema,
                )
            )

        stream = model_kit.generate(
            prompt_tokens=input_tokens,
            request_id=request_id,
            images_b64=images_b64,
            max_tokens=max_tokens,
            prompt_progress_reporter=prompt_progress_reporter,
            top_logprobs=top_logprobs,
            sampler=sampler,
            logits_processors=logits_processors,
        )
    else:
        prompt_progress_callback = BatchedMlxLmReporterAdapter(
            prompt_progress_reporter, emit_begin=True
        )
        # Add outlines logits processor if json_schema is provided
        if json_schema is not None:
            logits_processors.append(
                JSONLogitsProcessor(
                    json_schema,
                    OutlinesTransformerTokenizer(model_kit.tokenizer._tokenizer),
                    tensor_library_name="mlx",
                )
            )

        stream = model_kit.generate(
            prompt_tokens=input_tokens,
            request_id=request_id,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_progress_callback=prompt_progress_callback,
            top_logprobs=top_logprobs,
        )

    while True:
        try:
            generation_result: BatchedGenerationResponse = next(stream)
        except StopIteration:
            break
        except RequestCancelled:
            yield construct_user_cancelled_result()
            return
        except StopPromptProcessing:
            yield construct_user_cancelled_result()
            return
        except Exception:
            distributed_group = (
                model_kit.group if isinstance(model_kit, DistributedModelKit) else None
            )
            log_mlx_generation_exception(
                reason="batched-generation",
                request_id=request_id,
                distributed_group=distributed_group,
            )
            raise

        # Token processor
        token = generation_result.token
        text += generation_result.text

        token_buffer.append(
            Token(
                token,
                tokenizer.decode(token),
                generation_result.token_logprob,
                from_draft=generation_result.from_draft,
            )
        )
        if top_logprobs and generation_result.top_logprobs is not None:
            top_logprobs_buffer.append(generation_result.top_logprobs)

        # Stop processor
        should_stop, should_buffer, stop_result = process_stop_string_check(
            stop_string_processor, token
        )
        if should_stop:
            yield _handle_stop_string_detected(
                tokenizer,
                stop_result,
                text,
                token_buffer,
                top_logprobs_buffer,
            )
            model_kit.remove(request_id)
            break  # stop generation

        # If we currently have generated a partial match with a stop sequence, or detected an
        # in-progress multi-byte string, generate new tokens until we know if the stop sequence
        # is hit or not (i.e., make sure not to yield yet)
        if should_buffer:
            continue

        # Standard yield - yield when a non-empty text segment is available or eos token is hit
        should_yield, stop_condition = should_yield_token(text, token, tokenizer)
        if should_yield:
            yield GenerationResult(
                text=text,
                tokens=token_buffer,
                stop_condition=stop_condition,
                top_logprobs=top_logprobs_buffer,
            )
            token_buffer = []
            top_logprobs_buffer = []
            text = ""

        # The batched generator has hit max_tokens, so we can't iterate further
        if generation_result.finish_reason == "length":
            yield GenerationResult(
                text="",
                tokens=[],
                stop_condition=GenerationStopCondition(
                    stop_reason="token_limit",
                    stop_string="",
                    stop_tokens=[],
                ),
                top_logprobs=[],
            )
            return


def stop_generation(
    model_kit: LoadedModelKit,
    request_id: str,
):
    """
    Register stop request based off of request_id
    """
    if request_id_is_empty(request_id):
        logger.debug("Ignoring empty stop request")
        return

    BatchedVisionModelKit = _load_batched_vision_model_kit()
    if isinstance(model_kit, (BatchedModelKit, BatchedVisionModelKit)) or (
        isinstance(model_kit, DistributedModelKit)
        and model_kit.uses_distributed_batching()
    ):
        model_kit.remove(request_id)
        return

    if not model_kit.cancel_request(request_id):
        logger.warning(f"Could not cancel {request_id=} (request not found)")


def unload(
    model_kit: LoadedModelKit,
    *,
    force: bool = False,
):
    """Shutdown a loaded model, blocking accidental unloads while requests run."""
    if not force and model_kit_has_active_requests(model_kit):
        raise RuntimeError("Cannot unload a model while requests are still active")
    model_kit.shutdown()


def tokenize(
    model_kit: LoadedModelKit,
    prompt: str,
) -> List[int]:
    """
    Convert a text prompt into a list of token IDs using the model's tokenizer.

    Args:
        model_kit (LoadedModelKit): The model kit instance containing the tokenizer
            to use for tokenization
        prompt (str): The raw text prompt to be tokenized

    Returns:
        List[int]: A list of integer token IDs representing the tokenized prompt,
            ready for model input
    """
    return model_kit.tokenize(prompt)
