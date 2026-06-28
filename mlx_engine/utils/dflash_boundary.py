"""Guarded DFlash boundary and dependency probe for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Optional

from mlx_engine.utils.dflash_snapshot import DFlashSnapshotError, load_dflash_snapshot_profile


DFLASH_ENV = "MLX_ENGINE_DFLASH"
DFLASH_TARGET_MODEL_ENV = "MLX_ENGINE_DFLASH_TARGET_MODEL"
DFLASH_DRAFTER_MODEL_ENV = "MLX_ENGINE_DFLASH_DRAFTER_MODEL"
DFLASH_MAX_DRAFT_TOKENS_ENV = "MLX_ENGINE_DFLASH_MAX_DRAFT_TOKENS"
DEFAULT_DFLASH_MAX_DRAFT_TOKENS = 4
DFLASH_EXPECTED_DTYPE = "bfloat16"
DFLASH_REQUIRED_TARGET_TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
DFLASH_RESOURCE_PORTS = (3180, 3181, 3182, 12444)
DFLASH_AVAILABLE_MEMORY_HEADROOM_RATIO = 0.25
DFLASH_AVAILABLE_MEMORY_HEADROOM_MIN_BYTES = 8 * 1024 * 1024 * 1024

_DFLASH_DEPENDENCY_MODULES = (
    "mlx_vlm.speculative.dflash",
    "mlx_vlm.speculative.drafters.qwen3_dflash.dflash",
)
_SUPPORTED_MODEL_MARKERS = ("qwen",)
_UNSUPPORTED_MODEL_MARKERS = ("moe", "a3b")


@dataclass(frozen=True, slots=True)
class DFlashBoundaryOptions:
    """Validated opt-in state for the guarded DFlash boundary."""

    enabled: bool
    target_model_path: Path | None = None
    drafter_model_path: Path | None = None
    max_draft_tokens: int = DEFAULT_DFLASH_MAX_DRAFT_TOKENS

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("dflash enabled must be a boolean")
        if self.target_model_path is not None and not isinstance(
            self.target_model_path, Path
        ):
            raise ValueError("dflash target_model_path must be a pathlib.Path")
        if self.drafter_model_path is not None and not isinstance(
            self.drafter_model_path, Path
        ):
            raise ValueError("dflash drafter_model_path must be a pathlib.Path")
        if (
            isinstance(self.max_draft_tokens, bool)
            or not isinstance(self.max_draft_tokens, int)
            or self.max_draft_tokens < 1
        ):
            raise ValueError("dflash_max_draft_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class DFlashTargetProfile:
    """Validated local DFlash target snapshot summary."""

    model_path: Path
    config_path: Path
    tokenizer_paths: tuple[Path, ...]
    safetensors_paths: tuple[Path, ...]
    architectures: tuple[str, ...]
    model_type: str
    dtype: str
    num_hidden_layers: int
    vocab_size: int
    tokenizer_vocab_size: int


@dataclass(frozen=True, slots=True)
class DFlashReadinessReport:
    """Structured readiness report for the DFlash boundary spike."""

    enabled: bool
    dependency_available: bool
    target_family: str | None
    drafter_family: str | None
    target_profile: DFlashTargetProfile | None = None
    cache_mode_blockers: tuple[str, ...] = ()
    route_blockers: tuple[str, ...] = ()
    resource_blockers: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


class DFlashUnavailableError(ValueError):
    """Raised when the DFlash boundary is opted into but not ready."""


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"{name} must be an integer") from exc


def _coerce_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return value if isinstance(value, Path) else Path(value)


def resolve_dflash_options(
    dflash_toggle: bool | None,
    dflash_target_model: str | Path | None,
    dflash_drafter_model: str | Path | None,
    dflash_max_draft_tokens: int | None,
) -> DFlashBoundaryOptions:
    """Resolve public DFlash kwargs/env into validated boundary options."""

    enabled = _env_flag(DFLASH_ENV) if dflash_toggle is None else dflash_toggle
    if not enabled:
        return DFlashBoundaryOptions(enabled=False)

    target_model_path = dflash_target_model
    if target_model_path is None:
        target_model_path = os.getenv(DFLASH_TARGET_MODEL_ENV, "")
    drafter_model_path = dflash_drafter_model
    if drafter_model_path is None:
        drafter_model_path = os.getenv(DFLASH_DRAFTER_MODEL_ENV, "")
    max_draft_tokens = (
        _env_int(DFLASH_MAX_DRAFT_TOKENS_ENV, DEFAULT_DFLASH_MAX_DRAFT_TOKENS)
        if dflash_max_draft_tokens is None
        else dflash_max_draft_tokens
    )
    return DFlashBoundaryOptions(
        enabled=True,
        target_model_path=_coerce_path(target_model_path),
        drafter_model_path=_coerce_path(drafter_model_path),
        max_draft_tokens=max_draft_tokens,
    )


def probe_dflash_dependency() -> tuple[bool, tuple[str, ...]]:
    """Return whether the optional DFlash dependency is importable."""

    missing_modules = tuple(
        module_name
        for module_name in _DFLASH_DEPENDENCY_MODULES
        if importlib.util.find_spec(module_name) is None
    )
    return len(missing_modules) == 0, missing_modules


def _read_model_metadata(model_path: Path | None) -> dict[str, Any]:
    if model_path is None or not model_path.exists():
        return {}
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        config = json.loads(config_path.read_text())
    except Exception:  # pragma: no cover - defensive probe
        return {}
    return config if isinstance(config, dict) else {}


def _classify_qwen_family(model_path: Path | None) -> str | None:
    metadata = _read_model_metadata(model_path)
    if not metadata:
        return None

    def _metadata_strings(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, str))
        return ()

    metadata_strings: list[str] = []
    for key in ("model_type", "architectures"):
        metadata_strings.extend(_metadata_strings(metadata.get(key)))
    for nested_key in ("text_config", "vision_config"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            for key in ("model_type", "architectures"):
                metadata_strings.extend(_metadata_strings(nested.get(key)))

    corpus = " ".join(metadata_strings).lower()
    if not any(marker in corpus for marker in _SUPPORTED_MODEL_MARKERS):
        return None
    if any(marker in corpus for marker in _UNSUPPORTED_MODEL_MARKERS):
        return None
    return "qwen"


def _normalize_dtype(dtype: Any) -> str:
    normalized = str(dtype).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return DFLASH_EXPECTED_DTYPE
    return normalized


def _collect_target_tokenizer_paths(
    model_path: Path,
    blockers: list[str],
) -> tuple[Path, ...]:
    if not model_path.exists():
        blockers.append(f"DFlash target snapshot path does not exist: {model_path}")
        return ()
    if not model_path.is_dir():
        blockers.append(f"DFlash target snapshot path is not a directory: {model_path}")
        return ()

    tokenizer_paths = tuple(
        model_path / filename for filename in DFLASH_REQUIRED_TARGET_TOKENIZER_FILES
    )
    missing_files = [path for path in tokenizer_paths if not path.exists()]
    if missing_files:
        blockers.append(
            "Missing DFlash target tokenizer/config files: "
            + ", ".join(str(path) for path in missing_files)
        )
    return tokenizer_paths


def _parse_target_profile(
    model_path: Path,
    blockers: list[str],
) -> DFlashTargetProfile | None:
    config_path = model_path / "config.json"
    tokenizer_paths = _collect_target_tokenizer_paths(model_path, blockers)
    config = _read_model_metadata(model_path)
    if not config:
        blockers.append(f"Missing or invalid DFlash target config file: {config_path}")
        return None

    if _classify_qwen_family(model_path) is None:
        blockers.append(f"DFlash target must be a Qwen-family snapshot: {model_path}")

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures or not all(
        isinstance(item, str) for item in architectures
    ):
        blockers.append("DFlash target config.architectures must be a non-empty string list")
        architectures_tuple: tuple[str, ...] = ()
    else:
        architectures_tuple = tuple(architectures)
        if not any("qwen" in item.lower() for item in architectures_tuple):
            blockers.append(
                "DFlash target config.architectures must describe a Qwen-family model"
            )

    model_type = str(config.get("model_type", "")).strip().lower()
    if not model_type.startswith("qwen"):
        blockers.append("DFlash target config.model_type must be Qwen-family")

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        dtype_value = _normalize_dtype(text_config.get("dtype", ""))
        if dtype_value and dtype_value != DFLASH_EXPECTED_DTYPE:
            blockers.append(
                f"DFlash target text_config.dtype must be {DFLASH_EXPECTED_DTYPE!r}"
            )
        num_hidden_layers = text_config.get("num_hidden_layers")
        vocab_size = text_config.get("vocab_size", config.get("vocab_size"))
    else:
        dtype_value = ""
        num_hidden_layers = config.get("num_hidden_layers")
        vocab_size = config.get("vocab_size")

    if not isinstance(num_hidden_layers, int):
        blockers.append("DFlash target num_hidden_layers must be an integer")
        num_hidden_layers_int = -1
    else:
        num_hidden_layers_int = num_hidden_layers

    if not isinstance(vocab_size, int):
        blockers.append("DFlash target vocab_size must be an integer")
        vocab_size_int = -1
    else:
        vocab_size_int = vocab_size

    tokenizer_vocab_size = -1
    vocab_path = model_path / "vocab.json"
    if vocab_path.exists():
        try:
            tokenizer_vocab = json.loads(vocab_path.read_text())
        except json.JSONDecodeError as exc:
            blockers.append(f"Invalid JSON in DFlash target vocab file {vocab_path}: {exc.msg}")
            tokenizer_vocab = {}
        if isinstance(tokenizer_vocab, dict):
            tokenizer_vocab_size = len(tokenizer_vocab)
        else:
            blockers.append(f"DFlash target vocab file must contain a JSON object: {vocab_path}")
        if tokenizer_vocab_size != -1 and vocab_size_int != -1:
            allowed_delta = max(1024, vocab_size_int // 100)
            if tokenizer_vocab_size > vocab_size_int:
                blockers.append(
                    "DFlash target tokenizer vocab size must not exceed config.vocab_size "
                    f"({tokenizer_vocab_size} > {vocab_size_int})"
                )
            elif vocab_size_int - tokenizer_vocab_size > allowed_delta:
                blockers.append(
                    "DFlash target tokenizer vocab size must stay close to config.vocab_size "
                    f"({tokenizer_vocab_size} vs {vocab_size_int})"
                )

    if blockers:
        return None

    return DFlashTargetProfile(
        model_path=model_path,
        config_path=config_path,
        tokenizer_paths=tokenizer_paths,
        safetensors_paths=tuple(sorted(model_path.glob("*.safetensors"))),
        architectures=architectures_tuple,
        model_type=model_type,
        dtype=dtype_value or DFLASH_EXPECTED_DTYPE,
        num_hidden_layers=num_hidden_layers_int,
        vocab_size=vocab_size_int,
        tokenizer_vocab_size=tokenizer_vocab_size,
    )


def _probe_reserved_port_conflicts() -> tuple[str, ...]:
    blockers: list[str] = []
    for port in DFLASH_RESOURCE_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.05)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                blockers.append(
                    f"Reserved DFlash resource port 127.0.0.1:{port} is already in use"
                )
    return tuple(blockers)


def _probe_available_memory_bytes() -> int | None:
    try:
        vm_stat = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
        )
        page_size_proc = subprocess.run(
            ["sysctl", "-n", "hw.pagesize"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # pragma: no cover - macOS command availability
        return None

    try:
        page_size = int(page_size_proc.stdout.strip())
    except ValueError:  # pragma: no cover - defensive parsing
        page_size = 4096

    available_pages = 0
    for line in vm_stat.stdout.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+(\d+)\.", line)
        if match:
            available_pages += int(match.group(2))
    if available_pages <= 0:
        return None
    return available_pages * page_size


def _estimate_snapshot_bytes(paths: tuple[Path, ...]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _format_gib(size_in_bytes: int | None) -> str:
    if size_in_bytes is None:
        return "unknown"
    return f"{size_in_bytes / (1024 ** 3):.2f} GiB"


def probe_dflash_readiness(
    options: DFlashBoundaryOptions,
) -> DFlashReadinessReport:
    """Probe the DFlash boundary without mutating generation state."""

    blockers: list[str] = []
    dependency_available, missing_modules = probe_dflash_dependency()

    if not options.enabled:
        return DFlashReadinessReport(
            enabled=False,
            dependency_available=dependency_available,
            target_family=None,
            drafter_family=None,
            blockers=(),
        )

    target_family = _classify_qwen_family(options.target_model_path)
    drafter_family = _classify_qwen_family(options.drafter_model_path)
    target_profile = None
    resource_blockers: tuple[str, ...] = ()

    if options.target_model_path is None or options.drafter_model_path is None:
        blockers.append(
            "DFlash requires explicit Qwen-family target and drafter model paths"
        )

    if target_family is None:
        blockers.append("DFlash target model must be a Qwen-family snapshot")
    if drafter_family is None:
        blockers.append("DFlash drafter model must be a Qwen-family snapshot")

    if (
        options.target_model_path is not None
        and options.drafter_model_path is not None
        and options.target_model_path.resolve() == options.drafter_model_path.resolve()
    ):
        blockers.append("DFlash target and drafter snapshots must be distinct")

    if not dependency_available:
        blockers.append(
            "Missing optional DFlash dependency modules: "
            + ", ".join(missing_modules)
        )

    if options.target_model_path is not None:
        target_profile = _parse_target_profile(options.target_model_path, blockers)

    drafter_profile = None
    if options.drafter_model_path is not None:
        try:
            drafter_profile = load_dflash_snapshot_profile(options.drafter_model_path)
        except DFlashSnapshotError as exc:
            blockers.extend(exc.blockers)

    if target_profile is not None and drafter_profile is not None:
        if target_profile.vocab_size != drafter_profile.vocab_size:
            blockers.append(
                "DFlash target and drafter vocab sizes must match "
                f"({target_profile.vocab_size} != {drafter_profile.vocab_size})"
            )
        max_target_layer_id = max(drafter_profile.target_layer_ids)
        if target_profile.num_hidden_layers <= max_target_layer_id:
            blockers.append(
                "DFlash target does not expose every configured target layer id "
                f"(num_hidden_layers={target_profile.num_hidden_layers}, "
                f"max_target_layer_id={max_target_layer_id})"
            )

        estimated_bytes = _estimate_snapshot_bytes(target_profile.safetensors_paths)
        estimated_bytes += _estimate_snapshot_bytes(drafter_profile.safetensors_paths)
        available_bytes = _probe_available_memory_bytes()
        if available_bytes is not None:
            headroom = max(
                int(estimated_bytes * DFLASH_AVAILABLE_MEMORY_HEADROOM_RATIO),
                DFLASH_AVAILABLE_MEMORY_HEADROOM_MIN_BYTES,
            )
            required_bytes = estimated_bytes + headroom
            if available_bytes < required_bytes:
                resource_blockers = (
                    f"Insufficient free memory for real-pair DFlash preflight: "
                    f"need at least {_format_gib(required_bytes)}, "
                    f"found {_format_gib(available_bytes)}",
                )

    port_blockers = _probe_reserved_port_conflicts()
    if port_blockers:
        resource_blockers = (*resource_blockers, *port_blockers)

    blockers.extend(resource_blockers)

    return DFlashReadinessReport(
        enabled=True,
        dependency_available=dependency_available,
        target_family=target_family,
        drafter_family=drafter_family,
        target_profile=target_profile,
        resource_blockers=resource_blockers,
        blockers=_dedupe_blockers(blockers),
    )


def validate_dflash_preload_compatibility(
    *,
    options: DFlashBoundaryOptions,
    loaded_model_path: Path,
    is_vlm_route: bool,
    vocab_only: bool,
    distributed: bool,
    max_seq_nums: int | None,
    kv_bits: int | None,
    kv_group_size: int | None,
    quantized_kv_start: int | None,
    vlm_prompt_cache_storage_root: Path | None,
    vlm_prompt_cache_min_save_tokens: int | None,
) -> DFlashReadinessReport:
    """Fail closed before DFlash can reach heavyweight model loading."""

    readiness = probe_dflash_readiness(options)
    route_blockers = list(readiness.route_blockers)
    cache_mode_blockers = list(readiness.cache_mode_blockers)

    if not options.enabled:
        return readiness

    if options.target_model_path is not None and loaded_model_path.resolve() != options.target_model_path.resolve():
        route_blockers.append(
            "DFlash target model path must match the loaded model path "
            f"({loaded_model_path} != {options.target_model_path})"
        )
    if is_vlm_route:
        route_blockers.append("DFlash is only supported for sequential text generation")
    if vocab_only:
        route_blockers.append("DFlash cannot be combined with vocab_only loads yet")
    if distributed:
        route_blockers.append("DFlash cannot be combined with distributed loading yet")
    if max_seq_nums is not None and max_seq_nums > 1:
        route_blockers.append("DFlash requires the sequential route (max_seq_nums <= 1)")
    if vlm_prompt_cache_storage_root is not None:
        route_blockers.append(
            "DFlash is not compatible with persistent VLM prompt-cache storage yet"
        )
    if vlm_prompt_cache_min_save_tokens is not None:
        route_blockers.append(
            "DFlash is not compatible with persistent VLM prompt-cache admission yet"
        )

    if kv_bits is not None:
        cache_mode_blockers.append("DFlash does not support kv_bits cache mode yet")
    if kv_group_size is not None:
        cache_mode_blockers.append("DFlash does not support kv_group_size cache mode yet")
    if quantized_kv_start is not None:
        cache_mode_blockers.append(
            "DFlash does not support quantized_kv_start cache mode yet"
        )

    blockers = _dedupe_blockers(
        [
            *readiness.blockers,
            *route_blockers,
            *cache_mode_blockers,
        ]
    )
    report = DFlashReadinessReport(
        enabled=readiness.enabled,
        dependency_available=readiness.dependency_available,
        target_family=readiness.target_family,
        drafter_family=readiness.drafter_family,
        target_profile=readiness.target_profile,
        cache_mode_blockers=tuple(cache_mode_blockers),
        route_blockers=tuple(route_blockers),
        resource_blockers=readiness.resource_blockers,
        blockers=blockers,
    )
    if blockers:
        raise DFlashUnavailableError(
            build_dflash_no_go_message(report)
        )
    return report


def validate_dflash_surface_compatibility(
    *,
    enabled: bool,
    surface_label: str,
    images_b64: Optional[list[str]],
    specprefill_toggle: Optional[bool],
    speculative_decoding_toggle: Optional[bool],
    num_draft_tokens: Optional[int],
    draft_model: Any | None,
    model_kit_draft_model: Any | None = None,
) -> tuple[str, ...]:
    """Fail closed for unsupported DFlash surfaces."""

    if not enabled:
        return ()

    blockers: list[str] = []
    if surface_label != "sequential":
        blockers.append("DFlash is only supported for sequential text generation")
    if images_b64 is not None and len(images_b64) > 0:
        blockers.append("DFlash is not enabled for VLM/image surfaces yet")
    if specprefill_toggle is True:
        blockers.append("DFlash cannot be combined with SpecPrefill yet")
    if speculative_decoding_toggle is True:
        blockers.append("DFlash cannot be combined with speculative decoding yet")
    if num_draft_tokens is not None:
        blockers.append("DFlash cannot be combined with num_draft_tokens yet")
    if draft_model is not None:
        blockers.append("DFlash cannot be combined with a draft_model kwarg yet")
    if model_kit_draft_model is not None:
        blockers.append(
            "DFlash cannot be combined with an already loaded draft_model yet"
        )
    return tuple(blockers)


def _collect_prompt_cache_layers(model_kit: Any) -> tuple[Any, ...]:
    prompt_cache = getattr(getattr(model_kit, "cache_wrapper", None), "cache", None)
    if prompt_cache is None:
        prompt_cache = getattr(model_kit, "prompt_cache", None)
    if prompt_cache is None:
        return ()
    if isinstance(prompt_cache, tuple):
        return prompt_cache
    if isinstance(prompt_cache, list):
        return tuple(prompt_cache)
    try:
        return tuple(prompt_cache)
    except TypeError:
        return (prompt_cache,)


def _dedupe_blockers(blockers: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for blocker in blockers:
        if blocker in seen:
            continue
        seen.add(blocker)
        deduped.append(blocker)
    return tuple(deduped)


def validate_dflash_runtime_compatibility(model_kit: Any) -> tuple[str, ...]:
    """Fail closed before DFlash mutates prompt caches or live history."""

    blockers: list[str] = []
    if getattr(model_kit, "draft_model", None) is not None:
        blockers.append(
            "DFlash cannot be combined with an already loaded draft_model yet"
        )
    for attr_name, label in (
        ("max_kv_size", "max_kv_size"),
        ("kv_bits", "kv_bits"),
        ("kv_group_size", "kv_group_size"),
        ("quantized_kv_start", "quantized_kv_start"),
    ):
        if getattr(model_kit, attr_name, None) is not None:
            blockers.append(f"DFlash does not support {label} cache mode yet")

    prompt_cache_layers = _collect_prompt_cache_layers(model_kit)
    if not prompt_cache_layers:
        blockers.append("DFlash requires a prompt cache before runtime execution")
    else:
        for cache in prompt_cache_layers:
            cache_type_name = type(cache).__name__
            if cache_type_name == "KVCache":
                continue
            if cache_type_name == "RotatingKVCache" or (
                getattr(cache, "max_size", None) is not None
                and getattr(cache, "keep", None) is not None
            ):
                blockers.append("DFlash does not support bounded/rotating cache layers yet")
                continue
            if cache_type_name in {"ArraysCache", "BatchKVCache"} or (
                getattr(cache, "lengths", None) is not None
                or getattr(cache, "left_padding", None) is not None
            ):
                blockers.append("DFlash does not support ragged cache layers yet")
                continue
            blockers.append(
                f"DFlash does not support non-rollback-safe cache layer {cache_type_name} yet"
            )

    target_model = getattr(model_kit, "model", model_kit)
    lm = (
        target_model.language_model
        if hasattr(target_model, "language_model")
        else target_model
    )
    if not hasattr(lm, "rollback_speculative_cache"):
        blockers.append(
            f"{type(lm).__name__} does not implement rollback_speculative_cache"
        )

    return _dedupe_blockers(blockers)


def build_dflash_no_go_message(
    readiness: DFlashReadinessReport,
    *,
    surface_blockers: tuple[str, ...] = (),
) -> str:
    blockers = [*surface_blockers, *readiness.blockers]
    if not blockers:
        return "DFlash boundary is wired, but no execution path exists yet"

    next_steps = (
        "Next steps: install the optional DFlash dependency, stage a local "
        "Qwen-family target/drafter pair, and keep the feature default-off "
        "until a real sequential prototype is implemented."
    )
    return "DFlash no-go: " + "; ".join(blockers) + ". " + next_steps


def build_dflash_runtime_no_go_message(blockers: tuple[str, ...]) -> str:
    if not blockers:
        return "DFlash boundary is wired, but no execution path exists yet"

    next_steps = (
        "Next steps: switch to a plain KVCache sequential path with a "
        "rollback-capable target model and keep DFlash default-off until a "
        "real sequential smoke passes."
    )
    return "DFlash no-go: " + "; ".join(blockers) + ". " + next_steps


def validate_dflash_boundary(
    *,
    options: DFlashBoundaryOptions,
    surface_label: str,
    images_b64: Optional[list[str]],
    specprefill_toggle: Optional[bool],
    speculative_decoding_toggle: Optional[bool],
    num_draft_tokens: Optional[int],
    draft_model: Any | None,
) -> None:
    """Raise if DFlash was opted in but the boundary is not ready."""

    if not options.enabled:
        return

    surface_blockers = validate_dflash_surface_compatibility(
        enabled=True,
        surface_label=surface_label,
        images_b64=images_b64,
        specprefill_toggle=specprefill_toggle,
        speculative_decoding_toggle=speculative_decoding_toggle,
        num_draft_tokens=num_draft_tokens,
        draft_model=draft_model,
    )
    readiness = probe_dflash_readiness(options)
    if surface_blockers or readiness.blockers:
        raise DFlashUnavailableError(
            build_dflash_no_go_message(
                readiness,
                surface_blockers=surface_blockers,
            )
        )
    return None
