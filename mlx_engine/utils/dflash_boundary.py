"""Guarded DFlash boundary and dependency probe for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Optional


DFLASH_ENV = "MLX_ENGINE_DFLASH"
DFLASH_TARGET_MODEL_ENV = "MLX_ENGINE_DFLASH_TARGET_MODEL"
DFLASH_DRAFTER_MODEL_ENV = "MLX_ENGINE_DFLASH_DRAFTER_MODEL"
DFLASH_MAX_DRAFT_TOKENS_ENV = "MLX_ENGINE_DFLASH_MAX_DRAFT_TOKENS"
DEFAULT_DFLASH_MAX_DRAFT_TOKENS = 4

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
class DFlashReadinessReport:
    """Structured readiness report for the DFlash boundary spike."""

    enabled: bool
    dependency_available: bool
    target_family: str | None
    drafter_family: str | None
    blockers: tuple[str, ...]


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


def _has_local_weights(model_path: Path | None) -> bool:
    if model_path is None or not model_path.exists():
        return False
    return any(model_path.glob("*.safetensors"))


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

    if options.drafter_model_path is not None and not _has_local_weights(
        options.drafter_model_path
    ):
        blockers.append(
            f"No local DFlash drafter weights found at {options.drafter_model_path}"
        )

    return DFlashReadinessReport(
        enabled=True,
        dependency_available=dependency_available,
        target_family=target_family,
        drafter_family=drafter_family,
        blockers=tuple(blockers),
    )


def validate_dflash_surface_compatibility(
    *,
    enabled: bool,
    surface_label: str,
    images_b64: Optional[list[str]],
    specprefill_toggle: Optional[bool],
    speculative_decoding_toggle: Optional[bool],
    num_draft_tokens: Optional[int],
    draft_model: Any | None,
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
    if speculative_decoding_toggle is True or num_draft_tokens is not None:
        blockers.append(
            "DFlash cannot be combined with loaded draft_model speculation yet"
        )
    if draft_model is not None:
        blockers.append("DFlash cannot be combined with a loaded draft_model yet")
    return tuple(blockers)


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

    raise NotImplementedError(
        "DFlash boundary is ready, but a sequential execution path is not "
        "implemented yet"
    )
