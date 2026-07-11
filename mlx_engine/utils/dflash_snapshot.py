"""Parse and validate native DFlash drafter snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open


DFLASH_EXPECTED_ARCHITECTURE = "DFlashDraftModel"
DFLASH_EXPECTED_MODEL_TYPE = "qwen3"
DFLASH_EXPECTED_DTYPE = "bfloat16"
DFLASH_EXPECTED_LAYER_COUNT = 6
DFLASH_EXPECTED_VOCAB_SIZE = 248320
DFLASH_EXPECTED_BLOCK_SIZE = 16
DFLASH_EXPECTED_MASK_TOKEN_ID = 248077
DFLASH_EXPECTED_TARGET_LAYER_IDS = (1, 10, 18, 27, 35, 44, 52, 61)
DFLASH_EXPECTED_SAFETENSORS_FORMAT = "pt"

_LAYER_KEY_RE = re.compile(r"^layers\.(\d+)\.")


@dataclass(frozen=True, slots=True)
class DFlashSnapshotProfile:
    """Validated local DFlash drafter snapshot summary."""

    model_path: Path
    config_path: Path
    safetensors_paths: tuple[Path, ...]
    architectures: tuple[str, ...]
    model_type: str
    dtype: str
    num_hidden_layers: int
    vocab_size: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    safetensors_formats: tuple[str, ...]
    tensor_names: tuple[str, ...]
    tensor_dtypes: tuple[str, ...]
    tensor_layer_count: int


class DFlashSnapshotError(ValueError):
    """Raised when a local drafter snapshot is not valid DFlash metadata."""

    def __init__(self, model_path: Path, blockers: list[str]):
        self.model_path = model_path
        self.blockers = tuple(blockers)
        message = (
            "DFlash snapshot invalid at "
            + str(model_path)
            + ": "
            + "; ".join(self.blockers)
        )
        super().__init__(message)


def _normalize_dtype(dtype: Any) -> str:
    normalized = str(dtype).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return DFLASH_EXPECTED_DTYPE
    return normalized


def _parse_json_file(path: Path, blockers: list[str]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        blockers.append(f"Missing DFlash config file: {path}")
        return {}
    except json.JSONDecodeError as exc:
        blockers.append(f"Invalid JSON in DFlash config file {path}: {exc.msg}")
        return {}
    if not isinstance(data, dict):
        blockers.append(f"DFlash config file must contain a JSON object: {path}")
        return {}
    return data


def _collect_safetensors_paths(
    model_path: Path, blockers: list[str]
) -> tuple[Path, ...]:
    if not model_path.exists():
        blockers.append(f"DFlash snapshot path does not exist: {model_path}")
        return ()
    if not model_path.is_dir():
        blockers.append(f"DFlash snapshot path is not a directory: {model_path}")
        return ()

    safetensors_paths = tuple(
        sorted(
            path
            for path in model_path.glob("*.safetensors")
            if path.is_file() and not path.name.endswith(".index.json")
        )
    )
    if not safetensors_paths:
        blockers.append(f"No safetensors weights found under {model_path}")
    return safetensors_paths


def _collect_tensor_header(
    safetensors_paths: tuple[Path, ...],
    blockers: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    tensor_names: list[str] = []
    tensor_dtypes: list[str] = []
    safetensors_formats: list[str] = []

    for safetensors_path in safetensors_paths:
        try:
            with safe_open(str(safetensors_path), framework="np") as file_handle:
                metadata = file_handle.metadata() or {}
                safetensors_formats.append(
                    str(metadata.get("format", "")).strip().lower()
                )
                names = tuple(file_handle.keys())
                tensor_names.extend(names)
                tensor_dtypes.extend(
                    _normalize_dtype(file_handle.get_slice(name).get_dtype())
                    for name in names
                )
        except Exception as exc:  # pragma: no cover - defensive snapshot probe
            blockers.append(
                f"Failed to read safetensors header {safetensors_path}: {exc}"
            )

    return tuple(safetensors_formats), tuple(tensor_names), tuple(tensor_dtypes)


def _parse_config_fields(
    config: dict[str, Any], blockers: list[str]
) -> tuple[Any, ...]:
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(item, str) for item in architectures)
    ):
        blockers.append("DFlash config.architectures must be a non-empty string list")
        architectures_tuple: tuple[str, ...] = ()
    else:
        architectures_tuple = tuple(architectures)
        if architectures_tuple != (DFLASH_EXPECTED_ARCHITECTURE,):
            blockers.append(
                "DFlash config.architectures must be "
                f"['{DFLASH_EXPECTED_ARCHITECTURE}']"
            )

    model_type = str(config.get("model_type", "")).strip().lower()
    if model_type != DFLASH_EXPECTED_MODEL_TYPE:
        blockers.append(
            f"DFlash config.model_type must be {DFLASH_EXPECTED_MODEL_TYPE!r}"
        )

    dtype = _normalize_dtype(config.get("dtype", ""))
    if dtype != DFLASH_EXPECTED_DTYPE:
        blockers.append(
            f"DFlash config.dtype must be {DFLASH_EXPECTED_DTYPE!r} (got {config.get('dtype')!r})"
        )

    num_hidden_layers = config.get("num_hidden_layers")
    if num_hidden_layers != DFLASH_EXPECTED_LAYER_COUNT:
        blockers.append(
            f"DFlash config.num_hidden_layers must be {DFLASH_EXPECTED_LAYER_COUNT}"
        )

    vocab_size = config.get("vocab_size")
    if vocab_size != DFLASH_EXPECTED_VOCAB_SIZE:
        blockers.append(
            f"DFlash config.vocab_size must be {DFLASH_EXPECTED_VOCAB_SIZE}"
        )

    dflash_config = config.get("dflash_config")
    if not isinstance(dflash_config, dict):
        blockers.append("DFlash config.dflash_config must be a JSON object")
        target_layer_ids = ()
        block_size = -1
        mask_token_id = -1
    else:
        target_layer_ids = dflash_config.get("target_layer_ids")
        if not isinstance(target_layer_ids, list) or not all(
            isinstance(item, int) for item in target_layer_ids
        ):
            blockers.append(
                "DFlash config.dflash_config.target_layer_ids must be an int list"
            )
            target_layer_ids = ()
        else:
            target_layer_ids = tuple(target_layer_ids)
            if target_layer_ids != DFLASH_EXPECTED_TARGET_LAYER_IDS:
                blockers.append(
                    "DFlash config.dflash_config.target_layer_ids must be "
                    f"{list(DFLASH_EXPECTED_TARGET_LAYER_IDS)}"
                )

        block_size = dflash_config.get("block_size")
        if block_size != DFLASH_EXPECTED_BLOCK_SIZE:
            blockers.append(
                f"DFlash config.dflash_config.block_size must be {DFLASH_EXPECTED_BLOCK_SIZE}"
            )

        mask_token_id = dflash_config.get("mask_token_id")
        if mask_token_id != DFLASH_EXPECTED_MASK_TOKEN_ID:
            blockers.append(
                f"DFlash config.dflash_config.mask_token_id must be {DFLASH_EXPECTED_MASK_TOKEN_ID}"
            )

    return (
        architectures_tuple,
        model_type,
        dtype,
        num_hidden_layers if isinstance(num_hidden_layers, int) else -1,
        vocab_size if isinstance(vocab_size, int) else -1,
        block_size if isinstance(block_size, int) else -1,
        mask_token_id if isinstance(mask_token_id, int) else -1,
        target_layer_ids if isinstance(target_layer_ids, tuple) else (),
    )


def load_dflash_snapshot_profile(model_path: Path) -> DFlashSnapshotProfile:
    """Load and validate a local DFlash drafter snapshot."""

    blockers: list[str] = []
    config_path = model_path / "config.json"
    config = _parse_json_file(config_path, blockers)
    safetensors_paths = _collect_safetensors_paths(model_path, blockers)

    (
        architectures,
        model_type,
        dtype,
        num_hidden_layers,
        vocab_size,
        block_size,
        mask_token_id,
        target_layer_ids,
    ) = _parse_config_fields(config, blockers)

    safetensors_formats, tensor_names, tensor_dtypes = _collect_tensor_header(
        safetensors_paths,
        blockers,
    )
    if not safetensors_formats or any(
        safetensors_format != DFLASH_EXPECTED_SAFETENSORS_FORMAT
        for safetensors_format in safetensors_formats
    ):
        blockers.append(
            "DFlash safetensors metadata.format must be "
            f"{DFLASH_EXPECTED_SAFETENSORS_FORMAT!r}"
        )

    safetensors_formats = tuple(
        sorted({format_name for format_name in safetensors_formats if format_name})
    )
    tensor_dtypes = tuple(
        sorted({tensor_dtype for tensor_dtype in tensor_dtypes if tensor_dtype})
    )

    if tensor_dtypes and any(
        tensor_dtype != DFLASH_EXPECTED_DTYPE for tensor_dtype in tensor_dtypes
    ):
        blockers.append(f"DFlash weights must all use {DFLASH_EXPECTED_DTYPE!r} dtype")

    layer_indices = sorted(
        {
            int(match.group(1))
            for name in tensor_names
            if (match := _LAYER_KEY_RE.match(name))
        }
    )
    if not layer_indices:
        blockers.append("DFlash safetensors must contain per-layer tensors")
    elif layer_indices != list(range(len(layer_indices))):
        blockers.append(
            "DFlash safetensors layer tensors must be contiguous from 0; "
            f"got {layer_indices}"
        )
    if layer_indices and len(layer_indices) != DFLASH_EXPECTED_LAYER_COUNT:
        blockers.append(
            f"DFlash safetensors layer count must be {DFLASH_EXPECTED_LAYER_COUNT}"
        )
    if blockers:
        raise DFlashSnapshotError(model_path, blockers)

    return DFlashSnapshotProfile(
        model_path=model_path,
        config_path=config_path,
        safetensors_paths=safetensors_paths,
        architectures=architectures,
        model_type=model_type,
        dtype=dtype,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        block_size=block_size,
        mask_token_id=mask_token_id,
        target_layer_ids=target_layer_ids,
        safetensors_formats=safetensors_formats,
        tensor_names=tensor_names,
        tensor_dtypes=tensor_dtypes,
        tensor_layer_count=len(layer_indices),
    )
