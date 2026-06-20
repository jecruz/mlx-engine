"""
Model Kit module with automatic compatibility patches.

This module automatically applies compatibility patches for various model types
by replacing classes in their respective modules with derived, compatible versions.
"""

import logging

logger = logging.getLogger(__name__)


def _is_headless_metal_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "No Metal device available" in str(exc)


def _apply_compatibility_patch(import_path: str, skip_message: str) -> None:
    try:
        module = __import__(import_path, fromlist=["apply_patches"])
    except (AttributeError, ModuleNotFoundError) as exc:
        logger.debug("%s: %s", skip_message, exc)
        return
    except RuntimeError as exc:
        if not _is_headless_metal_error(exc):
            raise
        logger.debug("%s: %s", skip_message, exc)
        return

    module.apply_patches()


_apply_compatibility_patch(
    "mlx_engine.model_kit.patches.gemma3n",
    "Skipping Gemma3n compatibility patches",
)
try:
    from .patches.ernie_4_5 import apply_patches as _apply_patches_ernie_4_5
except ModuleNotFoundError as exc:
    if exc.name != "outlines_core":
        raise
    _apply_patches_ernie_4_5 = None
    logger.debug(
        "Skipping Ernie 4.5 compatibility patches because outlines_core is unavailable."
    )

if _apply_patches_ernie_4_5 is not None:
    _apply_patches_ernie_4_5()

_apply_compatibility_patch(
    "mlx_engine.model_kit.patches.lfm2_vl",
    "Skipping LFM2-VL compatibility patches",
)
try:
    from .patches.qwen3_5 import apply_patches as _apply_patches_qwen3_5
except (AttributeError, ModuleNotFoundError) as exc:
    _apply_patches_qwen3_5 = None
    logger.debug(
        "Skipping Qwen3.5 compatibility patches because a required mlx_vlm symbol is unavailable: %s",
        exc,
    )
except RuntimeError as exc:
    if not _is_headless_metal_error(exc):
        raise
    _apply_patches_qwen3_5 = None
    logger.debug(
        "Skipping Qwen3.5 compatibility patches because the Metal device is unavailable: %s",
        exc,
    )

if _apply_patches_qwen3_5 is not None:
    _apply_patches_qwen3_5()
