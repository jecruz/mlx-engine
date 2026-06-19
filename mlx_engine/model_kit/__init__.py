"""
Model Kit module with automatic compatibility patches.

This module automatically applies compatibility patches for various model types
by replacing classes in their respective modules with derived, compatible versions.
"""

import logging

from .patches.gemma3n import apply_patches as _apply_patches_gemma3n
from .patches.lfm2_vl import apply_patches as _apply_patches_lfm2_vl

logger = logging.getLogger(__name__)

try:
    from .patches.ernie_4_5 import apply_patches as _apply_patches_ernie_4_5
except ModuleNotFoundError as exc:
    if exc.name != "outlines_core":
        raise
    _apply_patches_ernie_4_5 = None
    logger.debug(
        "Skipping Ernie 4.5 compatibility patches because outlines_core is unavailable."
    )

_apply_patches_gemma3n()
if _apply_patches_ernie_4_5 is not None:
    _apply_patches_ernie_4_5()
_apply_patches_lfm2_vl()
try:
    from .patches.qwen3_5 import apply_patches as _apply_patches_qwen3_5
except (AttributeError, ModuleNotFoundError) as exc:
    _apply_patches_qwen3_5 = None
    logger.debug(
        "Skipping Qwen3.5 compatibility patches because a required mlx_vlm symbol is unavailable: %s",
        exc,
    )

if _apply_patches_qwen3_5 is not None:
    _apply_patches_qwen3_5()
