"""
`mlx_engine` is LM Studio's LLM inferencing engine for Apple MLX
"""

from importlib import import_module

__all__ = [
    "load_model",
    "load_draft_model",
    "is_draft_model_compatible",
    "unload_draft_model",
    "create_generator",
    "stop_generation",
    "tokenize",
    "unload",
]

from pathlib import Path
import os

from .utils.disable_hf_download import patch_huggingface_hub
from .utils.register_models import register_models
from .utils.logger import setup_logging

patch_huggingface_hub()
register_models()
setup_logging()


def _set_outlines_cache_dir(cache_dir: Path | str):
    """
    Set the cache dir for Outlines.

    Outlines reads the OUTLINES_CACHE_DIR environment variable to
    determine where to read/write its cache files
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    os.environ["OUTLINES_CACHE_DIR"] = str(cache_dir)


_set_outlines_cache_dir(Path("~/.cache/lm-studio/.internal/outlines"))

_GENERATE_EXPORTS = {
    "load_model",
    "load_draft_model",
    "is_draft_model_compatible",
    "unload_draft_model",
    "create_generator",
    "tokenize",
    "unload",
    "stop_generation",
}


def __getattr__(name):
    if name not in _GENERATE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    generate_module = import_module(".generate", __name__)
    value = getattr(generate_module, name)
    globals()[name] = value
    return value
