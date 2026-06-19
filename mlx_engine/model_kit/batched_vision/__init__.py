from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_engine.model_kit.batched_vision.model_kit import BatchedVisionModelKit


def __getattr__(name: str):
    """Lazily expose batched-vision entry points without eager heavy imports."""
    if name == "BatchedVisionModelKit":
        from mlx_engine.model_kit.batched_vision.model_kit import BatchedVisionModelKit

        return BatchedVisionModelKit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BatchedVisionModelKit"]
