from typing import TYPE_CHECKING

from .config import EngineConfig

if TYPE_CHECKING:
    from .engine import Engine, ForwardOutput
    from .sample import BatchSamplingArgs

__all__ = ["Engine", "EngineConfig", "ForwardOutput", "BatchSamplingArgs"]

_LAZY = {
    "Engine": "freetoken.engine.engine",
    "ForwardOutput": "freetoken.engine.engine",
    "BatchSamplingArgs": "freetoken.engine.sample",
}


def __getattr__(name: str):
    """Lazy exports: Engine/sampling pull the CUDA layer/kernel stack (flashlib,
    triton, …), which machines running the MLX backend don't have. Config-only
    consumers (server args, tokenizer workers, the MLX scheduler) must be able to
    import this package without them."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
