from typing import TYPE_CHECKING

from .config import (
    AttentionGroupConfig,
    BaseAttentionGroupConfig,
    DSV4AttentionGroupConfig,
    FullAttentionGroupConfig,
    KVCacheGroupSpec,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)

if TYPE_CHECKING:
    from .blocks import BaseLLMModel
    from .weight import load_moe_expert_sources, load_weight


def create_model(model_config: ModelConfig) -> "BaseLLMModel":
    from .register import get_model_class

    return get_model_class(model_config.architectures[0], model_config)


# Model/weight machinery is lazy: `.blocks` and `.weight` pull the CUDA layer and
# kernel stack (flashlib, pinned-tensor extension, …), which machines running the
# MLX backend don't have. Config-only consumers (utils.hf, server args, tokenizer
# workers, the MLX scheduler) import this package for the dataclasses above and
# must keep working without those native deps.
_LAZY = {
    "BaseLLMModel": "freetoken.models.blocks",
    "load_weight": "freetoken.models.weight",
    "load_moe_expert_sources": "freetoken.models.weight",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


__all__ = [
    "BaseLLMModel",
    "create_model",
    "load_weight",
    "load_moe_expert_sources",
    "AttentionGroupConfig",
    "BaseAttentionGroupConfig",
    "DSV4AttentionGroupConfig",
    "FullAttentionGroupConfig",
    "LinearGatedDeltaGroupConfig",
    "ModelConfig",
    "RotaryConfig",
    "SWAAttentionGroupConfig",
    "KVCacheGroupSpec",
]
