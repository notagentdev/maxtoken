"""Whole-MoE verify rows: the sparse block in four kernel launches.

A speculative verify window runs the MoE block at 2-3 rows, where the stock
path costs ~15 dispatches per layer (router qmm, softmax, top-k, gathered
gate/up, SwiGLU, gathered down, score weighting, shared expert, scalar gate,
sums) and the gathers run far from the roofline at these widths. MTPLX's
whole-MoE stages (vendored in _a3b_whole_moe_kernels.py, Apache-2.0) compute
the ENTIRE block in four fixed Metal launches per layer.

The kernels are exact for one contract — the Qwen3.5-35B-A3B family with the
packed gate/up layout our FTW store v2 writes (ftw_mlx.py) and the shared
expert packed by moe_pack.py: router q8g64 (256, 512), routed gate_up q4g64
(256, 1024, 256), routed down q4g64 (256, 2048, 64), shared gate_up/down
q4g64, scalar gate q8g64. Install validates every block against it and runs
a fixture parity check against the stock forward; any miss leaves the block
on the stock path and says why.

Scope: B=1, S in {2, 3}, bf16 activations — exactly the spec loop's verify
windows. Decode (S=1, batched) and prefill keep their existing paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

from maxtoken.utils import init_logger

logger = init_logger(__name__)

# id(block) -> {rows: bound call}
_ROUTES: Dict[int, Dict[int, Callable[[Any], Any]]] = {}


@dataclass(frozen=True)
class _Storage:
    weight: Any
    scales: Any = None
    biases: Any = None


@dataclass(frozen=True)
class _Binding:
    """Attribute-compatible with the vendored launchers' expectations."""

    block: Any
    stock_call: Any
    variant: str
    router: _Storage
    routed_gate_up: _Storage
    routed_down: _Storage
    shared_gate_up: _Storage
    shared_down: _Storage
    shared_scalar_gate: _Storage


def enabled() -> bool:
    return os.environ.get("MAXTOKEN_MLX_WHOLE_MOE", "1") != "0"


def m1_enabled() -> bool:
    """Route single decode rows (B=1, S=1) through the three-launch M1
    stages too (MAXTOKEN_MLX_WHOLE_MOE_M1=0 keeps the compiled block).

    Measured on Ornith-1.5-35B, interleaved A/B at B=1: 12.31 -> 12.16 ms
    per token (+1%). The block alone drops 129 -> 76 us per call, but the
    wide command buffers already hide the stock path's launches, so in situ
    the block is bandwidth-bound and only ~0.12 ms of the 2 ms shows up.
    Kept on for the launches it saves under CPU contention; do not expect
    more from it."""
    return os.environ.get("MAXTOKEN_MLX_WHOLE_MOE_M1", "1") != "0"


def _quantized(mod, *, bits, group_size, wshape, mshape) -> _Storage | None:
    import mlx.core as mx

    if (
        int(getattr(mod, "bits", 0) or 0) != bits
        or int(getattr(mod, "group_size", 0) or 0) != group_size
        or str(getattr(mod, "mode", "")) != "affine"
    ):
        return None
    w = getattr(mod, "weight", None)
    s = getattr(mod, "scales", None)
    b = getattr(mod, "biases", None)
    if (
        w is None
        or s is None
        or b is None
        or tuple(w.shape) != wshape
        or tuple(s.shape) != mshape
        or tuple(b.shape) != mshape
        or w.dtype != mx.uint32
        or s.dtype != mx.bfloat16
        or b.dtype != mx.bfloat16
    ):
        return None
    return _Storage(weight=w, scales=s, biases=b)


def _binding(block) -> _Binding | str:
    """Build the exact binding for one block, or a reason it cannot."""
    for attr in ("gate", "switch_mlp", "shared_expert", "shared_expert_gate"):
        if not hasattr(block, attr):
            return f"missing {attr}"
    if int(getattr(block, "num_experts", 0)) != 256:
        return "needs 256 experts"
    if int(getattr(block, "top_k", 0)) != 8:
        return "needs top-8 routing"
    if not bool(getattr(block, "norm_topk_prob", False)):
        return "needs normalized top-k scores"
    if getattr(block, "sharding_group", None) is not None:
        return "sharded blocks unsupported"
    switch = block.switch_mlp
    shared = block.shared_expert
    if not hasattr(switch, "gate_up_proj") or not hasattr(shared, "gate_up_proj"):
        return "gate/up not packed (moe_pack)"
    router = _quantized(
        block.gate, bits=8, group_size=64, wshape=(256, 512), mshape=(256, 32)
    )
    routed_gate_up = _quantized(
        switch.gate_up_proj,
        bits=4,
        group_size=64,
        wshape=(256, 1024, 256),
        mshape=(256, 1024, 32),
    )
    routed_down = _quantized(
        switch.down_proj,
        bits=4,
        group_size=64,
        wshape=(256, 2048, 64),
        mshape=(256, 2048, 8),
    )
    shared_gate_up = _quantized(
        shared.gate_up_proj, bits=4, group_size=64, wshape=(1024, 256), mshape=(1024, 32)
    )
    shared_down = _quantized(
        shared.down_proj, bits=4, group_size=64, wshape=(2048, 64), mshape=(2048, 8)
    )
    scalar_gate = _quantized(
        block.shared_expert_gate, bits=8, group_size=64, wshape=(1, 512), mshape=(1, 32)
    )
    parts = {
        "router": router,
        "routed gate_up": routed_gate_up,
        "routed down": routed_down,
        "shared gate_up": shared_gate_up,
        "shared down": shared_down,
        "scalar gate": scalar_gate,
    }
    for name, part in parts.items():
        if part is None:
            return f"{name} storage off-contract"
    return _Binding(
        block=block,
        stock_call=type(block).__call__,
        variant="target_q8g64_q4g64",
        router=router,
        routed_gate_up=routed_gate_up,
        routed_down=routed_down,
        shared_gate_up=shared_gate_up,
        shared_down=shared_down,
        shared_scalar_gate=scalar_gate,
    )


def _parity(block, routes, stock_call) -> float:
    """Max abs diff between the kernel routes and the stock forward."""
    import mlx.core as mx

    worst = 0.0
    for rows, call in routes.items():
        fixture = mx.arange(rows * 2048, dtype=mx.float32).reshape(1, rows, 2048)
        value = (
            mx.sin(fixture * 0.013) * 0.25 + mx.cos(fixture * 0.007) * 0.0625
        ).astype(mx.bfloat16)
        ours = call(value)
        stock = stock_call(block, value)
        mx.eval(ours, stock)
        worst = max(
            worst,
            float(
                mx.abs(ours.astype(mx.float32) - stock.astype(mx.float32)).max()
            ),
        )
    return worst


# The kernels compute in float with bf16 stores between stages; the stock
# path rounds at different points, so outputs differ in low bf16 bits. The
# reference gates its install at 0.5 absolute on this block's output scale.
PARITY_LIMIT = 0.5


def install(model) -> int:
    """Bind the whole-MoE verify routes on every eligible block."""
    if not enabled():
        return 0
    from . import _a3b_whole_moe_kernels as kernels

    text = getattr(model, "language_model", model)
    layers = list(getattr(getattr(text, "model", None), "layers", ()) or ())
    installed = 0
    reasons: Dict[str, int] = {}
    checked = False
    for layer in layers:
        block = getattr(layer, "mlp", None)
        if block is None or not hasattr(block, "switch_mlp"):
            continue
        made = _binding(block)
        if isinstance(made, str):
            reasons[made] = reasons.get(made, 0) + 1
            continue
        routes = {
            2: kernels.bind_target_m2(made),
            3: kernels.bind_target_m3(made),
        }
        if m1_enabled():
            routes[1] = kernels.bind_target_m1(made)
        if not checked:
            # One fixture parity check against the stock forward; the blocks
            # share code and layout, so the first proves the contract and
            # keeps install O(1) evals.
            diff = _parity(block, routes, made.stock_call)
            if diff > PARITY_LIMIT:
                logger.warning(
                    f"whole-MoE verify: parity {diff:.3f} exceeds "
                    f"{PARITY_LIMIT}; keeping the stock path"
                )
                return 0
            checked = True
        _ROUTES[id(block)] = routes
        installed += 1
    if installed:
        logger.info(
            "whole-MoE verify: %d blocks route %s through fused launches",
            installed,
            "1-3 row windows (M1 decode on)" if m1_enabled() else "2-3 row windows",
        )
    for reason, count in reasons.items():
        logger.info(f"whole-MoE verify: {count} blocks skipped ({reason})")
    return installed


def route(block, x):
    """The bound call for this forward, or None for the stock path."""
    routes = _ROUTES.get(id(block))
    if not routes:
        return None
    import mlx.core as mx

    if x.ndim != 3 or x.shape[0] != 1 or x.dtype != mx.bfloat16:
        return None
    return routes.get(int(x.shape[1]))


def uninstall() -> None:
    _ROUTES.clear()
