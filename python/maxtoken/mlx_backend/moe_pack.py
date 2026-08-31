"""Gate/up projections packed into one matmul for Qwen-family MoE blocks.

A sparse MoE block runs gate and up as two matmuls that read the same
activation and differ only in which output rows they produce, so the two
matrices can be concatenated along the output-feature axis and evaluated as
ONE matmul whose result is split. Affine quantization groups run along the
input axis, so the concatenation leaves every group intact: each output
element is produced by exactly the dot product that produced it before —
bitwise identical to the unpacked pair.

The win is dispatch count, not arithmetic. mx.gather_qmm at a verify window's
sizes (a few rows, top-8 of 256 experts, 512-wide intermediates) is
launch-bound — the microbench puts ~70 us of the ~120 us per MoE layer at
S=1 beyond the byte cost — and dropping one gathered matmul per layer helps
every decode step, verify window and prefill chunk alike.

For the routed experts the packed weight comes from the FTW mapped store
(ftw_mlx.py writes gate and up interleaved per expert since manifest v2), so
packing costs no resident memory. The shared expert is dense and small
(~0.5 MB a layer), so its pair is concatenated in place at install.

The recipe is MTPLX's moe_packed_projections (Apache-2.0); this is an
independent implementation narrowed to what our serving paths need.
"""

from __future__ import annotations

from typing import Any

from maxtoken.utils import init_logger

logger = init_logger(__name__)

_CLASSES: tuple | None = None


def classes() -> tuple:
    """(PackedQuantizedProjection, PackedSwitchGLU, PackedGateUpMLP).

    Defined lazily so importing this module never pulls mlx; ``__call__``
    must live on real classes — Python resolves ``obj(...)`` on the type, so
    an instance-attribute forward would silently never run."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_next import swiglu
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    class PackedQuantizedProjection(nn.Module):
        """One affine-quantized projection holding two stacked matrices."""

        def __init__(self, weight, scales, biases, *, group_size, bits, mode):
            super().__init__()
            self.weight = weight
            self.scales = scales
            if biases is not None:
                self.biases = biases
            self.group_size = int(group_size)
            self.bits = int(bits)
            self.mode = str(mode)
            self.freeze()

        def __call__(self, x):
            return mx.quantized_matmul(
                x,
                self["weight"],
                scales=self["scales"],
                biases=self["biases"] if "biases" in self else None,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )

        def gather(self, x, indices, sorted_indices):
            return mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                self["biases"] if "biases" in self else None,
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )

    class PackedSwitchGLU(nn.Module):
        """SwitchGLU with gate/up as a single gathered projection.

        Call-for-call the same semantics as mlx_lm's SwitchGLU, including
        the token-sorting path; only the two expert matmuls become one."""

        def __init__(self, gate_up_proj, down_proj, activation, split_at):
            super().__init__()
            self.gate_up_proj = gate_up_proj
            self.down_proj = down_proj
            self.activation = activation
            self._split_at = int(split_at)

        def __call__(self, x, indices):
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = _gather_sort(x, indices)
            packed = self.gate_up_proj.gather(x, idx, do_sort)
            x_gate, x_up = mx.split(packed, [self._split_at], axis=-1)
            x = self.down_proj(
                self.activation(x_up, x_gate), idx, sorted_indices=do_sort
            )
            if do_sort:
                x = _scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

    class PackedGateUpMLP(nn.Module):
        """Shared-expert MLP with gate/up packed into one projection."""

        def __init__(self, gate_up_proj, down_proj, split_at):
            super().__init__()
            self.gate_up_proj = gate_up_proj
            self.down_proj = down_proj
            self._split_at = int(split_at)

        def __call__(self, x):
            gate, up = mx.split(self.gate_up_proj(x), [self._split_at], axis=-1)
            return self.down_proj(swiglu(gate, up))

    _CLASSES = (PackedQuantizedProjection, PackedSwitchGLU, PackedGateUpMLP)
    return _CLASSES


def _quant_metadata(module) -> tuple | None:
    if "scales" not in module:
        return None
    return (
        int(getattr(module, "group_size", 64)),
        int(getattr(module, "bits", 4)),
        str(getattr(module, "mode", "affine")),
    )


def pack_shared_experts(model) -> int:
    """Concatenate every shared expert's gate/up pair in place.

    Dense and tiny (intermediate 512 on the A3B family), so the concat's
    resident cost is negligible; the saved dispatch is the same as the
    routed pack's. Returns the number of blocks packed."""
    import mlx.core as mx

    PackedQuantizedProjection, _, PackedGateUpMLP = classes()
    packed = 0
    for layer in _layers(model):
        block = getattr(layer, "mlp", None)
        shared = getattr(block, "shared_expert", None)
        if shared is None or not hasattr(shared, "gate_proj"):
            continue
        gate, up = shared.gate_proj, shared.up_proj
        gq, uq = _quant_metadata(gate), _quant_metadata(up)
        if gq is None or gq != uq or "bias" in gate or "bias" in up:
            continue
        if gate["weight"].shape != up["weight"].shape:
            continue
        if ("biases" in gate) != ("biases" in up):
            continue
        split_at = int(gate["weight"].shape[0])
        weight = mx.concatenate([gate["weight"], up["weight"]], axis=0)
        scales = mx.concatenate([gate["scales"], up["scales"]], axis=0)
        biases = None
        if "biases" in gate:
            biases = mx.concatenate([gate["biases"], up["biases"]], axis=0)
        evals = [weight, scales] + ([biases] if biases is not None else [])
        mx.eval(*evals)
        group_size, bits, mode = gq
        proj = PackedQuantizedProjection(
            weight, scales, biases, group_size=group_size, bits=bits, mode=mode
        )
        block.shared_expert = PackedGateUpMLP(proj, shared.down_proj, split_at)
        packed += 1
    if packed:
        logger.info(
            "MoE pack: %d shared experts packed (gate+up -> one matmul)", packed
        )
    return packed


def _layers(model):
    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", None)
    return list(getattr(inner, "layers", ()) or ())
