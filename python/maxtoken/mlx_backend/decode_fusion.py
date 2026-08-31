"""Fewer kernel launches per decode step on hybrid MoE models.

A decode step on Ornith-1.5-35B-A3B moves about 1.4 GB of weights, which the
memory system could stream in 5 ms; the step takes 14.5. The difference is
not bandwidth but launches: forty layers of gated-delta blocks and MoE blocks,
each a chain of small kernels -- four input projections, a convolution, two
norms, the recurrence, a router, top-k, three gathered matmuls, a shared
expert with its own gate, and the elementwise glue between them -- add up to
well over a thousand launches per token, and at that count the step is paid
in dispatch latency rather than bytes.

Two things shorten the chain without touching the arithmetic:

* The gated-delta block's four input projections read the same activations
  and are 4-bit affine matrices of the same group size, so their quantized
  rows are concatenated once at load into ONE linear, and the block's forward
  splits its output. Three launches fewer per recurrent layer, bit-identical
  (measured: 0.155 -> 0.111 ms for the projections alone on a chain).

* The MoE block is a pure function of its input, so every instance is
  ``mx.compile``d for the decode shape. The compiler fuses the score
  normalization, the expert weighting, the shared expert's gate and the sums
  into a handful of kernels (measured: 0.218 -> 0.146 ms per block on a
  chain). Prefill keeps the eager path -- compile traces per shape, and a
  prompt's chunk lengths vary -- as does anything that arrives with a mask.

Both are installed by patching the module classes, so the model's own
``__call__`` keeps working for every caller, prefill included. Nothing here
applies to the expert slot cache, which routes the MoE blocks itself.
"""

from __future__ import annotations

from typing import Any, Dict, List

from maxtoken.utils import init_logger

logger = init_logger(__name__)

_PATCHED: Dict[str, Any] = {}
_COMPILED: Dict[int, Any] = {}  # id(moe block) -> compiled decode forward

# Batch sizes the compiled MoE forward is allowed to trace; beyond this the
# eager path serves (continuous batching at odd widths would otherwise
# collect a trace per width).
MAX_COMPILED_BATCH = 8

# Verify windows of the speculative loop are a few rows wide at B=1. The same
# compiled forward serves them: mx.compile traces once per shape, and the
# widths are bounded by the loop's MAX_WINDOW, so the trace count stays small.
# Without this the whole 40-layer MoE chain of a verify runs eager, and on a
# launch-bound model the verify pays in dispatch latency what speculation
# saved in steps.
MAX_COMPILED_WIDTH = 6


def _text_model(model):
    return getattr(model, "language_model", model)


def _layers(model) -> List[Any]:
    inner = getattr(_text_model(model), "model", None)
    return list(getattr(inner, "layers", ()) or ())


# -- gated-delta input projections ---------------------------------------------

_GDN_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _fusable(lins) -> bool:
    import mlx.nn as nn

    if not all(isinstance(lin, nn.QuantizedLinear) for lin in lins):
        return False
    first = lins[0]
    return all(
        int(lin.bits) == int(first.bits)
        and int(lin.group_size) == int(first.group_size)
        and str(getattr(lin, "mode", "affine")) == str(getattr(first, "mode", "affine"))
        and "bias" not in lin
        and lin.weight.shape[1] == first.weight.shape[1]
        for lin in lins
    )


def _fuse_linears(lins):
    """One QuantizedLinear whose output is the concatenation of ``lins``'."""
    import mlx.core as mx
    import mlx.nn as nn

    fused = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(fused)
    fused.weight = mx.concatenate([lin.weight for lin in lins], axis=0)
    fused.scales = mx.concatenate([lin.scales for lin in lins], axis=0)
    fused.biases = mx.concatenate([lin.biases for lin in lins], axis=0)
    fused.group_size = int(lins[0].group_size)
    fused.bits = int(lins[0].bits)
    fused.mode = str(getattr(lins[0], "mode", "affine"))
    mx.eval(fused.parameters())
    fused.freeze()
    return fused


def _make_gdn_call(original):
    """mlx-lm's gated-delta forward with the four input projections replaced by
    the fused one. Everything else -- mask, padded batches, cache bookkeeping,
    the recurrence -- is the stock code."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    def fused_call(self, inputs, mask=None, cache=None):
        fused = _PATCHED["gdn_fused"].get(id(self))
        # Decode only. At one row the fused and the separate projections run
        # the same single-row kernel and agree bit for bit; at prefill widths
        # MLX tiles the wider matrix differently and the last bf16 bit moves,
        # which would make a prefill differ from the stock path -- and from
        # the prefix cache's contract that a restored prefix equals a kept one.
        if (
            fused is None
            or inputs.shape[1] != 1
            or getattr(self, "sharding_group", None) is not None
        ):
            return original(self, inputs, mask, cache)
        B, S, _ = inputs.shape
        proj, splits = fused
        qkv, z, b, a = mx.split(proj(inputs), splits, axis=-1)
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    return fused_call


def _fuse_gdn_projections(model) -> int:
    gdns = [
        layer.linear_attn
        for layer in _layers(model)
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn")
    ]
    gdns = [g for g in gdns if all(hasattr(g, n) for n in _GDN_PROJECTIONS)]
    if not gdns:
        return 0
    cls = type(gdns[0])
    if any(type(g) is not cls for g in gdns):
        return 0
    fused_by_id = _PATCHED.setdefault("gdn_fused", {})
    for g in gdns:
        lins = [getattr(g, n) for n in _GDN_PROJECTIONS]
        if not _fusable(lins):
            continue
        sizes = [int(lin.weight.shape[0]) for lin in lins]
        splits = [sizes[0], sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2]]
        fused_by_id[id(g)] = (_fuse_linears(lins), splits)
        # The originals stay on the module (checkpoint layout, weight
        # accounting, the capture path); only the forward stops reading them.
    if fused_by_id and "gdn_original" not in _PATCHED:
        _PATCHED["gdn_original"] = cls.__call__
        _PATCHED["gdn_class"] = cls
        cls.__call__ = _make_gdn_call(cls.__call__)
    return len(fused_by_id)


# -- MoE blocks ----------------------------------------------------------------


def _moe_classes():
    classes = []
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

        classes.append(Qwen3NextSparseMoeBlock)
    except Exception:  # noqa: BLE001 -- an mlx-lm without this family
        pass
    return classes


def _compile_moe_blocks(model) -> int:
    import mlx.core as mx

    from . import whole_moe

    classes = tuple(_moe_classes())
    if not classes:
        return 0
    blocks = [
        layer.mlp for layer in _layers(model)
        if isinstance(getattr(layer, "mlp", None), classes)
        and getattr(layer.mlp, "sharding_group", None) is None
    ]
    if not blocks:
        return 0
    originals = _PATCHED.setdefault("moe_originals", {})
    for cls in classes:
        if cls in originals:
            continue
        original = cls.__call__
        originals[cls] = original

        def patched(self, x, _orig=original):
            # Whole-MoE verify rows first: the entire block in four fused
            # launches (whole_moe.py), bound only for the shapes and storage
            # its install proved. Everything else keeps the compiled or
            # stock path.
            wm = whole_moe.route(self, x)
            if wm is not None:
                return wm(x)
            fn = _COMPILED.get(id(self))
            if (
                fn is not None
                and x.ndim == 3
                and (
                    (x.shape[1] == 1 and x.shape[0] <= MAX_COMPILED_BATCH)
                    or (x.shape[0] == 1 and x.shape[1] <= MAX_COMPILED_WIDTH)
                )
            ):
                return fn(x)
            return _orig(self, x)

        cls.__call__ = patched
    for block in blocks:
        original = originals[type(block)]
        _COMPILED[id(block)] = mx.compile(lambda x, _b=block, _o=original: _o(_b, x))
    return len(blocks)


# -- MoE expert-sort threshold -------------------------------------------------


def _tune_switch_sort() -> int:
    """Lower SwitchGLU's token-sort threshold (MAXTOKEN_MLX_MOE_SORT_MIN).

    Stock mlx-lm sorts tokens by expert only from 64 (token, expert) pairs
    upward; below that the unsorted gather_qmm serves each pair on its own,
    re-reading an expert's weights once per pair. A speculative verify window
    is 4 tokens x top-8 = 32 pairs, so its expert reads never dedupe and the
    verify's cost grows linearly with its rows (measured ~5.3 ms per extra
    row on Ornith-1.5-35B). Sorting groups the pairs by expert so shared
    experts stream once. Off unless the env var is set — the stock threshold
    is the tested default for everything else."""
    import os

    raw = os.environ.get("MAXTOKEN_MLX_MOE_SORT_MIN", "").strip()
    if not raw:
        return 0
    try:
        threshold = max(1, int(raw))
    except ValueError:
        return 0
    import mlx.core as mx
    from mlx_lm.models import switch_layers as sl

    if "switch_sort_original" not in _PATCHED:
        _PATCHED["switch_sort_original"] = sl.SwitchGLU.__call__

        def sorted_call(self, x, indices):
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= threshold
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = sl._gather_sort(x, indices)
            if self.training:
                idx = mx.stop_gradient(idx)
            x_up = self.up_proj(x, idx, sorted_indices=do_sort)
            x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
            x = self.down_proj(
                self.activation(x_up, x_gate), idx, sorted_indices=do_sort
            )
            if do_sort:
                x = sl._scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

        sl.SwitchGLU.__call__ = sorted_call
        logger.info(
            "MoE expert sort: threshold lowered to %d (token, expert) pairs",
            threshold,
        )
    return threshold


# -- entry points --------------------------------------------------------------


def install(model) -> Dict[str, int]:
    """Fuse and compile what the model offers. Call AFTER any weight surgery
    (mapped experts), since the compiled blocks bind the parameter arrays they
    were traced with. Returns the counts of what was installed."""
    # Before the MoE blocks are compiled: a compiled trace bakes the branch
    # SwitchGLU takes at its width, so the threshold must be in force first —
    # and the packed shared experts must already sit on the blocks.
    _tune_switch_sort()
    from . import moe_pack, whole_moe

    shared_packed = moe_pack.pack_shared_experts(model)
    report = {
        "gdn_fused": _fuse_gdn_projections(model),
        "moe_compiled": _compile_moe_blocks(model),
        "shared_packed": shared_packed,
        "whole_moe": whole_moe.install(model),
    }
    if any(report.values()):
        logger.info(
            "decode fusion: %d gated-delta blocks with fused input projections, "
            "%d MoE blocks compiled for decode",
            report["gdn_fused"], report["moe_compiled"],
        )
    return report


def uninstall() -> None:
    cls = _PATCHED.pop("gdn_class", None)
    original = _PATCHED.pop("gdn_original", None)
    if cls is not None and original is not None:
        cls.__call__ = original
    switch_original = _PATCHED.pop("switch_sort_original", None)
    if switch_original is not None:
        from mlx_lm.models import switch_layers as sl

        sl.SwitchGLU.__call__ = switch_original
    for moe_cls, original in (_PATCHED.pop("moe_originals", None) or {}).items():
        moe_cls.__call__ = original
    _PATCHED.clear()
    _COMPILED.clear()
