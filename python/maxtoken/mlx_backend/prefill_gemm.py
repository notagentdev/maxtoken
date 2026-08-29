"""Prefill through fp16 GEMMs instead of the 4-bit matmul, where that is faster.

A prefill chunk is compute-bound: the same weights multiply hundreds or
thousands of rows, so the question is how many FLOPs a second the kernel
sustains, not how many bytes it streams. Measured on an M1 Max at the 27B's
shapes (bf16 activations, 4-bit affine weights, group size 64):

    shape               rows   qmm 4-bit    fp16 GEMM   dequantize + GEMM
    5120 -> 17408       2048   5.6 TFLOPS   9.0         7.8 (bf16)
    5120 -> 17408        256   5.4          8.1         6.3
    5120 -> 10240       2048   5.7          9.0         7.7
    17408 -> 5120       2048   5.7          5.1         5.3   <- no gain; stays quantized

So for the projections whose K is the hidden size -- gate and up, q/k/v/o,
the gated-delta's in/out -- a chunk dequantizes the matrix to fp16 once (the
cost that the 4-bit kernel pays again per tile) and runs a plain fp16 GEMM
on fp16 activations; the result returns in the activations' dtype. The down
projection (K = 17408) keeps the quantized path: the GEMM is no faster there.
Half precision is safe for these operands: activations entering a projection
peak near 400 on the 27B (measured), a 4-bit value times its scale is small,
and both are far inside fp16's range; fp16 also carries more mantissa than
the bf16 the model computes in.

Decode never enters this path (rows < PREFILL_MIN_ROWS), so the single-row
kernel and the verify kernel are untouched. The dequantized matrix is a
transient of the chunk's graph, freed once its GEMM ran; with the prefill
paced every few layers (prefill_pacing.py) at most that many layers' worth
exists at once.
"""

from __future__ import annotations

from typing import Any, Dict

from maxtoken.utils import init_logger

logger = init_logger(__name__)

_PATCHED: Dict[str, Any] = {}

# Below this many rows the 4-bit kernel is bandwidth-bound and already right.
PREFILL_MIN_ROWS = 64
# Above this K the fp16 GEMM measured no faster than the quantized matmul.
MAX_K = 8192


def eligible(rows: int, K: int, N: int, bits: int, dtype) -> bool:
    import mlx.core as mx

    return (
        rows >= PREFILL_MIN_ROWS
        and int(bits) == 4
        and int(K) <= MAX_K
        and dtype in (mx.bfloat16, mx.float16)
    )


def gemm_fp16(x, weight, scales, biases, *, group_size: int, bits: int):
    """``x @ dequantize(weight)^T`` through an fp16 GEMM, in ``x``'s dtype."""
    import mlx.core as mx

    w16 = mx.dequantize(
        weight, scales.astype(mx.float16), biases.astype(mx.float16),
        group_size=group_size, bits=bits,
    )
    y = mx.matmul(x.astype(mx.float16), w16.T)
    return y if x.dtype == mx.float16 else y.astype(x.dtype)


def install() -> Dict[str, int]:
    """Route wide quantized linears through the fp16 GEMM, process-wide. Wraps
    whatever ``QuantizedLinear.__call__`` is at install time (the verify kernel
    patches it first, for 2-4 rows), so each patch keeps its own row range."""
    import mlx.nn as nn

    if _PATCHED:
        return {"already": 1}
    original = nn.QuantizedLinear.__call__
    stats = {"gemm": 0, "other": 0}

    def patched(self, x):
        if (
            x.ndim >= 2
            and str(getattr(self, "mode", "affine")) == "affine"
            and getattr(self, "biases", None) is not None
        ):
            rows = x.size // int(x.shape[-1])
            K = int(x.shape[-1])
            N = int(self.weight.shape[0])
            if eligible(rows, K, N, int(self.bits), x.dtype):
                y = gemm_fp16(
                    x.reshape(rows, K), self.weight, self.scales, self.biases,
                    group_size=int(self.group_size), bits=int(self.bits),
                )
                stats["gemm"] += 1
                y = y.reshape(*x.shape[:-1], N)
                return y + self["bias"] if "bias" in self else y
        stats["other"] += 1
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _PATCHED["original"] = original
    _PATCHED["stats"] = stats
    logger.info(
        f"prefill gemm: fp16 GEMM for 4-bit linears with >= {PREFILL_MIN_ROWS} rows and K <= {MAX_K}"
    )
    return stats


def uninstall() -> None:
    import mlx.nn as nn

    if not _PATCHED:
        return
    nn.QuantizedLinear.__call__ = _PATCHED.pop("original")
    _PATCHED.clear()


def stats() -> Dict[str, int]:
    return dict(_PATCHED.get("stats") or {})
