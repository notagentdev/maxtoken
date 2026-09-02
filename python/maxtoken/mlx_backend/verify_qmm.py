"""A quantized matmul for the 2-4 rows of a speculative verify window.

MLX's quantized matmul has an excellent path for a single row and a poor one
for a handful. Measured on this machine, a 5120x5120 4-bit linear costs
0.325 ms at one row and 0.450 ms at two (+38%), while the same shape in
bfloat16 costs 0.534 ms and 0.535 ms (+0.001) — unquantized, the second row is
free, because the weights were already streamed. Across a whole 27B forward
that gap is 36 ms per extra window token, which is what makes speculative
decoding lose on a quantized model: the window costs nearly as much per token
as running the tokens separately, so there is nothing left for the drafter to
win. See docs/mlx.md for the full measurement.

The fix is to dequantize each weight ONCE and reuse it across every row of the
window, which is exactly the reuse the multi-row path fails to exploit:

    for each of the 8 weights in a packed word:
        w = float((pack >> shift) & 0xF) * scale + bias    # once
        for r in range(M):                                  # ...for all rows
            acc[col][r] += x[r][k] * w

Geometry: one simdgroup owns 4 output columns, its 32 lanes stride over the
K/8 packed words, and each lane keeps 4*M accumulators in registers. A final
``simd_sum`` reduces across lanes and the first 4*M lanes write the tile.

One kernel is compiled per row count rather than padding everything up to 4.
Padding looked tidy and cost 16% at M=2: the kernel's time is flat in M (9.20 /
9.24 / 8.91 ms for M=2/3/4 on a 32-matmul chain), so a padded M=2 pays M=4's
FMAs for rows it then throws away. Three cached kernels are cheaper than that.

The approach is taken from MTPLX's verify_kernels (Apache-2.0,
https://github.com/youssofal/MTPLX) — the dequantize-once-reuse-across-rows
structure and the simdgroup-per-4-columns geometry are theirs. This is an
independent implementation of it, narrowed to the 4-bit affine case our
checkpoints actually use.
"""

from __future__ import annotations

import os
from typing import Any

from maxtoken.utils import init_logger

logger = init_logger(__name__)

# Simdgroups per threadgroup: each owns 4 output columns, so a threadgroup
# covers 4*NSG columns and N must divide by that.
NSG = max(1, min(24, int(os.environ.get("MAXTOKEN_VERIFY_QMM_NSG", "8") or 8)))
MROWS = 4          # rows the kernel is compiled for; 2 and 3 are padded up
PACK = 8           # 4-bit weights per 32-bit word
# "half": 16-bit products and per-pack partial sums, float accumulation across
# packs (the default, measured ~10-15% faster on the 27B's shapes). "float":
# the original all-float path, kept for A/B and for exactness comparisons.
MATH = os.environ.get("MAXTOKEN_VERIFY_QMM_MATH", "half").strip().lower() or "half"
if MATH not in ("half", "float"):
    MATH = "half"
# Split-K for the deep-K, narrow-N shapes (the MLP's down projection:
# K=17408, N=5120). There a threadgroup owns 4*NSG columns and scans the
# whole K, so only N/(4*NSG) threadgroups exist to hide memory latency —
# 160 at NSG=8, far under what 32 GPU cores want in flight. Splitting K
# into segments multiplies the threadgroups; each writes fp32 partials and
# one mx.sum folds them.
#
# MEASURED (2026-09-02, M1 Max) and left OFF by default: isolated, splits=4
# lifts the down shape 121 -> 154 GB/s (+27%, bit-identical); through the
# full served 27B verify it LOSES ~1 tok/s (33.2 -> 32.0). Inside the real
# forward the wide command buffers already overlap the down matmul with its
# neighbor layers' kernels, so the occupancy gap split-K fixes is hidden
# there, and only the partial-buffer + reduction overhead remains. Kept for
# A/Bs: MAXTOKEN_VERIFY_QMM_SPLITK=<n>.
try:
    SPLITK = max(0, int(os.environ.get("MAXTOKEN_VERIFY_QMM_SPLITK", "0") or 0))
except ValueError:
    SPLITK = 0
SPLITK_MIN_K = 12288

_KERNELS: dict[tuple, Any] = {}
_PATCHED: dict[str, Any] = {}


def _pack_block(m: int, math: str) -> str:
    """Dequantize one packed word per column, then FMA it into every row.

    The dequantized value lives in a register across the row loop; that reuse
    is the entire point of the kernel. In ``half`` math the products of a pack
    accumulate in 16-bit registers and are folded into the float accumulators
    once per pack: this GPU issues half FMAs about 1.6x faster than float ones
    (measured: 2.43 vs 1.47 T/s on independent chains), the 8-term partial
    sums stay far inside half's range (|x| entering a projection peaks near
    400 on the 27B), and the bf16 result has fewer bits than the half partials.
    """
    if math == "float":
        lines = ["_Pragma(\"unroll\")", "for (int ki = 0; ki < 8; ++ki) {"]
        for j in range(4):
            lines.append(f"    float w{j} = float((p{j} >> (ki * 4)) & 0xFu) * s{j} + b{j};")
        for j in range(4):
            for r in range(m):
                lines.append(f"    acc[{j} * {m} + {r}] += float(v{r}[ki]) * w{j};")
        lines.append("}")
        return "\n        ".join(lines)
    n_acc = 4 * m
    lines = [f"half hacc[{n_acc}];", "_Pragma(\"unroll\")",
             f"for (int i = 0; i < {n_acc}; ++i) {{ hacc[i] = half(0.0); }}"]
    for r in range(m):
        lines.append(f"half h{r}[8];")
        lines.append("_Pragma(\"unroll\")")
        lines.append(f"for (int i = 0; i < 8; ++i) {{ h{r}[i] = half(float(v{r}[i])); }}")
    lines += ["_Pragma(\"unroll\")", "for (int ki = 0; ki < 8; ++ki) {"]
    for j in range(4):
        lines.append(f"    half w{j} = half((p{j} >> (ki * 4)) & 0xFu) * hs{j} + hb{j};")
    for j in range(4):
        for r in range(m):
            lines.append(f"    hacc[{j} * {m} + {r}] = fma(h{r}[ki], w{j}, hacc[{j} * {m} + {r}]);")
    lines.append("}")
    lines += ["_Pragma(\"unroll\")", f"for (int i = 0; i < {n_acc}; ++i) {{ acc[i] += float(hacc[i]); }}"]
    return "\n        ".join(lines)


def _kernel(m: int, group_size: int, dtype, nsg: int):
    import mlx.core as mx

    key = (m, group_size, dtype, nsg, MATH)
    cached = _KERNELS.get(key)
    if cached is not None:
        return cached

    xloads = "\n        ".join(
        f"Vec8 v{r} = xv[({r} * K + k_base) / 8];" for r in range(m)
    )
    n_acc = 4 * m
    if MATH == "float":
        sb = "\n            ".join(
            f"float s{j} = float(scales[(n0 + {j}) * K_by_gs + gi]);\n"
            f"            float b{j} = float(biases[(n0 + {j}) * K_by_gs + gi]);"
            for j in range(4)
        )
    else:
        sb = "\n            ".join(
            f"half hs{j} = half(float(scales[(n0 + {j}) * K_by_gs + gi]));\n"
            f"            half hb{j} = half(float(biases[(n0 + {j}) * K_by_gs + gi]));"
            for j in range(4)
        )
    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int NSG = {nsg};

        uint sg   = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_pack = K / 8;
        int K_by_gs   = K / GS;
        int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(sg)) * 4;
        if (n0 + 3 >= N) {{ return; }}

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = 0.0f; }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = int(lane); pack < K_by_pack; pack += 32) {{
            int k_base = pack * 8;
            int gi = k_base / GS;
            uint32_t p0 = w_q[(n0 + 0) * K_by_pack + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_pack + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_pack + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_pack + pack];
            {xloads}
            {sb}
            {_pack_block(m, MATH)}
        }}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = simd_sum(acc[i]); }}

        if (lane < {n_acc}) {{
            int j   = int(lane) / {m};
            int row = int(lane) - j * {m};
            y[row * N + n0 + j] = T(acc[int(lane)]);
        }}
    """
    tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"ft_verify_qmm_m{m}_gs{group_size}_nsg{nsg}_{MATH}_{tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNELS[key] = kernel
    return kernel


def _kernel_splitk(m: int, group_size: int, dtype, nsg: int, splits: int):
    """The same tile geometry, but each threadgroup owns one K-SEGMENT.

    Partials are written in fp32 to ``y[(z, row, col)]`` and summed by the
    caller — numerically at least as tight as the single-pass float
    accumulator, since segment sums stay fp32 end to end."""
    import mlx.core as mx

    key = ("sk", m, group_size, dtype, nsg, splits, MATH)
    cached = _KERNELS.get(key)
    if cached is not None:
        return cached

    xloads = "\n        ".join(
        f"Vec8 v{r} = xv[({r} * K + k_base) / 8];" for r in range(m)
    )
    n_acc = 4 * m
    if MATH == "float":
        sb = "\n            ".join(
            f"float s{j} = float(scales[(n0 + {j}) * K_by_gs + gi]);\n"
            f"            float b{j} = float(biases[(n0 + {j}) * K_by_gs + gi]);"
            for j in range(4)
        )
    else:
        sb = "\n            ".join(
            f"half hs{j} = half(float(scales[(n0 + {j}) * K_by_gs + gi]));\n"
            f"            half hb{j} = half(float(biases[(n0 + {j}) * K_by_gs + gi]));"
            for j in range(4)
        )
    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int NSG = {nsg};
        constexpr int SPLITS = {splits};

        uint sg   = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_pack = K / 8;
        int K_by_gs   = K / GS;
        int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(sg)) * 4;
        if (n0 + 3 >= N) {{ return; }}
        int z = int(threadgroup_position_in_grid.z);
        int words_per_split = (K_by_pack + SPLITS - 1) / SPLITS;
        int w0 = z * words_per_split;
        int w1 = min(w0 + words_per_split, K_by_pack);

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = 0.0f; }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = w0 + int(lane); pack < w1; pack += 32) {{
            int k_base = pack * 8;
            int gi = k_base / GS;
            uint32_t p0 = w_q[(n0 + 0) * K_by_pack + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_pack + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_pack + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_pack + pack];
            {xloads}
            {sb}
            {_pack_block(m, MATH)}
        }}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = simd_sum(acc[i]); }}

        if (lane < {n_acc}) {{
            int j   = int(lane) / {m};
            int row = int(lane) - j * {m};
            y[(z * {m} + row) * N + n0 + j] = acc[int(lane)];
        }}
    """
    tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"ft_verify_qmm_sk{splits}_m{m}_gs{group_size}_nsg{nsg}_{MATH}_{tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNELS[key] = kernel
    return kernel


def eligible(m: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    """Whether this shape may take the kernel.

    K % 64 keeps every lane's Vec8 activation load and every group boundary
    aligned; N % (4*NSG) means a threadgroup always owns whole 4-column tiles,
    which is what lets the bounds check never fire in the hot loop.
    """
    import mlx.core as mx

    return (
        int(bits) == 4
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and 2 <= int(m) <= MROWS
        and int(K) % 64 == 0
        and int(N) % (4 * NSG) == 0
    )


def verify_qmm(x2, w_q, scales, biases, *, group_size: int):
    """(M, K) x (N, K)^T -> (M, N) for M in 2..4, 4-bit affine."""
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    # Deep-K shapes (the MLP's down projection) run a few percent faster with
    # half the simdgroups per threadgroup; everything else prefers NSG.
    nsg = min(NSG, 4) if K >= 12288 else NSG
    if SPLITK > 1 and K >= SPLITK_MIN_K:
        kernel = _kernel_splitk(M, group_size, x2.dtype, nsg, SPLITK)
        cols = 4 * nsg
        (parts,) = kernel(
            inputs=[mx.contiguous(x2), w_q, scales, biases, K, N],
            template=[("T", x2.dtype)],
            grid=(32 * nsg, N // cols, SPLITK),
            threadgroup=(32 * nsg, 1, 1),
            output_shapes=[(SPLITK, M, N)],
            output_dtypes=[mx.float32],
        )
        return parts.sum(axis=0).astype(x2.dtype)
    kernel = _kernel(M, group_size, x2.dtype, nsg)
    cols = 4 * nsg
    (y,) = kernel(
        inputs=[mx.contiguous(x2), w_q, scales, biases, K, N],
        template=[("T", x2.dtype)],
        grid=(32 * nsg, N // cols, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x2.dtype],
    )
    return y


def install() -> dict[str, int]:
    """Route small-M quantized linears through the kernel, process-wide.

    Patching ``QuantizedLinear.__call__`` rather than the model is deliberate:
    a verify window has to get through every projection in every layer, and
    there is no single call site to wrap. Anything the kernel does not cover —
    8-bit, odd shapes, M=1, M>4 — falls through to the stock path untouched,
    so prefill and ordinary decode keep running exactly as before.
    """
    import mlx.nn as nn

    if _PATCHED:
        return {"already": 1}
    original = nn.QuantizedLinear.__call__
    stats = {"kernel": 0, "stock": 0}

    def patched(self, x):
        if x.ndim >= 2 and 2 <= x.shape[-2] <= MROWS and x.size == x.shape[-1] * x.shape[-2]:
            K = int(self.weight.shape[1]) * PACK
            N = int(self.weight.shape[0])
            m = int(x.shape[-2])
            if (
                getattr(self, "biases", None) is not None
                and str(getattr(self, "mode", "affine")) == "affine"
                and eligible(m, K, N, int(self.bits), int(self.group_size), x.dtype)
            ):
                y = verify_qmm(
                    x.reshape(m, K), self.weight, self.scales, self.biases,
                    group_size=int(self.group_size),
                )
                stats["kernel"] += 1
                y = y.reshape(*x.shape[:-1], N)
                return y + self["bias"] if "bias" in self else y
        stats["stock"] += 1
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _PATCHED["original"] = original
    _PATCHED["stats"] = stats
    logger.info(
        f"verify qmm: small-M kernel installed (M=2..{MROWS}, 4-bit affine, "
        f"{MATH} math, NSG={NSG})"
    )
    return stats


def uninstall() -> None:
    import mlx.nn as nn

    if not _PATCHED:
        return
    nn.QuantizedLinear.__call__ = _PATCHED.pop("original")
    _PATCHED.clear()


def stats() -> dict[str, int]:
    return dict(_PATCHED.get("stats") or {})
