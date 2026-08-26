"""The small-M verify kernel must agree with MLX, on every shape it claims.

A wrong matmul here would not crash — it would quietly change what the model
says, and only in speculative mode, which is exactly the kind of fault that
survives a test suite. So the kernel is checked against stock
``QuantizedLinear`` on the real projection shapes, and its eligibility rule is
checked to reject everything it cannot handle rather than producing garbage.

Tolerance is relative and loose (2%): the kernel accumulates in float32 in a
different order than MLX does and writes bfloat16, so bit-exactness is not the
contract. Agreement to bf16's own resolution is.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="the kernel needs Metal")
nn = pytest.importorskip("mlx.nn")

from freetoken.mlx_backend.verify_qmm import (  # noqa: E402
    MROWS,
    eligible,
    install,
    uninstall,
    verify_qmm,
)


def _rel_error(a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.abs(a32 - b32).max() / mx.maximum(mx.abs(a32).max(), 1e-6)).item())


@pytest.mark.parametrize("shape", [(5120, 5120), (2048, 4096), (2560, 2560)])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("m", [2, 3, 4])
def test_matches_stock_quantized_linear(shape, group_size, m):
    K, N = shape
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=group_size, bits=4)
    mx.eval(lin.parameters())
    assert eligible(m, K, N, 4, group_size, mx.bfloat16)

    x = mx.random.normal((m, K)).astype(mx.bfloat16)
    mx.eval(x)
    got = verify_qmm(x, lin.weight, lin.scales, lin.biases, group_size=group_size)
    mx.eval(got)
    assert got.shape == (m, N)
    assert _rel_error(lin(x), got) < 0.02


def test_rejects_what_it_cannot_do():
    """Eligibility must be conservative: one wrong yes is a silent wrong answer."""
    assert not eligible(1, 5120, 5120, 4, 64, mx.bfloat16), "M=1 belongs to stock qmv"
    assert not eligible(MROWS + 1, 5120, 5120, 4, 64, mx.bfloat16), "M>4 not compiled"
    assert not eligible(2, 5120, 5120, 8, 64, mx.bfloat16), "8-bit not implemented"
    assert not eligible(2, 5120, 5120, 4, 17, mx.bfloat16), "group size unsupported"
    assert not eligible(2, 100, 5120, 4, 64, mx.bfloat16), "K must divide by 64"
    assert not eligible(2, 5120, 100, 4, 64, mx.bfloat16), "N must fill whole tiles"
    assert not eligible(2, 5120, 5120, 4, 64, mx.float32), "float32 has no Vec8 path"


def test_patch_routes_only_small_m_and_restores():
    """Installed, the patch must change small-M results not at all, leave M=1
    and prefill-sized calls on the stock path, and come off cleanly."""
    K = N = 2560
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    mx.eval(lin.parameters())
    xs = {m: mx.random.normal((m, K)).astype(mx.bfloat16) for m in (1, 2, 4, 8)}
    mx.eval(list(xs.values()))
    before = {m: lin(x) for m, x in xs.items()}
    mx.eval(list(before.values()))

    original = nn.QuantizedLinear.__call__
    stats = install()
    try:
        after = {m: lin(x) for m, x in xs.items()}
        mx.eval(list(after.values()))
        for m in xs:
            assert _rel_error(before[m], after[m]) < 0.02, f"M={m} diverged"
        assert stats["kernel"] == 2, "only M=2 and M=4 should take the kernel"
    finally:
        uninstall()
    assert nn.QuantizedLinear.__call__ is original
