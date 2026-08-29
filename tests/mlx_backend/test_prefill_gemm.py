"""The fp16 prefill GEMM must agree with the 4-bit matmul it replaces, and only
take over wide inputs: decode rows and the verify window keep their kernels."""

import pytest

mx = pytest.importorskip("mlx.core", reason="needs Metal")
nn = pytest.importorskip("mlx.nn")

from maxtoken.mlx_backend import prefill_gemm  # noqa: E402


def _rel_error(a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.abs(a32 - b32).max() / mx.maximum(mx.abs(a32).max(), 1e-6)).item())


@pytest.mark.parametrize("rows", [64, 256, 1024])
def test_gemm_matches_the_quantized_matmul(rows):
    lin = nn.QuantizedLinear(512, 1024, bias=False, group_size=64, bits=4)
    mx.eval(lin.parameters())
    x = mx.random.normal((rows, 512)).astype(mx.bfloat16)
    mx.eval(x)
    want = lin(x)
    got = prefill_gemm.gemm_fp16(x, lin.weight, lin.scales, lin.biases, group_size=64, bits=4)
    mx.eval(want, got)
    assert got.dtype == mx.bfloat16 and got.shape == want.shape
    assert _rel_error(want, got) < 0.02


def test_routing_takes_wide_inputs_only_and_restores():
    lin = nn.QuantizedLinear(512, 1024, bias=False, group_size=64, bits=4)
    wide = nn.QuantizedLinear(9216, 256, bias=False, group_size=64, bits=4)  # K above MAX_K
    mx.eval(lin.parameters(), wide.parameters())
    xs = {rows: mx.random.normal((1, rows, 512)).astype(mx.bfloat16) for rows in (1, 4, 63, 64, 300)}
    mx.eval(list(xs.values()))
    before = {rows: lin(x) for rows, x in xs.items()}
    xw = mx.random.normal((1, 300, 9216)).astype(mx.bfloat16)
    mx.eval(list(before.values()), xw)
    original = nn.QuantizedLinear.__call__
    stats = prefill_gemm.install()
    try:
        after = {rows: lin(x) for rows, x in xs.items()}
        mx.eval(list(after.values()))
        for rows in xs:
            assert _rel_error(before[rows], after[rows]) < 0.02, f"rows={rows} diverged"
        assert stats["gemm"] == 2, "only the 64- and 300-row inputs take the GEMM"
        mx.eval(wide(xw))
        assert stats["gemm"] == 2, "K above MAX_K stays on the quantized path"
    finally:
        prefill_gemm.uninstall()
    assert nn.QuantizedLinear.__call__ is original


def test_eligibility_rules():
    assert prefill_gemm.eligible(64, 5120, 17408, 4, mx.bfloat16)
    assert not prefill_gemm.eligible(4, 5120, 17408, 4, mx.bfloat16), "verify window stays"
    assert not prefill_gemm.eligible(2048, 17408, 5120, 4, mx.bfloat16), "down_proj gains nothing"
    assert not prefill_gemm.eligible(2048, 5120, 5120, 8, mx.bfloat16), "8-bit not measured"
    assert not prefill_gemm.eligible(2048, 5120, 5120, 4, mx.float32)
