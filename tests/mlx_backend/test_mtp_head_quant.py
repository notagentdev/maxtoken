"""The MTP head's quantization scheme is its own, not the trunk's.

MTPLX packs every draft head at INT4/g64 (`mtplx_mtp_quantization`) while the
trunk may be g32 with 8-bit attention (Qwen3.8-27B "Optimized-Speed"). Reading
the trunk's block for the head raised `[quantized_matmul] ... incompatible`
on every request (2026-09-11)."""
import numpy as np

from maxtoken.mlx_backend.mtp_draft import head_quant_scheme


def _packed(out, inp, bits, group):
    return {
        "fc.weight": np.zeros((out, inp * bits // 32), dtype=np.uint32),
        "fc.scales": np.zeros((out, inp // group), dtype=np.float16),
        "fc.biases": np.zeros((out, inp // group), dtype=np.float16),
        "norm.weight": np.zeros((out,), dtype=np.float16),
    }


def test_head_record_wins_over_trunk_scheme():
    cfg = {
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "mtplx_mtp_quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    assert head_quant_scheme(cfg, _packed(5120, 10240, 4, 64)) == {
        "bits": 4, "group_size": 64, "mode": "affine"}


def test_contract_group_when_no_head_record():
    cfg = {
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "mtplx_mtp_contract": {"mtp_quant_group_size": 64, "mtp_quant_mode": "affine"},
    }
    assert head_quant_scheme(cfg, _packed(5120, 10240, 4, 64))["group_size"] == 64


def test_trunk_scheme_is_the_fallback():
    cfg = {"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}
    assert head_quant_scheme(cfg, _packed(5120, 10240, 4, 64)) == {
        "bits": 4, "group_size": 64, "mode": "affine"}


def test_shapes_override_a_wrong_config():
    cfg = {
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "mtplx_mtp_quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
    }
    # tensors packed at g64 despite the record: the shapes decide
    assert head_quant_scheme(cfg, _packed(5120, 10240, 4, 64))["group_size"] == 64


def test_eight_bit_head_shapes():
    cfg = {"quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
           "mtplx_mtp_quantization": {"bits": 8, "group_size": 64, "mode": "affine"}}
    s = head_quant_scheme(cfg, _packed(1024, 4096, 8, 64))
    assert (s["bits"], s["group_size"]) == (8, 64)
