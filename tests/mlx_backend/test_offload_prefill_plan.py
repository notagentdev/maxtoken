"""The slot-cache prefill's chunk plan: few full passes, fine restore points."""

from maxtoken.mlx_backend.worker import offload_prefill_plan


def test_long_prompt_takes_few_passes_and_keeps_the_last_boundary():
    # A 9k-token agent prompt: 6 passes instead of 36, snapshots at every
    # chunk end and at the last 256-boundary before the end (8960).
    plan = offload_prefill_plan(0, 8999, chunk=2048, boundary=256)
    assert [n for _, n, _ in plan] == [2048, 2048, 2048, 2048, 768, 39]
    assert [pos for pos, _, _ in plan] == [0, 2048, 4096, 6144, 8192, 8960]
    assert [s for _, _, s in plan] == [True, True, True, True, True, False]
    assert sum(n for _, n, _ in plan) == 8999


def test_short_remainder_after_restore_is_one_chunk():
    assert offload_prefill_plan(8960, 9000, chunk=2048, boundary=256) == [(8960, 40, False)]


def test_end_on_a_boundary_never_snapshots_the_end():
    # Snapshots are for positions strictly before the end (the last token is
    # the first decode step's input, exactly as before).
    assert offload_prefill_plan(0, 512, chunk=2048, boundary=256) == [(0, 512, False)]


def test_chunk_rounds_down_to_boundary_multiples_and_never_below_one():
    assert [n for _, n, _ in offload_prefill_plan(0, 2000, chunk=1000, boundary=256)] == [
        768, 768, 256, 208,
    ]
    assert [n for _, n, _ in offload_prefill_plan(0, 600, chunk=17, boundary=256)] == [
        256, 256, 88,
    ]


def test_chunk_256_reproduces_the_old_boundary_walk():
    plan = offload_prefill_plan(0, 700, chunk=256, boundary=256)
    assert plan == [(0, 256, True), (256, 256, True), (512, 188, False)]


def test_restart_mid_chunk_realigns_to_absolute_boundaries():
    # A restore at 2304 (a boundary) still stops on the absolute 2048-grid.
    plan = offload_prefill_plan(2304, 6000, chunk=2048, boundary=256)
    assert [pos for pos, _, _ in plan] == [2304, 4096, 5888]
    assert [n for _, n, _ in plan] == [1792, 1792, 112]


def test_env_default_is_2048(monkeypatch):
    monkeypatch.delenv("MAXTOKEN_MLX_OFFLOAD_PREFILL_CHUNK", raising=False)
    assert [n for _, n, _ in offload_prefill_plan(0, 5000)][0] == 2048
    monkeypatch.setenv("MAXTOKEN_MLX_OFFLOAD_PREFILL_CHUNK", "1024")
    assert [n for _, n, _ in offload_prefill_plan(0, 5000)][0] == 1024
