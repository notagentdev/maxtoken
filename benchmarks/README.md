# benchmarks

Run from the repo root with `PYTHONPATH=python:.`. Each script's `--help` /
docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `mt serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

`mlx` serves fully resident; `mlx-offload` serves the experts from the slot
cache (sized with `--cache` / `--cache-rate`, else auto):

```bash
python benchmarks/bench_decode_moe.py --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --backend mlx
python benchmarks/bench_decode_moe.py --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --backend mlx-offload --cache-rate 0.6
```
