# macOS / Apple silicon (MLX backend)

On Apple-silicon Macs, FreeToken serves models through [MLX](https://github.com/ml-explore/mlx)
instead of the CUDA engine. The API server, tokenizer workers, terminal shell and both
client APIs (OpenAI- and Anthropic-compatible, including streaming, stop sequences and
usage accounting) are identical between the two backends — only the scheduler process
differs: on macOS it executes models via [mlx-lm](https://github.com/ml-explore/mlx-lm)
on the Metal GPU.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx]"
```

The CUDA-only native dependencies (flashlib, apache-tvm-ffi, triton, the C++
extensions) are skipped automatically on Darwin.

## Serve

```bash
ft serve --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit
```

`--backend auto` (the default) resolves to `mlx` on macOS, so no extra flag is needed;
pass `--backend mlx` to be explicit. Any model in the mlx-lm model zoo works — for MoE
models pick a quantized `mlx-community/...-4bit` checkpoint that fits your unified
memory. Then talk to it exactly as on Linux:

```bash
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "OLMoE-1B-7B-0125-Instruct-4bit",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

## Benchmarks

`benchmarks/bench_decode_moe.py` supports `--backend mlx` and measures through the
real serving path (spawned `ft serve`, streamed `/v1/chat/completions`, SSE arrival
stamps). Reference numbers from an Apple-silicon Mac (32 GB unified memory), AIME-25
prompt, 256 decode tokens:

| model (4-bit) | decode tok/s | ms/token | TTFT (warm) | memory |
|---|---|---|---|---|
| OLMoE-1B-7B-Instruct | 214.7 | 4.66 | 73 ms | 3.7 GiB |
| Qwen3-30B-A3B-Instruct-2507 | 63.9 | 15.6 | 268 ms | 16.1 GiB |

The reported `vram` figure is `mx.get_active_memory()` from the serving process —
live Metal allocations, surfaced through the same `/v1/stats` field the CUDA engine
uses.

## Behavior and limitations vs. the CUDA engine

- Concurrent requests are stepped round-robin (one token each per turn), so several
  streams progress together; each request owns its KV cache.
- Tensor parallelism (`--tp-size > 1`) is rejected — MLX uses the unified memory of
  one chip.
- Runtime cache rebuilds (`/v1/cache/rebuild`) are reported as `failed`: MLX has no
  fixed KV/expert pools to resize.
- No prefix cache yet: `cached_tokens` is always 0.
- CUDA-specific flags (`--moe-cache-*`, `--attention-backend`, `--cuda-graph-*`,
  `--num-pages`, …) are accepted but ignored by the MLX scheduler.

## Tests on macOS

The shared frontend suites run and pass on macOS:

```bash
pytest tests/mlx_backend tests/server tests/tokenizer tests/daemon -m "not slow"
```

The engine-internal suites (`tests/engine`, `tests/kernels`, `tests/moe`,
`tests/kvcache`, `tests/scheduler`, `tests/models`, `tests/dsv4`) exercise the CUDA
engine itself and need its native deps (flashlib etc.); they are Linux-only.
