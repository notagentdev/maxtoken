# macOS / Apple silicon (MLX backend)

On Apple-silicon Macs, FreeToken serves models through [MLX](https://github.com/ml-explore/mlx)
instead of the CUDA engine. The API server, tokenizer workers, terminal shell and both
client APIs (OpenAI- and Anthropic-compatible, including streaming, stop sequences and
usage accounting) are identical between the two backends — only the scheduler process
differs: on macOS it executes models via [mlx-lm](https://github.com/ml-explore/mlx-lm)
on the Metal GPU.

The backend has two serving modes:

- **resident** (default): the whole model lives in unified memory, plain mlx-lm
  execution.
- **expert offload** (`--moe-backend offload`): FreeToken's core idea on Apple
  silicon. Only the dense weights stay resident; the MoE experts are served from a
  per-layer LRU **slot cache** (`mx.gather_qmm` over the slots — the same kernel as
  resident serving, validated bit-identical), with misses fetched from the
  checkpoint's safetensors by direct byte-range reads. Prefill streams full layers
  (double-buffered, prefetched) and admits each chunk's hottest experts into the
  cache device-side. Decode is *speculate-and-verify*: steps run fully lazily
  against a device-side slot LUT with **zero CPU syncs**; one per-token check
  validates that every routed expert was resident, and a miss rolls the KV/GDN
  caches back one step, installs the experts and re-runs. An adaptive controller
  falls back to per-layer synchronous serving when redos get too frequent.
  Greedy outputs are token-identical to the resident model in both modes.

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
memory.

To serve a MoE model **larger than the memory you want to give it**, enable the
expert cache (this is what FreeToken is for):

```bash
# ~12 GiB total instead of ~18 GiB resident (Ornith-1.5 is a 35B-A3B MoE):
ft serve --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
    --moe-backend offload --moe-cache-rate 0.6
```

`--moe-cache-size N` (total expert slots), `--moe-cache-rate R` (fraction of each
layer's experts) or `--moe-cache-auto` (fill `--memory-ratio` of unified memory)
size the cache. It can be resized at runtime, without a restart or reload
(elastic memory management):

```bash
curl -X POST localhost:1919/v1/cache/rebuild -H 'Content-Type: application/json' \
     -d '{"moe_cache_size": 4096}'
```

Then talk to it exactly as on Linux:

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

| model (4-bit) | mode | decode tok/s | TTFT (warm) | memory |
|---|---|---|---|---|
| OLMoE-1B-7B-Instruct | resident | 214.7 | 73 ms | 3.7 GiB |
| Qwen3-30B-A3B-Instruct-2507 | resident | 63.9 | 268 ms | 16.1 GiB |
| Ornith-1.5-35B-A3B | resident | 66.6 | 205 ms | 18.3 GiB |
| Ornith-1.5-35B-A3B | offload, cache 60% | 15.1 | 6.9 s | **11.9 GiB** |
| Ornith-1.5-35B-A3B | offload, cache 35% | 9.1 | 41.8 s¹ | **7.7 GiB** |

¹ cold page cache (first pass over the weights right after download); warm prefill
streams at SSD/page-cache speed, see the 60% row.

The offload rows are the point: the same 18 GiB checkpoint serving inside a
choose-your-own memory budget on a 32 GiB machine that the resident row nearly
fills — the cache size dials memory against decode speed.

The reported `vram` figure is `mx.get_active_memory()` from the serving process —
live Metal allocations, surfaced through the same `/v1/stats` field the CUDA engine
uses.

## Behavior and limitations vs. the CUDA engine

- Concurrent requests are stepped round-robin (one token each per turn), so several
  streams progress together; each request owns its KV cache.
- Tensor parallelism (`--tp-size > 1`) is rejected — MLX uses the unified memory of
  one chip.
- Runtime cache rebuilds (`/v1/cache/rebuild`) resize the expert slot cache live
  (`moe_cache_size` only); KV is per-request on MLX, so there is no page pool to
  resize.
- No prefix cache yet: `cached_tokens` is always 0.
- Remaining CUDA-specific flags (`--attention-backend`, `--cuda-graph-*`,
  `--num-pages`, …) are accepted but ignored by the MLX scheduler;
  `--moe-backend offload` and the `--moe-cache-*` sizing flags are honored.

## Tests on macOS

The shared frontend suites run and pass on macOS:

```bash
pytest tests/mlx_backend tests/server tests/tokenizer tests/daemon -m "not slow"
```

The engine-internal suites (`tests/engine`, `tests/kernels`, `tests/moe`,
`tests/kvcache`, `tests/scheduler`, `tests/models`, `tests/dsv4`) exercise the CUDA
engine itself and need its native deps (flashlib etc.); they are Linux-only.
