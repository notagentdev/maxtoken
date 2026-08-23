# macOS / Apple silicon (MLX backend)

On Apple-silicon Macs, FreeToken serves models through [MLX](https://github.com/ml-explore/mlx)
instead of the CUDA engine. The API server, tokenizer workers, terminal shell and both
client APIs (OpenAI- and Anthropic-compatible, including streaming, stop sequences and
usage accounting) are identical between the two backends — only the scheduler process
differs: on macOS it executes models via [mlx-lm](https://github.com/ml-explore/mlx-lm)
on the Metal GPU.

The backend has three serving modes:

- **resident** (default): the whole model lives in unified memory, plain mlx-lm
  execution.
- **zero-copy mapped experts** (`--moe-backend offload`, the offload default):
  the switch-GLU expert tensors are repacked once into a page-aligned store
  (FTW-MLX, the Apple-silicon analogue of the CUDA engine's FTW format) and
  memory-mapped straight into MLX via DLPack — the GPU reads the file-backed
  pages through unified memory, the same trick llama.cpp's Metal backend uses.
  Serving is a plain `gather_qmm` over the mapped store: **resident-kernel
  speed**, zero copies, no cache management. Residency is OS-managed: hot
  experts live in the page cache, cold ones fault in from SSD once, and under
  memory pressure clean pages are evicted (never swapped) — graceful
  degradation instead of OOM. The repack lives under
  `~/.cache/freetoken/mlx-ftw/` (`FREETOKEN_MLX_FTW_DIR` overrides) and costs
  one streaming copy of the expert weights on first serve.
- **expert slot cache** (`--moe-backend offload` plus an explicit
  `--moe-cache-size`/`--moe-cache-rate`): FreeToken's core idea on Apple
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

To serve a MoE model **larger than the memory you want to commit to it**, enable
expert offload (this is what FreeToken is for):

```bash
# Zero-copy mapped store (default): resident-speed serving, experts live in
# reclaimable page cache instead of allocated memory:
ft serve --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --moe-backend offload

# Hard memory budget via the expert slot cache (~12 GiB total instead of ~18):
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
| Ornith-1.5-35B-A3B | resident | 66.6 | 205 ms | 18.3 GiB allocated |
| Ornith-1.5-35B-A3B | **offload, mapped (default)** | **67.8** | **209 ms** | ~18 GiB borrowed² (1.3 GiB owned) |
| Ornith-1.5-35B-A3B | offload, slot cache 60% | 15.1 | 6.9 s | **11.9 GiB** hard budget |
| Ornith-1.5-35B-A3B | offload, slot cache 35% | 8.9 | 21 s¹ | **7.7 GiB** hard budget |

¹ cold page cache (first pass over the weights); warm prefill streams at
SSD/page-cache speed.
² no free lunch: at full speed the expert weights occupy RAM in the mapped mode
too (that is why it is fast). The difference is the KIND of memory — the store
is clean file-backed page cache the OS can *drop* under pressure and re-fault
from SSD later, while the resident row's 18.3 GiB are dirty allocations that
have to be *written to swap* first. Only ~1.3 GiB (dense weights + KV) is
owned, non-reclaimable memory. `mx.get_active_memory` reports the mapped bytes
as well, so `/v1/stats` shows ~18 GiB either way.

Measured under memory pressure (a competing process holding 10 GiB of dirty
memory on the 32 GiB machine): with a *passive* competitor both modes recover
to full speed after one slow request (the OS pages the idle competitor out).
With a competitor *actively using* its 10 GiB, both modes stall — the working
sets genuinely don't fit and the SSD becomes the bottleneck for everyone; the
mapped store is not magic against that. Its demonstrated advantages are on the
edges: the serving process survives even an extreme storm (14 GiB hot
aggressor) without crashing, returns to full speed within seconds of the
pressure ending (13.7 -> 64.2 tok/s across two requests, no restart), and its
17 GiB of experts never cause swap *writes* — eviction is free, refault is a
read.

The offload rows are the point: the same 18 GiB checkpoint at full speed with
OS-elastic residency (mapped), or inside a chosen hard budget (slot cache),
on a 32 GiB machine the resident mode nearly fills.

The reported `vram` figure is `mx.get_active_memory()` from the serving process —
live Metal allocations, surfaced through the same `/v1/stats` field the CUDA engine
uses.

## Behavior and limitations vs. the CUDA engine

- **Continuous batching** (resident and mapped-expert serving): concurrent
  requests decode in one batched forward per step, with new prompts joining
  mid-flight (mlx-lm's `BatchGenerator`; per-request sampling params, stop
  sequences, aborts and prefix-cache donation all work per row). Measured on
  Ornith-1.5-35B (mapped): aggregate decode 62 → 94 → 116 tok/s at batch
  1 → 2 → 4 (engine-level); through the full HTTP server 58 → 47 → 74 →
  99 tok/s at 1 → 2 → 4 → 8 concurrent streams — the server path currently
  adds per-round overhead at batch ≥ 2 (~40 ms vs 24 ms per round in-process;
  under investigation, `FREETOKEN_MLX_TRACE=1` logs per-round timings).
  Greedy requests keep the batched argmax fast path even when sampling
  defaults fill in top-p/top-k. The slot-cache offload path remains
  round-robin (its speculate/verify loop is per-request).
- Tensor parallelism (`--tp-size > 1`) is rejected — MLX uses the unified memory of
  one chip.
- Runtime cache rebuilds (`/v1/cache/rebuild`) resize the expert slot cache live
  (`moe_cache_size` only); KV is per-request on MLX, so there is no page pool to
  resize.
- **Prefix cache** (on by default; `--cache-type naive` disables): generation
  caches are snapshotted at 256-token boundaries during prefill and at request
  end, and a new request restores the longest token-prefix match, prefilling
  only the remainder. This includes **hybrid models** (GDN/SSM + attention),
  whose recurrent state cannot be trimmed to arbitrary positions and for which
  prefix reuse is broken upstream (mlx-lm#980) — boundary snapshots restore
  exactly at a snapshot instead. Restoration is bit-identical to having kept
  the original cache alive (a cold recompute can differ in bf16 rounding —
  inherent to chunked prefill, as in every serving engine's prefix cache).
  Snapshots are copy-on-write references, budgeted at 15% of unified memory
  (`FREETOKEN_MLX_PREFIX_CACHE_MB` overrides), LRU-evicted. Report the reuse
  per request with `--enable-cache-report` (usage `cached_tokens`). Measured
  on Ornith-1.5-35B with a 2.4k-token system prompt: first request 19.5 s,
  follow-ups **1.0 s** (`cached_tokens=2304`).
- Remaining CUDA-specific flags (`--attention-backend`, `--cuda-graph-*`,
  `--num-pages`, …) are accepted but ignored by the MLX scheduler;
  `--moe-backend offload` and the `--moe-cache-*` sizing flags are honored.
- Offload decode cost is bounded by miss *density*, not I/O bandwidth: every miss
  needs a CPU-side routing decision (file reads cannot be issued from the GPU
  graph), so a fine-grained MoE routing ~25+ fresh experts per token pays either
  per-layer syncs or speculative re-runs. `FREETOKEN_MLX_ADMIT_FILTER=1` enables
  an experimental admission filter (inline-serve first-offense misses) that helps
  small expert pools with tight caches and hurts long-tail pools — measure before
  keeping it on.

## Tests on macOS

The shared frontend suites run and pass on macOS:

```bash
pytest tests/mlx_backend tests/server tests/tokenizer tests/daemon -m "not slow"
```

The engine-internal suites (`tests/engine`, `tests/kernels`, `tests/moe`,
`tests/kvcache`, `tests/scheduler`, `tests/models`, `tests/dsv4`) exercise the CUDA
engine itself and need its native deps (flashlib etc.); they are Linux-only.
