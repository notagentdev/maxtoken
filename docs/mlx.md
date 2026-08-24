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
  `FREETOKEN_MLX_MLOCK=1` pins the whole store into memory (use only when it
  fits with headroom): the first prefill starts fully warm (measured 2× faster
  server warm-up on Ornith-35B, 11 s → 5.2 s) and expert pages can never be
  evicted under memory pressure — at the price of the store's elasticity.
  `FREETOKEN_MLX_PREFETCH=1` (experimental madvise read-ahead before big
  prefills, inspired by llama.cpp's second-stream expert uploads) measured
  net-negative on this hardware and stays off by default.
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

## Web console

`ft serve` ships a built-in GUI at **`http://localhost:1919/`** (any platform,
not just macOS): live throughput/usage/cost cards, model status, streaming chat
with TTFT / tok/s / `cached_tokens` per response, a request log, and the elastic
MoE expert-cache slider (slot-cache mode) that applies `/v1/cache/rebuild` live.
It is a single self-contained HTML file served from the same origin as the APIs.

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
| Qwen3-Coder-Next-80B³ | offload, slot cache 20% | 8.4 | 5.1 s | **10.3 GiB** hard budget |
| Qwen3-Coder-Next-80B³ | offload, slot cache 5% | 5.3 | 6.5 s | **3.4 GiB** hard budget |

¹ cold page cache (first pass over the weights); warm prefill streams at
SSD/page-cache speed.
³ a **42 GiB checkpoint on the 32 GiB machine** — the model genuinely does not
fit, which is the case the slot cache exists for (Qwen3-Next: 48 layers ×
512 experts × top-10; 40.5 GiB of experts over a 1.3 GiB dense core). Decode
scales with the budget: 5.3 / 6.4 / 8.4 / 8.6 tok/s engine-level at 5 / 10 /
20 / 30% cache (3.4 / 5.5 / 10.3 / 13.5 GiB active) — diminishing returns
past 20%; through the HTTP server 7.6 tok/s at 20%. Prefill streams every
expert layer, so TTFT has a ~5 s floor regardless of prompt length (the full
40 GiB pass; the SSD covers 42 GiB in ~12 s).

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
  adds per-round overhead at batch ≥ 2 (~40 ms vs 24 ms per round in-process).
  An extensive investigation ruled out: spawn QoS (workers now self-promote to
  USER_INTERACTIVE anyway), ZMQ polling and reply IPC, the prefix store, disk
  I/O (0 MB/s during slow runs), GPU idle downclocking, CPU core contention
  with the frontend/detokenizer, and buffer-pool churn from mlx-lm's
  per-admission `mx.clear_cache` (worth ~10-15%, together with staggered
  admissions). Profiling places the remaining delta inside the mx evals
  themselves when the worker runs as part of the full server; the dominant
  factor is still open. Diagnosis knobs: `FREETOKEN_MLX_TRACE=1` (per-round
  timings), `FREETOKEN_MLX_PROFILE=<path>` (cProfile of the scheduler loop).
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
- **Speculative decoding** (`--draft-model`, slot-cache offload serving): a
  small same-vocabulary model (e.g. `mlx-community/Qwen3-0.6B-4bit` for Qwen3
  targets) drafts `--draft-tokens` (default 3) per step; the target verifies
  the window in one batched forward and commits the matched prefix plus one
  target token. Output is distribution-exact (greedy verified token-identical
  on Qwen3-Coder-Next-80B). Measured acceptance 2.4–2.7 tokens/verify — but on
  the 32 GiB test machine at 20% cache the net throughput is parity with plain
  decode (~6 tok/s both), not a win: offload decode is bound by expert
  *fetches*, whose volume scales with generated text and which speculation
  cannot reduce (it only amortizes the per-forward overhead, measured small:
  an all-hit 4-token window forward costs 60 ms vs 37 ms for one token). The
  flag is for machines/rates where fetches are cheaper (bigger cache, faster
  SSD); measure before keeping it on. The verify window is auto-clamped so
  `(k+1) × top_k` fits the per-layer slot budget. Lessons from the Vates
  project (studied at `../vates`): its 31–37 tok/s on the same model come from
  a model-trained MTP draft head (absent from the Qwen3-Coder-Next release —
  the checkpoint ships no `mtp.*` tensors), C++ async demand reads that hide
  the fetch latency, and non-uniform per-layer pool capacities; the latter two
  are the identified next levers here.
- The zero-copy **mapped** mode does not work for models larger than physical
  memory (e.g. the 42 GiB Qwen3-Coder-Next-80B on 32 GiB): Metal requires every
  buffer a command batch references to be residency-managed, so the first full
  forward aborts with an out-of-memory command-buffer error, and chunking the
  graph merely turns that into page-cache thrash (the expert sweep cycles
  40 GiB through ~26 GiB of cache with zero reuse). Beyond-memory models are
  what the expert slot cache (`--moe-cache-*`) is for.
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
