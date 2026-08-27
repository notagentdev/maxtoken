# macOS / Apple silicon (MLX backend)

MaxToken serves models through [MLX](https://github.com/ml-explore/mlx) on the Metal
GPU, via [mlx-lm](https://github.com/ml-explore/mlx-lm). The upstream CUDA engine was
removed in 0.0.1 (untestable on this platform, and it forced a torch dependency on
every install), so this is the only backend: API server, tokenizer workers, terminal
shell and both client APIs (OpenAI- and Anthropic-compatible, including streaming,
stop sequences and usage accounting) all sit on top of it.

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
  `~/.cache/maxtoken/mlx-ftw/` (`MAXTOKEN_MLX_FTW_DIR` overrides) and costs
  one streaming copy of the expert weights on first serve.
  `MAXTOKEN_MLX_MLOCK=1` pins the whole store into memory (use only when it
  fits with headroom): the first prefill starts fully warm (measured 2× faster
  server warm-up on Ornith-35B, 11 s → 5.2 s) and expert pages can never be
  evicted under memory pressure — at the price of the store's elasticity.
  `MAXTOKEN_MLX_PREFETCH=1` (experimental madvise read-ahead before big
  prefills, inspired by llama.cpp's second-stream expert uploads) measured
  net-negative on this hardware and stays off by default.
- **expert slot cache** (`--moe-backend offload` plus an explicit
  `--moe-cache-size`/`--moe-cache-rate`): MaxToken's core idea on Apple
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
pip install -e .
```

Pure Python: MLX ships its own Metal kernels, so nothing is compiled and no CUDA
ecosystem (torch, triton, flashlib) is installed.

## Web console

`mt serve` ships a built-in GUI at **`http://localhost:1919/`** (any platform,
not just macOS): live throughput/usage/cost cards, model status, streaming chat
with TTFT / tok/s / `cached_tokens` per response, a request log, and the elastic
MoE expert-cache slider (slot-cache mode) that applies `/admin/cache/rebuild` live.
It is a single self-contained HTML file served from the same origin as the APIs.

## Serve

```bash
mt serve --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit
```

`--backend auto` (the default) resolves to `mlx` on macOS, so no extra flag is needed;
pass `--backend mlx` to be explicit. Any model in the mlx-lm model zoo works — for MoE
models pick a quantized `mlx-community/...-4bit` checkpoint that fits your unified
memory.

To serve a MoE model **larger than the memory you want to commit to it**, enable
expert offload (this is what MaxToken is for):

```bash
# Zero-copy mapped store (default): resident-speed serving, experts live in
# reclaimable page cache instead of allocated memory:
mt serve --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --moe-backend offload

# Hard memory budget via the expert slot cache (~12 GiB total instead of ~18):
mt serve --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
    --moe-backend offload --moe-cache-rate 0.6
```

`--moe-cache-size N` (total expert slots), `--moe-cache-rate R` (fraction of each
layer's experts) or `--moe-cache-auto` (fill `--memory-ratio` of unified memory)
size the cache. It can be resized at runtime, without a restart or reload
(elastic memory management):

```bash
curl -X POST localhost:1919/admin/cache/rebuild -H 'Content-Type: application/json' \
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
real serving path (spawned `mt serve`, streamed `/v1/chat/completions`, SSE arrival
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
| DeepSeek-V4-Flash 2-bit⁴ | offload, slot cache 20% | 3.9 | 7.3 s | **20.5 GiB** hard budget |
| DeepSeek-V4-Flash 2-bit⁴ | offload, slot cache 10% | 3.3 | 6.5 s | **13.1 GiB** hard budget |

¹ cold page cache (first pass over the weights); warm prefill streams at
SSD/page-cache speed.
³ a **42 GiB checkpoint on the 32 GiB machine** — the model genuinely does not
fit, which is the case the slot cache exists for (Qwen3-Next: 48 layers ×
512 experts × top-10; 40.5 GiB of experts over a 1.3 GiB dense core). Decode
scales with the budget: 5.3 / 6.4 / 8.4 / 8.6 tok/s engine-level at 5 / 10 /
20 / 30% cache (3.4 / 5.5 / 10.3 / 13.5 GiB active) — diminishing returns
past 20%; through the HTTP server 7.6 tok/s at 20%. Long prefill chunks
stream every expert layer (a full 40 GiB pass — a multi-second TTFT floor
regardless of prompt length; the SSD covers 42 GiB in ~12 s), but chunks of
≤ 32 tokens (`MAXTOKEN_MLX_BANK_TOKENS`) — the short rest-prompt after a
prefix-cache restore — are served from a transient bank of only the routed
non-resident experts: a chat follow-up's TTFT drops from ~7 s to ~2 s and
now scales with the remainder, not with the model. The 32-token gate is
measured, not guessed: novel text densifies fast (64 fresh tokens already
route ~350 of 512 experts/layer), and a sweep over delta sizes 64–2048
shows the full-layer stream beating the bank at EVERY size from 64 up
(0.76x at 64, 0.2–0.3x at 256+) — so mid-size agent deltas (a pasted file,
a tool result) correctly stay on the streamed path at ~5–10 s per 2k
tokens, and raising the gate would make them slower, not faster. On that path a
cross-layer read-ahead additionally overlaps the next layer's fetches with
the current layer's compute (layer L+1's gate scores layer L's hidden,
re-based to L+1's RMSNorm — recall 0.946 measured; `MAXTOKEN_MLX_XLAYER=0`
disables): banked TTFT −23%. The same machinery measured *negative* on the
single-token decode path (the ~1 ms compute window cannot hide what the
extra python/graph breaks cost), so decode deliberately stays clean.
Between requests the scheduler also rebalances the slot budget by observed
per-layer miss pressure (`MAXTOKEN_MLX_REBALANCE=0` disables) — layers
differ widely in routing diversity, and an even split starves the diverse
ones.

⁴ **304 B parameters, an 86 GiB checkpoint — 2.7x this machine's RAM.** The
mixed 2-bit quant is `mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`
(experts 2-bit, attention/embeddings 6–8 bit). Two things are needed beyond
the usual: `MAXTOKEN_MLX_PREIMPORT=optiq` (mlx-lm ships no `deepseek_v4`;
the `mlx-optiq` package registers it on import), and a load path that does
not build-then-quantize — stock `mlx_lm.load` materializes a transient near
the FULL model size before `load_weights` overwrites it, which OS-kills the
process at this scale. Decode scales with the budget (3.3 -> 3.9 tok/s at
10% -> 20%, miss rate 48.7% -> 37.6%) and stops there: 30% needs 27.9 GiB
and exceeds Metal's ~26.8 GiB working-set limit. For comparison, the quant's
own publisher documents ~2.5 tok/s for their (cache-less) SSD streaming of
the same weights on an M3 Max. Cross-layer read-ahead disables itself on
this model — DeepSeek-V4 routes by hashing token ids, so its gate cannot be
scored from the hidden state alone.

² no free lunch: at full speed the expert weights occupy RAM in the mapped mode
too (that is why it is fast). The difference is the KIND of memory — the store
is clean file-backed page cache the OS can *drop* under pressure and re-fault
from SSD later, while the resident row's 18.3 GiB are dirty allocations that
have to be *written to swap* first. Only ~1.3 GiB (dense weights + KV) is
owned, non-reclaimable memory. `mx.get_active_memory` reports the mapped bytes
as well, so `/admin/stats` shows ~18 GiB either way.

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
live Metal allocations, surfaced through the same `/admin/stats` field the CUDA engine
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
  factor is still open. Diagnosis knobs: `MAXTOKEN_MLX_TRACE=1` (per-round
  timings), `MAXTOKEN_MLX_PROFILE=<path>` (cProfile of the scheduler loop).
  Greedy requests keep the batched argmax fast path even when sampling
  defaults fill in top-p/top-k. The slot-cache offload path remains
  round-robin (its speculate/verify loop is per-request).
- Tensor parallelism (`--tp-size > 1`) is rejected — MLX uses the unified memory of
  one chip.
- Runtime cache rebuilds (`/admin/cache/rebuild`) resize the expert slot cache live
  (`moe_cache_size`) and/or move the context-window ceiling (`max_seq_len`,
  clamped to [1024, model max]) — KV is per-request on MLX, so the ceiling IS
  this backend's capacity knob: admission, generation caps and the
  `context_length` published by `/v1/models` follow it immediately, and the
  console exposes it as a slider next to the expert-cache one. (The CUDA
  engine's runtime knob is the KV page pool instead; it ignores
  `max_seq_len`.)
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
  (`MAXTOKEN_MLX_PREFIX_CACHE_MB` overrides), LRU-evicted. Report the reuse
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
  SSD); measure before keeping it on. It also applies to resident and mapped
  serving (where it excludes continuous batching, being per-request), but
  measured NEGATIVE in every regime this machine could test:

  | target | drafter | baseline | speculative |
  |---|---|---|---|
  | Qwen3-Coder-Next-80B, slot cache | Qwen3-0.6B | 7.4 tok/s | 6.4 (fetch-bound) |
  | Qwen3.8-27B dense, resident | Qwen3.5-4B | 17.3 tok/s | 4.7 (drafter too costly) |
  | Ornith-35B-A3B, mapped | — | 67 tok/s | impossible: no compatible drafter is
  cheaper than a 3B active path |

  `--draft-model mtp` uses the checkpoint's OWN multi-token-prediction head
  instead of a second model, where one ships (`mtp.safetensors` /
  `model-mtp-head.safetensors`). That fixes everything the two-model variant
  gets wrong about cost and compatibility: the head is ONE layer (228 MB on
  Qwen3.8-27B, ~10x smaller than the smallest compatible standalone drafter),
  it shares the trunk's tokenizer and `lm_head` by construction, and it is
  trained on this exact model — measured 2.10 accepted tokens per verify at
  k=3 while costing 13% of the loop, against 1.80 for the 4B foreign drafter
  that cost most of it. Wiring follows the checkpoint's own `mtplx_runtime.json`
  contract: post-norm trunk hidden, post-norm chained hidden, concat order
  [embedding, hidden]. The state that produced the last committed token sits at
  index `accepted` of the verify window, NOT at its end — feeding the last one
  drifts the head off the committed path (acceptance 0.25 vs 1.10 out of 3).

  **It still does not pay on a hybrid model, and now the reason is exact.**
  A depth sweep on Qwen3.8-27B-MTPLX (sampled, temp 0.3/0.9/40, fixed seed):

  | | tok/s | vs plain | accepted/verify |
  |---|---|---|---|
  | plain decode | **15.7** | — | — |
  | MTP k=1 | 11.3 | 0.72x | 1.52 |
  | MTP k=3 | 6.7 | 0.42x | 1.80 |
  | MTP k=5 | 3.6 | 0.23x | 1.71 |
  | MTP k=6 | 4.1 | 0.26x | 1.92 |

  Depth does not rescue it: acceptance saturates near 1.9 while the draft cost
  grows linearly with k, so every step past k=1 buys drafts that are mostly
  rejected. And k=1 — the cheapest speculation possible, one head forward and
  one two-token window — still loses 28%. Per round it costs ~135 ms against
  ~64 ms for a plain step, i.e. a whole extra forward's worth of overhead, and
  1.5 accepted tokens cannot pay for it.

  Profiling the round names the cost exactly, and it is NOT the rollback
  (snapshotting all 48 recurrent states measures 0.6 ms) nor the drafter
  (3.9 ms). It is the verify forward: **91 ms for a 2-token window against
  54 ms for a 1-token step.** A bandwidth-bound decode should barely notice
  a second token; this one charges ~38 ms for it, and keeps charging
  linearly (3 tokens 130 ms, 4 tokens 168 ms).

  It is NOT the gated-delta kernel, though this file claimed so twice. That
  kernel does walk time serially, which makes the story tempting, but timing it
  alone at the model's real shapes settles it: T=1 costs 0.263 ms and T=2
  costs 0.295 ms, so across all 48 recurrent layers the serial recursion adds
  **1.5 ms of the 36.3 ms** — four percent. Splitting the forward by layer type
  finds the cost spread evenly instead (GatedDeltaNet +0.47 ms x48 = 22.5 ms,
  full attention +0.57 ms x16 = 9.2 ms), which is the signature of something
  every layer pays, not of one kernel.

  That something is **MLX's quantized matmul**. The same linear, same shape,
  same machine:

  | 5120 -> 5120 | T=1 | T=2 |
  |---|---|---|
  | bfloat16 | 0.534 ms | 0.535 ms (**+0.001**) |
  | 4-bit affine | 0.325 ms | 0.450 ms (**+38%**) |

  Unquantized, the second token is free — exactly right for a matmul whose
  weights were already streamed. Quantized, it costs 38% more, and a chain of
  32 real MLP-shaped projections (evaluated once, so no per-call sync inflates
  it) shows why: T=1 runs at 207 GiB/s, near this machine's ceiling, while T=2
  drops to 145 GiB/s and T=4 to 78 GiB/s. Cost grows roughly linearly in T
  rather than reusing the weight read. MLX 0.32.2 behaves identically, and
  feeding the window as a batch `(T,1,H)` instead of a sequence `(1,T,H)`
  changes nothing — the same kernel, the same price.

  So the honest break-even is `56 + 38k` ms per round, i.e. k=1 needs
  **1.68 accepted tokens**. Greedy at k=3 accepts 2.10, which does NOT win once
  the window costs `56 + 3*38 = 170` ms. Sampled k=1 accepted 1.52 under
  sample-and-match, where a draft survives just when the target's own sample
  happens to equal it.

  So we replaced that rule with proper rejection sampling: the drafter samples
  from its own distribution and records `q`, a draft is accepted with
  probability `min(1, p/q)`, and on rejection the token is drawn from the
  normalized residual `(p-q)+`. It is distribution-exact (statistically tested
  in `tests/mlx_backend/test_speculative_accept.py`) and theoretically the best
  rule available — and it still does not pay:

  | | accepted/verify | tok/s | vs plain |
  |---|---|---|---|
  | plain decode | — | **15.1** | — |
  | rejection k=1 | 1.55 | 11.6 | 0.77x |
  | rejection k=2 | 1.68 | 7.9 | 0.52x |
  | rejection k=3 | 1.75 | 6.4 | 0.43x |

  1.52 to 1.55 against the 1.68 needed. Acceptance under rejection sampling is
  exactly `1 - TV(p, q)`, a property of the head rather than of the code around
  it, so no acceptance rule can invent overlap that is not there.

  **What settles it is that another engine wins on this exact setup.** mtplx
  2.9.1, same checkpoint (verified byte-identical), same machine, same prompt
  and sampling, measured through its OpenAI endpoint:

  | | tok/s | tokens per step |
  |---|---|---|
  | mtplx, MTP off | 17.18 | 1.03 |
  | mtplx, MTP on (turbo) | **26.60** | 2.04 |
  | MaxToken, plain decode | 17.77 | 1.00 |
  | MaxToken, MTP k=1 | 11.58 | 1.52 |

  (Counting SSE chunks would have read mtplx as 12.8 tok/s — a speculative
  decoder emits several tokens per chunk. Take the token count from
  `stream_options.include_usage`, never from the chunk count.)

  Our plain decode is the faster of the two baselines, so the engine is fine;
  the speculation path is not. And their per-draft acceptance is *the same as
  ours*: 2.04 tokens per step at depth 2 is 1 + 0.52 + 0.52^2, i.e. ~52%,
  matching our 1.52 at k=1. They are not drafting better. They are paying
  ~7 ms per extra window token where we pay 38 — their multi-token quantized
  matmul is simply about five times more efficient than the one MLX ships.

  **So we built the kernel, and speculation now pays.** Two changes, both
  aimed at the round budget: a winning round may cost 1.52 x 56.2 = 85 ms, and
  it cost 143.

  *A small-M quantized matmul* (`verify_qmm.py`): dequantize each weight once
  and reuse it across every row of the window, one simdgroup per 4 output
  columns. The real 27B forward drops from 92.20 to 69.93 ms at T=2 — the
  extra window token from 36.6 ms to 14.3 — which moves the k=1 break-even
  from 1.68 accepted tokens to 1.23, under the measured 1.52.

  *No replay forward*: a rejected round used to roll the caches back and
  recompute the committed prefix in a forward of its own, 56 ms to produce
  nothing new, on 48% of rounds. It is now carried into the NEXT window, where
  it costs one extra row. The carry is capped at `MAX_WINDOW` — a round with no
  room left runs the carry alone, which still commits a token and always
  absorbs the carry, so it cannot grow without bound.

  | | tok/s |
  |---|---|
  | plain decode | 17.79 |
  | MTP k=1, before | 10.59 |
  | MTP k=1, + verify kernel | 12.98 |
  | **MTP k=1, + no replay** | **17.97** |
  | MTP k=2 (deeper is worse) | 15.28 |

  The gain over plain decode is 1%, so this is a beginning and not a victory —
  but it is the first configuration where speculation is not a loss, and the
  remaining distance to mtplx's 26.60 is now a known quantity rather than a
  mystery. Their `vk_k` is a split-K morphology where ours does the whole K
  reduction per simdgroup, and ours runs at 194 GiB/s against stock qmv's 221,
  so the headroom is real and measurable. Depth stays at 1: k=2 rejects more
  often, and every rejection lengthens the carried window.


  The two-model pattern is structural in a different way. A drafter has to be
  roughly an order of magnitude cheaper than the target's *active* path, share
  its tokenizer, and keep its own KV cache in lockstep. Sparse MoE targets defeat the first condition (Ornith
  activates 3B — nothing compatible is cheaper), and a hybrid drafter pays
  snapshot/restore of its recurrent state every round. This is exactly the case
  for **built-in MTP heads** instead: one extra layer, sharing the target's
  vocabulary and cache by construction. Qwen3-Next-80B-A3B-Instruct ships one
  (shard 41 of the bf16 repo); the Coder derivative does not. The verify loop
  here is ready for such a drafter — only the adapter that queries an MTP head
  instead of a second model is missing. The verify window is auto-clamped so
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
- **Reasoning budget** (`--max-reasoning-tokens N`, or Anthropic's
  `thinking.budget_tokens` per request — the lower wins): stops a request once
  it has spent N tokens inside its thinking block, reporting
  `finish_reason: "length"`. Guards against a model looping in `<think>` until
  its whole output budget is gone and answering nothing (seen on research-grade
  MoEs). Tokens in the answer never count against it; unlimited by default.
  Adjustable at runtime without a restart — the console has a slider for it,
  or `POST /admin/cache/rebuild {"max_reasoning_tokens": N}` (0 = off). Unlike the
  other knobs there it needs no engine work: the budget is enforced in the
  frontend, so it applies to the next request immediately.
- Remaining CUDA-specific flags (`--attention-backend`, `--cuda-graph-*`,
  `--num-pages`, …) are accepted but ignored by the MLX scheduler;
  `--moe-backend offload` and the `--moe-cache-*` sizing flags are honored.
- Offload decode cost is bounded by miss *density*, not I/O bandwidth: every miss
  needs a CPU-side routing decision (file reads cannot be issued from the GPU
  graph), so a fine-grained MoE routing ~25+ fresh experts per token pays either
  per-layer syncs or speculative re-runs. `MAXTOKEN_MLX_ADMIT_FILTER=1` enables
  an experimental admission filter (inline-serve first-offense misses) that helps
  small expert pools with tight caches and hurts long-tail pools — measure before
  keeping it on.

## Sampling: do not greedy-decode a thinking model

A Qwen-family reasoning model at `temperature=0` is a known failure pattern:
greedy decoding makes reasoning *termination* deterministic-worst-case, so the
model loops inside its thinking block instead of closing it. Observed here
repeatedly — a research MoE degenerating into `Ich bin. Ich bin. …`, an MTP
probe producing `map map map map`, and empty `content` whenever the loop ate
the whole token budget.

Practical rules:

- **Thinking on:** sample. `temperature 0.6, top_p 0.95, top_k 20` is the
  Qwen-family default (`--sampling-defaults model`, the server's default, fills
  exactly this from the checkpoint's `generation_config.json`). Give it a
  generous `max_tokens` and cap runaway thinking with `--max-reasoning-tokens`
  rather than with a tight output budget.
- **Thinking off / short answers:** lower and narrower is fine, e.g.
  `temperature 0.3, top_p 0.9, top_k 40`.
- **Benchmarks:** greedy is the right choice for *comparing two code paths*
  (identical output proves equivalence), and every speculative-decoding
  measurement here uses it for that reason. It is the wrong choice for judging
  a model's quality or its termination behavior — those numbers are the
  worst case, not the typical one.

## Tests on macOS

The shared frontend suites run and pass on macOS:

```bash
pytest tests/mlx_backend tests/server tests/tokenizer tests/daemon -m "not slow"
```

The engine-internal suites (`tests/engine`, `tests/kernels`, `tests/moe`,
`tests/kvcache`, `tests/scheduler`, `tests/models`, `tests/dsv4`) exercise the CUDA
engine itself and need its native deps (flashlib etc.); they are Linux-only.
