# Changelog

All notable changes to MaxToken. The project forked from
[FreeToken](https://github.com/FlashML-org/FreeToken) on 2026-08-23; the
history before that is upstream's. `0.0.1` was the fork's working version
before its first tagged release.

## [0.2.0] - 2026-09-07

First public release of MaxToken: an Apple-silicon-only MoE serving engine.
Every number below was measured through the HTTP serving path on a 32 GB
M1 Max; the methodology and the negative results are in
[docs/mlx.md](docs/mlx.md).

### Changed (breaking)

- The project is MaxToken. The CLI is `mt` (`maxtoken` alias), the
  distribution and import package `maxtoken`; `FREETOKEN_*` environment
  variables still configure their `MAXTOKEN_*` successors.
- The CUDA engine is gone. macOS on Apple silicon via MLX is the only
  backend; installing pulls no torch, triton or flashlib. Endpoints of
  removed commands and the `/v1` aliases of our own endpoints were dropped.

### Added

- **MLX execution backend** via mlx-lm: API server, tokenizer workers,
  terminal shell, OpenAI- and Anthropic-compatible APIs (chat completions,
  Responses, `/v1/messages`, `count_tokens`), streaming, stop sequences,
  usage accounting.
- **Expert offload** — models bigger than the machine: a per-layer LRU slot
  cache under a hard budget, misses read from the checkpoint by direct
  byte-range reads; speculate-and-verify decode with zero CPU syncs;
  streamed prefill with hottest-expert admission; a transient expert bank
  for short chunks; cross-layer read-ahead; miss-pressure slot rebalancing;
  live resize (`/admin/cache/rebuild`, console slider). Qwen3-Coder-Next-80B
  (42 GiB) serves at 7.4–10 tok/s inside a 10 GiB budget and ~5 tok/s inside
  3.4 GiB; DeepSeek-V4-Flash 2-bit (86 GiB checkpoint, 304 B parameters) at
  ~3.3 tok/s inside 13 GiB.
- **FTW-MLX zero-copy mapped expert store** (v2 writes gate and up
  interleaved per expert): file-backed experts read by the GPU through
  unified memory at resident-kernel speed with OS-elastic residency.
  Ornith-1.5-35B-A3B at 75–77 tok/s with ~1.3 GiB owned memory; idle
  re-warming after memory pressure evicts the store.
- **Prefix cache** across requests, hybrid (GDN/SSM) aware, with an SSD tier
  that survives restarts: a 9k-token agent prompt restores in ~0.5 s.
- **Continuous batching** on the resident and mapped paths.
- **Speculative decoding**: `--draft-model` with a second model, or the
  checkpoint's own multi-token-prediction head (`--draft-model mtp`, or a
  sibling artifact's `mtp.safetensors`); distribution-exact rejection
  sampling with residual correction; a Metal kernel for the 2–4-row verify
  matmul; whole-MoE verify kernels (vendored from MTPLX, Apache-2.0) for
  1–3-row windows. Ornith-1.5-35B-A3B 89–94 tok/s; Qwen3.8-27B 32–35 tok/s
  with prefill at 95–99 tok/s.
- **Prefill**: fp16 GEMMs for the hidden-size projections, paced wide
  chunks, decode rounds interleaved between chunks, SSE keepalives on long
  prompts.
- **Web console** at `/`: chat with per-response TTFT, tok/s and
  `cached_tokens`; live throughput and request log; cache slider; engine
  reload; runtime sliders for the context window, the reasoning budget and
  the sampling defaults. Also: LM-Studio-style model listing so agents learn
  the real context window, `GET /v1/models/{id}`, `--enable-cache-report`,
  a reasoning token budget, the `developer` role.
- Serving of architectures mlx-lm does not register
  (`MAXTOKEN_MLX_PREIMPORT`), and tolerance of unfamiliar MoE routers.

### Fixed

- The slot-cache prefill streamed every expert layer once per 256 tokens —
  a 1710-token prompt on the 80B took 55.9 s to the first token in seven
  passes; it takes 21.0 s in one.
- Slot rebalancing silently dropped the overflow of layers capped at their
  expert count.
- Widening Metal command buffers without a paced prefill panicked the
  machine; long prompts no longer starve other requests or the system
  (bounded buffer cache, pacing, yields between chunks).
- Sampled speculative requests were greedy in disguise (a scatter through a
  negative-stride view); prefill pacing fired on verify windows; the last
  prompt token was fed to the trunk twice; a window-cache rollback restored
  the wrong state.
- A failed batched forward, an eviction, or a prompt over the context length
  no longer takes the server down.

### Documented

- Start the server from a regular shell: a background-band launcher's
  macOS QoS clamp is inherited by the Metal threads, cannot be dropped from
  inside, and halves batched decode.
- What was measured and lost, so nobody repeats it: madvise read-ahead,
  admission filtering on long-tail expert pools, deeper draft windows,
  Split-K verify, quantizing the MTP head, mmap slices for small expert
  parts, snapshotting the MTP head's history across prefix hits.

### Removed

- Upstream's Linux/CUDA installer, wheel CI, kernel-cache package and
  community assets.

[0.2.0]: https://github.com/notagentdev/maxtoken/releases/tag/v0.2.0
