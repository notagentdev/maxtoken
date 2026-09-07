# Supported models

MaxToken loads MLX checkpoints — the quantized safetensors conversions that
mlx-lm serves — and everything mlx-lm's model registry knows is a candidate.
The checkpoints below are the ones this repository was measured on (a 32 GB
M1 Max, through the real HTTP path; numbers in the README and in
[mlx.md](mlx.md)):

| Model | Checkpoint | Served as |
|---|---|---|
| Ornith-1.5-35B-A3B (Qwen3.5-family hybrid MoE) | [ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit) | mapped store; native MTP head as drafter |
| Qwen3-Coder-Next-80B-A3B | [mlx-community/Qwen3-Coder-Next-4bit](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit) | slot cache (42 GiB checkpoint on 32 GB) |
| DeepSeek-V4-Flash (304 B, 2-bit) | [mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit) | slot cache; needs `MAXTOKEN_MLX_PREIMPORT=optiq` |
| Qwen3-30B-A3B | [mlx-community/Qwen3-30B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit) | resident / mapped |
| Qwen3.8-27B (dense hybrid) | a 4-bit conversion that ships `mtp.safetensors` | resident; `--draft-model mtp` |
| OLMoE-1B-7B | [mlx-community/OLMoE-1B-7B-0125-Instruct-4bit](https://huggingface.co/mlx-community/OLMoE-1B-7B-0125-Instruct-4bit) | test model (small, fast) |

Other checkpoints of the same architectures (`qwen3_moe`, `qwen3_next`,
`qwen3_5_moe`, `olmoe`, …) work the same way; a per-expert or stacked expert
layout is detected from the weight names. An architecture mlx-lm does not
register (for example the `qwen4_exp` "Niwaki" family, which needs mlx-vlm
and a custom loader) does not load.

## Serving modes

`mt serve --model <ckpt> [--moe-backend offload] [--moe-cache-*]`:

- **resident** (default) — the whole model in unified memory, plain mlx-lm.
- **mapped** (`--moe-backend offload`, model fits) — the dense core is
  resident, the routed experts are a zero-copy memory-mapped store whose
  residency the OS manages: resident-kernel speed, elastic memory.
- **slot cache** (`--moe-backend offload` with `--moe-cache-size`, `-rate`
  or `-auto`) — a hard budget of expert slots in memory, misses read from
  the SSD; the only way to serve a checkpoint bigger than the machine.

[mlx.md](mlx.md) has the measurements for each mode and the honest negative
results.

## Notes

- Multimodal checkpoints are served text-only.
- Thinking models must not be greedy-decoded (see mlx.md, "Sampling"): the
  server fills sampling defaults from the checkpoint's `generation_config.json`.
