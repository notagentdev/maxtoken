<div align="center">

# MaxToken

**Serve MoE models bigger than your memory — on the Mac you already own.**

An independent fork of [FlashML's FreeToken](https://github.com/FlashML-org/FreeToken)
that turns the edge-native MoE serving engine into an Apple-silicon-first runtime:
maximum tokens out of minimum memory.

</div>

## What it does

MaxToken serves Mixture-of-Experts models whose weights do not fit the memory you
want to give them. On Apple silicon it executes via [MLX](https://github.com/ml-explore/mlx),
keeping only the dense core resident and serving the routed experts from an
elastic, hard-budgeted slot cache backed by direct SSD reads — or, when the model
fits, from a zero-copy memory-mapped store at resident-kernel speed.

Measured on a 32 GB M1 Max (all through the real HTTP serving path):

| model (4-bit) | memory | decode |
|---|---|---|
| Qwen3-Coder-Next-**80B** (42 GiB checkpoint) | **10 GiB** hard budget | ~10 tok/s |
| Qwen3-Coder-Next-**80B** | **3.4 GiB** hard budget | ~5 tok/s |
| Ornith-1.5-**35B**-A3B (18 GiB checkpoint) | 1.3 GiB owned + page cache | ~67 tok/s |

The 80B rows are the point: a checkpoint 1.3× the machine's total RAM, serving
usable tokens inside a quarter of its size — with follow-up TTFT of ~2-3 s via
the prefix cache and short-remainder banked prefill. The workload is memory- and
I/O-bound, not compute-bound: it runs cool enough for fanless MacBooks that CPU
runtimes grill at 100 °C.

> **An informal experiment, not a controlled benchmark:** we copied the same
> two checkpoints into LM Studio (a resident-only MLX runtime) on the same
> machine. The 35B served, about 10 tok/s slower than MaxToken; the 42 GiB 80B
> could not be loaded at all — resident-only runtimes need the entire model
> inside Metal's working-set limit, which is exactly the constraint MaxToken's
> expert offload removes. One run, default settings on both sides; read it as
> an illustration of the category difference, not as a measured comparison.

Everything is served through **OpenAI- and Anthropic-compatible APIs** (Claude
Code and Codex point at it directly), with a built-in single-file **web console**
(chat, live throughput, request log, elastic cache slider) at the server root.

## Quick start

```bash
git clone <this-repo> && cd maxtoken
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx]"

# a model that fits: zero-copy mapped experts, resident speed
ft serve --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --moe-backend offload

# a model that does NOT fit: hard memory budget via the expert slot cache
ft serve --model mlx-community/Qwen3-Coder-Next-4bit \
    --moe-backend offload --moe-cache-rate 0.2
```

Then open `http://localhost:1919/` for the console, or point any OpenAI/Anthropic
client at it. The cache budget resizes live (`/v1/cache/rebuild` or the console
slider), and between requests the scheduler rebalances slots across layers by
observed miss pressure. See **[docs/mlx.md](docs/mlx.md)** for the full macOS
guide: serving modes, benchmarks, speculative decoding (`--draft-model`), and
the honest negative results.

The CUDA engine inherited from upstream remains intact for Linux/NVIDIA
machines; the API server, tokenizer workers, shell and both client APIs are
shared between the backends.

## Relationship to FreeToken

MaxToken began as the upstream FreeToken engine and diverged into its own
project: the MLX backend, the FTW-MLX zero-copy mapped store, the MLX expert
slot cache with speculate-and-verify decode, the hybrid-capable prefix cache,
continuous batching on MLX, banked short-chunk prefill with cross-layer
read-ahead, miss-pressure slot rebalancing, draft-model speculative decoding,
and the web console were developed here. The distribution is `maxtoken`; the
import package deliberately stays `freetoken` so upstream diffs remain readable.

If you use the underlying engine for research, cite the FreeToken
[paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

Upstream FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang)
and reused design and code from [SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).
The MLX backend additionally learned from [mlx-lm](https://github.com/ml-explore/mlx-lm),
llama.cpp's Metal mmap path, and the measured ablations of
[Vates](https://github.com/AMOS144/Vates).

## License

[Apache License 2.0](LICENSE).
