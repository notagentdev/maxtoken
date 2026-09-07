# Install

MaxToken runs on Apple silicon via [MLX](https://github.com/ml-explore/mlx).
There is no Linux or CUDA build: the upstream CUDA engine was removed in
0.0.1 (see the README).

## Requirements

- macOS on Apple silicon (M1 or newer). The numbers in this repository were
  measured on a 32 GB M1 Max; a smaller machine serves smaller checkpoints,
  or serves a bigger one from the SSD under a hard memory budget
  (`--moe-cache-rate`, see [mlx.md](mlx.md)).
- Python >= 3.11.
- Xcode command-line tools (`xcode-select --install`). MLX ships prebuilt
  Metal kernels; nothing is compiled at install time.

## Install from source

```bash
git clone https://github.com/notagentdev/maxtoken.git && cd maxtoken
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pip install -e ".[dev]"` adds pytest. Installing pulls no CUDA ecosystem:
no torch, no triton.

## Verify

```bash
mt --version
python -m pytest tests/mlx_backend -q
```

Then head to [quickstart.md](quickstart.md). The full macOS guide — serving
modes, offload budgets, speculative decoding, benchmarks and the measured
negative results — is [mlx.md](mlx.md).
