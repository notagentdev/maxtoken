# Contributing to MaxToken

Thanks for helping make MaxToken better. This page covers how to report
issues and how to submit pull requests.

## Reporting issues

Open an issue in this repository. A report we cannot reproduce is a report
we cannot fix, so include:

- your Mac: chip, unified memory, macOS version, and how much memory was
  free when it happened (`vm_stat` or Activity Monitor);
- the MaxToken version (`mt --version`) and, when building from source, the
  commit (`git rev-parse --short HEAD`);
- the exact checkpoint (the Hugging Face id, e.g.
  `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, not just the model name);
- the full `mt serve` command and the full log, not a screenshot of the last
  line;
- whether the server was started from a terminal or from a background
  launcher (the latter is throttled by macOS, see [docs/mlx.md](docs/mlx.md)).

Performance reports need numbers: the tokens/s the server logs, the prompt
length, and whether the request was the first after start or a warm one.

## Pull requests

### AI policy

AI-assisted contributions are welcome. You are responsible for everything in
your PR, however it was produced: a human must understand what changed and
why, have run it on real hardware, and be able to explain it to a reviewer.
PRs whose descriptions do not match their code, or whose benchmark numbers
were not actually measured, are closed without review.

### What a PR needs

- One change per PR. Unrelated fixes go in separate PRs.
- State what you tested on: chip, memory, macOS version, the checkpoint's
  Hugging Face id, and the exact command.
- For performance changes, A/B end-to-end results: the same model, prompt
  and settings on `main` and on your branch, measured back to back on the
  same machine state, with tokens/s (and TTFT if prefill is affected) for
  both. This repository documents negative results as carefully as positive
  ones; a change that loses is still a useful PR if it says so.
- For bug fixes, a test that fails before the change and passes after, where
  the code allows it.

## Development setup

```bash
git clone https://github.com/notagentdev/maxtoken.git && cd maxtoken
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

See [docs/install.md](docs/install.md) for requirements and
[tests/README.md](tests/README.md) for the suite's layout and the
`needs_weights` tests that run against a local checkpoint.

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

- `type`: `feat`, `fix`, `perf`, `refactor`, `build`, `docs`, `test`, `chore`.
- `scope`: the module you touched, e.g. `mlx`, `server`, `shell`, `deps`,
  `README`. Optional.
- `subject`: imperative, lowercase, no trailing period.
- Breaking changes: add `!` after the scope and explain in the body.

Examples from the history:

```
perf(mlx): split-K verify kernel — built, measured, defaulted off
fix(mlx): offload prefill streams one pass per 2048 tokens; rebalance keeps its budget
docs: the batch-server mystery's resolution and the unclamped numbers
```

## License

By contributing to MaxToken, you agree that your contributions will be
licensed under the [LICENSE](LICENSE) in the root of this repository.
