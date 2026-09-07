# MaxToken test suite

Directories mirror the `python/maxtoken/` subsystem a test primarily exercises —
put a new test next to the module it protects.

| directory      | subsystem under test |
|----------------|----------------------|
| `mlx_backend/` | `maxtoken.mlx_backend` — the MLX scheduler, expert offload (slot cache, rebalance, read-ahead), the mapped FTW store, prefix cache and its SSD tier, decode fusion, speculative decoding, the offload prefill plan |
| `server/`      | `maxtoken.server` — OpenAI/Anthropic/Responses APIs, streaming, tool-call and reasoning parsers, accounting, reload/maintenance state, supervisor |
| `tokenizer/`   | `maxtoken.tokenizer` — tokenize/detokenize request plumbing, thinking-mode resolution |
| `daemon/`      | `maxtoken.daemon` — supervisor import safety, serve manager |
| `shell/`       | `maxtoken.shell` — the terminal chat client |

The CLI surface itself is not unit-tested: `mt` dispatch, `mt ctl` and
`mt launch` are thin and change often, and a test written after the fact only
restates whatever the code currently does — running the command is the real
gate. What does live here is the logic reachable *through* those commands when
it has its own failure mode (`server/test_parser_auto_selection.py` covers the
architecture -> parser inference behind `--tool-call-parser auto`).

## Running

```bash
python -m pytest tests/ -q                 # full suite, ~30 s on an M1 Max
python -m pytest tests/mlx_backend -q      # one subsystem
```

The suite needs no model: MLX tests build small synthetic checkpoints in a
temporary directory and skip themselves where the hardware is missing.
`needs_weights`-marked tests skip unless the env var pointing at a real local
checkpoint is set:

| env var                   | used by |
|---------------------------|---------|
| `MAXTOKEN_TEST_MLX_MODEL` | `mlx_backend/test_prefix_cache_weights.py` — a local MLX checkpoint directory; exercises prefix-cache restores against a real model |

Anything that measures throughput lives in `benchmarks/`, not here: a
performance claim in this repository is an A/B measured back to back on the
same machine state, and the numbers are recorded in `docs/mlx.md`.

## What earns a place here

A test must be able to fail on a plausible regression of production code.

Weigh that against how the bug would be found otherwise. Application-facing
surfaces — the terminal UI, the access-log filter, the request ring, the status
and health endpoints, the CLI — announce their own breakage the moment anyone
uses them, and they change shape often enough that a unit test mostly restates
today's implementation. What earns a unit test is logic that fails *silently*:
a wrong number that still looks like a number, a model family that quietly
stops having its tool calls parsed, a cache that returns the wrong slot, a
chunk plan that streams the checkpoint seven times instead of once. Prefer
covering that where it lives rather than through the surface that happens to
expose it.

Writing a test in the same commit as the feature is right; what matters is
where its expectations come from. A test that checks the result against
something independent — the stock mlx-lm forward, a round-trip, a numpy
mirror, the live registry — keeps working when the implementation is
rewritten, and can disagree with it. A test whose assertions restate the
branch structure the author just wrote can only ever agree, so it never fails
on anything except a deliberate change, and then it is edited to match. If you
cannot name what the expectation is checked *against*, the test is unlikely to
earn its place. Bug-repro tests should drive the current call pattern of the
fixed code, not the pre-fix one.
