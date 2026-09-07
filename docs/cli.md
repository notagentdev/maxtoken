# CLI reference

```
mt <command> [args]
```

| Command | Purpose |
|---|---|
| `mt serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `mt shell` | Chat with a server in the terminal |
| `mt ctl` | Query and manage a running server over HTTP |
| `mt launch` | Configure and launch a coding agent against a server |

`mt --version` prints the installed version. Every command supports `--help`.

## mt serve

```bash
mt serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag; add `--moe-backend offload` for MoE
checkpoints. Tool-call and reasoning parsers, sampling defaults and the
context length resolve from the checkpoint. Environment knobs for the MLX
backend (`MAXTOKEN_MLX_*`) are documented next to the behaviour they tune in
[mlx.md](mlx.md).

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir or HF repo id of an MLX checkpoint |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--max-running-requests` | 4 | Max concurrently running requests (continuous batching on the resident and mapped paths) |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Context length |
| `--decode-log-interval` | 40 | Status line (cache miss rate, speculative acceptance) every N decode steps |

### Experts and memory

See [mlx.md](mlx.md) for what each mode does and what it costs.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-backend` | auto (= resident) | `offload`: routed experts from a zero-copy mapped store when the model fits, or from the slot cache below when a `--moe-cache-*` budget is given. The other accepted values (`fused`, `cpu`, `hybrid`) are leftovers of the removed CUDA engine |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | — | Slot-cache budget as slots / fraction of all experts / sized from free memory (mutually exclusive); selects the SSD-backed slot cache |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--cache-type` | radix | `radix` (prefix reuse, hybrid-model aware) or `naive` (off) |
| `--prefix-cache-dir` | `~/.maxtoken/prefix-cache` | SSD tier of the prefix cache: per-block KV and recurrent snapshots that survive restarts |
| `--prefix-cache-disk-gb` | 20 | Budget of that tier; `0` disables it |

### Speculative decoding

| Flag | Default | Meaning |
|---|---|---|
| `--draft-model` | off | `mtp` uses the checkpoint's own multi-token-prediction head; a path to a sibling artifact's `mtp.safetensors` works for conversions that ship without it; a second model's path drafts with that model |
| `--draft-tokens` | 3 | Draft depth per round (2 measured best on Ornith-1.5-35B, 3 on Qwen3.8-27B) |

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits (`cached_tokens`) in each response's usage block |

## mt shell

```bash
mt shell                                                          # attach to a running server
mt shell --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --moe-backend offload   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## mt ctl

```bash
mt ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /admin/stats` | Throughput, latency, memory |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /admin/cache/status` | Expert-cache table |
| `cache --moe N [--wait 300]` | `POST /admin/cache/rebuild` | Live resize of the expert slot cache without a restart (`k`/`m` suffixes) |
| `requests [--since N] [--limit N]` | `GET /admin/requests` | Recent request ring |

## mt launch

```bash
mt launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |
