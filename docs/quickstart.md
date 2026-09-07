# Quick start

Assumes MaxToken is installed — see [install.md](install.md).

## Launch a server

```bash
mt serve --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --moe-backend offload
```

`--model` takes a local directory or a Hugging Face repo id of an MLX
checkpoint (4-bit conversions from `mlx-community` and friends). MoE
checkpoints want `--moe-backend offload`: experts are served from a
zero-copy memory-mapped store when they fit, or from an SSD-backed slot
cache under a hard budget when they do not (`--moe-cache-rate 0.2`). Dense
checkpoints need no flag. Everything else — tool-call and reasoning parsers,
sampling defaults, context length — resolves from the checkpoint; see
[cli.md](cli.md) for the flags and [mlx.md](mlx.md) for the serving modes.

Start the server from a regular terminal, not from a background launcher:
macOS clamps a background task's Metal threads and the clamp is inherited
(details in [mlx.md](mlx.md)).

The server is ready when

```bash
curl http://127.0.0.1:1919/health
```

reports `"status": "ok"`. `http://127.0.0.1:1919/` serves the built-in web
console (chat, live throughput, request log, cache slider).

## Send a request

Check what is being served:

```bash
curl http://127.0.0.1:1919/v1/models
```

Then use that id as the `model` field:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Ornith-1.5-35B-A3B-MLX-4bit",
    "messages": [{"role": "user", "content": "What is a Mixture-of-Experts model?"}],
    "max_tokens": 256,
    "stream": true
  }'
```

MaxToken serves the OpenAI API (`/v1/chat/completions`, `/v1/responses`,
`/v1/models`) and the Anthropic API (`/v1/messages`,
`/v1/messages/count_tokens`), so a client library for either works by pointing
its base URL at the server.

## Chat in the terminal

```bash
mt shell                                                          # attach to the server above
mt shell --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --moe-backend offload   # serve + chat, one process
```

`/help` lists the in-shell commands. Attach mode loads no model, so it also
drives a server on another machine (`--server URL`).

## Use a coding agent

```bash
mt launch claude   # claude / codex / dsh / hermes / openclaw / opencode
```

Writes that agent's provider config, installs its CLI if missing, and starts it
against your server. `--dry-run` previews the changes.
