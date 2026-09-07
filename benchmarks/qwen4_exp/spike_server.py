"""Spike server: Qwen3.8-Flash-Next (Niwaki, qwen4_exp) over an OpenAI-style
API, one request at a time, on MaxToken's stores (niwaki_stores.py).

This is the feasibility path, NOT MaxToken's serving engine: no prefix cache,
no batching, no reload — but enough of /v1 and /admin for a chat client and
for the MaxToken web console (already open in a browser tab) to show live
numbers.

    MLX_MAX_OPS_PER_BUFFER=40 MLX_MAX_MB_PER_BUFFER=256 \\
    ~/.venvs/qwen4-spike/bin/python benchmarks/qwen4_exp/spike_server.py <model_dir> --port 1234

Env: MAXTOKEN_SPIKE_THINK=1 enables the chat template's thinking mode
(reasoning streams as reasoning_content); MAX_WIRED_GB / MIN_FREE_GB tune the
memory watchdog (aborts before the GPU driver does); PREFILL_STEP sets the
prefill chunk (256).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # bench_mlx_decode (watchdog)
sys.path.insert(0, _HERE)

from bench_mlx_decode import Watchdog, vm_gb  # noqa: E402
from niwaki_stores import load_niwaki_with_stores  # noqa: E402

# Module scope on purpose: with `from __future__ import annotations` the
# endpoint's `request: Request` is a string that FastAPI resolves against the
# module globals — imported inside build_app it did not resolve and FastAPI
# treated `request` as a required query parameter (HTTP 422).
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402

VERSION = "0.2.0-spike"
THINK = os.environ.get("MAXTOKEN_SPIKE_THINK", "0") == "1"
PREFILL_STEP = int(os.environ.get("PREFILL_STEP", "256"))
STATE: dict = {
    "lock": threading.Lock(),
    "requests": 0,
    "active": 0,
    "completed": 0,
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "decode_win": deque(maxlen=4096),   # (t, n) for the 5 s sliding rate
    "prefill_win": deque(maxlen=4096),
    "ring": deque(maxlen=512),          # (idx, record)
    "ring_next": 0,
    "last_rebuild": None,
    "reasoning_budget": 0,
    "context_override": 0,
    "instance_id": uuid.uuid4().hex,
    "ready_at": None,
    "store_bytes": 0,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _defaults(model_dir: str) -> dict:
    gc = {}
    p = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(p):
        gc = json.load(open(p))
    eos = gc.get("eos_token_id", [])
    if isinstance(eos, int):
        eos = [eos]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    return {
        "generation": {
            "temperature": float(gc.get("temperature", 1.0)),
            "top_p": float(gc.get("top_p", 0.95)),
            "top_k": int(gc.get("top_k", 20)),
            "max_output_tokens": 2048,
        },
        "eos": [int(e) for e in eos],
        "context": int(tc.get("max_position_embeddings", 32768)),
        "num_experts": int(tc.get("num_experts", 0)),
        "num_moe_layers": int(tc.get("num_hidden_layers", 0)) - len(cfg.get("niwaki", {}).get("shared_only_layers", [])),
    }


def _rate(win) -> float:
    now = time.monotonic()
    while win and win[0][0] < now - 5.0:
        win.popleft()
    if not win:
        return 0.0
    return sum(n for _, n in win) / max(now - win[0][0], 1e-9)


def _record(rec: dict) -> None:
    STATE["ring"].append((STATE["ring_next"], rec))
    STATE["ring_next"] += 1


def _split_thinking(chunks):
    """Yield (kind, text) with kind in {reasoning, content}; the <think> block
    that a thinking-mode template opens becomes reasoning."""
    if not THINK:
        for c in chunks:
            yield "content", c
        return
    phase, pending = "reasoning", ""
    for c in chunks:
        pending += c
        if phase == "reasoning":
            if pending.startswith("<think>"):
                pending = pending[len("<think>"):]
            elif "<think>".startswith(pending):
                continue  # tag still arriving
            i = pending.find("</think>")
            if i >= 0:
                if pending[:i]:
                    yield "reasoning", pending[:i]
                pending = pending[i + len("</think>"):].lstrip("\n")
                phase = "content"
                if pending:
                    yield "content", pending
                pending = ""
                continue
            keep = 7  # a split "</think>" may be in flight
            if len(pending) > keep:
                yield "reasoning", pending[:-keep]
                pending = pending[-keep:]
        else:
            yield "content", pending
            pending = ""
    if pending:
        yield ("reasoning" if phase == "reasoning" else "content"), pending


def _generate(messages, params, *, path="/v1/chat/completions", stream=True):
    """Yields (kind, text, info); the last item is ("done", "", stats). Records
    the request in the ring and the totals whatever happens."""
    from mlx_vlm import stream_generate

    model, processor, d = STATE["model"], STATE["processor"], STATE["defaults"]
    g = d["generation"]
    tok = getattr(processor, "tokenizer", processor)
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=THINK)
    kwargs = dict(
        max_tokens=int(params.get("max_tokens") or g["max_output_tokens"]),
        temperature=float(params.get("temperature", g["temperature"])),
        top_p=float(params.get("top_p", g["top_p"])),
        top_k=int(params.get("top_k", g["top_k"])),
        prefill_step_size=PREFILL_STEP,
        eos_tokens=d["eos"],
    )
    t0 = time.perf_counter()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    first = None
    n = 0
    prompt_tokens = None
    last = None
    error = None
    STATE["requests"] += 1
    STATE["active"] += 1

    def chunks():
        nonlocal first, n, prompt_tokens, last
        for ch in stream_generate(model, processor, prompt, **kwargs):
            now = time.monotonic()
            if first is None:
                first = time.perf_counter() - t0
                prompt_tokens = getattr(ch, "prompt_tokens", None)
                if prompt_tokens:
                    STATE["prefill_win"].append((now, int(prompt_tokens)))
                    STATE["prompt_tokens_total"] += int(prompt_tokens)
            n += 1
            STATE["decode_win"].append((now, 1))
            STATE["completion_tokens_total"] += 1
            last = ch
            if ch.text:
                yield ch.text

    try:
        for kind, text in _split_thinking(chunks()):
            yield kind, text, None
    except Exception as exc:  # noqa: BLE001 -- the probe must report, not die
        error = f"{type(exc).__name__}: {exc}"
        log(f"request failed: {error}")
    finally:
        STATE["active"] -= 1
    finish = getattr(last, "finish_reason", None) or ("length" if n >= kwargs["max_tokens"] else "stop")
    dt = time.perf_counter() - t0
    dec = (n - 1) / (dt - first) if first is not None and n > 1 else 0.0
    if error is None:
        STATE["completed"] += 1
    _record({
        "ts": started, "method": "POST", "path": path, "status": 500 if error else 200,
        "model": STATE["name"], "duration_ms": int(dt * 1000),
        "ttft_ms": int(first * 1000) if first is not None else None,
        "prompt_tokens": prompt_tokens, "completion_tokens": n, "stream": stream, "error": error,
    })
    log(f"request {STATE['requests']}: prompt {prompt_tokens} tok, TTFT {first or 0:.1f}s, "
        f"{n} tok in {dt:.1f}s = {dec:.1f} tok/s, finish={finish}, wired {vm_gb()[0]:.1f} GB"
        + (f", ERROR {error}" if error else ""))
    yield "done", "", {"prompt_tokens": prompt_tokens or 0, "completion_tokens": n,
                       "finish_reason": "error" if error else finish, "error": error}


def _geometry() -> dict:
    d = STATE["defaults"]
    return {
        "num_pages": 0, "num_mamba_slots": 0, "page_size": 1,
        "moe_cache_size": 0,  # mapped store: no slot budget to move
        "swa_full_tokens_ratio": 0.0,
        "num_experts": d["num_experts"], "num_moe_layers": d["num_moe_layers"],
        "unit_bytes": {"kv_per_token": 0, "moe_per_expert": 0, "mamba_per_slot": 0, "swa_per_token": 0},
        "expert_store": "zero-copy mapped (file-backed)",
    }


def build_app(name: str):
    app = FastAPI(version=VERSION)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    d = STATE["defaults"]

    def uptime() -> int:
        return max(0, int(time.monotonic() - STATE["ready_at"])) if STATE["ready_at"] else 0

    console = os.path.join(_HERE, "..", "..", "python", "maxtoken", "server", "console.html")

    @app.get("/", include_in_schema=False)
    @app.get("/console", include_in_schema=False)
    def root():
        # The MaxToken web console, the same single file `mt serve` ships (at
        # /console there, / as well here); it polls /health, /admin/stats,
        # /admin/requests and /admin/cache/status.
        if os.path.exists(console):
            return FileResponse(console, media_type="text/html")
        return JSONResponse({"status": "ok", "model": name, "console": "not found"})

    @app.get("/v1/models/{model_id}")
    def model_by_id(model_id: str):
        ctx = STATE["context_override"] or d["context"]
        return {"id": name, "object": "model", "owned_by": "MaxToken", "max_model_len": ctx, "context_length": ctx}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": name, "instance_id": STATE["instance_id"], "uptime_s": uptime(),
                "maintenance": "serving", "version": VERSION, "engine": "maxtoken-spike-qwen4_exp"}

    @app.get("/v1/models")
    def models():
        ctx = STATE["context_override"] or d["context"]
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "MaxToken",
                                            "max_model_len": ctx, "context_length": ctx}]}

    @app.get("/api/v0/models")
    def lms_models():
        ctx = STATE["context_override"] or d["context"]
        return {"object": "list", "data": [{"id": name, "object": "model", "type": "llm", "publisher": "MaxToken",
                                            "arch": "qwen4_exp", "quantization": "3bit", "state": "loaded",
                                            "max_context_length": ctx, "loaded_context_length": ctx}]}

    @app.get("/admin/stats")
    def stats():
        import mlx.core as mx

        ring = list(STATE["ring"])
        durs = sorted(r["duration_ms"] for _, r in ring)
        p95 = durs[max(0, -(-95 * len(durs) // 100) - 1)] if durs else 0
        ttfts = [r["ttft_ms"] for _, r in ring if r["ttft_ms"] is not None]
        return {
            "instance_id": STATE["instance_id"],
            "model": {"id": name, "ctx": STATE["context_override"] or d["context"], "attn": "hybrid_linear",
                      "moe": True, "sampling": {k: d["generation"][k] for k in ("temperature", "top_p", "top_k")}},
            "uptime_s": uptime(),
            "kv": None, "mamba": None, "swa": None,
            "vram_bytes": max(0, int(mx.get_active_memory()) - STATE["store_bytes"]),
            "throughput": {"decode_tps": round(_rate(STATE["decode_win"]), 1),
                           "prefill_tps": round(_rate(STATE["prefill_win"]), 1)},
            "requests": {"active": STATE["active"], "completed": STATE["completed"], "p95_ms": int(p95),
                         "ttft_mean_ms": int(sum(ttfts) / len(ttfts)) if ttfts else 0,
                         "prompt_tokens_total": STATE["prompt_tokens_total"],
                         "completion_tokens_total": STATE["completion_tokens_total"]},
        }

    @app.get("/admin/requests")
    def requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        matched = [(i, r) for i, r in STATE["ring"] if i >= since]
        out = [r for _, r in matched[:limit]]
        nxt = matched[len(out) - 1][0] + 1 if len(out) < len(matched) else STATE["ring_next"]
        return {"entries": out, "next_cursor": nxt}

    @app.get("/admin/cache/status")
    def cache_status():
        return {
            "state": "serving", "last_rebuild": STATE["last_rebuild"], "geometry": _geometry(),
            "context": {"current": STATE["context_override"] or d["context"], "ceiling": d["context"]},
            "reasoning_budget": STATE["reasoning_budget"],
            "generation": dict(d["generation"]),
            "prefix_disk": None,
        }

    @app.post("/admin/cache/rebuild")
    async def cache_rebuild(request: Request):
        body = await request.json()
        applied = {}
        if "temperature" in body:
            d["generation"]["temperature"] = float(body["temperature"])
            applied["temperature"] = d["generation"]["temperature"]
        if body.get("max_output_tokens"):
            d["generation"]["max_output_tokens"] = int(body["max_output_tokens"])
            applied["max_output_tokens"] = d["generation"]["max_output_tokens"]
        if "max_reasoning_tokens" in body:
            STATE["reasoning_budget"] = int(body["max_reasoning_tokens"] or 0)
            applied["max_reasoning_tokens"] = STATE["reasoning_budget"]
        if body.get("max_seq_len"):
            STATE["context_override"] = int(body["max_seq_len"])
            applied["max_seq_len"] = STATE["context_override"]
        if "moe_cache_size" in body:
            return JSONResponse({"status": "failed", "completed": False,
                                 "error": "mapped expert store: no slot budget to resize (spike server)"},
                                status_code=422)
        STATE["last_rebuild"] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **applied}
        log(f"settings applied: {applied}")
        return {"status": "ok", "completed": True, **applied}

    @app.post("/admin/reload")
    def reload():
        return JSONResponse({"status": "failed", "error": "spike server: reload not supported — restart the process"},
                            status_code=501)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        messages = body.get("messages") or []
        stream = bool(body.get("stream", False))
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def event(delta, finish=None, usage=None):
            payload = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": name,
                       "choices": [] if usage is not None and delta is None else
                       [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n"

        def run():
            with STATE["lock"]:
                yield from _generate(messages, body, stream=stream)

        if stream:
            def sse():
                yield event({"role": "assistant", "content": ""})
                stats = None
                for kind, text, info in run():
                    if kind == "done":
                        stats = info
                        break
                    yield event({"reasoning_content": text} if kind == "reasoning" else {"content": text})
                yield event({}, finish=stats["finish_reason"] if stats else "stop")
                if want_usage and stats:
                    yield event(None, usage={"prompt_tokens": stats["prompt_tokens"],
                                             "completion_tokens": stats["completion_tokens"],
                                             "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"]})
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        content, reasoning, stats = [], [], None
        for kind, text, info in run():
            if kind == "done":
                stats = info
            elif kind == "reasoning":
                reasoning.append(text)
            else:
                content.append(text)
        if stats and stats.get("error"):
            return JSONResponse({"error": {"message": stats["error"], "type": "server_error"}}, status_code=500)
        msg = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": stats["finish_reason"]}],
                             "usage": {"prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
                                       "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"]}})

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--cache-rate", type=float, default=None, help="slot cache instead of the mapped store")
    p.add_argument("--served-model-name", default=None)
    args = p.parse_args()

    wd = Watchdog(min_free_gb=float(os.environ.get("MIN_FREE_GB", "0.3")),
                  max_wired_gb=float(os.environ.get("MAX_WIRED_GB", "26.5")))
    wd.start()

    model_dir = os.path.abspath(args.model_dir)
    name = args.served_model_name or os.path.basename(model_dir.rstrip("/"))
    STATE["name"] = name
    STATE["defaults"] = _defaults(model_dir)
    g = STATE["defaults"]["generation"]
    log(f"loading {name} (thinking {'on' if THINK else 'off'}, prefill step {PREFILL_STEP}, "
        f"sampling {g['temperature']}/{g['top_p']}/{g['top_k']})")
    model, processor = load_niwaki_with_stores(model_dir, cache_rate=args.cache_rate, log=log)
    STATE["model"], STATE["processor"] = model, processor
    store = getattr(model, "_maxtoken_mapped_store", None)
    if store is not None:
        try:
            STATE["store_bytes"] = os.path.getsize(store.path)
        except OSError:
            pass

    if os.environ.get("WARMUP", "1") == "1":
        t0 = time.perf_counter()
        for _ in _generate([{"role": "user", "content": "Hi"}], {"max_tokens": 8}, path="/warmup", stream=False):
            pass
        log(f"warm-up done in {time.perf_counter()-t0:.1f}s")
    STATE["ready_at"] = time.monotonic()

    import uvicorn

    log(f"serving {name} on http://{args.host}:{args.port}  (OpenAI /v1/chat/completions, /v1/models, /health, /admin/*)")
    uvicorn.run(build_app(name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
