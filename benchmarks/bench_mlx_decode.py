#!/usr/bin/env python3
"""In-process decode and prefill probe for an MLX checkpoint, with a memory watchdog.

Measures what the engine itself does -- no server, no ZMQ -- so a change to the
forward, a Metal knob or a sampler can be judged in isolation:

    python benchmarks/bench_mlx_decode.py --model ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
    MLX_MAX_OPS_PER_BUFFER=400 MLX_MAX_MB_PER_BUFFER=256 \\
        python benchmarks/bench_mlx_decode.py --model ... --wired-gb 16 --prefill 8192

Reports ms/token for greedy decode (async-pipelined, the engine's own regime),
the wired and free memory before/after, and -- with --prefill N -- the peak
wired memory while N prompt tokens are processed in the worker's chunk size.
The watchdog samples vm_stat every 200 ms and kills the process the moment
free memory drops under --min-free-gb, because the alternative on a machine
this size is a kernel panic in the GPU driver (seen 2026-08-28 with wide
command buffers: 28.7 GB wired, 66 MB free, machine down).

The Metal command-buffer limits are read by MLX once, from the environment:
set them on the command line as above, never from inside the process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
sys.path.insert(0, ROOT)


def _host_vm_pages() -> tuple[int, int, int, int] | None:
    """(free, inactive, speculative, wired) page counts via host_statistics64,
    i.e. WITHOUT forking. The watchdog samples from a thread while MLX's Metal
    threads hold malloc locks, and a fork() there (which subprocess.run does on
    macOS) aborts the process in libSystem's atfork handler — seen 2026-09-07
    as EXC_BREAKPOINT in _os_unfair_lock_unowned_abort, silently, mid-decode."""
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

        class VmStat64(ctypes.Structure):
            _fields_ = [
                ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
                ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
                ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
                ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
                ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
                ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
                ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
                ("speculative_count", ctypes.c_uint32), ("decompressions", ctypes.c_uint64),
                ("compressions", ctypes.c_uint64), ("swapins", ctypes.c_uint64),
                ("swapouts", ctypes.c_uint64), ("compressor_page_count", ctypes.c_uint32),
                ("throttled_count", ctypes.c_uint32), ("external_page_count", ctypes.c_uint32),
                ("internal_page_count", ctypes.c_uint32),
                ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
            ]

        stats = VmStat64()
        count = ctypes.c_uint32(ctypes.sizeof(VmStat64) // 4)
        libc.mach_host_self.restype = ctypes.c_uint32
        host = libc.mach_host_self()
        if libc.host_statistics64(host, 4, ctypes.byref(stats), ctypes.byref(count)) != 0:
            return None
        return stats.free_count, stats.inactive_count, stats.speculative_count, stats.wire_count
    except (OSError, AttributeError):
        return None


def vm_gb() -> tuple[float, float]:
    """(wired GB, available GB). "Available" is free + inactive + speculative:
    macOS keeps "free" tiny on purpose and lets the file cache hold the rest,
    so free alone reads as an emergency while there is nothing wrong. What
    killed the machine was wired memory, which no cache can give back."""
    page = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 16384
    pages = _host_vm_pages()
    if pages is not None:
        free, inactive, spec, wired = pages
        return wired * page / 2**30, (free + inactive + spec) * page / 2**30
    # Fallback only (it forks): keep it out of any thread that runs beside Metal.
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                vals[k.strip()] = int(v)
    avail = vals.get("Pages free", 0) + vals.get("Pages inactive", 0) + vals.get("Pages speculative", 0)
    return vals.get("Pages wired down", 0) * page / 2**30, avail * page / 2**30


class Watchdog(threading.Thread):
    def __init__(self, min_free_gb: float, max_wired_gb: float):
        super().__init__(daemon=True)
        self.min_free_gb = min_free_gb
        self.max_wired_gb = max_wired_gb
        self.peak_wired = 0.0
        self.min_free = 1e9
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            wired, free = vm_gb()
            self.peak_wired = max(self.peak_wired, wired)
            self.min_free = min(self.min_free, free)
            if free < self.min_free_gb or wired > self.max_wired_gb:
                print(f"\nWATCHDOG: available {free:.2f} GB (limit {self.min_free_gb}), wired {wired:.1f} GB "
                      f"(limit {self.max_wired_gb}) "
                      f"(wired {wired:.1f} GB) -- aborting before the kernel does", flush=True)
                os._exit(3)
            time.sleep(0.2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", type=int, default=60)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--prefill", type=int, default=0, help="also prefill this many tokens (memory probe)")
    p.add_argument("--chunk", type=int, default=2048, help="prefill chunk size (the worker's default)")
    p.add_argument("--wired-gb", type=float, default=None, help="mx.set_wired_limit in GB (default: leave MLX's 0)")
    p.add_argument("--min-free-gb", type=float, default=1.5, help="abort when free+inactive+speculative falls under this")
    p.add_argument("--max-wired-gb", type=float, default=26.0, help="abort when wired memory exceeds this")
    p.add_argument("--fusion", action="store_true", help="install decode_fusion (the worker's default)")
    p.add_argument("--pace", type=int, default=0, help="prefill: evaluate every N layers to bound in-flight memory (0 = whole chunk lazily)")
    p.add_argument("--cache-limit-gb", type=float, default=None, help="mx.set_cache_limit in GB")
    args = p.parse_args()

    wd = Watchdog(args.min_free_gb, args.max_wired_gb)
    wd.start()
    print(f"env: ops/buffer={os.environ.get('MLX_MAX_OPS_PER_BUFFER', 'default')} "
          f"mb/buffer={os.environ.get('MLX_MAX_MB_PER_BUFFER', 'default')}  before: "
          f"wired {vm_gb()[0]:.1f} GB available {vm_gb()[1]:.1f} GB", flush=True)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    if args.wired_gb is not None:
        mx.set_wired_limit(int(args.wired_gb * 2**30))
    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 2**30))
    t = time.perf_counter()
    model, tok = load(args.model, lazy=True)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter() - t:.1f}s, active {mx.get_active_memory() / 2**30:.1f} GiB, "
          f"wired {vm_gb()[0]:.1f} GB available {vm_gb()[1]:.1f} GB", flush=True)
    if args.fusion:
        from maxtoken.mlx_backend import decode_fusion

        print("fusion:", decode_fusion.install(model), flush=True)

    msg = ("Write a detailed essay on how the printing press changed Europe between 1450 and 1600, "
           "with dates and names.")
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True)
    if isinstance(ids, str):
        ids = tok.encode(ids)
    ids = [int(i) for i in ids]

    res = []
    for _ in range(args.reps):
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array(ids[:-1])[None], cache=cache))
        y = mx.array([ids[-1]])
        for _ in range(3):
            y = mx.argmax(model(y[None], cache=cache)[:, -1, :], axis=-1)
            mx.eval(y)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.tokens):
            y = mx.argmax(model(y[None], cache=cache)[:, -1, :], axis=-1)
            mx.async_eval(y)
        mx.eval(y)
        mx.synchronize()
        res.append((time.perf_counter() - t0) / args.tokens * 1000)
    wired, free = vm_gb()
    print(f"decode: {' / '.join(f'{r:.2f}' for r in res)} ms/token = {1000 / min(res):.1f} tok/s (best)  "
          f"after: wired {wired:.1f} GB available {free:.1f} GB", flush=True)

    if args.prefill:
        prompt = (ids * (args.prefill // len(ids) + 1))[: args.prefill]
        cache = make_prompt_cache(model)
        text = getattr(model, "language_model", model)
        t0 = time.perf_counter()
        pos = 0
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        inner = text.model
        while pos < len(prompt):
            chunk = mx.array(prompt[pos : pos + args.chunk])[None]
            if args.pace <= 0:
                mx.eval(inner(chunk, cache=cache))
            else:
                # The model's own layer loop, with a synchronous eval every
                # `pace` layers: bounds what is in flight to that many layers'
                # temporaries, whatever the command-buffer limits allow.
                h = inner.embed_tokens(chunk)
                fa_mask = create_attention_mask(h, cache[inner.fa_idx])
                ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
                for i, layer in enumerate(inner.layers):
                    h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=cache[i])
                    if (i + 1) % args.pace == 0:
                        mx.eval(h)
                mx.eval(inner.norm(h))
            pos += int(chunk.shape[1])
        mx.synchronize()
        dt = time.perf_counter() - t0
        print(f"prefill {args.prefill} tokens in chunks of {args.chunk}: {dt:.2f}s "
              f"({args.prefill / dt:.0f} tok/s)  peak wired {wd.peak_wired:.1f} GB, min available {wd.min_free:.2f} GB",
              flush=True)
    wd.stop.set()
    print(f"peak wired {wd.peak_wired:.1f} GB, min available {wd.min_free:.2f} GB over the run", flush=True)


if __name__ == "__main__":
    main()
