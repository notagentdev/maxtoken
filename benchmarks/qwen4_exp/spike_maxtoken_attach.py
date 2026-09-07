"""Qwen3.8-Flash-Next (Niwaki, qwen4_exp) on MaxToken's stores, in-process:
loads through niwaki_stores.load_niwaki_with_stores (PLE on disk, experts
from the mapped store or the slot cache) and streams a generation under the
memory watchdog, printing TTFT, decode rate and the wired peak.

    ~/.venvs/qwen4-spike/bin/python spike_maxtoken_attach.py <model_dir> [max_tokens]
    MOE_CACHE_RATE=0.3   slot cache instead of the mapped store
    LONG_PROMPT=1500     a ~1.5k-token prompt (prefill measurement)
    PREFILL_STEP=256     mlx-vlm's chunked prefill step
    MLX_MAX_OPS_PER_BUFFER=40 MLX_MAX_MB_PER_BUFFER=256 keep the wired peak
    under the 26 GB the watchdog aborts at (measured 2026-09-07).
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from bench_mlx_decode import Watchdog, vm_gb  # noqa: E402
from niwaki_stores import load_niwaki_with_stores  # noqa: E402

model_dir = os.path.abspath(sys.argv[1])
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128

wd = Watchdog(min_free_gb=float(os.environ.get("MIN_FREE_GB", "0.3")),
              max_wired_gb=float(os.environ.get("MAX_WIRED_GB", "26")))
wd.start()
print(f"before load: wired {vm_gb()[0]:.1f} GB, available {vm_gb()[1]:.1f} GB", flush=True)

import mlx.core as mx  # noqa: E402

rate = os.environ.get("MOE_CACHE_RATE")
model, processor = load_niwaki_with_stores(model_dir, cache_rate=float(rate) if rate else None)
print(f"wired {vm_gb()[0]:.1f} GB available {vm_gb()[1]:.1f} GB", flush=True)

text_layers = list(getattr(model, "language_model", model).model.layers)
tok = getattr(processor, "tokenizer", processor)
long_tokens = int(os.environ.get("LONG_PROMPT", "0"))
if long_tokens:
    para = ("In 1450 Johannes Gutenberg completed a workable printing press in Mainz, combining a screw press, "
            "oil-based ink and metal type cast from a hand mould. By 1480 presses operated in more than a hundred "
            "towns; by 1500 an estimated twenty million volumes had been printed. Venice, with Aldus Manutius, "
            "became the centre of scholarly printing; Paris and Lyon followed. ")
    content = "Summarize the following notes in three sentences.\n\n" + para * (long_tokens // 90)
else:
    content = "Explain in three sentences why the sky is blue."
prompt = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                 tokenize=False, enable_thinking=False)

try:
    from mlx_vlm import stream_generate
except ImportError:  # older layout
    from mlx_vlm.generate import stream_generate


def ple_lookups() -> int:
    total = 0
    for layer in text_layers:
        emb = getattr(getattr(getattr(layer, "ple", None), "ple_embedding", None), "ngram_embedding", None)
        total += int(getattr(emb, "lookups", 0) or 0)
    return total


for rep in range(2):
    t2 = time.perf_counter()
    n, first, text, last = 0, None, [], None
    print(f"rep {rep}: generating ...", flush=True)
    gen_kwargs = {"max_tokens": max_tokens, "temperature": 0.7}
    if os.environ.get("PREFILL_STEP"):
        gen_kwargs["prefill_step_size"] = int(os.environ["PREFILL_STEP"])
    for chunk in stream_generate(model, processor, prompt, **gen_kwargs):
        now = time.perf_counter() - t2
        if first is None:
            first = now
            print(f"  first token at {first:.1f}s: {chunk.text!r}", flush=True)
        n += 1
        text.append(chunk.text)
        last = chunk
        if n % 16 == 0:
            print(f"  {n} tokens at {now:.1f}s ({(n-1)/(now-first):.1f} tok/s decode); ple lookups so far {ple_lookups()}",
                  flush=True)
    dt = time.perf_counter() - t2
    print(f"rep {rep}: {n} tokens in {dt:.1f}s; TTFT {first:.1f}s; decode {(n-1)/(dt-first):.1f} tok/s; "
          f"reported prompt_tps={getattr(last, 'prompt_tps', None)} generation_tps={getattr(last, 'generation_tps', None)} "
          f"peak={getattr(last, 'peak_memory', None)}; watchdog peak wired {wd.peak_wired:.1f} GB", flush=True)
    print("  text:", "".join(text)[:400].replace("\n", " "), flush=True)
