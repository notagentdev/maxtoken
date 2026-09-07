"""Step 2 of the qwen4_exp spike: STOCK mlx-vlm + the checkpoint's own Niwaki
loader, no MaxToken code. Measures whether the 34 GiB model loads on this
32 GB machine at all, and if so its prefill/decode speed — under the same
memory watchdog benchmarks/bench_mlx_decode.py uses, because the alternative
on a machine this size is a swap storm or a GPU-driver panic.

    ~/.venvs/qwen4-spike/bin/python spike_stock_baseline.py <model_dir> [max_tokens]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_mlx_decode import Watchdog, vm_gb  # noqa: E402

model_dir = os.path.abspath(sys.argv[1])
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
sys.path.insert(0, model_dir)  # niwaki_flash_load.py ships in the checkpoint

wd = Watchdog(min_free_gb=float(os.environ.get("MIN_FREE_GB", "1.5")),
              max_wired_gb=float(os.environ.get("MAX_WIRED_GB", "26")))
wd.start()
w0, a0 = vm_gb()
print(f"before load: wired {w0:.1f} GB, available {a0:.1f} GB", flush=True)

import mlx.core as mx  # noqa: E402

t0 = time.perf_counter()
try:
    from niwaki_flash_load import load
    model, processor = load(model_dir)
except Exception as exc:  # report the exact failure, this is a probe
    print(f"LOAD FAILED after {time.perf_counter()-t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
    raise
print(f"loaded (lazy) in {time.perf_counter()-t0:.1f}s; active {mx.get_active_memory()/2**30:.2f} GiB; "
      f"wired {vm_gb()[0]:.1f} GB available {vm_gb()[1]:.1f} GB", flush=True)

tok = getattr(processor, "tokenizer", processor)
messages = [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)

from mlx_vlm import generate  # noqa: E402

t1 = time.perf_counter()
try:
    res = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False, temperature=0.7)
except TypeError as exc:
    print(f"generate signature mismatch: {exc}", flush=True)
    res = generate(model, processor, prompt, max_tokens=max_tokens)
dt = time.perf_counter() - t1
text = getattr(res, "text", res)
print(f"generated in {dt:.1f}s; prompt_tps={getattr(res, 'prompt_tps', None)} "
      f"generation_tps={getattr(res, 'generation_tps', None)} "
      f"prompt_tokens={getattr(res, 'prompt_tokens', None)} gen_tokens={getattr(res, 'generation_tokens', None)} "
      f"peak_memory={getattr(res, 'peak_memory', None)}", flush=True)
print(f"active {mx.get_active_memory()/2**30:.2f} GiB peak {mx.get_peak_memory()/2**30:.2f} GiB; "
      f"watchdog peak wired {wd.peak_wired:.1f} GB, min available {wd.min_free:.1f} GB", flush=True)
print("--- text ---")
print(str(text)[:600])
