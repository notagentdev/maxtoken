"""Ticket benchmark for the Ternary Bonsai 2 pack.

Bonsai keeps its weights in a Hadamard-rotated basis and transforms activations
inside every packed linear, so it cannot be loaded by `mlx_lm.load`; it ships
its own runtime. Everything above that — slots, shared prefill, scoring — is
unchanged, which is the point of keeping the head model-agnostic.

    ~/.venvs/bonsai/bin/python benchmarks/system_one/demo_tickets_bonsai.py
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "downloads" / "Ternary-Bonsai-2-27B-mlx-2bit"
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(PACK / "runtime"))
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.gpu)

from vision_artifact import load_vl_model  # noqa: E402

import demo_tickets  # noqa: E402
from maxtoken.system_one import SystemOne  # noqa: E402


class PackTokenizer:
    """The pack ships its chat template as a file, not in the tokenizer."""

    def __init__(self, tokenizer, template: str):
        self._tokenizer = tokenizer
        self._template = template

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("tokenize", False)
        return self._tokenizer.apply_chat_template(
            messages, chat_template=self._template, **kwargs)

    def encode(self, text, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, ids, **kwargs):
        return self._tokenizer.decode(ids, **kwargs)

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id


def main() -> None:
    t0 = time.perf_counter()
    model, processor, _ = load_vl_model(str(PACK))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    head = SystemOne(
        model.language_model,
        PackTokenizer(tokenizer, (PACK / "chat_template.jinja").read_text()),
    )
    print(f"loaded in {time.perf_counter() - t0:.1f} s, "
          f"active {mx.get_active_memory() / 2**30:.2f} GiB\n", flush=True)
    demo_tickets.main(str(PACK), head=head)


if __name__ == "__main__":
    main()
