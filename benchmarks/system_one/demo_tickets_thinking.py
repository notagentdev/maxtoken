"""Think first, then decide: a hybrid of free reasoning and typed readout.

System One answers in one forward pass and therefore cannot compute anything
on the way. This lets the model write a reasoning block first, then reads the
constrained answer from the context that block produced. It costs generation —
the whole point of the single-pass design — so the question is whether the
decisions get better enough to justify that.

    python benchmarks/system_one/demo_tickets_thinking.py <model> [max_think]
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from demo_tickets import QUESTIONS, TICKETS  # noqa: E402
from maxtoken.system_one import Question, SystemOne, ece  # noqa: E402

CLOSE = "</think>\n\n"


def think(head, text, sampler, max_tokens):
    """Generate the reasoning block, stopping at its closing tag."""
    produced = []
    for step in stream_generate(head.model, head.tokenizer, prompt=text,
                                max_tokens=max_tokens, sampler=sampler):
        produced.append(step.text)
        if "</think>" in "".join(produced[-4:]):
            break
    body = "".join(produced)
    return body.split("</think>")[0], len(produced)


def decide_after_thinking(head, state, name, spec, sampler, max_think):
    question = Question.from_spec(name, spec, head.use_slots)
    messages = [{"role": "system", "content": head.system},
                {"role": "user", "content": f"{state}\n\n{question.render()}"}]
    opened = head.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    reasoning, tokens = think(head, opened, sampler, max_think)
    full = opened + reasoning + CLOSE
    ids = head.tokenizer.encode(full, add_special_tokens=False)
    cache = make_prompt_cache(head.model)
    logits = head._forward(ids, cache)
    answer = head._answer(question, head._score(question, cache, logits[-1]))
    return answer, tokens


def main(model_path, max_think=256):
    head = SystemOne.load(model_path)
    sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20)

    correct = esc_correct = 0
    confidences, hits = [], []
    spent = 0
    started = time.perf_counter()
    for text, gold_cat, gold_esc in TICKETS:
        cat, n1 = decide_after_thinking(head, text, "category", QUESTIONS["category"],
                                        sampler, max_think)
        esc, n2 = decide_after_thinking(head, text, "escalate", QUESTIONS["escalate"],
                                        sampler, max_think)
        spent += n1 + n2
        hit = cat.value == gold_cat
        correct += hit
        esc_correct += esc.value == gold_esc
        confidences.append(cat.probability)
        hits.append(hit)
        print(f"{'ok ' if hit else 'MISS'} {str(cat.value):8s} p={cat.probability:.3f}  "
              f"esc={str(esc.value):5s} p={esc.probability:.3f}  "
              f"{n1 + n2:4d} think tok | {text[:44]}...", flush=True)

    n = len(TICKETS)
    seconds = time.perf_counter() - started
    print(f"\ncategory accuracy   {correct}/{n}")
    print(f"escalation accuracy {esc_correct}/{n}")
    print(f"thinking tokens     {spent / n:.0f} per ticket (2 questions)")
    print(f"time                {seconds / n * 1000:.0f} ms/ticket")
    print(f"ECE                 {ece(confidences, hits):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 256)
