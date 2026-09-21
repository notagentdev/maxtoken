"""Exercise the System One head on a small labelled ticket set.

Reports accuracy, latency, calibration error before and after a temperature
refit, and the full distribution for one case so the numbers are inspectable.

    python benchmarks/system_one/demo_tickets.py downloads/<model>
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from maxtoken.system_one import SystemOne, ece, fit_temperature  # noqa: E402

QUESTIONS = {
    "category": {
        "type": "choice",
        "prompt": "What kind of ticket is this?",
        "options": ["bug", "feature", "question", "spam"],
    },
    "escalate": {
        "type": "bool",
        "prompt": "Does this need escalation to an engineer today?",
    },
    "severity": {"type": "score", "prompt": "How severe is it?", "min": 0, "max": 10},
}

# (text, category, escalate)
TICKETS = [
    ("Checkout returns a 500 for every card payment since the 14:00 deploy. "
     "No orders are going through at all.", "bug", True),
    ("Could you add a dark mode to the dashboard? The white background is "
     "rough during night shifts.", "feature", False),
    ("Where do I find the invoice for last month? I looked under Billing but "
     "only see the current one.", "question", False),
    ("EARN $$$ FROM HOME! Click here for the secret method the banks hate.",
     "spam", False),
    ("The export button spins forever on datasets above ~50k rows. Smaller "
     "exports finish fine.", "bug", False),
    ("Please support SAML single sign-on, our security team requires it "
     "before we can roll out further.", "feature", False),
    ("All users in the EU region are locked out, login returns 'tenant not "
     "found'. This is production.", "bug", True),
    ("Is there an API rate limit on the search endpoint, and if so what is "
     "it?", "question", False),
    ("Hi dear, I represent a prince and require your assistance with a "
     "transfer of forty million.", "spam", False),
    ("Data loss: deleting one project also removed two unrelated projects for "
     "three of our customers.", "bug", True),
    ("It would be nice if the CSV export kept the column order from the "
     "table view.", "feature", False),
    ("Does the retention setting apply retroactively to data we already "
     "stored?", "question", False),
]


def main(model_path: str, head=None) -> None:
    if head is None:
        print(f"loading {model_path} ...", flush=True)
        t0 = time.perf_counter()
        head = SystemOne.load(model_path)
        print(f"loaded in {time.perf_counter() - t0:.1f} s\n", flush=True)

    rows, correct, confidences = [], [], []
    esc_correct = 0
    prefill_ms = decide_ms = 0.0

    for text, gold_cat, gold_esc in TICKETS:
        d = head.decide(text, QUESTIONS)
        cat = d["category"]
        hit = cat.value == gold_cat
        correct.append(hit)
        confidences.append(cat.probability)
        esc_correct += int(d["escalate"].value == gold_esc)
        prefill_ms += d.prefill_ms
        decide_ms += d.decide_ms
        rows.append(
            (list(QUESTIONS["category"]["options"]),
             [cat.distribution[o] for o in QUESTIONS["category"]["options"]],
             gold_cat)
        )
        mark = "ok " if hit else "MISS"
        print(f"{mark} {cat.value:<8} p={cat.probability:.3f}  "
              f"esc={str(d['escalate'].value):<5} p={d['escalate'].probability:.3f}  "
              f"sev={d['severity'].value:<3} E={d['severity'].expectation:.1f}  "
              f"| {text[:52]}...", flush=True)

    n = len(TICKETS)
    print(f"\ncategory accuracy   {sum(correct)}/{n}")
    print(f"escalation accuracy {esc_correct}/{n}")
    print(f"prefill  {prefill_ms / n:.0f} ms/ticket ({rows and ''}shared across "
          f"{len(QUESTIONS)} questions)")
    print(f"decide   {decide_ms / n:.0f} ms/ticket  -> {(prefill_ms + decide_ms) / n:.0f} "
          f"ms total per ticket")

    print(f"\nECE before refit    {ece(confidences, correct):.3f}")
    # The head stores raw candidate log-probabilities implicitly; recover them
    # from the distributions for the refit.
    import math
    scores = [[math.log(max(p, 1e-12)) for p in probs] for _, probs, _ in rows]
    labels = [opts.index(gold) for opts, _, gold in rows]
    t = fit_temperature(scores, labels)
    head.temperature = t
    refit = []
    for (opts, probs, gold), hit in zip(rows, correct):
        scaled = [math.log(max(p, 1e-12)) / t for p in probs]
        top = max(scaled)
        z = sum(math.exp(s - top) for s in scaled)
        refit.append(math.exp(max(scaled) - top) / z)
    print(f"fitted temperature  {t:.2f}")
    print(f"ECE after refit     {ece(refit, correct):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "downloads/Qwen3.8-2B-Distill")
