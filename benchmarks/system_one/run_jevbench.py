"""Score JevBench's public tasks with our System One head.

Uses JevBench's own `tasks`, `scoring` and `summarize` modules, so accuracy,
Brier, ECE and paraphrase consistency are their definitions, not ours. Cost is
deliberately absent: a local model has no per-token tariff, and JevBench's own
runner leaves that null rather than plot a zero.

The request mapping follows their SemIf adapter: every option is rendered as
"<label>: <description>" and the label set is exactly `task.labels`.

    python benchmarks/system_one/run_jevbench.py <model> <out.json>
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = Path("/Users/dev/projects/jevbench")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(BENCH))

from jevbench.scoring import score_task  # noqa: E402
from jevbench.summarize import summarize  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

from maxtoken.system_one import SystemOne  # noqa: E402

SPLITS = ("easy", "original", "hard")


def load_head(model_path: str) -> SystemOne:
    """Strict load, falling back for checkpoints carrying unused tensors.

    Gemma 4 shares KV across its last `num_kv_shared_layers` layers, so the
    converter's copies of their k/v projections are dead weight mlx-lm does
    not declare. The fallback is announced, never silent.
    """
    try:
        return SystemOne.load(model_path)
    except ValueError as error:
        if "not in model" not in str(error):
            raise
        count = str(error).split("Received ", 1)[-1].split(" ", 1)[0]
        print(f"note: {count} weights not declared by the model class; "
              "retrying with strict=False", flush=True)
        return SystemOne.load(model_path, strict=False)


def options_for(task):
    """(label, description) pairs exactly as JevBench's SemIf adapter builds them."""
    kind = task.question["type"]
    criteria = task.question.get("criteria")
    if kind == "noul":
        return [(k, (criteria or {}).get(k) or f"The proposition is {k}.")
                for k in task.labels]
    if kind == "choice":
        return [(k, (criteria or {}).get(k) or k) for k in task.labels]
    return [(str(i), criteria[i]) for i in range(len(criteria))]


def main(model_path: str, out_path: str) -> None:
    tasks = []
    for name in SPLITS:
        tasks.extend(load_jsonl(str(BENCH / "datasets" / "public" / f"{name}.jsonl")))
    print(f"{len(tasks)} public tasks; loading {model_path} ...", flush=True)
    head = load_head(model_path)

    records = []
    correct = 0
    for i, task in enumerate(tasks, 1):
        pairs = options_for(task)
        state = task.state if isinstance(task.state, str) else json.dumps(
            task.state, ensure_ascii=False)
        question = {"type": "choice", "prompt": task.question["instructions"],
                    "options": [f"{k}: {d}" for k, d in pairs]}
        started = time.perf_counter()
        answer = head.decide(state, {"decision": question})["decision"]
        latency = time.perf_counter() - started
        probabilities = list(answer.distribution.values())
        if len(probabilities) != len(pairs):
            raise ValueError(f"{task.id}: duplicate option descriptions")
        probs = {label: p for (label, _), p in zip(pairs, probabilities)}
        scored = score_task(probs, task)
        correct += bool(scored["correct"])
        records.append({
            "task_id": task.id, "family": task.family, "split": task.split,
            "group": task.group, "ts": time.time(), "status": "ok", "ok": True,
            "valid": scored["valid"], "correct": scored["correct"],
            "predicted": scored.get("predicted"), "ordinal_ev": scored.get("ordinal_ev"),
            "probs": scored.get("probs"), "probs_as_returned": probs,
            "strict_valid": scored.get("strict_valid", False),
            "renormalized": scored.get("renormalized", False),
            "probs_source": "native", "model": model_path, "error": None,
            "schema_error": scored.get("error"), "status_code": None,
            "latency_s": latency, "usage": {}, "cost_usd": None,
            "cost_basis": "self_hosted_local_weights",
        })
        if i % 40 == 0:
            print(f"  {i}/{len(tasks)}  running accuracy {correct/i:.3f}", flush=True)

    summary = summarize(tasks, records)
    # Keep the raw distributions beside the aggregates: a temperature refit
    # needs them, and re-running the model to recover them is the expensive
    # way to learn something the first pass already knew.
    summary["per_task"] = [
        {"id": t.id, "family": t.family, "labels": list(t.labels),
         "expected": t.expected,
         "probs": [r["probs_as_returned"][l] for l in t.labels]}
        for t, r in zip(tasks, records)
    ]
    Path(out_path).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    print(f"\nmacro accuracy      {summary['macro_accuracy']:.4f}")
    print(f"plain accuracy      {summary['accuracy']:.4f}  ({summary['n_correct']}/{summary['n_scorable']})")
    print(f"schema validity     {summary['schema_validity']:.4f}")
    print(f"brier mean          {summary['brier_mean']:.4f}")
    print(f"ECE                 {summary['ece']['ece']:.4f}")
    if summary.get("ordinal_mae") is not None:
        print(f"ordinal MAE         {summary['ordinal_mae']:.4f}")
    pc = summary["paraphrase_consistency"]
    print(f"paraphrase agree    {pc['agreement']} on {pc['both_valid']} pairs")
    lat = summary["latency"]
    print(f"latency p50/p95     {lat['p50_s']*1000:.0f} / {lat['p95_s']*1000:.0f} ms"
          f"  over {lat['n']} decisions")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
