"""Option-order falsification on JevBench's hard tier.

Scores every hard task twice in one process — options as given and in reverse
display order — and reports whether the decision survives. JevBench documented
one entrant falling from 72% to 21% under exactly this change, so a score that
does not survive it is a measurement of position bias, not of judgement.

    python benchmarks/system_one/jevbench_reversal.py <model> <out.json>
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
from jevbench.tasks import load_jsonl  # noqa: E402

from maxtoken.system_one import SystemOne  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_jevbench import load_head, options_for  # noqa: E402


def score_once(head, task, pairs):
    state = task.state if isinstance(task.state, str) else json.dumps(
        task.state, ensure_ascii=False)
    question = {"type": "choice", "prompt": task.question["instructions"],
                "options": [f"{k}: {d}" for k, d in pairs]}
    answer = head.decide(state, {"decision": question})["decision"]
    probabilities = list(answer.distribution.values())
    probs = {label: p for (label, _), p in zip(pairs, probabilities)}
    return probs, score_task(probs, task)


def main(model_path: str, out_path: str) -> None:
    tasks = load_jsonl(str(BENCH / "datasets/public/hard.jsonl"))
    print(f"{len(tasks)} hard tasks, scored twice; loading {model_path} ...", flush=True)
    head = load_head(model_path)

    rows = []
    started = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        pairs = options_for(task)
        forward, fs = score_once(head, task, pairs)
        reverse, rs = score_once(head, task, list(reversed(pairs)))
        # Total variation distance between the two distributions, aligned by label.
        tv = 0.5 * sum(abs(forward[k] - reverse[k]) for k in forward)
        rows.append({
            "id": task.id, "family": task.family,
            "forward_correct": bool(fs["correct"]), "reverse_correct": bool(rs["correct"]),
            "forward_pred": fs.get("predicted"), "reverse_pred": rs.get("predicted"),
            "flipped": fs.get("predicted") != rs.get("predicted"),
            "tv_distance": tv,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(tasks)}", flush=True)

    n = len(rows)
    fwd = sum(r["forward_correct"] for r in rows)
    rev = sum(r["reverse_correct"] for r in rows)
    flips = sum(r["flipped"] for r in rows)
    tvs = sorted(r["tv_distance"] for r in rows)
    Path(out_path).write_text(json.dumps({
        "model": model_path, "n": n, "forward_correct": fwd, "reverse_correct": rev,
        "argmax_flips": flips, "mean_tv": sum(tvs) / n, "max_tv": tvs[-1],
        "rows": rows,
    }, indent=2) + "\n")

    print(f"\nforward   {fwd}/{n} = {fwd/n:.4f}")
    print(f"reversed  {rev}/{n} = {rev/n:.4f}   ({(rev-fwd)/n:+.4f})")
    print(f"argmax flips      {flips}/{n} = {flips/n:.3f}")
    print(f"mean TV distance  {sum(tvs)/n:.3f}   max {tvs[-1]:.3f}")
    print(f"{(time.perf_counter()-started)/60:.1f} min; wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
