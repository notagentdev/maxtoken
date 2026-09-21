"""Two-fold temperature refit on a saved System One run.

A scalar fitted on the cases it is then scored on always looks good, so the
231 tasks are split in half by a seeded shuffle: fit on one half, measure on
the other, then swap. Reported numbers are out-of-fold only.

    python benchmarks/system_one/refit_temperature.py /tmp/jb-<model>-probs.json
"""

import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, "/Users/dev/projects/jevbench")

from jevbench.metrics import ece_top_label  # noqa: E402

from maxtoken.system_one import fit_temperature  # noqa: E402


def rescale(probs, t):
    scaled = [math.log(max(p, 1e-12)) / t for p in probs]
    top = max(scaled)
    exp = [math.exp(s - top) for s in scaled]
    total = sum(exp)
    return [e / total for e in exp]


def measure(rows):
    pairs, brier = [], []
    for r in rows:
        gold = r["labels"].index(str(r["expected"]))
        best = max(range(len(r["probs"])), key=r["probs"].__getitem__)
        pairs.append((r["probs"][best], best == gold))
        brier.append(sum((p - (i == gold)) ** 2 for i, p in enumerate(r["probs"])))
    return (ece_top_label(pairs)["ece"], sum(brier) / len(brier),
            sum(c for _, c in pairs) / len(pairs))


def main(path: str) -> None:
    data = json.loads(Path(path).read_text())
    rows = [r for r in data["per_task"] if r["expected"] is not None]
    print(f"{len(rows)} scorable tasks from {path}")

    ece_raw, brier_raw, acc = measure(rows)
    print(f"\nraw          ECE {ece_raw:.4f}   Brier {brier_raw:.4f}   accuracy {acc:.4f}")

    order = list(range(len(rows)))
    random.Random(20260920).shuffle(order)
    folds = [[rows[i] for i in order[::2]], [rows[i] for i in order[1::2]]]

    out = []
    for i, (fit_rows, test_rows) in enumerate(((folds[0], folds[1]), (folds[1], folds[0]))):
        scores = [[math.log(max(p, 1e-12)) for p in r["probs"]] for r in fit_rows]
        labels = [r["labels"].index(str(r["expected"])) for r in fit_rows]
        t = fit_temperature(scores, labels)
        scaled = [{**r, "probs": rescale(r["probs"], t)} for r in test_rows]
        e, b, a = measure(scaled)
        print(f"fold {i}: T={t:.2f} fitted on {len(fit_rows)}  ->  "
              f"held-out ECE {e:.4f}  Brier {b:.4f}  accuracy {a:.4f}")
        out.extend(scaled)

    e, b, a = measure(out)
    print(f"\nout-of-fold  ECE {e:.4f}   Brier {b:.4f}   accuracy {a:.4f}")
    print(f"             ECE {ece_raw - e:+.4f}   Brier {brier_raw - b:+.4f} "
          "(positive = the refit helped)")
    print("\nAccuracy cannot change: a temperature is monotone, so the argmax is fixed.")


if __name__ == "__main__":
    main(sys.argv[1])
