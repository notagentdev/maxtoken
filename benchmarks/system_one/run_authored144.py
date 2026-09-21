"""Score SemIf's authored-144 fixture with our System One head.

Emits predictions in SemIf's schema so their own evaluator produces the
number, which makes ours directly comparable to their published
`mean_family_balanced_accuracy`. The prompt is OURS, not theirs — this
measures our head on their fixture, not a replication of their system.

    python benchmarks/system_one/run_authored144.py <model> <out.jsonl> [gold.jsonl]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from maxtoken.system_one import SystemOne  # noqa: E402

GOLD = Path("/Users/dev/projects/semif/benchmarks/data/authored144.jsonl")


def main(model_path: str, out_path: str, gold: Path = GOLD) -> None:
    rows = [json.loads(line) for line in Path(gold).read_text().splitlines() if line.strip()]
    print(f"{len(rows)} rows; loading {model_path} ...", flush=True)
    head = SystemOne.load(model_path)

    out = Path(out_path)
    correct = 0
    started = time.perf_counter()
    with out.open("w") as sink:
        for i, row in enumerate(rows, 1):
            descriptions = [o["description"] for o in row["options"]]
            question = {"type": "choice", "prompt": row["question"],
                        "options": descriptions}
            decision = head.decide(row["state"], {"decision": question})
            answer = decision["decision"]
            probabilities = list(answer.distribution.values())
            if len(probabilities) != len(descriptions):
                raise ValueError(f"row {row['id']}: duplicate option descriptions")
            chosen = max(range(len(probabilities)), key=probabilities.__getitem__)
            correct += int(chosen == row["label"])
            sink.write(json.dumps({
                "id": row["id"],
                "option_ids": [o["id"] for o in row["options"]],
                "probabilities": probabilities,
            }) + "\n")
            if i % 24 == 0:
                print(f"  {i}/{len(rows)}  running accuracy {correct/i:.3f}", flush=True)

    seconds = time.perf_counter() - started
    print(f"plain accuracy {correct}/{len(rows)} = {correct/len(rows):.4f}")
    print(f"{seconds:.1f} s total, {seconds/len(rows)*1000:.0f} ms/row")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else GOLD)
