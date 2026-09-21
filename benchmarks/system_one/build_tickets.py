"""Build a balanced ticket fixture from Tobi-Bueck/customer-support-tickets.

The twelve authored tickets saturate: every model from 4B up scores 12/12 on
category, so the set cannot separate them. This draws a stratified sample with
FOREIGN labels instead — the dataset's own type, priority and queue — so the
benchmark measures agreement with the data's annotators, not with us.

Three question families per ticket, 20 tickets per (type x priority) cell:
type is the four-way call, priority is the graded one, queue is the eight-way
routing decision that exercises more answer slots than anything else we run.

The source is CC-BY-NC-4.0 and synthetic. Neither it nor the fixture belongs
in this repository; the manifest of selected rows and this script do, which is
enough to rebuild it byte-for-byte.

    python benchmarks/system_one/build_tickets.py
"""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "sources" / "tickets.csv"
FIXTURE = HERE / "sources" / "tickets720.jsonl"
MANIFEST = HERE / "manifests" / "tickets720-manifest.json"
SEED = 20260921
PER_CELL = 20

TYPES = {
    "Incident": "Something that worked before is broken now for this customer.",
    "Request": "The customer wants something provided, enabled or explained; nothing is broken.",
    "Problem": "An underlying or recurring cause behind repeated failures, not a single outage.",
    "Change": "The customer asks for a system, plan or configuration to be modified.",
}
PRIORITIES = {
    "low": "Can wait for the normal queue; no one is blocked.",
    "medium": "Should be handled soon; work is hindered but not stopped.",
    "high": "Needs attention now; the customer is blocked or money is at stake.",
}
QUEUES = {
    "Technical Support": "Technical faults in the product itself.",
    "Product Support": "How the product is meant to be used, and its features.",
    "Customer Service": "Accounts, orders and general customer care.",
    "IT Support": "The customer's own devices, access and internal IT.",
    "Billing and Payments": "Invoices, charges, refunds and payment methods.",
    "Returns and Exchanges": "Sending goods back or swapping them.",
    "Service Outages and Maintenance": "Planned or unplanned service interruptions.",
    "Sales and Pre-Sales": "Buying decisions, quotes and pre-purchase questions.",
}

FAMILIES = (
    ("ticket_type", "type", TYPES,
     "Which kind of ticket is this?"),
    ("priority", "priority", PRIORITIES,
     "How urgently does this need to be handled?"),
    ("queue", "queue", QUEUES,
     "Which department should handle this ticket?"),
)


def usable(row):
    body = (row.get("body") or "").strip()
    return (
        (row.get("language") or "").strip().lower() == "en"
        and body and (row.get("subject") or "").strip()
        and 120 <= len(body) <= 2500
        and (row.get("type") or "").strip() in TYPES
        and (row.get("priority") or "").strip() in PRIORITIES
        and (row.get("queue") or "").strip() in QUEUES
    )


def main() -> None:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with SOURCE.open(newline="") as handle:
        rows = [(i, r) for i, r in enumerate(csv.DictReader(handle))]
    pool = [(i, r) for i, r in rows if usable(r)]

    cells = defaultdict(list)
    for i, r in pool:
        cells[(r["type"].strip(), r["priority"].strip())].append((i, r))
    rng = random.Random(SEED)
    picked = []
    for key in sorted(cells):
        bucket = sorted(cells[key], key=lambda p: p[0])
        rng.shuffle(bucket)
        if len(bucket) < PER_CELL:
            raise ValueError(f"cell {key} has only {len(bucket)} rows")
        picked.extend(bucket[:PER_CELL])
    picked.sort(key=lambda p: p[0])

    out = []
    for index, row in picked:
        state = f"Subject: {row['subject'].strip()}\n\n{row['body'].strip()}"
        group = f"ticket-{index:05d}"
        for family, column, labels, prompt in FAMILIES:
            gold = row[column].strip()
            options = [{"id": k, "description": v} for k, v in labels.items()]
            out.append({
                "id": f"{group}-{family}",
                "family": family,
                "group_id": group,
                "state": state,
                "question": prompt,
                "options": options,
                "label": list(labels).index(gold),
                "target_distribution": None,
                "split": "test",
                "provenance": {
                    "source": "Tobi-Bueck/customer-support-tickets",
                    "source_file": SOURCE.name,
                    "source_sha256": digest,
                    "source_row": index,
                    "license": "cc-by-nc-4.0",
                    "kind": "synthetic_third_party_labelled",
                    "label_basis": f"dataset column {column!r}, unmodified",
                    "rights": "not redistributed; rebuild with build_tickets.py",
                },
            })

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    MANIFEST.write_text(json.dumps({
        "source": "Tobi-Bueck/customer-support-tickets",
        "source_file": SOURCE.name,
        "source_sha256": digest,
        "seed": SEED,
        "per_cell": PER_CELL,
        "tickets": len(picked),
        "rows": len(out),
        "families": [f[0] for f in FAMILIES],
        "selected_source_rows": [i for i, _ in picked],
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print(f"{len(picked)} tickets x {len(FAMILIES)} questions = {len(out)} rows")
    print(f"wrote {FIXTURE}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
