# System One decisions — head and benchmarks

`python/maxtoken/system_one.py` answers typed questions about a state by
scoring **only the allowed answers** and softmaxing over that set. No token is
generated, so a structured-output error is impossible by construction and a
decision costs one forward pass instead of an answer sentence.

Options are presented as uppercase letter slots (`A.`, `B.`, …) and only those
single tokens are scored. Two invariants are checked at run time and raise
rather than produce quiet nonsense: every letter must be one exact round-trip
token, and appending it must not re-tokenize the end of the prompt.

All questions about one state share a single prefill; each question then runs
from a fork of that cache. Where the questions are of similar length they are
scored in **one batched forward pass** instead (right padding, each row read at
its own last real position); the head falls back to the sequential path when
padding would waste more than `batch_waste` (default 1.3) of the real tokens.

## The ticket benchmark (the one that matters here)

`build_tickets.py` draws a stratified sample from
[`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets):
20 tickets per (type × priority) cell, English only, 120–2500 characters of
body, giving **240 tickets × 3 questions = 720 labelled decisions**.

The labels are the dataset's own `type`, `priority` and `queue` columns,
unmodified. That is the point: the benchmark measures agreement with a third
party's annotation, not with ours. The source is synthetic and CC-BY-NC-4.0,
so neither it nor the built fixture is committed — `manifests/tickets720-manifest.json`
holds the seed, the source checksum and the selected row numbers, which is
enough to rebuild it byte-for-byte.

```bash
curl -sL -o benchmarks/system_one/sources/tickets.csv \
  https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/dataset-tickets-multi-lang-4-20k.csv
python benchmarks/system_one/build_tickets.py
python benchmarks/system_one/run_authored144.py <model> out.jsonl \
  benchmarks/system_one/sources/tickets720.jsonl
```

### Metric

The headline is the **mean of the two learnable families**, `ticket_type`
(4 options) and `queue` (8 options). `priority` is reported beside it and
deliberately excluded: its ceiling is 0.400 against a 0.333 chance level even
for the 27B, while the same model goes from 0.408 to 0.688 on `ticket_type`.
A family where a far stronger model does not improve carries no signal in the
text; folding it into one number would only dilute the others.

### Measured, 2026-09-21, M1 Max 32 GB

| model | ticket_type | queue | **headline** | priority | ms/row |
|---|---|---|---|---|---|
| chance | 0.250 | 0.125 | **0.188** | 0.333 | |
| Qwen3.5-4B-MTPLX 4bit | 0.408 | 0.254 | 0.331 | 0.375 | 439 |
| Qwen3.8-2B-Distill 4bit | 0.467 | 0.275 | 0.371 | 0.350 | **178** |
| Qwen3.8-9B-Distill 4bit | 0.554 | 0.292 | 0.423 | 0.408 | 2159 |
| Qwen3.8-27B-MTPLX 4bit | **0.688** | 0.279 | 0.483 | 0.400 | 12004 |
| **Ornith-1.5-35B-A3B 4bit** | 0.679 | **0.300** | **0.490** | 0.392 | **599** |

Two results worth keeping:

**Ornith matches the dense 27B (0.490 vs 0.483) at a twentieth of the cost.**
A System One decision pays only prefill, and prefill scales with the active
parameters, not the stored ones: 3B active against 27B dense. The full 720-row
run took Ornith 7 minutes and the 27B 2 hours 24 minutes. At 12 s per decision
a three-question ticket would cost the 27B 36 seconds, which is not a usable
product.

**The ranking is workload-dependent.** The 4B is last here, below the 2B,
although it beats it by 21 points on SemIf's `authored144`. Its prediction
histogram explains it: against 60 true cases per class it answers `Request`
169 times and `Problem` 5 times. Recognising `Problem` at all only appears in
the large models (47 and 65 against 5 and 2). Choosing a model on one fixture
is how you end up four places out on another.

## External fixtures used for calibration of the above

Run through the same head, with the authors' own evaluators, so the numbers sit
beside their published ones.

| suite | rows | what it adds |
|---|---|---|
| SemIf `authored144` | 144 | their published Qwen3.5-4B figure is 0.8132; ours is 0.8378 |
| SemIf `perturbations108` | 108 | option reversal, criterion paraphrase, irrelevant context |
| WANLI (rebuilt, checksum-verified) | 256 | external check; theirs 0.637, ours 0.6321 — i.e. we reproduce them |
| JevBench public tasks | 231 | per-task outcomes for 25 systems on identical items |

On JevBench's 111 hard tasks our 27B reaches 0.7568 against Jev 1.13.0's
0.7297, with brier_hard 0.3414 against 0.3396 — indistinguishable. Ornith
reaches 0.7027 at 457 ms, which is Jev's latency class on a laptop.

## Order reversal — read this before believing any of it

`jevbench_reversal.py` scores every hard task twice, options as given and
reversed. Every published number in this field, ours included, comes from a
single option order.

| model | forward | flips | order-robust | agree | accuracy when agreeing |
|---|---|---|---|---|---|
| Qwen3.5-4B | 0.6306 | 50/111 | 0.6081 | 55% | 0.7377 |
| Gemma-4-E4B | 0.4955 | 23/111 | 0.4820 | 79% | 0.5227 |
| Qwen3.8-9B | 0.6306 | 30/111 | 0.5946 | 73% | 0.6790 |
| Ornith-35B-A3B | 0.7027 | 32/111 | 0.6667 | 71% | 0.7722 |
| Qwen3.8-27B | 0.7568 | 23/111 | 0.7252 | 79% | 0.8182 |

Our 27B's 0.7568 becomes 0.6937 reversed, so "matches Jev" is a tie at best —
and ours is the only number in that comparison that was tested this way.

**A low flip rate is not quality.** Gemma flips as rarely as the 27B and agrees
with itself as often, but is right 0.5227 of the time when it agrees against
the 27B's 0.8182: it is rigid, not grounded. Order agreement works as an
automation gate only on a calibrated model — on Ornith it lifts 0.7027 to
0.7722 at 71% coverage for two 457 ms passes, which beats the 27B ungated.

## Two traps that produce quiet nonsense

**Put the format instruction after the options.** Before them, the answer
position is dominated by markdown scaffolding and only ~9% of the mass sits on
a real option, so the restricted softmax normalises over noise. Never pre-seed
an `"Answer: "` prefix either; the model reads it as a numbered list and emits
a digit.

**Candidate sets must be prefix-free.** `"1"` is a token prefix of `"10"`, so
scoring it bare hands it every continuation's mass and the argmax sticks at 1
on every model. Letter slots avoid this; the string path detects it and closes
each candidate with the end-of-turn token.

## Calibration

`refit_temperature.py` fits one scalar on half the cases and measures it on the
other, then swaps. On Ornith's hard tier ECE goes 0.1755 → 0.1107/0.0860 per
fold with T ≈ 1.75–2.20, i.e. the model is overconfident — but Jev's published
0.0606 is not reached. About half of that calibration gap is a scalar; the rest
is not. Note that pooling the two folds reports a *lower* ECE than either fold,
because opposite-direction miscalibration cancels in the bins; trust the folds.

## Letter slots or the answer strings themselves

`SYSTEM_ONE_SLOTS=0` scores the option strings directly instead of `A`/`B`/`C`,
over a token trie: one forward pass per divergence node rather than one per
candidate, and only a candidate that is a strict prefix of another is closed
with the end-of-turn token. Verified against per-candidate scoring on an
11-value score question: identical to 0.000e+00, distribution summing to
1.000000.

It was built on the expectation that removing the letter indirection would
help. Measured on the 720-decision ticket fixture, it does the opposite
depending on the model:

| | ticket_type | queue | headline | ms/row |
|---|---|---|---|---|
| Qwen3.5-4B, slots | 0.408 | 0.254 | 0.331 | 439 |
| Qwen3.5-4B, trie | **0.487** | 0.267 | **0.377** | 1165 |
| Ornith-35B-A3B, slots | **0.679** | **0.300** | **0.490** | **599** |
| Ornith-35B-A3B, trie | 0.546 | 0.267 | 0.406 | 1453 |

The trie gains the weak model 4.6 points and costs the strong one 8.4. The
prediction histograms say why: against 60 true cases per class Ornith answers
`Request` 97 times with slots and 111 times with the strings, i.e. it falls
back towards the class prior carried by the answer words themselves. Letter
slots break that prior — the meaning lives in the prompt's option list and is
never scored — which helps a model that reads the context and hurts one that
cannot reliably map a letter to a meaning.

Slots therefore stay the default: they win on the best model and are 2.4x
faster. The trie is kept because it lifts the 16-option ceiling and scores
arbitrary answer strings exactly, and because the choice is now measured
rather than assumed.
