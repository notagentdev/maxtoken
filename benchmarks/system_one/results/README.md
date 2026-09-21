# Raw results

Every number quoted in [`../README.md`](../README.md) is reproducible from the
files here. `summary.json` carries them machine-readably; the rest are the
row-level outputs the summaries were computed from. `SHA256SUMS` covers all of
them (`cd results && shasum -a 256 -c SHA256SUMS`).

Measured 2026-09-21 on an Apple M1 Max with 32 GB, mlx-lm 0.31.3. Every run
used the letter-slot readout with the calibration temperature left at 1.0, so
these are raw model probabilities with no post-hoc correction.

| directory | contents |
|---|---|
| `tickets720/` | per-decision probabilities on the 720-row ticket fixture |
| `authored144/` | predictions and SemIf evaluator output on their 144 authored cases |
| `perturbations108/` | SemIf's stability report per model (read `systems.direct_logits`) |
| `wanli256/` | predictions and evaluator output on the rebuilt WANLI subset |
| `jevbench/` | `jb-*.json` summaries on the 231 public tasks, `reversal-*.json` the order-reversal runs on the hard tier |

`jb-ornith-opt.json` is the same Ornith run with narrow MLX command buffers
(40 ops / 256 MB): bit-identical quality, 8% slower at p50 and 27% at p95.
`jb-ornith-probs.json` additionally keeps the per-task distributions, which is
what `refit_temperature.py` reads.

The prediction files hold decision ids, option ids and probabilities — our own
measurements. They contain no ticket text, so nothing from the CC-BY-NC source
is redistributed here; rebuild the fixture with `build_tickets.py` and the
manifest to line them up again.
