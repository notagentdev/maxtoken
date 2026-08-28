"""How much work MLX puts into one Metal command buffer, and the lesson that
came with choosing it.

MLX encodes the lazy graph into command buffers and commits one whenever it
holds more than a handful of ops or a few tens of megabytes of freshly
allocated arrays (``MLX_MAX_OPS_PER_BUFFER``, ``MLX_MAX_MB_PER_BUFFER``, read
once from the environment when the Metal device comes up). The GPU idles at
every boundary, and a decode step on a deep model is hundreds of small
kernels, so the limits are worth a lot of throughput. Measured on
Ornith-1.5-35B-A3B (40 layers, 4-bit, M1 Max, in-process):

    MLX_MAX_OPS_PER_BUFFER  MLX_MAX_MB_PER_BUFFER   ms/token
    default                 default                 14.5
    400                      256                    12.7
    400                     1024                    12.2
    400                     2000                    11.9

The first time the worker set them (400 / 2000, 2026-08-28) an agent's long
prompts took the machine down in a GPU-driver kernel panic --
``IOGPUGroupMemory::remove_memory_object() memory object not found`` -- with
28.7 GB wired and 66 MB free. Everything an uncompleted command buffer
references stays wired, a prefill chunk allocates gigabytes, the encoder runs
far ahead of the GPU, and the prefix store was allowed 15% of RAM on top of
an 18 GB model. Wide buffers were the last straw, not the only one.

What ships now is the combination that measured safe: 400 ops / 1024 MB
here, a prefill that evaluates every few layers so its in-flight set stays
bounded whatever these limits allow (prefill_pacing.py; peak 23.0 GB on an
8 192-token prompt against 21.8 at MLX's defaults and >26 unpaced), and a
prefix store budgeted from the memory actually left after the model loaded
(worker). A value the environment already sets always wins over these.
"""

from __future__ import annotations

import os

DEFAULTS = {
    "MLX_MAX_OPS_PER_BUFFER": "400",
    "MLX_MAX_MB_PER_BUFFER": "1024",
}

KNOBS = tuple(DEFAULTS)


def apply_dispatch_defaults() -> dict[str, str]:
    """Install the defaults above for every limit the environment leaves unset.
    Must run before the Metal device comes up. Returns what is in force."""
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in KNOBS}


def dispatch_limits() -> dict[str, str | None]:
    """The limits in force for this process (None = MLX's own default)."""
    return {name: os.environ.get(name) for name in KNOBS}
