"""How much work MLX puts into one Metal command buffer -- and why the worker
no longer decides that for you.

MLX encodes the lazy graph into command buffers and commits one whenever it
holds more than a handful of ops or a few tens of megabytes of freshly
allocated arrays (``MLX_MAX_OPS_PER_BUFFER``, ``MLX_MAX_MB_PER_BUFFER``, read
once from the environment when the Metal device comes up). The GPU idles at
every boundary, and a decode step on a deep model is hundreds of small
kernels, so raising the limits is worth a lot of throughput. Measured on
Ornith-1.5-35B-A3B (40 layers, 4-bit, M1 Max, in-process):

    MLX_MAX_OPS_PER_BUFFER  MLX_MAX_MB_PER_BUFFER   ms/token
    default                 default                 14.45
    default                 2000                    13.2
    400                     2000                    11.9

For a few hours on 2026-08-28 the worker set 400 / 2000 as defaults. Then
an agent drove the server with long prompts and the machine went down with
a kernel panic in the GPU driver -- ``IOGPUGroupMemory::remove_memory_object()
memory object not found`` -- with 28.7 GB wired at the moment of the panic
and 66 MB free. A command buffer references every buffer its kernels touch,
and the driver keeps those resident until the buffer completes: with hundreds
of ops and gigabytes of outputs per buffer, the in-flight working set grows
by whole layers of weights and whole prefill chunks of activations, and on a
machine that was already deep in swap that was the end of it. The elastic
residency that makes the mapped expert store safe on this machine (docs) is
exactly what a wide buffer defeats.

So nothing is set here any more. The knobs are yours: export the two
variables before ``mt serve`` if the machine has the headroom (a resident
model well inside physical memory, no other model processes), and measure
the wired memory while a long prompt prefills before trusting it.
"""

from __future__ import annotations

import os

KNOBS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")


def dispatch_limits() -> dict[str, str | None]:
    """The limits in force for this process (None = MLX's own default)."""
    return {name: os.environ.get(name) for name in KNOBS}
