"""How much work MLX puts into one Metal command buffer.

MLX encodes the lazy graph into command buffers and commits one whenever it
holds more than a handful of ops or a few tens of megabytes of referenced
arrays; the GPU then idles at every boundary. A decode step on a deep model is
made of hundreds of small kernels over a couple of gigabytes of weights, so the
defaults cut it into dozens of buffers, each costing a round trip nobody sees
in a profile. Measured on Ornith-1.5-35B-A3B (40 layers, 4-bit, M1 Max):

    MLX_MAX_OPS_PER_BUFFER  MLX_MAX_MB_PER_BUFFER   ms/token
    default                 default                 14.45
    default                 2000                    13.2
    200                     2000                    12.0
    400                     2000                    11.9
    1000                    2000                    11.8

The two limits are read once, when the Metal device comes up, from the
process environment -- so they are set here as defaults before anything
touches MLX, and only when the environment does not already set them: a
value the user chose wins.
"""

from __future__ import annotations

import os

DEFAULTS = {
    "MLX_MAX_OPS_PER_BUFFER": "400",
    "MLX_MAX_MB_PER_BUFFER": "2000",
}


def apply_dispatch_defaults() -> dict[str, str]:
    """Install the defaults above for every limit the environment leaves unset.
    Returns what is in force afterwards."""
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in DEFAULTS}
