"""MaxToken inference runtime."""

import os as _os

# Before anything imports transformers (and with it torch): cap the native
# BLAS/OpenMP thread pools. Nothing in this runtime does torch compute, but
# libomp's default busy-wait worker pool steals cores from the MLX worker's
# graph encode -- measured on an M1 Max: single-stream decode 73 -> 85 tok/s
# and four concurrent streams 91 -> 129 just from pinning these. setdefault:
# an operator who really wants torch parallelism can still override.
for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("KMP_BLOCKTIME", "0"),
):
    _os.environ.setdefault(_var, _val)

from maxtoken.version import __version__


def _adopt_legacy_env() -> None:
    """Let a ``FREETOKEN_*`` variable still configure its ``MAXTOKEN_*`` successor.

    The environment names are a public contract — anything a user put in a shell
    profile, a launch script, or a service unit. The rename would have broken all
    of it silently, which is the worst way for a rename to fail: the process
    starts, the setting is simply ignored. Doing this once at import covers every
    read site, and the new name always wins where both are set.
    """
    for name, value in list(_os.environ.items()):
        if not name.startswith("FREETOKEN_"):
            continue
        _os.environ.setdefault("MAXTOKEN_" + name[len("FREETOKEN_"):], value)


_adopt_legacy_env()

__all__ = ["__version__"]
