"""``python -m maxtoken`` — the same commands as the ``mt`` script.

This used to jump straight into the server, so it accepted serve flags but
silently ignored a subcommand: ``python -m maxtoken serve --model ...`` died on
"unrecognized arguments: serve" while every document showed ``mt serve``. The
trap was known well enough to be commented at the one call site that had to
avoid it. Both spellings now work.
"""

from .cli import main

assert __name__ == "__main__"

raise SystemExit(main())
