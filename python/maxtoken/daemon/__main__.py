from __future__ import annotations

from maxtoken.daemon import main  # package dispatch: client verb → client, else → server

raise SystemExit(main(prog="python -m maxtoken.daemon"))
