"""Allow `python -m controlkit` to run the CLI."""

from __future__ import annotations

from controlkit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

