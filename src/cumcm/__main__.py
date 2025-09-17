"""Entrypoint for ``python -m cumcm`` delegating to the CLI."""

from __future__ import annotations

from .cli import main


def _run() -> int:
    return main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_run())
