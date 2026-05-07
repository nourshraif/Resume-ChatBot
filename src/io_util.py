"""Console helpers for Windows (cp1252) vs Unicode resume text."""

import sys


def configure_utf8_stdout() -> None:
    """Avoid UnicodeEncodeError when printing Arabic, bullets, etc."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
