"""Freeze the completed A3 learner-audit sensitivity package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from when_answers_surprise.empirical.a3_learner_audit_freeze import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
