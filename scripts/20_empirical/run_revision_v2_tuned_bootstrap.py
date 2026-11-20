"""Run the chunked tuned-Ridge first-stage refit bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from when_answers_surprise.empirical.revision_v2_tuned_bootstrap import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
