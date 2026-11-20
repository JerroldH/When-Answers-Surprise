"""Resume an interrupted holdout WRDS export using event-ID batches."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.data.holdout_2025_recovery import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
