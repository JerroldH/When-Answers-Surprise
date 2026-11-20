"""Build the CIZ bridge and outcome-blind inputs for the 2025 holdout."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.empirical.holdout_2025 import build_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(build_main())
