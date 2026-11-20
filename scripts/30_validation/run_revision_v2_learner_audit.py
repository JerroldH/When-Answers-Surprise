"""Run the focused revision-v2 A3 learner audit."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.empirical.revision_v2_learner_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
