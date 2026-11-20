"""Run diagnostics over existing S&P 500 LM tone baseline outputs.

This stage reuses baseline artifacts and writes diagnostic comparators. It does
not rescore transcripts or rebuild the baseline sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.sp500_lm_diagnostics import DiagnosticConfig, run_lm_diagnostics  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
ANALYSIS_ROOT = DATA_ROOT / "analysis"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnostics over existing S&P 500 LM tone outputs."
    )
    parser.add_argument(
        "--sample",
        choices=["full", "smoke"],
        default="full",
        help="Use existing full or smoke LM tone outputs.",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parity-tolerance", type=float, default=1e-8)
    parser.add_argument("--min-train-rows", type=int, default=120)
    args = parser.parse_args()

    suffix = "" if args.sample == "full" else "_smoke"
    input_dir = args.input_dir or (ANALYSIS_ROOT / f"lm_baseline{suffix}")
    output_dir = args.output_dir or (ANALYSIS_ROOT / f"lm_baseline{suffix}_diagnostics")

    summary = run_lm_diagnostics(
        DiagnosticConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            parity_tolerance=args.parity_tolerance,
            min_train_rows=args.min_train_rows,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
