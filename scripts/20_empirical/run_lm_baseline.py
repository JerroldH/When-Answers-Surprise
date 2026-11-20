"""Run the S&P 500 earnings-call LM tone baseline analysis.

This starts the empirical analysis layer after WRDS staging is complete. It
writes baseline outputs used by diagnostics and paper-result freezing; it is not
a lightweight import check.
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

from when_answers_surprise.sp500_lm import LMBaselineConfig, run_lm_pipeline  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
OUTPUT_DIR = DATA_ROOT / "analysis" / "lm_baseline"
SMOKE_OUTPUT_DIR = DATA_ROOT / "analysis" / "lm_baseline_smoke"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "wrds_sp500_lm"
SMOKE_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "wrds_sp500_lm_smoke"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the S&P 500 earnings-call LM tone baseline."
    )
    parser.add_argument("--sample-end", default="2024-12-31")
    parser.add_argument(
        "--max-events-per-year",
        type=int,
        default=None,
        help="Optional smoke-test cap per event year.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=120,
        help="Minimum prior rows for expanding first-stage folds.",
    )
    parser.add_argument(
        "--no-model-robustness",
        action="store_true",
        help="Skip ElasticNet/HGB first-stage comparison.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=None,
        help="Optional isolated transcript/IBES panel; default uses data_root/derived.",
    )
    args = parser.parse_args()

    smoke_mode = args.max_events_per_year is not None
    config = LMBaselineConfig(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        output_dir=args.output_dir or (SMOKE_OUTPUT_DIR if smoke_mode else OUTPUT_DIR),
        figures_dir=args.figures_dir or (SMOKE_FIGURES_DIR if smoke_mode else FIGURES_DIR),
        sample_end=args.sample_end,
        min_train_rows=args.min_train_rows,
        max_events_per_year=args.max_events_per_year,
        run_model_robustness=not args.no_model_robustness,
        panel_path=args.panel_path,
    )
    summary = run_lm_pipeline(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
