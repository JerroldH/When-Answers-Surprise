"""Run the S&P 500 LM net-tone learner-invariance audit.

The audit compares first-stage learners on the common response sample. It may
reuse or build feature caches, but it should not be confused with the baseline
or paper-freeze stages.
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

from when_answers_surprise.sp500_lm_learner_audit import (  # noqa: E402
    LearnerAuditConfig,
    run_lm_learner_invariance_audit,
)


DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
ANALYSIS_ROOT = DATA_ROOT / "analysis"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the S&P 500 LM net-tone learner-invariance audit. "
            "The audit reuses or builds feature-ready first-stage caches, "
            "checks fixed-Ridge parity, then compares learner-specific "
            "innovations on one frozen response sample."
        )
    )
    parser.add_argument(
        "--sample",
        choices=["full", "smoke"],
        default="full",
        help="Use full baseline outputs or the smoke baseline output directory.",
    )
    parser.add_argument("--sample-end", default="2024-12-31")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild first-stage feature caches before running learner comparisons.",
    )
    parser.add_argument(
        "--max-events-per-year",
        type=int,
        default=None,
        help="Optional per-year cap for cache construction. Smoke defaults to 5 when omitted.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=120,
        help="Minimum prior observations for outer and inner expanding-year folds.",
    )
    parser.add_argument("--parity-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--nonlinear",
        choices=["auto", "hist_gradient_boosting", "extra_trees"],
        default="auto",
        help="Nonlinear sklearn learner. Auto runs a first-stage smoke/runtime check.",
    )
    parser.add_argument(
        "--nonlinear-smoke-seconds",
        type=float,
        default=30.0,
        help="Maximum acceptable runtime for the auto nonlinear smoke check.",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--panel-path", type=Path, default=None)
    args = parser.parse_args()

    suffix = "" if args.sample == "full" else "_smoke"
    max_events_per_year = args.max_events_per_year
    if args.sample == "smoke" and max_events_per_year is None:
        max_events_per_year = 5

    input_dir = args.input_dir or (ANALYSIS_ROOT / f"lm_baseline{suffix}")
    output_dir = args.output_dir or (ANALYSIS_ROOT / f"lm_baseline{suffix}_learner_audit")
    cache_dir = args.cache_dir or (ANALYSIS_ROOT / f"lm_baseline{suffix}_learner_audit_cache")

    summary = run_lm_learner_invariance_audit(
        LearnerAuditConfig(
            project_root=PROJECT_ROOT,
            data_root=DATA_ROOT,
            input_dir=input_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            sample_end=args.sample_end,
            min_train_rows=args.min_train_rows,
            max_events_per_year=max_events_per_year,
            parity_tolerance=args.parity_tolerance,
            refresh_cache=args.refresh_cache,
            nonlinear=args.nonlinear,
            nonlinear_smoke_seconds=args.nonlinear_smoke_seconds,
            panel_path=args.panel_path,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
