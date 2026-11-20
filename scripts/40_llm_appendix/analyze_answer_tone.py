"""Analyze appendix LLM answer-tone scores.

This step reads answer_llm_scores.parquet from --input-dir and computes the
A0/A1/A2/A3 filtration ladder, response tables, and diagnostics. It never calls an
LLM provider.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.llm.answer_pipeline import (  # noqa: E402
    BASELINE_DIR,
    FEATURE_CACHE_DIR,
    analyze_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze appendix LLM answer-tone scores.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=FEATURE_CACHE_DIR)
    parser.add_argument(
        "--score-cache",
        type=Path,
        required=True,
        help="Read-only existing score cache; no scoring/provider calls are made.",
    )
    parser.add_argument("--input-dir", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--max-failure-rate", type=float, default=0.02)
    parser.add_argument("--min-duplicate-exact-agreement", type=float, default=0.95)
    args = parser.parse_args()

    analyze_scores(
        baseline_dir=args.baseline_dir,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=args.output_dir,
        score_cache=args.score_cache,
        min_train_rows=args.min_train_rows,
        max_failure_rate=args.max_failure_rate,
        min_duplicate_exact_agreement=args.min_duplicate_exact_agreement,
    )


if __name__ == "__main__":
    main()
