"""Analyze prospective 2025 LLM answer-tone innovations without API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.llm.holdout_answer import analyze_llm_holdout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--development-response", type=Path, required=True)
    parser.add_argument("--holdout-response", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--variant-workers", type=int, default=1)
    args = parser.parse_args()
    result = analyze_llm_holdout(
        baseline_dir=args.baseline_dir,
        feature_dir=args.feature_dir,
        score_cache=args.score_cache,
        development_response=args.development_response,
        holdout_response=args.holdout_response,
        output_dir=args.output_dir,
        min_train_rows=args.min_train_rows,
        variant_workers=max(args.variant_workers, 1),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
