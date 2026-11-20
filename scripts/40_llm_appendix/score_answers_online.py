"""Score appendix answer tone with an online LLM provider.

This step only creates or updates answer_llm_scores.parquet plus scoring
diagnostics. Run 97_analyze_llm_answer_tone.py afterward to compute the
filtration ladder and market-response tables.
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
    default_output_dir_for_config,
    score_online,
)
from when_answers_surprise.llm.tone_scoring import provider_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score appendix LLM answer tone online.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=FEATURE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--historical-score-cache",
        type=Path,
        default=None,
        help="Reuse exact transcript-turn-text scores from an earlier scorer panel.",
    )
    parser.add_argument(
        "--provider",
        default="fake",
        choices=["fake", "gemini_adc", "gemini", "openai", "deepseek", "qwen"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--duplicate-fraction", type=float, default=0.01)
    parser.add_argument("--max-duplicate-rows", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=1000)
    args = parser.parse_args()

    config = provider_config(args.provider, model=args.model)
    output_dir = args.output_dir or default_output_dir_for_config(config)
    score_online(
        baseline_dir=args.baseline_dir,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=output_dir,
        score_config=config,
        max_rows=None if args.allow_full_run else args.max_rows,
        duplicate_fraction=args.duplicate_fraction,
        max_duplicate_rows=args.max_duplicate_rows,
        workers=max(args.workers, 1),
        flush_every=max(args.flush_every, 1),
        historical_score_cache=args.historical_score_cache,
    )


if __name__ == "__main__":
    main()
