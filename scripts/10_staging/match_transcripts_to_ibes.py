from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.ibes_events import build_ibes_event_dataset, write_ibes_matching_report  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
STAGING_ROOT = DATA_ROOT / "staging"
DERIVED_ROOT = DATA_ROOT / "derived"
REPORT_ROOT = PROJECT_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match S&P 500 transcript events to pre-event IBES EPS snapshots."
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=STAGING_ROOT / "ciq_transcript_text.parquet",
    )
    parser.add_argument(
        "--ibes",
        type=Path,
        default=STAGING_ROOT / "ibes_summary_snapshots.parquet",
    )
    parser.add_argument(
        "--event-csv",
        type=Path,
        default=DERIVED_ROOT / "transcript_ibes_event_level.csv",
    )
    parser.add_argument(
        "--event-parquet",
        type=Path,
        default=DERIVED_ROOT / "transcript_ibes_event_level.parquet",
    )
    parser.add_argument(
        "--merged",
        type=Path,
        default=DERIVED_ROOT / "transcript_ibes_panel.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_ROOT / "transcript_ibes_matching_report.md",
    )
    args = parser.parse_args()

    outputs = build_ibes_event_dataset(args.transcripts, args.ibes)
    args.event_parquet.parent.mkdir(parents=True, exist_ok=True)
    outputs.event_level.to_parquet(args.event_parquet, index=False)
    outputs.merged_transcripts.to_parquet(args.merged, index=False)

    outputs.event_level.to_csv(args.event_csv, index=False)

    write_ibes_matching_report(
        outputs,
        args.report,
        transcripts_path=args.transcripts,
        ibes_csv_path=args.ibes,
        event_csv_path=args.event_csv,
        event_parquet_path=args.event_parquet,
        merged_path=args.merged,
    )
    print("S&P 500 transcript/IBES matching complete.")
    print(f"Transcript events: {outputs.audit['transcript_events']}")
    print(f"Matched events: {outputs.audit['matched_events']}")
    print(f"Unmatched events: {outputs.audit['unmatched_events']}")
    print(f"Match quality: {outputs.audit['match_quality_counts']}")
    print(f"Event Parquet: {args.event_parquet}")
    print(f"Merged Parquet: {args.merged}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
