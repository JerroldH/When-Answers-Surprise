from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.wrds_sp500 import (  # noqa: E402
    build_ciq_transcript_text,
    normalize_ciq_components,
    read_partitioned_parquet,
    select_latest_ciq_transcript_events,
)


DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
RAW_ROOT = DATA_ROOT / "raw" / "wrds_exports"
STAGING_ROOT = DATA_ROOT / "staging"
REPORT_ROOT = PROJECT_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage S&P 500 Capital IQ transcript events, components, and text."
    )
    parser.add_argument(
        "--events-dir", type=Path, default=RAW_ROOT / "ciq_transcript_events_sp500"
    )
    parser.add_argument(
        "--components-dir", type=Path, default=RAW_ROOT / "ciq_transcript_components_sp500"
    )
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)
    parser.add_argument(
        "--report", type=Path, default=REPORT_ROOT / "ciq_transcript_staging_report.md"
    )
    args = parser.parse_args()

    events_raw = read_partitioned_parquet(args.events_dir)
    components_raw = read_partitioned_parquet(args.components_dir)
    events = select_latest_ciq_transcript_events(events_raw)
    components = normalize_ciq_components(components_raw)
    transcript_text, audit = build_ciq_transcript_text(events, components)

    args.staging_root.mkdir(parents=True, exist_ok=True)
    events_path = args.staging_root / "ciq_transcript_events.parquet"
    components_path = args.staging_root / "ciq_transcript_components.parquet"
    text_path = args.staging_root / "ciq_transcript_text.parquet"
    events.to_parquet(events_path, index=False)
    components.to_parquet(components_path, index=False)
    transcript_text.to_parquet(text_path, index=False)

    audit_dict = asdict(audit) | {
        "events_path": str(events_path),
        "components_path": str(components_path),
        "text_path": str(text_path),
    }
    _write_report(args.report, "Capital IQ Transcript Staging Report", audit_dict)
    print(json.dumps(audit_dict, indent=2, sort_keys=True))


def _write_report(path: Path, title: str, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"- {key}: {value}" for key, value in audit.items())
    path.write_text(f"# {title}\n\n{rows}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
