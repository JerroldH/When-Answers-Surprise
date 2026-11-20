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

from when_answers_surprise.ibes_staging import merge_summary_with_actuals  # noqa: E402
from when_answers_surprise.wrds_sp500 import read_partitioned_parquet  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
RAW_ROOT = DATA_ROOT / "raw" / "wrds_exports"
STAGING_ROOT = DATA_ROOT / "staging"
REPORT_ROOT = PROJECT_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage adjusted IBES Summary Statistics with actual EPS fields."
    )
    parser.add_argument(
        "--summary-dir", type=Path, default=RAW_ROOT / "ibes_statsum_epsus_sp500"
    )
    parser.add_argument("--actuals-dir", type=Path, default=RAW_ROOT / "ibes_act_epsus_sp500")
    parser.add_argument(
        "--output", type=Path, default=STAGING_ROOT / "ibes_summary_snapshots.parquet"
    )
    parser.add_argument(
        "--report", type=Path, default=REPORT_ROOT / "ibes_staging_report.md"
    )
    args = parser.parse_args()

    summary = read_partitioned_parquet(args.summary_dir)
    actuals = read_partitioned_parquet(args.actuals_dir)
    staged, audit = merge_summary_with_actuals(summary, actuals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged.to_parquet(args.output, index=False)

    audit_dict = asdict(audit) | {"output": str(args.output)}
    _write_report(args.report, "IBES Staging Report", audit_dict)
    print(json.dumps(audit_dict, indent=2, sort_keys=True))


def _write_report(path: Path, title: str, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"- {key}: {value}" for key, value in audit.items())
    path.write_text(f"# {title}\n\n{rows}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
