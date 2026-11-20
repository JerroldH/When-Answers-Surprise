from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.wrds_sp500 import (  # noqa: E402
    build_security_master,
    normalize_sp500_membership,
    read_partitioned_parquet,
)


DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
RAW_ROOT = DATA_ROOT / "raw" / "wrds_exports"
STAGING_ROOT = DATA_ROOT / "staging"
REPORT_ROOT = PROJECT_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build S&P 500 membership and identifier crosswalk staging tables."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_ROOT / "security_master_report.md")
    args = parser.parse_args()

    membership_raw = read_partitioned_parquet(args.raw_root / "crsp_sp500_membership")
    names = read_partitioned_parquet(args.raw_root / "crsp_dsenames_sp500")
    ccm = _read_optional(args.raw_root / "crsp_ccmxpf_linktable_sp500")
    ibes = _read_optional(args.raw_root / "wrdsapps_ibcrsphist_sp500")
    ciq = _read_optional(args.raw_root / "ciq_gvkey_sp500")

    membership = normalize_sp500_membership(membership_raw)
    security_master = build_security_master(membership, names, ccm, ibes, ciq)

    args.staging_root.mkdir(parents=True, exist_ok=True)
    membership_path = args.staging_root / "universe_sp500_membership.parquet"
    master_path = args.staging_root / "security_master.parquet"
    membership.to_parquet(membership_path, index=False)
    security_master.to_parquet(master_path, index=False)

    audit = {
        "membership_rows": int(len(membership)),
        "membership_unique_permno": int(membership["permno"].nunique()),
        "security_master_rows": int(len(security_master)),
        "security_master_unique_permno": int(security_master["permno"].nunique()),
        "gvkey_nonmissing_rows": _nonmissing(security_master, "gvkey"),
        "ibes_ticker_nonmissing_rows": _nonmissing(security_master, "ibes_ticker"),
        "ciq_companyid_nonmissing_rows": _nonmissing(security_master, "ciq_companyid"),
        "membership_path": str(membership_path),
        "security_master_path": str(master_path),
    }
    _write_report(args.report, "Security Master Report", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


def _read_optional(path: Path):
    try:
        return read_partitioned_parquet(path)
    except FileNotFoundError:
        return None


def _nonmissing(data, column: str) -> int:
    return int(data[column].notna().sum()) if column in data else 0


def _write_report(path: Path, title: str, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"- {key}: {value}" for key, value in audit.items())
    path.write_text(f"# {title}\n\n{rows}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
