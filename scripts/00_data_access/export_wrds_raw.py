"""Download the private WRDS raw layer for the S&P 500 case study.

This is the first stage of the private pipeline:
WRDS PostgreSQL -> raw partitioned parquet. It is intentionally separate from
staging, analysis, diagnostics, learner audit, and paper-result freezing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "wrds_exports"
DEFAULT_CONFIGS = [
    "crsp_sp500_membership.yaml",
    "crsp_dsenames_sp500.yaml",
    "crsp_ccmxpf_linktable_sp500.yaml",
    "wrdsapps_ibcrsphist_sp500.yaml",
    "ciq_gvkey_sp500.yaml",
    "crsp_dsi_sp500.yaml",
    "crsp_dsf_sp500.yaml",
    "comp_fundq_sp500.yaml",
    "comp_funda_sp500.yaml",
    "comp_secd_sp500.yaml",
    "ibes_statsum_epsus_sp500.yaml",
    "ibes_act_epsus_sp500.yaml",
    "ciq_transcript_events_sp500.yaml",
    "ciq_transcript_components_sp500.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all S&P 500 WRDS raw exports with one WRDS connection."
    )
    parser.add_argument("--force", action="store_true", help="Re-export completed partitions.")
    parser.add_argument(
        "--only",
        action="append",
        choices=DEFAULT_CONFIGS,
        help="Run only this config filename. Repeat to run several.",
    )
    args = parser.parse_args()

    selected = args.only or DEFAULT_CONFIGS
    command = [sys.executable, "-m", "wrds_exporter.cli", "run"]
    if args.force:
        command.append("--force")
    for filename in selected:
        command.extend(["--config", str(CONFIG_ROOT / filename)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
