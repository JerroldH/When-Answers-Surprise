"""Build S&P 500 staging and derived tables from existing WRDS raw exports.

This orchestrates only the data-build DAG:
raw partitioned parquet -> staging tables -> derived transcript/IBES panel.
It does not run the LM tone baseline, diagnostics, learner audit, or paper
freeze steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from when_answers_surprise.common.manifests import manifest_digest, write_manifest_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build S&P 500 staging and derived tables from downloaded WRDS raw parquet."
    )
    parser.add_argument(
        "--include-raw-checksums",
        action="store_true",
        # Raw parquet can be large; public replication only needs processed checksums by default.
        help="Include raw parquet checksums in manifests. This can be slow for large WRDS exports.",
    )
    args = parser.parse_args()

    run("build_security_master.py")
    run("stage_ibes.py")
    run("stage_ciq_transcripts.py")
    run("match_transcripts_to_ibes.py")
    write_manifests(include_raw=args.include_raw_checksums)
    print(f"S&P 500 WRDS data build complete: {DATA_ROOT}")


def run(script_name: str) -> None:
    command = [sys.executable, str(SCRIPT_ROOT / script_name)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_manifests(*, include_raw: bool) -> None:
    """Record generated files without forcing a checksum pass over licensed raw data."""

    manifest_dir = DATA_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(DATA_ROOT.rglob("*")):
        if not path.is_file() or manifest_dir in path.parents:
            continue
        if not include_raw and "raw" in path.relative_to(DATA_ROOT).parts:
            continue
        digest = manifest_digest(path)
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "stage": path.relative_to(DATA_ROOT).parts[0],
                "bytes": digest["bytes"],
                "sha256": digest["sha256"],
                "hash_mode": digest["hash_mode"],
            }
        )
    write_manifest_csv(manifest_dir / "files.csv", rows)
    write_manifest_csv(
        manifest_dir / "checksums.csv",
        [
            {
                "sha256": row["sha256"],
                "hash_mode": row["hash_mode"],
                "relative_path": row["relative_path"],
            }
            for row in rows
        ],
    )


if __name__ == "__main__":
    main()
