"""Freeze validated revision-v2 paper-facing artifacts with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025" / "analysis"
DEFAULT_REVISION_DIR = ANALYSIS_ROOT / "revision_v2"
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "revision_v2_frozen"
SUBDIRECTORIES = ("event_audit", "learner_audit", "bootstrap")


def freeze_revision_v2(source: Path, output: Path) -> dict[str, object]:
    """Copy compact final artifacts only after all required audits close."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Frozen revision-v2 directory is not empty: {output}")
    summaries = {
        "revision": _read_summary(source / "build_summary.json"),
        "event_audit": _read_summary(source / "event_audit" / "build_summary.json"),
        "learner_audit": _read_summary(source / "learner_audit" / "build_summary.json"),
        "bootstrap": _read_summary(source / "bootstrap" / "build_summary.json"),
    }
    if any(summary.get("status") != "complete" for summary in summaries.values()):
        raise ValueError("All revision-v2 components must be complete before freezing")
    if (
        int(summaries["bootstrap"].get("valid_replicates", 0)) != 200
        or int(summaries["bootstrap"].get("requested_replicates", 0)) != 200
    ):
        raise ValueError("Revision-v2 freeze requires exactly 200 valid bootstrap replicates")
    if not bool(summaries["bootstrap"].get("all_fold_checks_time_forward", False)):
        raise ValueError("Revision-v2 freeze requires all bootstrap folds to be time-forward")
    response_calls = int(summaries["revision"]["response_calls"])
    if any(
        int(summaries[name].get("fixed_response_calls", response_calls)) != response_calls
        for name in ("event_audit", "bootstrap")
    ):
        raise ValueError("Revision-v2 components do not use one fixed response sample")

    output.mkdir(parents=True, exist_ok=False)
    copied = []
    root_names = [
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".npy"}
    ]
    root_names.append(source / "response_model_sample_revision_v2.parquet")
    for path in root_names:
        copied.append(_copy_one(path, output / path.name, output))
    for directory in SUBDIRECTORIES:
        destination = output / directory
        for path in (source / directory).iterdir():
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".npy"}:
                copied.append(_copy_one(path, destination / path.name, output))

    manifest = {
        "status": "frozen",
        "source": str(source),
        "output": str(output),
        "response_calls": response_calls,
        "bootstrap_replicates": 200,
        "summaries": summaries,
        "files": copied,
    }
    (output / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _copy_one(source: Path, destination: Path, root: Path) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.relative_to(root).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def _read_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze validated revision-v2 artifacts.")
    parser.add_argument("--source", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = freeze_revision_v2(args.source, args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "response_calls": manifest["response_calls"],
                "bootstrap_replicates": manifest["bootstrap_replicates"],
                "output": manifest["output"],
                "file_count": len(manifest["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
