"""Fail-closed WRDS export orchestration for the isolated 2025 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = PROJECT_ROOT / "configs" / "wrds_holdout_2025" / "bundle.yaml"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "wrds_holdout_2025_ciz"
DEFAULT_RAW_ROOT = DEFAULT_DATA_ROOT / "raw" / "wrds_exports"
DEFAULT_CONTRACT_ROOT = DEFAULT_DATA_ROOT / "export_contract"
SOURCE_TABLE_PATTERN = re.compile(
    r"\b(?:from|join)\s+([a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)", re.IGNORECASE
)


def export_holdout_wrds(
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    raw_root: Path = DEFAULT_RAW_ROOT,
    contract_root: Path = DEFAULT_CONTRACT_ROOT,
) -> dict[str, object]:
    """Resolve the export bundle, run it once, and record SQL/data provenance."""

    username = os.environ.get("WRDS_USER", "").strip()
    if not username:
        raise EnvironmentError(
            "WRDS_USER is not set. Set it for the current shell session before exporting."
        )
    _require_absent_or_empty(raw_root, "holdout raw export")
    _require_absent_or_empty(contract_root, "holdout export contract")
    try:
        import wrds_exporter  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "wrds_exporter is unavailable in this Python environment; no export was started"
        ) from exc

    bundle_path = bundle_path.resolve()
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    resolved_dir = contract_root / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=False)
    raw_root = raw_root.resolve()
    configs = _resolve_bundle_configs(
        bundle_path=bundle_path,
        bundle=bundle,
        raw_root=raw_root,
        resolved_dir=resolved_dir,
    )
    command = [sys.executable, "-m", "wrds_exporter.cli", "run"]
    for path in configs:
        command.extend(["--config", str(path)])
    started = datetime.now(timezone.utc)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    completed = datetime.now(timezone.utc)

    config_records = []
    for path in configs:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        sql = str(payload["query"]["sql"])
        query_name = str(payload["query"]["name"])
        export_dir = raw_root / query_name
        config_records.append(
            {
                "query_name": query_name,
                "resolved_config": path.relative_to(contract_root).as_posix(),
                "resolved_config_sha256": _sha256(path),
                "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                "source_tables": sorted(
                    {match.lower() for match in SOURCE_TABLE_PATTERN.findall(sql)}
                ),
                "row_count": _parquet_row_count(export_dir),
                "parquet_files": len(list(export_dir.glob("**/part-*.parquet"))),
            }
        )
    manifest = {
        "status": "complete",
        "contract": bundle["contract"],
        "username_environment_variable": "WRDS_USER",
        "username_recorded": False,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "raw_root": str(raw_root),
        "overwrite": False,
        "resume": False,
        "queries": config_records,
    }
    (contract_root / "wrds_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _resolve_bundle_configs(
    *,
    bundle_path: Path,
    bundle: dict[str, object],
    raw_root: Path,
    resolved_dir: Path,
) -> list[Path]:
    config_root = bundle_path.parent
    paths = [
        config_root / str(name) for name in bundle.get("custom_configs", [])
    ] + [
        (config_root / str(name)).resolve()
        for name in bundle.get("passthrough_configs", [])
    ]
    if not paths:
        raise ValueError("Holdout WRDS bundle contains no configs")
    resolved = []
    for source in paths:
        if not source.exists():
            raise FileNotFoundError(source)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        keep_partition = bool(payload.pop("holdout_keep_partition", False))
        payload["export"]["output_dir"] = str(raw_root)
        payload["export"]["overwrite"] = False
        payload["export"]["resume"] = False
        partition = payload.get("query", {}).get("partition")
        if partition and partition.get("type") == "year" and not keep_partition:
            partition["start"] = int(bundle["partition_start"])
            partition["end"] = int(bundle["partition_end"])
        destination = resolved_dir / source.name
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=1000),
            encoding="utf-8",
        )
        resolved.append(destination)
    return resolved


def _require_absent_or_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty {label} directory: {path.resolve()}"
        )


def _parquet_row_count(directory: Path) -> int:
    files = sorted(directory.glob("**/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Export produced no parquet parts under {directory}")
    return int(sum(pq.ParquetFile(path).metadata.num_rows for path in files))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export isolated 2024--2025 WRDS inputs for the FinNLP holdout."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT_ROOT)
    args = parser.parse_args()
    manifest = export_holdout_wrds(
        bundle_path=args.bundle,
        raw_root=args.raw_root,
        contract_root=args.contract_root,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "queries": len(manifest["queries"]),
                "raw_root": manifest["raw_root"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
