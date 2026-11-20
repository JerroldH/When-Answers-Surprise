"""Recover an interrupted holdout WRDS export without overwriting completed data.

The recovery path is deliberately separate from the first-run exporter.  It
validates and inventories completed query directories, quarantines only the
known zero-row transcript-component attempt, and fetches transcript components
in small batches keyed by the already-exported event transcript IDs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq
import yaml

from when_answers_surprise.data.holdout_2025_export import (
    DEFAULT_BUNDLE,
    DEFAULT_CONTRACT_ROOT,
    DEFAULT_RAW_ROOT,
    SOURCE_TABLE_PATTERN,
)
from wrds_exporter.config import ExportJobConfig, load_config
from wrds_exporter.connection import close_connection, connect_wrds
from wrds_exporter.exporter import WRDSExporter
from wrds_exporter.logging_config import configure_logging
from wrds_exporter.partitioning import generate_partitions


COMPONENT_QUERY_NAME = "ciq_transcript_components_holdout"
RECOVERY_VERSION = "event-id-batches-v1"
DEFAULT_BATCH_SIZE = 250
COMPONENT_COLUMNS = (
    "transcriptcomponentid",
    "transcriptid",
    "componentorder",
    "transcriptcomponenttypeid",
    "componenttypename",
    "transcriptpersonid",
    "personname",
    "proid",
    "speaker_companyname",
    "speakertypeid",
    "componenttext",
)
COMPONENT_SQL_TEMPLATE = """
SELECT
  c.transcriptcomponentid,
  c.transcriptid,
  c.componentorder,
  c.transcriptcomponenttypeid,
  ct.transcriptcomponenttypename AS componenttypename,
  c.transcriptpersonid,
  p.transcriptpersonname AS personname,
  p.proid,
  p.companyname AS speaker_companyname,
  p.speakertypeid,
  c.componenttext
FROM ciq_transcripts.ciqtranscriptcomponent AS c
LEFT JOIN ciq_transcripts.ciqtranscriptcomponenttype AS ct
  ON c.transcriptcomponenttypeid = ct.transcriptcomponenttypeid
LEFT JOIN ciq_transcripts.ciqtranscriptperson AS p
  ON c.transcriptpersonid = p.transcriptpersonid
WHERE c.transcriptid IN ({validated_transcript_ids})
""".strip()


def prepare_holdout_recovery(
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    raw_root: Path = DEFAULT_RAW_ROOT,
    contract_root: Path = DEFAULT_CONTRACT_ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Validate partial outputs and create an immutable recovery contract."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    bundle_path = bundle_path.resolve()
    raw_root = raw_root.resolve()
    contract_root = contract_root.resolve()
    recovery_root = contract_root / "recovery"
    manifest_path = recovery_root / "recovery_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        _validate_existing_recovery_contract(
            manifest,
            bundle_path=bundle_path,
            raw_root=raw_root,
            contract_root=contract_root,
            batch_size=batch_size,
        )
        _validate_event_contract(recovery_root / "event_id_contract.json")
        _validate_initial_inventory(manifest, raw_root)
        _validate_recovery_files(manifest, recovery_root)
        return manifest

    resolved_configs = contract_root / "resolved_configs"
    if not resolved_configs.is_dir():
        raise FileNotFoundError(
            f"Missing resolved WRDS configs from the interrupted export: {resolved_configs}"
        )
    ordered_paths = _ordered_resolved_configs(bundle_path, resolved_configs)
    configs = [load_config(path) for path in ordered_paths]

    recovery_root.mkdir(parents=True, exist_ok=False)
    completed_before_recovery: list[dict[str, Any]] = []
    for config, path in zip(configs, ordered_paths, strict=True):
        if config.query.name == COMPONENT_QUERY_NAME:
            continue
        query_dir = raw_root / config.query.name
        if not query_dir.exists():
            continue
        completed_before_recovery.append(
            _validate_completed_export(config, path, query_dir)
        )

    interrupted_record = _quarantine_interrupted_components(
        raw_root / COMPONENT_QUERY_NAME,
        recovery_root / "interrupted_component_query",
        recovery_root=recovery_root,
    )
    event_contract = _build_event_contract(
        raw_root / "ciq_transcript_events_holdout",
        recovery_root / "event_id_contract.json",
        batch_size=batch_size,
    )
    recovery_configs = _write_recovery_configs(
        ordered_paths,
        recovery_root / "resolved_configs",
    )
    manifest = {
        "status": "prepared",
        "recovery_version": RECOVERY_VERSION,
        "prepared_at_utc": _utc_now(),
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "raw_root": str(raw_root),
        "contract_root": str(contract_root),
        "batch_size": batch_size,
        "event_id_contract": "event_id_contract.json",
        "event_id_contract_sha256": _sha256(
            recovery_root / "event_id_contract.json"
        ),
        "event_counts": {
            year: int(record["event_count"])
            for year, record in event_contract["years"].items()
        },
        "completed_before_recovery": completed_before_recovery,
        "interrupted_component_query": interrupted_record,
        "recovery_configs": recovery_configs,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def run_holdout_recovery(
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    raw_root: Path = DEFAULT_RAW_ROOT,
    contract_root: Path = DEFAULT_CONTRACT_ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = 3,
    connection_factory: Callable[[str], Any] = connect_wrds,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Resume the export using one WRDS connection and finalize provenance."""

    username = os.environ.get("WRDS_USER", "").strip()
    if not username:
        raise EnvironmentError("WRDS_USER is not set for this Git Bash session")
    preparation = prepare_holdout_recovery(
        bundle_path=bundle_path,
        raw_root=raw_root,
        contract_root=contract_root,
        batch_size=batch_size,
    )
    raw_root = raw_root.resolve()
    contract_root = contract_root.resolve()
    recovery_root = contract_root / "recovery"
    logger = configure_logging()
    db = connection_factory("WRDS_USER")
    try:
        export_component_batches(
            db,
            event_contract_path=recovery_root / "event_id_contract.json",
            query_dir=raw_root / COMPONENT_QUERY_NAME,
            batch_size=batch_size,
            max_retries=max_retries,
            sleep_fn=sleep_fn,
            logger=logger,
        )
        _run_remaining_configs(
            db,
            bundle_path=bundle_path.resolve(),
            raw_root=raw_root,
            contract_root=contract_root,
            logger=logger,
        )
    finally:
        close_connection(db)

    manifest = finalize_holdout_recovery(
        bundle_path=bundle_path,
        raw_root=raw_root,
        contract_root=contract_root,
    )
    preparation.update(
        {
            "status": "complete",
            "completed_at_utc": manifest["completed_at_utc"],
            "wrds_export_manifest": "../wrds_export_manifest.json",
            "wrds_export_manifest_sha256": _sha256(
                contract_root / "wrds_export_manifest.json"
            ),
        }
    )
    _atomic_write_json(recovery_root / "recovery_manifest.json", preparation)
    return manifest


def export_component_batches(
    db: Any,
    *,
    event_contract_path: Path,
    query_dir: Path,
    batch_size: int,
    max_retries: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Fetch components for fixed event IDs in small, resumable batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    logger = logger or logging.getLogger("holdout_2025_recovery")
    event_contract = _validate_event_contract(event_contract_path)
    if int(event_contract["batch_size"]) != batch_size:
        raise ValueError(
            "batch_size differs from the frozen event-ID recovery contract"
        )
    contract_hash = _sha256(event_contract_path)
    query_dir = query_dir.resolve()
    query_dir.mkdir(parents=True, exist_ok=True)
    state_path = query_dir / "_state.json"
    state = _load_or_initialize_component_state(
        state_path,
        contract_hash=contract_hash,
        batch_size=batch_size,
    )

    for year_text, year_record in sorted(event_contract["years"].items()):
        event_ids = [int(value) for value in year_record["transcript_ids"]]
        for batch_index, start in enumerate(range(0, len(event_ids), batch_size)):
            batch_ids = event_ids[start : start + batch_size]
            batch_key = f"year={year_text}/batch={batch_index:05d}"
            target = query_dir / f"year={year_text}" / f"part-{batch_index:05d}.parquet"
            if _completed_component_batch_is_valid(
                state.get("batches", {}).get(batch_key), target, batch_ids
            ):
                logger.info("Skipping completed component batch %s", batch_key)
                continue
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite unverified component batch: {target}"
                )
            state["batches"][batch_key] = {
                "status": "running",
                "year": int(year_text),
                "batch_index": batch_index,
                "expected_transcript_count": len(batch_ids),
                "expected_transcript_ids_sha256": _ids_sha256(batch_ids),
                "started_at_utc": _utc_now(),
                "attempt": 0,
            }
            _atomic_write_json(state_path, state)
            frame: pd.DataFrame | None = None
            sql = render_component_sql(batch_ids)
            for attempt in range(1, max_retries + 2):
                state["batches"][batch_key]["attempt"] = attempt
                _atomic_write_json(state_path, state)
                try:
                    frame = db.raw_sql(sql)
                    break
                except Exception as exc:
                    state["batches"][batch_key].update(
                        {"status": "failed", "error": _safe_error(exc)}
                    )
                    _atomic_write_json(state_path, state)
                    if attempt > max_retries:
                        raise
                    delay = float(2 ** (attempt - 1))
                    logger.warning(
                        "Component batch %s failed on attempt %s/%s; retrying in %.1fs: %s",
                        batch_key,
                        attempt,
                        max_retries + 1,
                        delay,
                        _safe_error(exc),
                    )
                    sleep_fn(delay)
            assert frame is not None
            try:
                frame = _validate_component_batch(
                    frame, batch_ids, batch_key=batch_key
                )
            except Exception as exc:
                state["batches"][batch_key].update(
                    {"status": "failed", "error": _safe_error(exc)}
                )
                _atomic_write_json(state_path, state)
                raise
            _write_parquet_atomic(frame, target)
            record = state["batches"][batch_key]
            record.update(
                {
                    "status": "completed",
                    "row_count": int(len(frame)),
                    "returned_transcript_count": int(
                        _normalized_id_series(frame["transcriptid"]).nunique()
                    ),
                    "file": target.relative_to(query_dir).as_posix(),
                    "file_bytes": target.stat().st_size,
                    "file_sha256": _sha256(target),
                    "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                    "completed_at_utc": _utc_now(),
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            logger.info(
                "Completed component batch %s rows=%s path=%s",
                batch_key,
                len(frame),
                target,
            )

    manifest = _validate_and_manifest_components(
        query_dir,
        event_contract=event_contract,
        event_contract_path=event_contract_path,
        state=state,
    )
    _atomic_write_json(query_dir / "_manifest.json", manifest)
    state["status"] = "complete"
    state["completed_at_utc"] = manifest["generated_at_utc"]
    _atomic_write_json(state_path, state)
    return manifest


def render_component_sql(transcript_ids: list[int]) -> str:
    """Render a direct component query after strict numeric-ID validation."""

    ids = _normalize_transcript_ids(transcript_ids)
    if not ids:
        raise ValueError("A component batch cannot be empty")
    id_sql = ", ".join(str(value) for value in ids)
    return COMPONENT_SQL_TEMPLATE.format(validated_transcript_ids=id_sql)


def finalize_holdout_recovery(
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    raw_root: Path = DEFAULT_RAW_ROOT,
    contract_root: Path = DEFAULT_CONTRACT_ROOT,
) -> dict[str, Any]:
    """Verify every query and write the canonical completed export manifest."""

    bundle_path = bundle_path.resolve()
    raw_root = raw_root.resolve()
    contract_root = contract_root.resolve()
    recovery_root = contract_root / "recovery"
    preparation = _read_json(recovery_root / "recovery_manifest.json")
    _validate_initial_inventory(preparation, raw_root)

    original_paths = _ordered_resolved_configs(
        bundle_path, contract_root / "resolved_configs"
    )
    recovery_paths = {
        path.name: path
        for path in (recovery_root / "resolved_configs").glob("*.yaml")
    }
    completed_original = {
        record["query_name"] for record in preparation["completed_before_recovery"]
    }
    records: list[dict[str, Any]] = []
    for original_path in original_paths:
        original_config = load_config(original_path)
        if original_config.query.name == COMPONENT_QUERY_NAME:
            records.append(
                _component_export_record(
                    raw_root / COMPONENT_QUERY_NAME,
                    recovery_root / "event_id_contract.json",
                )
            )
            continue
        if original_config.query.name in completed_original:
            config = original_config
            config_path = original_path
            config_mode = "original_completed"
        else:
            config_path = recovery_paths[original_path.name]
            config = load_config(config_path)
            config_mode = "recovery_resume"
        record = _validate_completed_export(
            config, config_path, raw_root / config.query.name
        )
        record["config_mode"] = config_mode
        records.append(record)

    manifest = {
        "status": "complete",
        "contract": yaml.safe_load(bundle_path.read_text(encoding="utf-8"))["contract"],
        "recovery_version": RECOVERY_VERSION,
        "username_environment_variable": "WRDS_USER",
        "username_recorded": False,
        "completed_at_utc": _utc_now(),
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "raw_root": str(raw_root),
        "overwrite": False,
        "resume": True,
        "queries": records,
    }
    _atomic_write_json(contract_root / "wrds_export_manifest.json", manifest)
    return manifest


def recovery_status(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    contract_root: Path = DEFAULT_CONTRACT_ROOT,
) -> dict[str, Any]:
    """Return a compact, read-only status summary for the recovery CLI."""

    recovery_root = contract_root.resolve() / "recovery"
    status: dict[str, Any] = {
        "prepared": (recovery_root / "recovery_manifest.json").exists(),
        "complete": (contract_root.resolve() / "wrds_export_manifest.json").exists(),
        "component_batches": {"completed": 0, "planned": 0},
        "queries": [],
    }
    event_contract_path = recovery_root / "event_id_contract.json"
    if event_contract_path.exists():
        event_contract = _validate_event_contract(event_contract_path)
        status["component_batches"]["planned"] = sum(
            int(record["batch_count"])
            for record in event_contract["years"].values()
        )
    if not raw_root.exists():
        return status
    for query_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        part_files = sorted(query_dir.glob("**/part-*.parquet"))
        state_path = query_dir / "_state.json"
        state = _read_json(state_path) if state_path.exists() else {}
        if state.get("batches") is not None:
            values = list(state.get("batches", {}).values())
            completed = sum(item.get("status") == "completed" for item in values)
            total = len(values)
            if query_dir.name == COMPONENT_QUERY_NAME:
                status["component_batches"]["completed"] = completed
        else:
            values = list(state.get("partitions", {}).values())
            completed = sum(item.get("status") == "completed" for item in values)
            total = len(values)
        status["queries"].append(
            {
                "query_name": query_dir.name,
                "part_files": len(part_files),
                "completed_units": completed,
                "known_units": total,
            }
        )
    return status


def _run_remaining_configs(
    db: Any,
    *,
    bundle_path: Path,
    raw_root: Path,
    contract_root: Path,
    logger: logging.Logger,
) -> None:
    preparation = _read_json(contract_root / "recovery" / "recovery_manifest.json")
    completed_original = {
        record["query_name"] for record in preparation["completed_before_recovery"]
    }
    original_paths = _ordered_resolved_configs(
        bundle_path, contract_root / "resolved_configs"
    )
    recovery_root = contract_root / "recovery" / "resolved_configs"
    for original_path in original_paths:
        original = load_config(original_path)
        if original.query.name == COMPONENT_QUERY_NAME:
            continue
        if original.query.name in completed_original:
            _validate_completed_export(
                original, original_path, raw_root / original.query.name
            )
            logger.info("Preserving completed query %s", original.query.name)
            continue
        recovery_path = recovery_root / original_path.name
        config = load_config(recovery_path)
        WRDSExporter(config, logger=logger).run(db=db)
        _validate_completed_export(config, recovery_path, raw_root / config.query.name)


def _build_event_contract(
    event_dir: Path, destination: Path, *, batch_size: int
) -> dict[str, Any]:
    years: dict[str, dict[str, Any]] = {}
    all_ids: list[int] = []
    for year_dir in sorted(event_dir.glob("year=*")):
        year_text = year_dir.name.split("=", 1)[1]
        files = sorted(year_dir.glob("part-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No event parquet files under {year_dir}")
        frames = [pd.read_parquet(path, columns=["transcriptid"]) for path in files]
        ids = _normalize_transcript_ids(
            pd.concat(frames, ignore_index=True)["transcriptid"].tolist()
        )
        if len(ids) != sum(len(frame) for frame in frames):
            raise ValueError(f"Duplicate transcriptid values in event partition {year_text}")
        years[year_text] = {
            "event_count": len(ids),
            "batch_count": int(math.ceil(len(ids) / batch_size)),
            "transcript_ids_sha256": _ids_sha256(ids),
            "transcript_ids": ids,
            "source_files": [_file_record(path, event_dir) for path in files],
        }
        all_ids.extend(ids)
    if sorted(years) != ["2024", "2025"]:
        raise ValueError(
            f"Expected event partitions 2024 and 2025, found {sorted(years)}"
        )
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Event transcript IDs overlap across holdout years")
    contract = {
        "recovery_version": RECOVERY_VERSION,
        "query_name": COMPONENT_QUERY_NAME,
        "batch_size": batch_size,
        "component_sql_template_sha256": hashlib.sha256(
            COMPONENT_SQL_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "created_at_utc": _utc_now(),
        "years": years,
    }
    _atomic_write_json(destination, contract)
    return contract


def _validate_event_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("recovery_version") != RECOVERY_VERSION:
        raise ValueError(f"Unsupported event-ID recovery contract: {path}")
    if contract.get("query_name") != COMPONENT_QUERY_NAME:
        raise ValueError(f"Wrong component query in event-ID contract: {path}")
    expected_template_hash = hashlib.sha256(
        COMPONENT_SQL_TEMPLATE.encode("utf-8")
    ).hexdigest()
    if contract.get("component_sql_template_sha256") != expected_template_hash:
        raise ValueError("Component SQL template changed after recovery preparation")
    for year_text, record in contract.get("years", {}).items():
        ids = _normalize_transcript_ids(record.get("transcript_ids", []))
        if len(ids) != int(record.get("event_count", -1)):
            raise ValueError(f"Event count mismatch in recovery year {year_text}")
        if _ids_sha256(ids) != record.get("transcript_ids_sha256"):
            raise ValueError(f"Event-ID checksum mismatch in recovery year {year_text}")
        for source in record.get("source_files", []):
            source_path = path.parents[2] / "raw" / "wrds_exports" / "ciq_transcript_events_holdout" / source["path"]
            if not source_path.exists() or _sha256(source_path) != source["sha256"]:
                raise ValueError(f"Frozen event source changed: {source_path}")
    return contract


def _validate_component_batch(
    frame: pd.DataFrame, expected_ids: list[int], *, batch_key: str
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"WRDS returned a non-DataFrame for {batch_key}")
    result = frame.copy()
    result.columns = [str(column).lower() for column in result.columns]
    missing_columns = set(COMPONENT_COLUMNS) - set(result.columns)
    if missing_columns:
        raise ValueError(
            f"Component batch {batch_key} missing columns: {sorted(missing_columns)}"
        )
    returned_ids = set(_normalized_id_series(result["transcriptid"]).tolist())
    expected = set(expected_ids)
    if returned_ids - expected:
        raise ValueError(
            f"Component batch {batch_key} returned unexpected transcript IDs"
        )
    if expected - returned_ids:
        raise ValueError(
            f"Component batch {batch_key} has events without components: "
            f"{sorted(expected - returned_ids)[:10]}"
        )
    if result["transcriptcomponentid"].isna().any():
        raise ValueError(f"Component batch {batch_key} has null component IDs")
    if result["transcriptcomponentid"].duplicated().any():
        raise ValueError(f"Component batch {batch_key} has duplicate component IDs")
    for column in ("transcriptid", "componentorder", "componenttext"):
        if result[column].isna().any():
            raise ValueError(f"Component batch {batch_key} has null {column}")
    return result.loc[:, list(COMPONENT_COLUMNS)]


def _validate_and_manifest_components(
    query_dir: Path,
    *,
    event_contract: dict[str, Any],
    event_contract_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    files = sorted(query_dir.glob("year=*/part-*.parquet"))
    expected_file_count = sum(
        int(record["batch_count"]) for record in event_contract["years"].values()
    )
    if len(files) != expected_file_count:
        raise ValueError(
            f"Expected {expected_file_count} component files, found {len(files)}"
        )
    expected_ids = {
        int(value)
        for record in event_contract["years"].values()
        for value in record["transcript_ids"]
    }
    returned_ids: set[int] = set()
    component_ids: set[int] = set()
    row_count = 0
    file_records: list[dict[str, Any]] = []
    for path in files:
        frame = pd.read_parquet(
            path, columns=["transcriptid", "transcriptcomponentid"]
        )
        ids = set(_normalized_id_series(frame["transcriptid"]).tolist())
        current_components = set(
            _normalized_id_series(frame["transcriptcomponentid"]).tolist()
        )
        if len(current_components) != len(frame):
            raise ValueError(f"Duplicate component IDs within {path}")
        if component_ids & current_components:
            raise ValueError("Duplicate component IDs across component batches")
        component_ids.update(current_components)
        returned_ids.update(ids)
        row_count += len(frame)
        file_records.append(_file_record(path, query_dir))
    if returned_ids != expected_ids:
        raise ValueError(
            "Final component coverage does not match the frozen event-ID contract"
        )
    completed_batches = sum(
        record.get("status") == "completed" for record in state["batches"].values()
    )
    if completed_batches != expected_file_count:
        raise ValueError("Component batch state is incomplete")
    return {
        "query_name": COMPONENT_QUERY_NAME,
        "mode": RECOVERY_VERSION,
        "generated_at_utc": _utc_now(),
        "event_id_contract": str(event_contract_path.resolve()),
        "event_id_contract_sha256": _sha256(event_contract_path),
        "component_sql_template_sha256": hashlib.sha256(
            COMPONENT_SQL_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "batch_size": int(event_contract["batch_size"]),
        "event_count": len(expected_ids),
        "events_with_components": len(returned_ids),
        "row_count": row_count,
        "unique_component_ids": len(component_ids),
        "file_count": len(files),
        "files": file_records,
    }


def _component_export_record(
    query_dir: Path, event_contract_path: Path
) -> dict[str, Any]:
    manifest_path = query_dir / "_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("mode") != RECOVERY_VERSION:
        raise ValueError(f"Unexpected component export mode: {manifest_path}")
    if manifest.get("event_id_contract_sha256") != _sha256(event_contract_path):
        raise ValueError("Component export does not match the event-ID contract")
    for record in manifest["files"]:
        path = query_dir / record["path"]
        if not path.exists() or _sha256(path) != record["sha256"]:
            raise ValueError(f"Component file changed after export: {path}")
    return {
        "query_name": COMPONENT_QUERY_NAME,
        "config_mode": RECOVERY_VERSION,
        "resolved_config": None,
        "resolved_config_sha256": None,
        "sql_sha256": manifest["component_sql_template_sha256"],
        "source_tables": [
            "ciq_transcripts.ciqtranscriptcomponent",
            "ciq_transcripts.ciqtranscriptcomponenttype",
            "ciq_transcripts.ciqtranscriptperson",
        ],
        "row_count": int(manifest["row_count"]),
        "parquet_files": int(manifest["file_count"]),
        "event_count": int(manifest["event_count"]),
        "event_id_contract_sha256": manifest["event_id_contract_sha256"],
    }


def _validate_completed_export(
    config: ExportJobConfig, config_path: Path, query_dir: Path
) -> dict[str, Any]:
    state_path = query_dir / "_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing exporter state for {config.query.name}")
    state = _read_json(state_path)
    if state.get("query_name") != config.query.name:
        raise ValueError(f"Query-name mismatch in {state_path}")
    if state.get("sql_hash") != config.sql_hash:
        raise ValueError(f"SQL hash mismatch in {state_path}")
    if state.get("config_hash") != config.config_hash:
        raise ValueError(f"Config hash mismatch in {state_path}")
    partitions = state.get("partitions", {})
    expected_names = {
        partition.name for partition in generate_partitions(config.query.partition)
    }
    if set(partitions) != expected_names:
        raise ValueError(
            f"Partition set mismatch for {config.query.name}: "
            f"expected {sorted(expected_names)}, found {sorted(partitions)}"
        )
    row_count = 0
    files: list[dict[str, Any]] = []
    for name in sorted(expected_names):
        record = partitions[name]
        if record.get("status") != "completed":
            raise ValueError(f"Query {config.query.name} partition {name} is incomplete")
        partition_rows = 0
        for item in record.get("files", []):
            path = query_dir / item["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(item["bytes"]):
                raise ValueError(f"File-size mismatch for {path}")
            rows = int(pq.ParquetFile(path).metadata.num_rows)
            if rows != int(item["rows"]):
                raise ValueError(f"Parquet row-count mismatch for {path}")
            partition_rows += rows
            files.append(_file_record(path, query_dir))
        if partition_rows != int(record.get("total_rows", -1)):
            raise ValueError(f"Partition row-count mismatch for {config.query.name} {name}")
        row_count += partition_rows
    return {
        "query_name": config.query.name,
        "resolved_config": str(config_path.resolve()),
        "resolved_config_sha256": _sha256(config_path),
        "sql_sha256": config.sql_hash,
        "source_tables": sorted(
            {
                match.lower()
                for match in SOURCE_TABLE_PATTERN.findall(config.query.sql)
            }
        ),
        "row_count": row_count,
        "parquet_files": len(files),
        "files": files,
    }


def _quarantine_interrupted_components(
    query_dir: Path, archive_dir: Path, *, recovery_root: Path
) -> dict[str, Any] | None:
    if not query_dir.exists():
        return None
    part_files = sorted(query_dir.glob("**/part-*.parquet*"))
    if part_files:
        raise ValueError(
            "Interrupted component query contains data files; refusing automatic quarantine"
        )
    state_path = query_dir / "_state.json"
    if not state_path.exists():
        raise ValueError(
            "Interrupted component directory has no state file; refusing automatic quarantine"
        )
    state = _read_json(state_path)
    if state.get("query_name") != COMPONENT_QUERY_NAME:
        raise ValueError("Interrupted component state has the wrong query name")
    records = list(state.get("partitions", {}).values())
    if not records or any(
        int(record.get("total_rows", 0)) != 0 or record.get("files")
        for record in records
    ):
        raise ValueError(
            "Interrupted component state is not the expected zero-row attempt"
        )
    _assert_descendant(query_dir, query_dir.parent)
    _assert_descendant(archive_dir, recovery_root)
    if archive_dir.exists():
        raise FileExistsError(archive_dir)
    shutil.move(str(query_dir), str(archive_dir))
    return {
        "archived_path": archive_dir.relative_to(recovery_root).as_posix(),
        "files": [
            _file_record(path, archive_dir)
            for path in sorted(archive_dir.rglob("*"))
            if path.is_file()
        ],
    }


def _write_recovery_configs(
    original_paths: list[Path], destination: Path
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for source in original_paths:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["export"]["overwrite"] = False
        payload["export"]["resume"] = True
        target = destination / source.name
        target.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=1000), encoding="utf-8"
        )
        records.append(
            {
                "path": target.relative_to(destination.parent).as_posix(),
                "sha256": _sha256(target),
            }
        )
    return records


def _ordered_resolved_configs(bundle_path: Path, resolved_dir: Path) -> list[Path]:
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    names = [
        Path(str(value)).name
        for key in ("custom_configs", "passthrough_configs")
        for value in bundle.get(key, [])
    ]
    paths = [resolved_dir / name for name in names]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing resolved configs: {missing}")
    return paths


def _load_or_initialize_component_state(
    path: Path, *, contract_hash: str, batch_size: int
) -> dict[str, Any]:
    if path.exists():
        state = _read_json(path)
        if state.get("recovery_version") != RECOVERY_VERSION:
            raise ValueError(f"Component state uses an incompatible mode: {path}")
        if state.get("event_id_contract_sha256") != contract_hash:
            raise ValueError("Component state event-ID contract changed")
        if int(state.get("batch_size", -1)) != batch_size:
            raise ValueError("Component state batch size changed")
        return state
    state = {
        "query_name": COMPONENT_QUERY_NAME,
        "recovery_version": RECOVERY_VERSION,
        "event_id_contract_sha256": contract_hash,
        "component_sql_template_sha256": hashlib.sha256(
            COMPONENT_SQL_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "batch_size": batch_size,
        "status": "running",
        "started_at_utc": _utc_now(),
        "batches": {},
    }
    _atomic_write_json(path, state)
    return state


def _completed_component_batch_is_valid(
    record: dict[str, Any] | None, target: Path, expected_ids: list[int]
) -> bool:
    if not record or record.get("status") != "completed":
        return False
    if record.get("expected_transcript_ids_sha256") != _ids_sha256(expected_ids):
        raise ValueError(f"Completed batch ID contract changed for {target}")
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.stat().st_size != int(record["file_bytes"]):
        raise ValueError(f"Completed batch size changed: {target}")
    if _sha256(target) != record["file_sha256"]:
        raise ValueError(f"Completed batch hash changed: {target}")
    return True


def _validate_existing_recovery_contract(
    manifest: dict[str, Any],
    *,
    bundle_path: Path,
    raw_root: Path,
    contract_root: Path,
    batch_size: int,
) -> None:
    expected = {
        "recovery_version": RECOVERY_VERSION,
        "bundle_sha256": _sha256(bundle_path),
        "raw_root": str(raw_root),
        "contract_root": str(contract_root),
        "batch_size": batch_size,
    }
    changed = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if changed:
        raise ValueError(f"Recovery contract changed: {changed}")


def _validate_initial_inventory(manifest: dict[str, Any], raw_root: Path) -> None:
    for query in manifest["completed_before_recovery"]:
        query_dir = raw_root / query["query_name"]
        for record in query["files"]:
            path = query_dir / record["path"]
            if not path.exists() or _sha256(path) != record["sha256"]:
                raise ValueError(f"Completed pre-recovery file changed: {path}")


def _validate_recovery_files(
    manifest: dict[str, Any], recovery_root: Path
) -> None:
    interrupted = manifest.get("interrupted_component_query")
    if interrupted:
        archive_root = recovery_root / interrupted["archived_path"]
        for record in interrupted["files"]:
            path = archive_root / record["path"]
            if not path.exists() or _sha256(path) != record["sha256"]:
                raise ValueError(f"Archived interrupted state changed: {path}")
    for record in manifest.get("recovery_configs", []):
        path = recovery_root / record["path"]
        if not path.exists() or _sha256(path) != record["sha256"]:
            raise ValueError(f"Recovery config changed: {path}")


def _normalize_transcript_ids(values: list[Any]) -> list[int]:
    series = pd.Series(values, dtype="object")
    if series.empty:
        return []
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or (numeric <= 0).any():
        raise ValueError("Transcript IDs must be positive integers")
    integers = numeric.astype("int64")
    if not (numeric.astype("float64") == integers.astype("float64")).all():
        raise ValueError("Transcript IDs must be integers")
    normalized = sorted(int(value) for value in integers.tolist())
    if len(normalized) != len(set(normalized)):
        raise ValueError("Transcript IDs must be unique")
    return normalized


def _normalized_id_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any():
        raise ValueError("Identifier column contains null values")
    integers = numeric.astype("int64")
    if not (numeric.astype("float64") == integers.astype("float64")).all():
        raise ValueError("Identifier column contains non-integer values")
    return integers


def _ids_sha256(values: list[int]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.tmp")
    if temp.exists():
        temp.unlink()
    frame.to_parquet(temp, index=False, compression="snappy")
    temp.replace(target)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temp.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_error(exc: Exception) -> str:
    text = re.sub(r"(?i)(password\s*[=:]\s*)\S+", r"\1<redacted>", str(exc))
    return text[:2000]


def _assert_descendant(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes intended root: {resolved_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely recover the interrupted FinNLP 2025 WRDS export."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    status_parser = subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_holdout_recovery(batch_size=args.batch_size)
    elif args.command == "run":
        result = run_holdout_recovery(batch_size=args.batch_size)
    else:
        result = recovery_status()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
