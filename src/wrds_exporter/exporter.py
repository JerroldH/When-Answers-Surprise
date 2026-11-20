from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date
import json
import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd

from wrds_exporter.config import ExportJobConfig
from wrds_exporter.connection import close_connection, connect_wrds
from wrds_exporter.partitioning import QueryPartition, generate_partitions
from wrds_exporter.state import ExportState, StateStore, utc_now_iso


class ExportError(RuntimeError):
    """Raised when a WRDS export cannot complete."""


ConnectionFactory = Callable[[str], Any]
SleepFn = Callable[[float], None]


class WRDSExporter:
    def __init__(
        self,
        config: ExportJobConfig,
        *,
        connection_factory: ConnectionFactory = connect_wrds,
        sleep_fn: SleepFn = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.connection_factory = connection_factory
        self.sleep_fn = sleep_fn
        self.logger = logger or logging.getLogger("wrds_exporter")
        self.query_dir = config.query_output_dir
        self.state_store = StateStore(
            self.query_dir,
            query_name=config.query.name,
            sql_hash=config.sql_hash,
            config_hash=config.config_hash,
        )

    def run(self, *, force: bool = False, db: Any | None = None) -> dict[str, int]:
        state = self.state_store.load(rebuild_corrupt=force)
        self.query_dir.mkdir(parents=True, exist_ok=True)
        owns_connection = db is None
        if db is None:
            db = self._connect_with_retry()
        summary = {"completed": 0, "skipped": 0, "failed": 0}
        try:
            for partition in generate_partitions(self.config.query.partition):
                if self._should_skip(state, partition, force=force):
                    self.logger.info("Skipping completed partition %s", partition.name)
                    summary["skipped"] += 1
                    continue
                try:
                    self._run_partition_with_retry(db, state, partition)
                    summary["completed"] += 1
                    self._write_manifest(state)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    summary["failed"] += 1
                    if not self.config.export.continue_on_error:
                        raise
            self._write_manifest(state)
            return summary
        finally:
            if owns_connection:
                close_connection(db)

    def _connect_with_retry(self) -> Any:
        max_retries = self.config.export.max_retries
        for attempt in range(max_retries + 1):
            try:
                return self.connection_factory(self.config.connection.username_env)
            except Exception as exc:
                if attempt >= max_retries or _is_non_retryable_sql_error(exc):
                    raise
                delay = _backoff_seconds(attempt)
                self.logger.warning(
                    "WRDS connection failed on attempt %s/%s; retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    _safe_error(exc),
                )
                self.sleep_fn(delay)
        raise ExportError("WRDS connection retry loop exited unexpectedly.")

    def _run_partition_with_retry(
        self,
        db: Any,
        state: ExportState,
        partition: QueryPartition,
    ) -> None:
        max_retries = self.config.export.max_retries
        for attempt in range(max_retries + 1):
            self._clean_partition_files(partition)
            try:
                self._run_partition_once(db, state, partition, attempt=attempt + 1)
                return
            except KeyboardInterrupt:
                self._mark_failed(state, partition, "KeyboardInterrupt")
                self.state_store.save(state)
                self.logger.exception("Interrupted while exporting %s", partition.name)
                raise
            except Exception as exc:
                self._mark_failed(state, partition, _safe_error(exc))
                self.state_store.save(state)
                if attempt >= max_retries or _is_non_retryable_sql_error(exc):
                    self.logger.exception(
                        "Partition %s failed without further retry", partition.name
                    )
                    raise
                delay = _backoff_seconds(attempt)
                self.logger.warning(
                    "Partition %s failed on attempt %s/%s; retrying in %.1fs: %s",
                    partition.name,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    _safe_error(exc),
                )
                self.sleep_fn(delay)

    def _run_partition_once(
        self,
        db: Any,
        state: ExportState,
        partition: QueryPartition,
        *,
        attempt: int,
    ) -> None:
        start_time = utc_now_iso()
        state.partitions[partition.name] = {
            "partition": partition.name,
            "status": "running",
            "files": [],
            "total_rows": 0,
            "started_at": start_time,
            "completed_at": None,
            "error": None,
            "sql_hash": self.config.sql_hash,
            "config_hash": self.config.config_hash,
            "start_date": partition.start_date.isoformat(),
            "end_date": partition.end_date.isoformat(),
            "attempt": attempt,
        }
        self.state_store.save(state)
        self.logger.info(
            "Starting partition %s [%s, %s)",
            partition.name,
            partition.start_date,
            partition.end_date,
        )

        iterator = self._raw_sql_iter(db, partition)
        stats = _PartitionStats(
            date_column=self.config.query.date_columns[0]
            if self.config.query.date_columns
            else None,
            non_null_columns=self.config.query.integrity.non_null_columns,
        )
        files: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(iterator):
            if not isinstance(chunk, pd.DataFrame):
                raise ExportError(
                    f"raw_sql returned a non-DataFrame chunk for {partition.name}."
                )
            final_path = self._part_path(partition, chunk_index)
            rows = _write_parquet_atomic(
                chunk,
                final_path,
                compression=self.config.export.compression,
            )
            file_info = {
                "path": final_path.relative_to(self.query_dir).as_posix(),
                "rows": rows,
                "bytes": final_path.stat().st_size,
            }
            files.append(file_info)
            stats.update(chunk)
            self.logger.info(
                "Partition %s chunk %05d rows=%s total_rows=%s path=%s",
                partition.name,
                chunk_index,
                rows,
                stats.total_rows,
                final_path,
            )

        record = state.partitions[partition.name]
        record.update(
            {
                "status": "completed",
                "files": files,
                "total_rows": stats.total_rows,
                "file_count": len(files),
                "total_bytes": sum(int(file_info["bytes"]) for file_info in files),
                "date_column": stats.date_column,
                "min_date": _date_to_iso(stats.min_date),
                "max_date": _date_to_iso(stats.max_date),
                "non_null": stats.non_null,
                "completed_at": utc_now_iso(),
                "error": None,
            }
        )
        self.state_store.save(state)
        self.logger.info(
            "Completed partition %s rows=%s files=%s bytes=%s",
            partition.name,
            record["total_rows"],
            record["file_count"],
            record["total_bytes"],
        )

    def _raw_sql_iter(self, db: Any, partition: QueryPartition) -> Iterable[pd.DataFrame]:
        params = {
            **_sql_params(self.config.query.params),
            **partition.params,
        }
        return db.raw_sql(
            self.config.query.sql,
            params=params,
            date_cols=self.config.query.date_columns,
            chunksize=self.config.export.chunksize,
            return_iter=True,
        )

    def _should_skip(
        self,
        state: ExportState,
        partition: QueryPartition,
        *,
        force: bool,
    ) -> bool:
        if force or not self.config.export.resume:
            return False
        record = state.partitions.get(partition.name)
        if not record:
            return False
        return (
            record.get("status") == "completed"
            and record.get("sql_hash") == self.config.sql_hash
            and record.get("config_hash") == self.config.config_hash
        )

    def _mark_failed(
        self,
        state: ExportState,
        partition: QueryPartition,
        error: str,
    ) -> None:
        record = state.partitions.get(partition.name, {"partition": partition.name})
        record.update(
            {
                "status": "failed",
                "completed_at": utc_now_iso(),
                "error": error,
                "sql_hash": self.config.sql_hash,
                "config_hash": self.config.config_hash,
                "start_date": partition.start_date.isoformat(),
                "end_date": partition.end_date.isoformat(),
            }
        )
        state.partitions[partition.name] = record

    def _part_path(self, partition: QueryPartition, chunk_index: int) -> Path:
        return self.query_dir / partition.name / f"part-{chunk_index:05d}.parquet"

    def _clean_partition_files(self, partition: QueryPartition) -> None:
        partition_dir = self.query_dir / partition.name
        if not partition_dir.exists():
            return
        for path in partition_dir.glob("part-*.parquet*"):
            if path.is_file():
                path.unlink()

    def _write_manifest(self, state: ExportState) -> None:
        manifest = {
            "query_name": self.config.query.name,
            "generated_at": utc_now_iso(),
            "sql_hash": self.config.sql_hash,
            "config_hash": self.config.config_hash,
            "partitions": [
                state.partitions[name]
                for name in sorted(state.partitions)
                if state.partitions[name].get("status") == "completed"
            ],
        }
        path = self.query_dir / "_manifest.json"
        tmp_path = path.with_name("_manifest.json.tmp")
        tmp_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)


class _PartitionStats:
    def __init__(self, *, date_column: str | None, non_null_columns: list[str]) -> None:
        self.date_column = date_column
        self.non_null_columns = non_null_columns
        self.total_rows = 0
        self.min_date: pd.Timestamp | None = None
        self.max_date: pd.Timestamp | None = None
        self.non_null = {column: 0 for column in non_null_columns}

    def update(self, chunk: pd.DataFrame) -> None:
        self.total_rows += int(len(chunk))
        if self.date_column and self.date_column in chunk:
            dates = pd.to_datetime(chunk[self.date_column], errors="coerce").dropna()
            if not dates.empty:
                chunk_min = dates.min()
                chunk_max = dates.max()
                self.min_date = chunk_min if self.min_date is None else min(self.min_date, chunk_min)
                self.max_date = chunk_max if self.max_date is None else max(self.max_date, chunk_max)
        for column in self.non_null_columns:
            if column in chunk:
                self.non_null[column] += int(chunk[column].notna().sum())


def _write_parquet_atomic(
    chunk: pd.DataFrame,
    final_path: Path,
    *,
    compression: str,
) -> int:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    chunk.to_parquet(tmp_path, index=False, compression=compression)
    tmp_path.replace(final_path)
    return int(len(chunk))


def _backoff_seconds(attempt: int) -> float:
    return float(min(60, 2**attempt))


def _is_non_retryable_sql_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    markers = (
        "programmingerror",
        "syntax error",
        "undefined table",
        "undefined column",
        "does not exist",
    )
    return any(marker in text for marker in markers)


def _safe_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:500]


def _date_to_iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return value.date().isoformat()


def _sql_params(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sql_params(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_sql_params(item) for item in value)
    return value
