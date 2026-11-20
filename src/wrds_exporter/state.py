from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class StateError(RuntimeError):
    """Raised when the exporter state file cannot be used safely."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ExportState:
    query_name: str
    sql_hash: str
    config_hash: str
    partitions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_name": self.query_name,
            "sql_hash": self.sql_hash,
            "config_hash": self.config_hash,
            "partitions": self.partitions,
        }


class StateStore:
    def __init__(
        self,
        query_dir: Path,
        *,
        query_name: str,
        sql_hash: str,
        config_hash: str,
    ) -> None:
        self.query_dir = query_dir
        self.path = query_dir / "_state.json"
        self.query_name = query_name
        self.sql_hash = sql_hash
        self.config_hash = config_hash

    def load(self, *, rebuild_corrupt: bool = False) -> ExportState:
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            if rebuild_corrupt:
                backup = self.path.with_name(
                    f"_state.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
                )
                self.path.replace(backup)
                return self._empty()
            raise StateError(
                f"State file is not valid JSON: {self.path}. Use --force to rebuild it."
            ) from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("partitions"), dict):
            raise StateError(f"State file has an invalid shape: {self.path}")
        return ExportState(
            query_name=str(raw.get("query_name", self.query_name)),
            sql_hash=str(raw.get("sql_hash", "")),
            config_hash=str(raw.get("config_hash", "")),
            partitions=dict(raw.get("partitions", {})),
        )

    def save(self, state: ExportState) -> None:
        self.query_dir.mkdir(parents=True, exist_ok=True)
        state.query_name = self.query_name
        state.sql_hash = self.sql_hash
        state.config_hash = self.config_hash
        tmp_path = self.path.with_name("_state.json.tmp")
        tmp_path.write_text(
            json.dumps(state.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _empty(self) -> ExportState:
        return ExportState(
            query_name=self.query_name,
            sql_hash=self.sql_hash,
            config_hash=self.config_hash,
        )
