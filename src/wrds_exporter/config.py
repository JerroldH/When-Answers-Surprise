from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from wrds_exporter.partitioning import PartitionConfig
from wrds_exporter.validation import validate_safe_name


class ConfigError(ValueError):
    """Raised when a WRDS export YAML file is invalid."""


@dataclass(frozen=True)
class ConnectionConfig:
    username_env: str = "WRDS_USER"


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path = Path("data/wrds_exports")
    format: str = "parquet"
    compression: str = "snappy"
    chunksize: int = 250_000
    overwrite: bool = False
    resume: bool = True
    max_retries: int = 3
    continue_on_error: bool = False


@dataclass(frozen=True)
class IntegrityConfig:
    non_null_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryConfig:
    name: str
    sql: str
    partition: PartitionConfig
    params: dict[str, Any] = field(default_factory=dict)
    date_columns: list[str] = field(default_factory=list)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)


@dataclass(frozen=True)
class ExportJobConfig:
    connection: ConnectionConfig
    export: ExportConfig
    query: QueryConfig

    @property
    def sql_hash(self) -> str:
        return hashlib.sha256(self.query.sql.encode("utf-8")).hexdigest()

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(_jsonable(asdict(self)), sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def query_output_dir(self) -> Path:
        return self.export.output_dir / self.query.name


def load_config(path: Path) -> ExportJobConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Could not read config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {path}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a YAML mapping.")
    try:
        connection = _connection_config(raw.get("connection") or {})
        export = _export_config(raw.get("export") or {})
        query = _query_config(_required_mapping(raw, "query"), base_dir=path.parent)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return ExportJobConfig(connection=connection, export=export, query=query)


def _connection_config(raw: dict[str, Any]) -> ConnectionConfig:
    username_env = str(raw.get("username_env", "WRDS_USER"))
    if not username_env:
        raise ConfigError("connection.username_env must be non-empty.")
    return ConnectionConfig(username_env=username_env)


def _export_config(raw: dict[str, Any]) -> ExportConfig:
    output_dir = Path(raw.get("output_dir", "data/wrds_exports"))
    export_format = str(raw.get("format", "parquet")).lower()
    if export_format != "parquet":
        raise ConfigError("Only export.format: parquet is supported.")
    chunksize = int(raw.get("chunksize", 250_000))
    if chunksize <= 0:
        raise ConfigError("export.chunksize must be positive.")
    max_retries = int(raw.get("max_retries", 3))
    if max_retries < 0:
        raise ConfigError("export.max_retries must be non-negative.")
    return ExportConfig(
        output_dir=output_dir,
        format=export_format,
        compression=str(raw.get("compression", "snappy")),
        chunksize=chunksize,
        overwrite=bool(raw.get("overwrite", False)),
        resume=bool(raw.get("resume", True)),
        max_retries=max_retries,
        continue_on_error=bool(raw.get("continue_on_error", False)),
    )


def _query_config(raw: dict[str, Any], *, base_dir: Path) -> QueryConfig:
    name = validate_safe_name(str(raw["name"]), label="query.name")
    sql = str(raw["sql"])
    if not sql.strip():
        raise ConfigError("query.sql must be non-empty.")
    partition_raw = _required_mapping(raw, "partition")
    params = _resolve_params(raw.get("params") or {}, base_dir=base_dir)
    if not isinstance(params, dict):
        raise ConfigError("query.params must be a mapping.")
    date_columns = [str(column) for column in raw.get("date_columns", [])]
    integrity_raw = raw.get("integrity") or {}
    if not isinstance(integrity_raw, dict):
        raise ConfigError("query.integrity must be a mapping.")
    integrity = IntegrityConfig(
        non_null_columns=[
            str(column) for column in integrity_raw.get("non_null_columns", [])
        ]
    )
    return QueryConfig(
        name=name,
        sql=sql,
        partition=PartitionConfig(
            type=str(partition_raw["type"]),
            start=partition_raw["start"],
            end=partition_raw["end"],
        ),
        params=params,
        date_columns=date_columns,
        integrity=integrity,
    )


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping.")
    return value


def _resolve_params(value: Any, *, base_dir: Path) -> Any:
    if isinstance(value, dict) and set(value) == {"from_file"}:
        return _read_param_file(base_dir / str(value["from_file"]))
    if isinstance(value, dict):
        return {key: _resolve_params(item, base_dir=base_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_params(item, base_dir=base_dir) for item in value]
    return value


def _read_param_file(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"Could not read parameter file: {path}") from exc
    values = []
    for line in lines:
        clean = line.split("#", 1)[0].strip()
        if clean:
            values.append(clean)
    if not values:
        raise ConfigError(f"Parameter file is empty: {path}")
    return values


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
