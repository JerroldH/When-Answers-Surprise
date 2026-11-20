from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_ACTUAL_COLUMNS = ("ACTUAL", "ANNDATS_ACT", "ANNTIMS_ACT", "CURR_ACT")
ACTUAL_MERGE_KEY = ("TICKER", "MEASURE", "FPEDATS")


@dataclass(frozen=True)
class IBESStagingAudit:
    summary_rows: int
    actual_rows: int
    actual_unique_keys: int
    staged_rows: int
    staged_actual_nonmissing: int
    duplicated_actual_keys: int


def stage_ibes_summary_with_actuals(
    summary_dir: Path,
    actuals_dir: Path,
    output_csv: Path,
) -> IBESStagingAudit:
    summary = _read_partitioned_parquet(summary_dir)
    actuals = _read_partitioned_parquet(actuals_dir)
    staged, audit = merge_summary_with_actuals(summary, actuals)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    staged.to_csv(output_csv, index=False)
    return audit


def merge_summary_with_actuals(
    summary: pd.DataFrame,
    actuals: pd.DataFrame,
) -> tuple[pd.DataFrame, IBESStagingAudit]:
    summary = _normalize_summary(summary)
    actuals = _normalize_actuals(actuals)

    missing_summary = set(ACTUAL_MERGE_KEY) - set(summary.columns)
    missing_actuals = set(ACTUAL_MERGE_KEY) - set(actuals.columns)
    if missing_summary:
        raise ValueError(f"Summary data missing columns: {sorted(missing_summary)}")
    if missing_actuals:
        raise ValueError(f"Actuals data missing columns: {sorted(missing_actuals)}")

    actual_columns = [
        column
        for column in (*ACTUAL_MERGE_KEY, *SUMMARY_ACTUAL_COLUMNS)
        if column in actuals.columns
    ]
    actual_key_duplicates = int(actuals.duplicated(list(ACTUAL_MERGE_KEY)).sum())
    actual_unique = _deduplicate_actuals(actuals)[actual_columns]

    summary_without_actuals = summary.drop(columns=list(SUMMARY_ACTUAL_COLUMNS), errors="ignore")
    staged = summary_without_actuals.merge(
        actual_unique,
        on=list(ACTUAL_MERGE_KEY),
        how="left",
        validate="many_to_one",
    )
    staged = staged[_summary_output_columns(staged)]

    audit = IBESStagingAudit(
        summary_rows=int(len(summary)),
        actual_rows=int(len(actuals)),
        actual_unique_keys=int(len(actual_unique)),
        staged_rows=int(len(staged)),
        staged_actual_nonmissing=int(staged["ACTUAL"].notna().sum())
        if "ACTUAL" in staged
        else 0,
        duplicated_actual_keys=actual_key_duplicates,
    )
    return staged, audit


def _read_partitioned_parquet(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("year=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No partitioned parquet files found under {directory}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def _normalize_summary(data: pd.DataFrame) -> pd.DataFrame:
    result = _uppercase_columns(data)
    for column in ("TICKER", "OFTIC", "MEASURE", "FPI", "FISCALP", "ESTFLAG"):
        if column in result:
            result[column] = result[column].astype("string").str.strip().str.upper()
    for column in ("STATPERS", "FPEDATS", "ANNDATS_ACT"):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    return result


def _normalize_actuals(data: pd.DataFrame) -> pd.DataFrame:
    result = _uppercase_columns(data)
    if "PENDS" in result and "FPEDATS" not in result:
        result = result.rename(columns={"PENDS": "FPEDATS"})
    if "VALUE" in result and "ACTUAL" not in result:
        result = result.rename(columns={"VALUE": "ACTUAL"})
    if "ANNDATS" in result and "ANNDATS_ACT" not in result:
        result = result.rename(columns={"ANNDATS": "ANNDATS_ACT"})
    if "ANNTIMS" in result and "ANNTIMS_ACT" not in result:
        result = result.rename(columns={"ANNTIMS": "ANNTIMS_ACT"})

    for column in ("TICKER", "OFTIC", "MEASURE", "PDICITY"):
        if column in result:
            result[column] = result[column].astype("string").str.strip().str.upper()
    for column in ("FPEDATS", "ANNDATS_ACT", "ACTDATS"):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    if "ACTUAL" in result:
        result["ACTUAL"] = pd.to_numeric(result["ACTUAL"], errors="coerce")
    return result


def _deduplicate_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ("TICKER", "MEASURE", "FPEDATS", "ACTDATS", "ACTTIMS", "ANNDATS_ACT")
        if column in actuals.columns
    ]
    ordered = actuals.sort_values(sort_columns, kind="stable") if sort_columns else actuals
    return ordered.drop_duplicates(list(ACTUAL_MERGE_KEY), keep="last").reset_index(drop=True)


def _summary_output_columns(data: pd.DataFrame) -> list[str]:
    preferred = [
        "TICKER",
        "CUSIP",
        "OFTIC",
        "CNAME",
        "STATPERS",
        "MEASURE",
        "FISCALP",
        "FPI",
        "ESTFLAG",
        "CURCODE",
        "NUMEST",
        "NUMUP",
        "NUMDOWN",
        "MEDEST",
        "MEANEST",
        "STDEV",
        "HIGHEST",
        "LOWEST",
        "USFIRM",
        "FPEDATS",
        "ACTUAL",
        "ANNDATS_ACT",
        "ANNTIMS_ACT",
        "CURR_ACT",
    ]
    return [column for column in preferred if column in data.columns] + [
        column for column in data.columns if column not in preferred
    ]


def _uppercase_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).strip().upper() for column in result.columns]
    return result
