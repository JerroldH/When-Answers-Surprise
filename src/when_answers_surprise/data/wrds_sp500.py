from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SAMPLE_START = pd.Timestamp("2018-01-01")
SAMPLE_END = pd.Timestamp("2025-12-31")
CIQ_EVENT_TYPES = (
    "Earnings Calls",
    "Company Conference Presentations",
    "Special Calls",
    "Shareholder/Analyst Calls",
    "Analyst/Investor Day",
)
WORD_PATTERN = re.compile(r"\b[\w]+(?:['-][\w]+)*\b", re.UNICODE)
TEXT_COLUMNS = (
    "full_text",
    "prepared_text",
    "qa_text",
    "question_text",
    "answer_text",
)
COUNT_COLUMNS = (
    "full_word_count",
    "prepared_word_count",
    "qa_word_count",
    "question_word_count",
    "answer_word_count",
    "component_count",
    "prepared_component_count",
    "question_component_count",
    "answer_component_count",
    "operator_component_count",
    "unknown_component_count",
)
CIQ_MIDNIGHT_PLACEHOLDER = re.compile(r"^0{1,2}:00:00(?:\.0+)?$")


@dataclass(frozen=True)
class CIQTextBuildAudit:
    event_rows: int
    event_unique_transcript_ids: int
    component_rows: int
    component_transcript_ids: int
    events_with_components: int
    events_missing_components: int
    duplicate_event_transcript_ids: int
    component_type_counts: dict[str, int]


def read_partitioned_parquet(directory: Path) -> pd.DataFrame:
    """Read the partition layout produced by wrds-export."""
    patterns = ("year=*/part-*.parquet", "month=*/part-*.parquet", "part-*.parquet")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No WRDS parquet part files found under {directory}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def normalize_ciq_event_times(values: pd.Series) -> pd.Series:
    """Return CIQ event times with date-only midnight placeholders set missing."""

    times = values.astype("string").str.strip().replace("", pd.NA)
    placeholders = times.str.fullmatch(CIQ_MIDNIGHT_PLACEHOLDER, na=False)
    return times.mask(placeholders, pd.NA)


def normalize_sp500_membership(
    membership: pd.DataFrame,
    *,
    sample_start: str | pd.Timestamp = SAMPLE_START,
    sample_end: str | pd.Timestamp = SAMPLE_END,
) -> pd.DataFrame:
    data = _lowercase_columns(membership)
    required = {"permno", "mbrstartdt", "mbrenddt"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"S&P 500 membership missing columns: {sorted(missing)}")

    data["permno"] = pd.to_numeric(data["permno"], errors="coerce").astype("Int64")
    data["mbrstartdt"] = pd.to_datetime(data["mbrstartdt"], errors="coerce").dt.normalize()
    data["mbrenddt"] = pd.to_datetime(data["mbrenddt"], errors="coerce").dt.normalize()
    start = pd.Timestamp(sample_start)
    end = pd.Timestamp(sample_end)
    overlaps = data["mbrstartdt"].le(end) & data["mbrenddt"].fillna(pd.Timestamp.max).ge(start)
    result = data.loc[overlaps].copy()
    result["sample_start"] = start
    result["sample_end"] = end
    result["in_sample_start"] = result["mbrstartdt"].where(
        result["mbrstartdt"].gt(start), start
    )
    result["in_sample_end"] = result["mbrenddt"].where(
        result["mbrenddt"].lt(end), end
    )
    return result.sort_values(["permno", "mbrstartdt"], kind="stable").reset_index(drop=True)


def build_security_master(
    membership: pd.DataFrame,
    names: pd.DataFrame,
    ccm_links: pd.DataFrame | None = None,
    ibes_links: pd.DataFrame | None = None,
    ciq_links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    members = normalize_sp500_membership(membership)
    universe = set(members["permno"].dropna().astype(int))
    name_rows = _lowercase_columns(names)
    if "permno" not in name_rows:
        raise ValueError("CRSP names data missing permno")
    name_rows["permno"] = pd.to_numeric(name_rows["permno"], errors="coerce").astype("Int64")
    name_rows = name_rows.loc[name_rows["permno"].isin(universe)].copy()
    if name_rows.empty:
        name_rows = members[["permno"]].drop_duplicates().copy()

    for column in ("namedt", "nameendt"):
        if column in name_rows:
            name_rows[column] = pd.to_datetime(name_rows[column], errors="coerce").dt.normalize()
    for column in ("ticker", "ncusip", "comnam", "tsymbol"):
        if column in name_rows:
            name_rows[column] = name_rows[column].astype("string").str.strip()
    if "ticker" in name_rows:
        name_rows["ticker_upper"] = name_rows["ticker"].str.upper()

    master = name_rows.copy()
    if ccm_links is not None and not ccm_links.empty:
        ccm = _lowercase_columns(ccm_links)
        if "lpermno" in ccm:
            ccm["permno"] = pd.to_numeric(ccm["lpermno"], errors="coerce").astype("Int64")
            if "gvkey" in ccm:
                ccm["gvkey"] = ccm["gvkey"].map(_normalize_gvkey).astype("string")
            ccm = ccm.loc[ccm["permno"].isin(universe)].copy()
            master = master.merge(
                ccm,
                on="permno",
                how="left",
                suffixes=("", "_ccm"),
            )

    if ibes_links is not None and not ibes_links.empty:
        ibes = _lowercase_columns(ibes_links)
        if "permno" in ibes:
            ibes["permno"] = pd.to_numeric(ibes["permno"], errors="coerce").astype("Int64")
            ibes = ibes.loc[ibes["permno"].isin(universe)].copy()
            ibes = ibes.rename(columns={"ticker": "ibes_ticker"})
            master = master.merge(ibes, on="permno", how="left", suffixes=("", "_ibes"))

    if ciq_links is not None and not ciq_links.empty and "gvkey" in master:
        ciq = _normalize_ciq_gvkey_links(ciq_links)
        if not ciq.empty:
            master = master.merge(ciq, on="gvkey", how="left", suffixes=("", "_ciq"))

    return _order_security_master_columns(master)


def select_latest_ciq_transcript_events(
    events: pd.DataFrame,
    *,
    event_types: Iterable[str] = CIQ_EVENT_TYPES,
    sample_start: str | pd.Timestamp = SAMPLE_START,
    sample_end: str | pd.Timestamp = SAMPLE_END,
) -> pd.DataFrame:
    data = _lowercase_columns(events)
    required = {
        "keydevid",
        "transcriptid",
        "mostimportantdateutc",
        "keydeveventtypename",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CIQ transcript events missing columns: {sorted(missing)}")

    for column in ("keydevid", "transcriptid", "companyid"):
        if column in data:
            data[column] = data[column].map(_normalize_id).astype("string")
    if "ticker" in data:
        data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()
    data["mostimportantdateutc"] = pd.to_datetime(
        data["mostimportantdateutc"], errors="coerce"
    ).dt.normalize()
    for column in ("transcriptcreationdate_utc",):
        if column in data:
            data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    start = pd.Timestamp(sample_start)
    end = pd.Timestamp(sample_end)
    selected_types = {str(value) for value in event_types}
    data = data.loc[
        data["mostimportantdateutc"].between(start, end, inclusive="both")
        & data["keydeveventtypename"].isin(selected_types)
    ].copy()

    data["_creation_date_sort"] = (
        data["transcriptcreationdate_utc"]
        if "transcriptcreationdate_utc" in data
        else pd.NaT
    )
    data["_creation_time_sort"] = (
        data["transcriptcreationtime_utc"].astype("string").fillna("")
        if "transcriptcreationtime_utc" in data
        else ""
    )
    data["_transcriptid_sort"] = pd.to_numeric(data["transcriptid"], errors="coerce")
    data = data.sort_values(
        [
            "keydevid",
            "_creation_date_sort",
            "_creation_time_sort",
            "_transcriptid_sort",
            "transcriptid",
        ],
        kind="stable",
        na_position="first",
    )
    latest = data.drop_duplicates("keydevid", keep="last").drop(
        columns=["_creation_date_sort", "_creation_time_sort", "_transcriptid_sort"]
    )
    if "mostimportanttimeutc" in latest:
        event_time = normalize_ciq_event_times(latest["mostimportanttimeutc"])
        latest["event_datetime_utc"] = pd.to_datetime(
            latest["mostimportantdateutc"].dt.strftime("%Y-%m-%d") + " " + event_time,
            errors="coerce",
            utc=True,
        )
    sort_columns = [
        column
        for column in ("mostimportantdateutc", "ticker", "keydevid", "transcriptid")
        if column in latest
    ]
    return latest.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def build_ciq_transcript_text(
    events: pd.DataFrame,
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, CIQTextBuildAudit]:
    event_rows = select_latest_ciq_transcript_events(events)
    duplicate_event_ids = int(event_rows["transcriptid"].duplicated().sum())
    if duplicate_event_ids:
        raise ValueError("Latest CIQ events contain duplicate transcript IDs")

    component_rows = normalize_ciq_components(components)
    text_rows, type_counts = aggregate_ciq_components(component_rows)
    merged = event_rows.merge(text_rows, on="transcriptid", how="left", validate="one_to_one")
    for column in TEXT_COLUMNS:
        if column not in merged:
            merged[column] = ""
        merged[column] = merged[column].fillna("")
    for column in COUNT_COLUMNS:
        if column not in merged:
            merged[column] = 0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0).astype(int)
    for column in ("has_prepared_text", "has_qa_text", "has_operator_messages", "has_unknown_component_types"):
        if column not in merged:
            merged[column] = False
        merged[column] = merged[column].where(merged[column].notna(), False).astype(bool)

    audit = CIQTextBuildAudit(
        event_rows=int(len(event_rows)),
        event_unique_transcript_ids=int(event_rows["transcriptid"].nunique()),
        component_rows=int(len(component_rows)),
        component_transcript_ids=int(component_rows["transcriptid"].nunique())
        if "transcriptid" in component_rows
        else 0,
        events_with_components=int(merged["component_count"].gt(0).sum()),
        events_missing_components=int(merged["component_count"].eq(0).sum()),
        duplicate_event_transcript_ids=duplicate_event_ids,
        component_type_counts=dict(sorted(type_counts.items())),
    )
    return _order_ciq_text_columns(merged), audit


def normalize_ciq_components(components: pd.DataFrame) -> pd.DataFrame:
    data = _lowercase_columns(components)
    rename = {
        "transcriptcomponenttypename": "componenttypename",
        "transcriptpersonname": "personname",
    }
    data = data.rename(columns={key: value for key, value in rename.items() if key in data})
    required = {"transcriptid", "componentorder", "componenttypename", "componenttext"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CIQ transcript components missing columns: {sorted(missing)}")
    data["transcriptid"] = data["transcriptid"].map(_normalize_id).astype("string")
    data["componentorder"] = pd.to_numeric(data["componentorder"], errors="coerce")
    if "transcriptcomponentid" in data:
        data["transcriptcomponentid"] = pd.to_numeric(
            data["transcriptcomponentid"], errors="coerce"
        )
    data["componenttypename"] = data["componenttypename"].astype("string").str.strip()
    data["componenttext"] = data["componenttext"].map(_clean_text)
    sort_columns = [
        column
        for column in ("transcriptid", "componentorder", "transcriptcomponentid")
        if column in data
    ]
    return data.sort_values(sort_columns, kind="stable", na_position="last").reset_index(drop=True)


def aggregate_ciq_components(
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, Counter[str]]:
    data = normalize_ciq_components(components)
    rows: list[dict[str, Any]] = []
    all_type_counts: Counter[str] = Counter()
    for transcriptid, group in data.groupby("transcriptid", sort=True, dropna=True):
        row, type_counts = _aggregate_one_transcript(str(transcriptid), group)
        rows.append(row)
        all_type_counts.update(type_counts)
    if not rows:
        return pd.DataFrame(columns=["transcriptid", *TEXT_COLUMNS, *COUNT_COLUMNS]), all_type_counts
    return pd.DataFrame(rows), all_type_counts


def _aggregate_one_transcript(
    transcriptid: str,
    components: pd.DataFrame,
) -> tuple[dict[str, Any], Counter[str]]:
    prepared_parts: list[str] = []
    question_parts: list[str] = []
    answer_parts: list[str] = []
    qa_parts: list[str] = []
    full_parts: list[str] = []
    type_counts: Counter[str] = Counter()

    for _, component in components.iterrows():
        component_type = str(component.get("componenttypename") or "").strip()
        text = _clean_text(component.get("componenttext"))
        type_counts[component_type or "<missing>"] += 1
        if not text:
            continue
        if component_type == "Presenter Speech":
            prepared_parts.append(text)
            full_parts.append(text)
        elif component_type == "Question":
            question_parts.append(text)
            qa_parts.append(text)
            full_parts.append(text)
        elif component_type == "Answer":
            answer_parts.append(text)
            qa_parts.append(text)
            full_parts.append(text)
        elif component_type in {"Question + Answer", "Question and Answer"}:
            qa_parts.append(text)
            full_parts.append(text)
        elif component_type == "Presenter Speech + Question + Answer":
            full_parts.append(text)

    prepared_text = "\n\n".join(prepared_parts)
    question_text = "\n\n".join(question_parts)
    answer_text = "\n\n".join(answer_parts)
    qa_text = "\n\n".join(qa_parts)
    full_text = "\n\n".join(full_parts)
    operator_count = sum(
        count for name, count in type_counts.items() if "Operator Message" in name
    )
    known_count = sum(
        type_counts.get(name, 0)
        for name in (
            "Presenter Speech",
            "Question",
            "Answer",
            "Question + Answer",
            "Question and Answer",
            "Presenter Speech + Question + Answer",
        )
    )
    unknown_count = len(components) - known_count - operator_count
    row = {
        "transcriptid": transcriptid,
        "full_text": full_text,
        "prepared_text": prepared_text,
        "qa_text": qa_text,
        "question_text": question_text,
        "answer_text": answer_text,
        "full_word_count": _word_count(full_text),
        "prepared_word_count": _word_count(prepared_text),
        "qa_word_count": _word_count(qa_text),
        "question_word_count": _word_count(question_text),
        "answer_word_count": _word_count(answer_text),
        "component_count": int(len(components)),
        "prepared_component_count": int(type_counts.get("Presenter Speech", 0)),
        "question_component_count": int(type_counts.get("Question", 0)),
        "answer_component_count": int(type_counts.get("Answer", 0)),
        "operator_component_count": int(operator_count),
        "unknown_component_count": int(unknown_count),
        "component_type_counts_json": json.dumps(dict(sorted(type_counts.items()))),
        "has_prepared_text": bool(prepared_text),
        "has_qa_text": bool(qa_text),
        "has_operator_messages": bool(operator_count),
        "has_unknown_component_types": bool(unknown_count),
    }
    return row, type_counts


def _order_security_master_columns(data: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "permno",
        "permco",
        "ticker",
        "ticker_upper",
        "ncusip",
        "cusip",
        "comnam",
        "namedt",
        "nameendt",
        "gvkey",
        "linkprim",
        "linktype",
        "linkdt",
        "linkenddt",
        "ibes_ticker",
        "sdate",
        "edate",
        "score",
        "ciq_companyid",
        "ciq_ticker",
        "ciq_companyname",
        "ciq_startdate",
        "ciq_enddate",
        "primaryflag",
    ]
    columns = [column for column in preferred if column in data.columns]
    columns.extend(column for column in data.columns if column not in columns)
    return data[columns].reset_index(drop=True)


def _order_ciq_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "ticker",
        "companyid",
        "gvkey",
        "keydevid",
        "transcriptid",
        "companyname",
        "headline",
        "mostimportantdateutc",
        "mostimportanttimeutc",
        "event_datetime_utc",
        "keydeveventtypename",
        "transcriptcollectiontypename",
        "transcriptpresentationtypename",
        "full_text",
        "prepared_text",
        "qa_text",
        "question_text",
        "answer_text",
        "full_word_count",
        "prepared_word_count",
        "qa_word_count",
        "question_word_count",
        "answer_word_count",
        "component_count",
        "prepared_component_count",
        "question_component_count",
        "answer_component_count",
        "operator_component_count",
        "unknown_component_count",
        "component_type_counts_json",
        "has_prepared_text",
        "has_qa_text",
        "has_operator_messages",
        "has_unknown_component_types",
        "transcriptcreationdate_utc",
        "transcriptcreationtime_utc",
        "audiolengthsec",
    ]
    columns = [column for column in preferred if column in data.columns]
    columns.extend(column for column in data.columns if column not in columns)
    return data[columns].reset_index(drop=True)


def _lowercase_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def _normalize_ciq_gvkey_links(data: pd.DataFrame) -> pd.DataFrame:
    ciq = _lowercase_columns(data)
    required = {"gvkey", "companyid"}
    missing = required - set(ciq.columns)
    if missing:
        raise ValueError(f"CIQ GVKEY link data missing columns: {sorted(missing)}")

    ciq["gvkey"] = ciq["gvkey"].map(_normalize_gvkey).astype("string")
    ciq["companyid"] = ciq["companyid"].map(_normalize_id).astype("string")
    ciq = ciq.loc[ciq["gvkey"].notna() & ciq["companyid"].notna()].copy()
    ciq = ciq.rename(
        columns={
            "ticker": "ciq_ticker",
            "companyid": "ciq_companyid",
            "companyname": "ciq_companyname",
            "startdate": "ciq_startdate",
            "enddate": "ciq_enddate",
        }
    )
    for column in ("ciq_startdate", "ciq_enddate"):
        if column in ciq:
            ciq[column] = pd.to_datetime(ciq[column], errors="coerce").dt.normalize()
    keep = [
        column
        for column in (
            "gvkey",
            "ciq_companyid",
            "ciq_ticker",
            "ciq_companyname",
            "ciq_startdate",
            "ciq_enddate",
            "primaryflag",
        )
        if column in ciq
    ]
    return ciq[keep].drop_duplicates().reset_index(drop=True)


def _normalize_gvkey(value: Any) -> str | None:
    text = _normalize_id(value)
    if text is None:
        return None
    return text.zfill(6) if text.isdigit() and len(text) < 6 else text


def _normalize_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split())


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))
