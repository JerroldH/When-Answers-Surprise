from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from when_answers_surprise.data.wrds_sp500 import normalize_ciq_event_times


ALLOWED_FPI = {6, 7, 8, 9}
TICKER_MAP = {"GOOG": "GOOGL"}
REQUIRED_IBES_COLUMNS = {
    "TICKER",
    "STATPERS",
    "FPEDATS",
    "FPI",
    "MEASURE",
    "MEANEST",
}
NUMERIC_COLUMNS = (
    "FPI",
    "MEANEST",
    "MEDEST",
    "STDEV",
    "NUMEST",
    "ACTUAL",
)
DATE_COLUMNS = ("STATPERS", "FPEDATS", "ANNDATS_ACT")
CALL_TIMING_BUCKETS = ("premarket", "market_hours", "after_close", "unknown")
ACTUAL_TIMING_REASONS = (
    "known_prior_day",
    "known_same_day_before_call",
    "known_exact_equality",
    "same_day_missing_time",
    "after_call_rejection",
)


def build_actual_announcement_datetime_utc(
    announcement_date: object,
    announcement_time: object,
    *,
    timezone: str = "America/New_York",
) -> pd.Timestamp:
    """Combine IBES date/time fields and convert the local timestamp to UTC."""

    if pd.isna(announcement_date):
        return pd.NaT
    parsed_time = _parse_ibes_time(announcement_time)
    if parsed_time is None:
        return pd.NaT
    date = pd.Timestamp(announcement_date).normalize()
    naive = date + pd.Timedelta(
        hours=parsed_time[0], minutes=parsed_time[1], seconds=parsed_time[2]
    )
    try:
        localized = naive.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    except (TypeError, ValueError):
        return pd.NaT
    return localized.tz_convert("UTC") if pd.notna(localized) else pd.NaT


def actual_known_before_call(
    call_datetime_utc: object,
    announcement_date: object,
    announcement_time: object,
    *,
    timezone: str = "America/New_York",
) -> tuple[pd.Timestamp, bool, str]:
    """Apply a conservative point-in-time availability rule to one IBES actual."""

    call = pd.to_datetime(call_datetime_utc, utc=True, errors="coerce")
    if pd.isna(call):
        return pd.NaT, False, "missing_call_datetime"
    if pd.isna(announcement_date):
        return pd.NaT, False, "missing_announcement_date"
    call_ny = call.tz_convert(timezone)
    announcement_day = pd.Timestamp(announcement_date).date()
    call_day = call_ny.date()
    actual_datetime = build_actual_announcement_datetime_utc(
        announcement_date, announcement_time, timezone=timezone
    )
    if announcement_day < call_day:
        return actual_datetime, True, "known_prior_day"
    if announcement_day > call_day:
        return actual_datetime, False, "after_call_rejection"
    if pd.isna(announcement_time):
        return actual_datetime, False, "same_day_missing_time"
    if pd.isna(actual_datetime):
        return actual_datetime, False, "invalid_announcement_time"
    if actual_datetime < call:
        return actual_datetime, True, "known_same_day_before_call"
    if actual_datetime == call:
        return actual_datetime, False, "known_exact_equality"
    return actual_datetime, False, "after_call_rejection"


def classify_call_timing_utc(
    call_datetime_utc: object,
    *,
    timezone: str = "America/New_York",
) -> str:
    """Classify a UTC call timestamp using the daily-return alignment buckets."""

    call = pd.to_datetime(call_datetime_utc, utc=True, errors="coerce")
    if pd.isna(call):
        return "unknown"
    local = call.tz_convert(timezone)
    minute = local.hour * 60 + local.minute
    if minute < 9 * 60 + 30:
        return "premarket"
    if minute < 16 * 60:
        return "market_hours"
    return "after_close"


def actual_timing_audit(event_level: pd.DataFrame) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Return complete overall and call-bucket timing counts, including zero cells."""

    data = event_level.copy()
    if "call_timing_bucket" not in data:
        data["call_timing_bucket"] = data["event_datetime_utc"].map(classify_call_timing_utc)
    observed_reasons = [
        str(value)
        for value in data["actual_rejection_reason"].dropna().unique()
        if str(value) not in {"", "nan"}
    ]
    reasons = list(dict.fromkeys([*ACTUAL_TIMING_REASONS, *sorted(observed_reasons)]))
    overall = {
        reason: int(data["actual_rejection_reason"].eq(reason).sum())
        for reason in reasons
    }
    by_bucket = []
    for bucket in CALL_TIMING_BUCKETS:
        bucket_rows = data.loc[data["call_timing_bucket"].eq(bucket)]
        for reason in reasons:
            by_bucket.append(
                {
                    "call_timing_bucket": bucket,
                    "actual_timing_reason": reason,
                    "events": int(bucket_rows["actual_rejection_reason"].eq(reason).sum()),
                }
            )
    return overall, by_bucket


def _parse_ibes_time(value: object) -> tuple[int, int, int] | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return int(value.hour), int(value.minute), int(value.second)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    if ":" in text:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return int(parsed.hour), int(parsed.minute), int(parsed.second)
        return None
    try:
        raw_digits = str(int(float(text)))
    except (TypeError, ValueError):
        return None
    if len(raw_digits) in {3, 4}:
        digits = raw_digits.zfill(4) + "00"
    else:
        digits = raw_digits.zfill(6)
    if len(digits) != 6:
        return None
    hour, minute, second = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour, minute, second


@dataclass(frozen=True)
class IBESEventOutputs:
    event_level: pd.DataFrame
    merged_transcripts: pd.DataFrame
    filtered_ibes: pd.DataFrame
    audit: dict[str, Any]


def build_ibes_event_dataset(
    transcripts_path: Path | pd.DataFrame,
    ibes_csv_path: Path | pd.DataFrame,
) -> IBESEventOutputs:
    transcripts = (
        transcripts_path.copy()
        if isinstance(transcripts_path, pd.DataFrame)
        else pd.read_parquet(transcripts_path)
    )
    _validate_transcripts(transcripts)
    ibes = load_ibes_summary(ibes_csv_path)

    events = transcripts[
        [
            "ticker",
            "transcriptid",
            "companyid",
            "headline",
            "mostimportantdateutc",
            "mostimportanttimeutc",
        ]
    ].copy()
    events["ticker"] = events["ticker"].astype("string").str.strip().str.upper()
    events["ibes_lookup_ticker"] = events["ticker"].replace(TICKER_MAP)
    events["transcriptid"] = events["transcriptid"].astype("string")
    events["call_date"] = pd.to_datetime(
        events["mostimportantdateutc"], errors="coerce"
    ).dt.normalize()
    event_time = normalize_ciq_event_times(events["mostimportanttimeutc"])
    events["event_datetime_utc"] = pd.to_datetime(
        events["mostimportantdateutc"].astype("string").str[:10] + " " + event_time,
        errors="coerce",
        utc=True,
    )

    ibes_by_ticker = _build_ibes_ticker_lookup(ibes)
    matched_rows = [
        _match_event(row, ibes_by_ticker.get(str(row["ibes_lookup_ticker"])))
        for _, row in events.iterrows()
    ]
    event_level = pd.DataFrame(matched_rows)
    event_level = event_level.sort_values(
        ["call_date", "ticker", "transcriptid"], kind="stable"
    ).reset_index(drop=True)
    event_level["call_timing_bucket"] = event_level["event_datetime_utc"].map(
        classify_call_timing_utc
    )

    if "event_datetime_utc" in transcripts and "event_datetime_utc" in event_level:
        left_datetime = pd.to_datetime(transcripts["event_datetime_utc"], utc=True, errors="coerce")
        right_datetime = event_level.set_index("transcriptid")["event_datetime_utc"].reindex(
            transcripts["transcriptid"].astype(str).to_numpy()
        )
        right_datetime = pd.to_datetime(right_datetime, utc=True, errors="coerce").reset_index(drop=True)
        comparable = left_datetime.notna() & right_datetime.notna()
        if comparable.any() and not (left_datetime[comparable].reset_index(drop=True) == right_datetime[comparable].reset_index(drop=True)).all():
            raise ValueError("Transcript and IBES event timestamps disagree")

    transcript_payload = transcripts.drop(columns="event_datetime_utc", errors="ignore")
    merged = transcript_payload.merge(
        event_level.drop(
            columns=[
                "companyid",
                "headline",
                "mostimportantdateutc",
                "mostimportanttimeutc",
            ],
            errors="ignore",
        ),
        on=["ticker", "transcriptid"],
        how="left",
        validate="one_to_one",
    )

    matched = event_level["ibes_matched"]
    timing_counts, timing_counts_by_bucket = actual_timing_audit(event_level)
    audit = {
        "transcript_events": int(len(event_level)),
        "matched_events": int(matched.sum()),
        "unmatched_events": int((~matched).sum()),
        "match_quality_counts": {
            str(name): int(count)
            for name, count in event_level["match_quality"]
            .value_counts(dropna=False)
            .items()
        },
        "match_rate_by_ticker": _match_rate_by_ticker(event_level),
        "forecast_age_days": _distribution_summary(
            event_level.loc[matched, "forecast_age_days"]
        ),
        "nonmissing_counts": {
            column: int(event_level[column].notna().sum())
            for column in (
                "actual_eps",
                "consensus_eps",
                "forecast_dispersion",
                "analyst_coverage",
            )
        },
        "unmatched_transcript_ids": event_level.loc[
            ~matched, "transcriptid"
        ].tolist(),
        "filtered_ibes_rows": int(len(ibes)),
        "filtered_ibes_tickers": sorted(ibes["match_ticker"].dropna().unique()),
        "actual_rejection_counts": timing_counts,
        "actual_timing_counts_by_call_timing_bucket": timing_counts_by_bucket,
        "actual_after_call_rejections": int(
            event_level["actual_rejection_reason"].eq("after_call_rejection").sum()
        ),
        "same_day_missing_time_rejections": int(
            event_level["actual_rejection_reason"].eq("same_day_missing_time").sum()
        ),
        "before_after_row_counts": {
            "events_before_matching": int(len(event_level)),
            "actual_known_before_call": int(event_level["actual_known_before_call"].fillna(False).sum()),
            "actual_unavailable_before_call": int(
                (~event_level["actual_known_before_call"].fillna(False)).sum()
            ),
        },
    }
    return IBESEventOutputs(
        event_level=event_level,
        merged_transcripts=merged,
        filtered_ibes=ibes,
        audit=audit,
    )


def load_ibes_summary(path: Path | pd.DataFrame) -> pd.DataFrame:
    ibes = path.copy() if isinstance(path, pd.DataFrame) else _read_ibes_table(path)
    ibes.columns = [str(column).strip().upper() for column in ibes.columns]
    missing = REQUIRED_IBES_COLUMNS - set(ibes.columns)
    if missing:
        raise ValueError(f"IBES CSV is missing required columns: {sorted(missing)}")

    for column in NUMERIC_COLUMNS:
        if column in ibes:
            ibes[column] = pd.to_numeric(ibes[column], errors="coerce")
    for column in DATE_COLUMNS:
        if column in ibes:
            ibes[column] = pd.to_datetime(ibes[column], errors="coerce").dt.normalize()

    official = (
        ibes["OFTIC"].astype("string").str.strip().str.upper()
        if "OFTIC" in ibes
        else pd.Series(pd.NA, index=ibes.index, dtype="string")
    )
    internal = ibes["TICKER"].astype("string").str.strip().str.upper()
    ibes["match_ticker"] = official.fillna(internal).replace(TICKER_MAP)
    ibes["internal_match_ticker"] = internal.replace(TICKER_MAP)
    ibes["ticker_source"] = np.where(official.notna(), "OFTIC", "TICKER")

    fiscal_period = (
        ibes["FISCALP"].astype("string").str.strip().str.upper()
        if "FISCALP" in ibes
        else pd.Series("QTR", index=ibes.index, dtype="string")
    )
    clean = ibes.loc[
        ibes["MEASURE"].astype("string").str.upper().eq("EPS")
        & fiscal_period.eq("QTR")
        & ibes["FPI"].isin(ALLOWED_FPI)
        & ibes["MEANEST"].notna()
        & ibes["STATPERS"].notna()
    ].copy()
    clean = clean.sort_values(
        ["match_ticker", "STATPERS", "FPEDATS", "FPI"], kind="stable"
    ).reset_index(drop=True)
    return clean


def _build_ibes_ticker_lookup(ibes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Index IBES rows under both official and internal ticker aliases."""

    primary = ibes.assign(_lookup_ticker=ibes["match_ticker"])
    internal_aliases = ibes.loc[
        ibes["internal_match_ticker"].notna()
        & ibes["internal_match_ticker"].ne(ibes["match_ticker"])
    ].assign(_lookup_ticker=lambda frame: frame["internal_match_ticker"])
    expanded = pd.concat([primary, internal_aliases], ignore_index=True)
    return {
        str(ticker): group.drop(columns="_lookup_ticker").reset_index(drop=True)
        for ticker, group in expanded.groupby("_lookup_ticker", sort=False)
    }


def _read_ibes_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported IBES file format: {path}")


def write_ibes_matching_report(
    outputs: IBESEventOutputs,
    output_path: Path,
    *,
    transcripts_path: Path,
    ibes_csv_path: Path,
    event_csv_path: Path,
    event_parquet_path: Path,
    merged_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit = outputs.audit
    rates = pd.DataFrame(audit["match_rate_by_ticker"])
    examples = outputs.event_level.loc[
        outputs.event_level["ticker"].isin(["AAPL", "MSFT", "NVDA"])
        & outputs.event_level["ibes_matched"],
        [
            "ticker",
            "transcriptid",
            "call_date",
            "STATPERS",
            "FPEDATS",
            "FPI",
            "consensus_eps",
            "actual_eps",
            "actual_announcement_datetime_utc",
            "actual_known_before_call",
            "actual_rejection_reason",
            "eps_surprise",
            "forecast_dispersion",
            "analyst_coverage",
            "forecast_age_days",
            "match_quality",
        ],
    ].groupby("ticker", sort=True).head(3)
    quality = pd.DataFrame(
        [
            {"match_quality": name, "events": count}
            for name, count in audit["match_quality_counts"].items()
        ]
    )
    nonmissing = pd.DataFrame(
        [
            {"variable": name, "nonmissing_events": count}
            for name, count in audit["nonmissing_counts"].items()
        ]
    )
    timing_by_bucket = pd.DataFrame(
        audit["actual_timing_counts_by_call_timing_bucket"]
    )
    age = audit["forecast_age_days"]

    text = f"""# IBES Event Matching Report

## Inputs and Outputs

- Clean transcript events: `{_display_path(transcripts_path)}`
- IBES Summary Statistics: `{_display_path(ibes_csv_path)}`
- Event-level CSV: `{_display_path(event_csv_path)}`
- Event-level Parquet: `{_display_path(event_parquet_path)}`
- Transcript + IBES Parquet: `{_display_path(merged_path)}`

## Matching Rule

The primary match requires the same cleaned ticker, `ANNDATS_ACT` within one
calendar day of the call date, `STATPERS` strictly before the call, and an
actual EPS announcement known strictly before the call timestamp. Exact
timestamp equality is reported as `known_exact_equality` but excluded from the
primary sample. IBES actual times are interpreted in `America/New_York` and
converted to UTC.
The latest eligible `STATPERS` is selected. Same-day announcements with a
missing time are rejected conservatively.

The backup match is used only when the primary match is unavailable. It selects
the closest fiscal period end on or before the call date, restricted to 120
days before the call, then takes the latest pre-call `STATPERS` for that period.
`OFTIC` is preferred for ticker matching; `GOOG` is mapped to `GOOGL` if needed.

## Attrition

- Transcript events: {audit['transcript_events']}
- Matched to IBES: {audit['matched_events']}
- Unmatched: {audit['unmatched_events']}
- Filtered IBES rows: {audit['filtered_ibes_rows']}
- Actual after-call rejections: {audit['actual_after_call_rejections']}
- Same-day missing-time rejections: {audit['same_day_missing_time_rejections']}
- Before/after timing counts: {audit['before_after_row_counts']}

{_markdown_table(quality)}

## Actual Timing by Call Bucket

{_markdown_table(timing_by_bucket)}

## Match Rate by Ticker

{_markdown_table(rates)}

## Forecast Age

- Minimum: {age['min']:.1f} days
- 25th percentile: {age['p25']:.1f} days
- Median: {age['median']:.1f} days
- 75th percentile: {age['p75']:.1f} days
- Maximum: {age['max']:.1f} days

## Variable Coverage

{_markdown_table(nonmissing)}

## Matched Examples

{_markdown_table(examples)}

## Unmatched Transcript IDs

{_format_list(audit['unmatched_transcript_ids'])}
"""
    output_path.write_text(text, encoding="utf-8")


def _match_event(event: pd.Series, ibes: pd.DataFrame | None) -> dict[str, Any]:
    ticker = str(event["ticker"])
    call_date = pd.Timestamp(event["call_date"])
    if ibes is None:
        return _unmatched_event_row(event)
    firm = ibes.loc[ibes["STATPERS"].lt(call_date)].copy()

    primary = firm.loc[
        firm["ANNDATS_ACT"].notna()
        & firm["ANNDATS_ACT"].sub(call_date).dt.days.abs().le(1)
    ].copy()
    if not primary.empty:
        selected = _latest_unique_snapshot(primary, call_date)
        quality = "primary_anndats"
        actual_datetime, actual_known, rejection_reason = actual_known_before_call(
            event.get("event_datetime_utc"), selected.get("ANNDATS_ACT"), selected.get("ANNTIMS_ACT")
        )
    else:
        eligible = firm.loc[
            firm["FPEDATS"].notna()
            & firm["FPEDATS"].le(call_date)
            & (call_date - firm["FPEDATS"]).dt.days.between(0, 120)
        ].copy()
        if eligible.empty:
            return _unmatched_event_row(event)
        target_fpedats = eligible["FPEDATS"].max()
        selected = _latest_unique_snapshot(
            eligible.loc[eligible["FPEDATS"].eq(target_fpedats)],
            call_date,
        )
        quality = "backup_fpedats"
        actual_datetime, actual_known, rejection_reason = actual_known_before_call(
            event.get("event_datetime_utc"), selected.get("ANNDATS_ACT"), selected.get("ANNTIMS_ACT")
        )
    if not actual_known:
        return _unmatched_event_row(
            event,
            rejection_reason=rejection_reason,
            actual_datetime=actual_datetime,
        )

    statpers = pd.Timestamp(selected["STATPERS"])
    mean_est = _float_or_nan(selected.get("MEANEST"))
    actual = _float_or_nan(selected.get("ACTUAL"))
    surprise = actual - mean_est if np.isfinite(actual) and np.isfinite(mean_est) else np.nan
    scaled = (
        surprise / abs(mean_est)
        if np.isfinite(surprise) and np.isfinite(mean_est) and mean_est != 0
        else np.nan
    )
    announcement = selected.get("ANNDATS_ACT")
    match_gap = (
        abs((pd.Timestamp(announcement) - call_date).days)
        if pd.notna(announcement)
        else np.nan
    )

    return {
        **_event_identity(event),
        "STATPERS": statpers,
        "FPEDATS": selected.get("FPEDATS"),
        "FPI": _float_or_nan(selected.get("FPI")),
        "MEANEST": mean_est,
        "MEDEST": _float_or_nan(selected.get("MEDEST")),
        "STDEV": _float_or_nan(selected.get("STDEV")),
        "NUMEST": _float_or_nan(selected.get("NUMEST")),
        "ACTUAL": actual,
        "ANNDATS_ACT": announcement,
        "ANNTIMS_ACT": selected.get("ANNTIMS_ACT"),
        "actual_announcement_datetime_utc": actual_datetime,
        "actual_known_before_call": True,
        "actual_rejection_reason": rejection_reason,
        "consensus_eps": mean_est,
        "actual_eps": actual,
        "eps_surprise": surprise,
        "scaled_eps_surprise": scaled,
        "forecast_dispersion": _float_or_nan(selected.get("STDEV")),
        "analyst_coverage": _float_or_nan(selected.get("NUMEST")),
        "forecast_age_days": int((call_date - statpers).days),
        "ibes_matched": True,
        "match_quality": quality,
        "match_date_gap_days": match_gap,
        "ibes_ticker": selected.get("TICKER"),
        "ibes_oftic": selected.get("OFTIC"),
        "ibes_cusip": selected.get("CUSIP"),
    }


def _latest_unique_snapshot(candidates: pd.DataFrame, call_date: pd.Timestamp) -> pd.Series:
    latest_statpers = candidates["STATPERS"].max()
    latest = candidates.loc[candidates["STATPERS"].eq(latest_statpers)].copy()
    if len(latest) > 1:
        latest["_period_gap"] = latest["FPEDATS"].sub(call_date).abs().dt.days
        latest = latest.sort_values(
            ["_period_gap", "FPEDATS", "FPI"], kind="stable"
        )
    return latest.iloc[0]


def _unmatched_event_row(
    event: pd.Series,
    *,
    rejection_reason: str = "unmatched",
    actual_datetime: object = pd.NaT,
) -> dict[str, Any]:
    row = _event_identity(event)
    for column in (
        "STATPERS",
        "FPEDATS",
        "FPI",
        "MEANEST",
        "MEDEST",
        "STDEV",
        "NUMEST",
        "ACTUAL",
        "ANNDATS_ACT",
        "ANNTIMS_ACT",
        "consensus_eps",
        "actual_eps",
        "eps_surprise",
        "scaled_eps_surprise",
        "forecast_dispersion",
        "analyst_coverage",
        "forecast_age_days",
        "match_date_gap_days",
        "ibes_ticker",
        "ibes_oftic",
        "ibes_cusip",
    ):
        row[column] = np.nan
    row["ibes_matched"] = False
    row["match_quality"] = "unmatched"
    row["actual_announcement_datetime_utc"] = actual_datetime
    row["actual_known_before_call"] = False
    row["actual_rejection_reason"] = rejection_reason
    return row


def _event_identity(event: pd.Series) -> dict[str, Any]:
    return {
        "transcriptid": str(event["transcriptid"]),
        "ticker": str(event["ticker"]),
        "companyid": event.get("companyid"),
        "headline": event.get("headline"),
        "mostimportantdateutc": event.get("mostimportantdateutc"),
        "mostimportanttimeutc": event.get("mostimportanttimeutc"),
        "event_datetime_utc": event.get("event_datetime_utc"),
        "call_date": event.get("call_date"),
    }


def _match_rate_by_ticker(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for ticker, group in data.groupby("ticker", sort=True):
        matched = int(group["ibes_matched"].sum())
        rows.append(
            {
                "ticker": ticker,
                "events": int(len(group)),
                "matched": matched,
                "match_rate": matched / len(group),
            }
        )
    return rows


def _distribution_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {name: np.nan for name in ("min", "p25", "median", "p75", "max")}
    return {
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def _float_or_nan(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _validate_transcripts(transcripts: pd.DataFrame) -> None:
    required = {
        "ticker",
        "transcriptid",
        "companyid",
        "headline",
        "mostimportantdateutc",
        "mostimportanttimeutc",
    }
    missing = required - set(transcripts.columns)
    if missing:
        raise ValueError(
            f"Transcript event file is missing required columns: {sorted(missing)}"
        )
    if transcripts["transcriptid"].duplicated().any():
        raise ValueError("Transcript event file contains duplicate transcript IDs")


def _markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "_None_"
    display = data.copy()
    for column in display.select_dtypes(include=["datetime", "datetimetz"]).columns:
        display[column] = display[column].dt.strftime("%Y-%m-%d")
    columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| "
        + " | ".join(str(value).replace("|", "\\|") for value in row)
        + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _format_list(values: list[str]) -> str:
    if not values:
        return "_None_"
    return ", ".join(f"`{value}`" for value in values)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
