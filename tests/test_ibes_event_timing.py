import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from when_answers_surprise.data.ibes_events import (  # noqa: E402
    _match_event,
    actual_timing_audit,
    actual_known_before_call,
    build_ibes_event_dataset,
    build_actual_announcement_datetime_utc,
    load_ibes_summary,
    write_ibes_matching_report,
)
from when_answers_surprise.empirical.pipeline import (  # noqa: E402
    align_call_to_trading_day,
    classify_call_timing,
)
from when_answers_surprise.empirical import pipeline as empirical_pipeline  # noqa: E402


CALL = pd.Timestamp("2024-01-10 15:00:00", tz="UTC")


def test_actual_timing_rules_cover_before_after_missing_and_exact_equality() -> None:
    cases = [
        ("2024-01-09", "", True, "known_prior_day"),
        ("2024-01-11", "12:00:00", False, "after_call_rejection"),
        ("2024-01-10", "09:00:00", True, "known_same_day_before_call"),
        ("2024-01-10", "11:00:00", False, "after_call_rejection"),
        ("2024-01-10", None, False, "same_day_missing_time"),
        ("2024-01-10", "10:00:00", False, "known_exact_equality"),
    ]

    for announcement_date, announcement_time, known, reason in cases:
        actual_dt, actual_known, rejection_reason = actual_known_before_call(
            CALL, announcement_date, announcement_time
        )
        assert actual_known is known
        assert rejection_reason == reason
        if announcement_time:
            assert pd.notna(actual_dt)


def test_actual_announcement_datetime_uses_new_york_dst() -> None:
    actual = build_actual_announcement_datetime_utc(
        "2024-07-01", "09:30:00", timezone="America/New_York"
    )

    assert actual == pd.Timestamp("2024-07-01 13:30:00", tz="UTC")


def test_loader_preserves_colon_formatted_announcement_times(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "TICKER": "ABC",
                "STATPERS": "2024-01-09",
                "FPEDATS": "2024-01-10",
                "FPI": 6,
                "MEASURE": "EPS",
                "MEANEST": 1.0,
                "ANNDATS_ACT": "2024-01-10",
                "ANNTIMS_ACT": "09:00:00",
                "ACTUAL": 1.2,
            },
            {
                "TICKER": "ABC",
                "STATPERS": "2024-01-08",
                "FPEDATS": "2024-01-10",
                "FPI": 6,
                "MEASURE": "EPS",
                "MEANEST": 1.0,
                "ANNDATS_ACT": "2024-01-10",
                "ANNTIMS_ACT": "11:00:00",
                "ACTUAL": 1.1,
            },
        ]
    )
    path = tmp_path / "ibes.csv"
    raw.to_csv(path, index=False)

    loaded = load_ibes_summary(path)

    assert loaded["ANNTIMS_ACT"].notna().all()
    loaded_times = set(loaded["ANNTIMS_ACT"].astype(str))
    assert loaded_times == {"09:00:00", "11:00:00"}
    before = actual_known_before_call(CALL, "2024-01-10", "09:00:00")
    after = actual_known_before_call(CALL, "2024-01-10", "11:00:00")
    assert before[1:] == (True, "known_same_day_before_call")
    assert after[1:] == (False, "after_call_rejection")


def test_call_timing_boundaries_are_explicit() -> None:
    assert classify_call_timing(pd.Timestamp("2024-01-10 09:29", tz="America/New_York")) == "premarket"
    assert classify_call_timing(pd.Timestamp("2024-01-10 09:30", tz="America/New_York")) == "market_hours"
    assert classify_call_timing(pd.Timestamp("2024-01-10 15:59:59", tz="America/New_York")) == "market_hours"
    assert classify_call_timing(pd.Timestamp("2024-01-10 16:00", tz="America/New_York")) == "after_close"


def test_daily_return_alignment_keeps_intraday_calls_same_day_and_moves_after_close() -> None:
    trading_days = pd.DatetimeIndex(["2024-01-10", "2024-01-11"])

    assert align_call_to_trading_day(
        pd.Timestamp("2024-01-10 08:00", tz="America/New_York"), trading_days
    ) == pd.Timestamp("2024-01-10")
    assert align_call_to_trading_day(
        pd.Timestamp("2024-01-10 10:00", tz="America/New_York"), trading_days
    ) == pd.Timestamp("2024-01-10")
    assert align_call_to_trading_day(
        pd.Timestamp("2024-01-10 16:00", tz="America/New_York"), trading_days
    ) == pd.Timestamp("2024-01-11")


def test_actual_timing_audit_has_complete_reason_by_call_bucket_grid() -> None:
    event_level = pd.DataFrame(
        {
            "event_datetime_utc": pd.to_datetime(
                [
                    "2024-01-10 13:00:00+00:00",
                    "2024-01-10 15:00:00+00:00",
                    "2024-01-10 22:00:00+00:00",
                ],
                utc=True,
            ),
            "actual_rejection_reason": [
                "known_prior_day",
                "known_same_day_before_call",
                "known_exact_equality",
            ],
        }
    )

    overall, by_bucket = actual_timing_audit(event_level)
    bucket_table = pd.DataFrame(by_bucket)

    assert overall["known_prior_day"] == 1
    assert overall["known_same_day_before_call"] == 1
    assert overall["known_exact_equality"] == 1
    assert overall["same_day_missing_time"] == 0
    assert overall["after_call_rejection"] == 0
    assert set(bucket_table["call_timing_bucket"]) == {
        "premarket",
        "market_hours",
        "after_close",
        "unknown",
    }
    assert (
        bucket_table.loc[
            bucket_table["call_timing_bucket"].eq("after_close")
            & bucket_table["actual_timing_reason"].eq("known_exact_equality"),
            "events",
        ].iloc[0]
        == 1
    )


def test_timing_robustness_skips_underpowered_after_close_group(monkeypatch) -> None:
    n_premarket = 120
    n_after_close = 7
    n_rows = n_premarket + n_after_close
    sample = pd.DataFrame(
        {
            "call_timing_bucket": ["premarket"] * n_premarket
            + ["after_close"] * n_after_close,
            "market_hours_call": [False] * n_rows,
            "permno": [f"firm_{index % 30}" for index in range(n_rows)],
            "event_trading_date": [
                pd.Timestamp("2024-01-01") + pd.Timedelta(days=index % 30)
                for index in range(n_rows)
            ],
        }
    )
    estimated_sample_sizes = []

    def fake_estimate_all_response_curves(subset: pd.DataFrame):
        estimated_sample_sizes.append(len(subset))
        coefficients = pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "outcome": "CAR_0_1",
                    "treatment": "answer_innovation_z",
                    "beta": 0.0,
                    "standard_error": 0.1,
                    "t_stat": 0.0,
                    "n_obs": len(subset),
                    "n_firms": subset["permno"].nunique(),
                    "n_event_dates": subset["event_trading_date"].nunique(),
                    "standard_error_type": "test",
                }
            ]
        )
        return coefficients, pd.DataFrame()

    monkeypatch.setattr(
        empirical_pipeline,
        "estimate_all_response_curves",
        fake_estimate_all_response_curves,
    )

    output = empirical_pipeline.estimate_timing_subsample_responses(sample)
    after_close = output.loc[output["sample_label"].eq("after_close_only")].iloc[0]

    assert n_after_close not in estimated_sample_sizes
    assert after_close["status"] == "insufficient_sample"
    assert after_close["n_sample_rows"] == n_after_close
    assert "rows=7<120" in after_close["insufficient_reason"]
    assert "firms=7<30" in after_close["insufficient_reason"]
    assert "event_dates=7<30" in after_close["insufficient_reason"]
    assert pd.isna(after_close["beta"])


def test_matching_report_contains_complete_timing_breakdown(tmp_path: Path) -> None:
    transcripts = pd.DataFrame(
        [
            {
                "ticker": "ABC",
                "transcriptid": "t1",
                "companyid": "c1",
                "headline": "earnings",
                "mostimportantdateutc": "2024-01-10",
                "mostimportanttimeutc": "15:00:00",
            }
        ]
    )
    ibes = pd.DataFrame(
        [
            {
                "TICKER": "ABC",
                "STATPERS": "2024-01-09",
                "FPEDATS": "2024-01-10",
                "FPI": 6,
                "MEASURE": "EPS",
                "MEANEST": 1.0,
                "ANNDATS_ACT": "2024-01-10",
                "ANNTIMS_ACT": "09:00:00",
                "ACTUAL": 1.2,
            }
        ]
    )
    transcripts_path = tmp_path / "transcripts.parquet"
    ibes_path = tmp_path / "ibes.csv"
    report_path = tmp_path / "matching_report.md"
    transcripts.to_parquet(transcripts_path, index=False)
    ibes.to_csv(ibes_path, index=False)

    outputs = build_ibes_event_dataset(transcripts_path, ibes_path)
    write_ibes_matching_report(
        outputs,
        report_path,
        transcripts_path=transcripts_path,
        ibes_csv_path=ibes_path,
        event_csv_path=tmp_path / "events.csv",
        event_parquet_path=tmp_path / "events.parquet",
        merged_path=tmp_path / "panel.parquet",
    )
    report = report_path.read_text(encoding="utf-8")

    assert "known strictly before the call timestamp" in report
    assert "known_exact_equality" in report
    for reason in (
        "known_prior_day",
        "known_same_day_before_call",
        "same_day_missing_time",
        "after_call_rejection",
    ):
        assert reason in report
    for bucket in ("premarket", "market_hours", "after_close", "unknown"):
        assert bucket in report


def test_after_call_actual_is_unmatched_and_cannot_enter_surprise() -> None:
    event = pd.Series(
        {
            "ticker": "ABC",
            "transcriptid": "t1",
            "companyid": "c1",
            "headline": "earnings",
            "mostimportantdateutc": "2024-01-10",
            "mostimportanttimeutc": "15:00:00",
            "event_datetime_utc": pd.Timestamp("2024-01-10 15:00:00", tz="UTC"),
            "call_date": pd.Timestamp("2024-01-10"),
        }
    )
    ibes = pd.DataFrame(
        [
            {
                "STATPERS": pd.Timestamp("2024-01-09"),
                "FPEDATS": pd.Timestamp("2024-01-10"),
                "ANNDATS_ACT": pd.Timestamp("2024-01-10"),
                "ANNTIMS_ACT": "11:00:00",
                "FPI": 6,
                "MEANEST": 1.0,
                "ACTUAL": 1.2,
                "STDEV": 0.1,
                "NUMEST": 4,
                "TICKER": "ABC",
                "OFTIC": "ABC",
                "CUSIP": "x",
            }
        ]
    )

    matched = _match_event(event, ibes)

    assert matched["ibes_matched"] is False
    assert matched["actual_known_before_call"] is False
    assert matched["actual_rejection_reason"] == "after_call_rejection"
    assert pd.isna(matched["scaled_eps_surprise"])
    assert pd.isna(matched["eps_surprise"])


def test_matching_uses_event_and_internal_ticker_aliases(tmp_path: Path) -> None:
    transcripts = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "transcriptid": f"t{index}",
                "companyid": f"c{index}",
                "headline": "earnings",
                "mostimportantdateutc": "2024-01-10",
                "mostimportanttimeutc": "15:00:00",
            }
            for index, ticker in enumerate(("GOOG", "NWS"), start=1)
        ]
    )
    ibes = pd.DataFrame(
        [
            {
                "TICKER": internal,
                "OFTIC": official,
                "STATPERS": "2024-01-09",
                "FPEDATS": "2023-12-31",
                "FPI": 6,
                "MEASURE": "EPS",
                "MEANEST": 1.0,
                "ANNDATS_ACT": "2024-01-10",
                "ANNTIMS_ACT": "09:00:00",
                "ACTUAL": 1.2,
            }
            for internal, official in (("GOOG", "GOOGL"), ("NWS", "NWSA"))
        ]
    )
    transcripts_path = tmp_path / "transcripts.parquet"
    ibes_path = tmp_path / "ibes.csv"
    transcripts.to_parquet(transcripts_path, index=False)
    ibes.to_csv(ibes_path, index=False)

    result = build_ibes_event_dataset(transcripts_path, ibes_path).event_level

    assert result.set_index("ticker")["ibes_matched"].astype(bool).to_dict() == {
        "GOOG": True,
        "NWS": True,
    }
