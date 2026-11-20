"""Daily and non-overlapping pre-event audit for revision-v2 residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from when_answers_surprise.empirical.diagnostics import add_nuisance_time_features, placebo_nuisance_controls
from when_answers_surprise.empirical.pipeline import aggregate_turn_innovations, response_controls, zscore
from when_answers_surprise.empirical.pre_event_audit import holm_adjust
from when_answers_surprise.empirical.revision_v2_inference import (
    curve_joint_tests,
    fit_joint_horizon_system,
    simultaneous_bands,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
DEFAULT_REVISION_DIR = DATA_ROOT / "analysis" / "revision_v2"
DEFAULT_CRSP_DIR = DATA_ROOT / "raw" / "wrds_exports" / "crsp_dsf_sp500"
DEFAULT_OUTPUT_DIR = DEFAULT_REVISION_DIR / "event_audit"

TREATMENTS = ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]
DAILY_OUTCOMES = [f"AR_m{offset}" for offset in range(20, 0, -1)]
BIN_DEFINITIONS = {
    "CAR_m20_m16": [f"AR_m{offset}" for offset in range(20, 15, -1)],
    "CAR_m15_m11": [f"AR_m{offset}" for offset in range(15, 10, -1)],
    "CAR_m10_m6": [f"AR_m{offset}" for offset in range(10, 5, -1)],
    "CAR_m5_m1": [f"AR_m{offset}" for offset in range(5, 0, -1)],
}


def read_crsp_daily(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("year=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CRSP parquet files under {directory}")
    columns = ["permno", "date", "ret", "vwretd"]
    return pd.concat((pd.read_parquet(path, columns=columns) for path in files), ignore_index=True)


def build_daily_pre_event_sample(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Attach exact market-calendar AR[-20,-1] outcomes to fixed call events."""

    required = {"transcriptid", "permno", "event_trading_date"}
    if missing := sorted(required.difference(events.columns)):
        raise ValueError(f"Daily pre-event events missing columns: {missing}")
    if events["transcriptid"].duplicated().any():
        raise ValueError("Daily pre-event audit requires unique transcript ids")

    returns = daily.copy()
    returns["permno"] = pd.to_numeric(returns["permno"], errors="coerce").astype("Int64")
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce").dt.normalize()
    returns["abnormal_return"] = pd.to_numeric(returns["ret"], errors="coerce") - pd.to_numeric(
        returns["vwretd"], errors="coerce"
    )
    returns = returns.dropna(subset=["permno", "date", "abnormal_return"])
    if returns.duplicated(["permno", "date"]).any():
        raise ValueError("CRSP daily data contain duplicate permno-date rows")

    market_days = pd.DatetimeIndex(sorted(returns["date"].unique()))
    day_position = pd.Series(np.arange(len(market_days)), index=market_days)
    base = events[["transcriptid", "permno", "event_trading_date"]].copy()
    base["transcriptid"] = base["transcriptid"].astype(str)
    base["event_trading_date"] = pd.to_datetime(
        base["event_trading_date"], errors="coerce"
    ).dt.normalize()
    base["event_position"] = base["event_trading_date"].map(day_position)
    if base["event_position"].isna().any():
        examples = base.loc[base["event_position"].isna(), "event_trading_date"].head().tolist()
        raise ValueError(f"Aligned event dates are absent from CRSP market calendar: {examples}")

    long_rows = []
    event_positions = base["event_position"].astype(int).to_numpy()
    for offset in range(20, 0, -1):
        positions = event_positions - offset
        dates = np.full(len(base), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = positions >= 0
        dates[valid] = market_days.to_numpy()[positions[valid]]
        piece = base[["transcriptid", "permno"]].copy()
        piece["offset"] = offset
        piece["date"] = dates
        long_rows.append(piece)
    long = pd.concat(long_rows, ignore_index=True)
    long = long.merge(
        returns[["permno", "date", "abnormal_return"]],
        on=["permno", "date"],
        how="left",
        validate="many_to_one",
    )
    wide = long.pivot(index="transcriptid", columns="offset", values="abnormal_return")
    wide = wide.rename(columns={offset: f"AR_m{offset}" for offset in wide.columns}).reset_index()
    out = events.copy()
    out["transcriptid"] = out["transcriptid"].astype(str)
    out = out.merge(wide, on="transcriptid", how="left", validate="one_to_one")
    for name, columns in BIN_DEFINITIONS.items():
        out[name] = out[columns].sum(axis=1, min_count=len(columns))
    return out


def safe_controls(data: pd.DataFrame) -> list[str]:
    work = add_nuisance_time_features(data)
    controls = placebo_nuisance_controls(work, response_controls(work))
    if "pre_ret_20" in controls or "pre_ret_60_minus_20" not in controls:
        raise ValueError("Daily pre-event controls overlap the audited outcomes")
    return controls


def run_event_audit(
    *, revision_dir: Path, crsp_dir: Path, output_dir: Path
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    response = pd.read_parquet(revision_dir / "response_model_sample_revision_v2.parquet")
    daily = read_crsp_daily(crsp_dir)
    sample = build_daily_pre_event_sample(response, daily)
    sample = add_nuisance_time_features(sample)
    controls = safe_controls(sample)

    daily_system = fit_joint_horizon_system(
        sample,
        outcomes=DAILY_OUTCOMES,
        treatments=TREATMENTS,
        controls=controls,
    )
    daily_coefficients = daily_system.coefficients.copy()
    daily_coefficients["p_holm_daily_60"] = holm_adjust(daily_coefficients["p_value"])
    daily_tests = curve_joint_tests(
        daily_coefficients, daily_system.covariance, daily_system.coefficient_index
    )
    daily_bands = simultaneous_bands(
        daily_coefficients,
        daily_system.covariance,
        daily_system.coefficient_index,
        seed=20260716,
    )

    bin_system = fit_joint_horizon_system(
        sample,
        outcomes=list(BIN_DEFINITIONS),
        treatments=TREATMENTS,
        controls=controls,
    )
    bin_coefficients = bin_system.coefficients.copy()
    bin_coefficients["p_holm_bin_12"] = holm_adjust(bin_coefficients["p_value"])
    bin_tests = curve_joint_tests(
        bin_coefficients, bin_system.covariance, bin_system.coefficient_index
    )

    question_ladder = estimate_question_ladder_timing(
        sample,
        revision_dir / "question_ladder_turn_residuals.parquet",
        controls=controls,
    )

    coverage = pd.DataFrame(
        [
            {
                "metric": "fixed_response_calls",
                "count": len(sample),
                "share": 1.0,
            },
            {
                "metric": "complete_daily_pre_event_calls",
                "count": int(sample[DAILY_OUTCOMES].notna().all(axis=1).sum()),
                "share": float(sample[DAILY_OUTCOMES].notna().all(axis=1).mean()),
            },
            {
                "metric": "joint_system_calls",
                "count": daily_system.n_obs,
                "share": float(daily_system.n_obs / max(len(sample), 1)),
            },
        ]
    )
    tables = {
        "daily_pre_event_coefficients_conditional": daily_coefficients,
        "daily_pre_event_joint_tests_conditional": daily_tests,
        "daily_pre_event_simultaneous_bands_conditional": daily_bands,
        "five_day_pre_event_coefficients_conditional": bin_coefficients,
        "five_day_pre_event_joint_tests_conditional": bin_tests,
        "daily_pre_event_sample_coverage": coverage,
        "question_ladder_daily_pre_event_coefficients": question_ladder[
            "daily_coefficients"
        ],
        "question_ladder_daily_pre_event_joint_tests": question_ladder["daily_tests"],
        "question_ladder_daily_pre_event_simultaneous_bands": question_ladder[
            "daily_bands"
        ],
        "question_ladder_one_day_responses": question_ladder["one_day"],
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    np.save(output_dir / "daily_pre_event_covariance_conditional.npy", daily_system.covariance)
    np.save(output_dir / "five_day_pre_event_covariance_conditional.npy", bin_system.covariance)
    daily_system.coefficient_index.to_csv(
        output_dir / "daily_pre_event_coefficient_index.csv", index=False
    )
    bin_system.coefficient_index.to_csv(
        output_dir / "five_day_pre_event_coefficient_index.csv", index=False
    )
    summary = {
        "status": "complete",
        "fixed_response_calls": len(sample),
        "joint_system_calls": daily_system.n_obs,
        "daily_outcomes": DAILY_OUTCOMES,
        "five_day_bins": BIN_DEFINITIONS,
        "return_history_control": "pre_ret_60_minus_20",
        "output_dir": str(output_dir),
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def estimate_question_ladder_timing(
    sample: pd.DataFrame,
    turn_residual_path: Path,
    *,
    controls: list[str],
) -> dict[str, pd.DataFrame]:
    """Re-estimate Q0-Q2 timing diagnostics on the same fixed call sample."""

    turns = pd.read_parquet(turn_residual_path)
    daily_rows = []
    test_rows = []
    band_rows = []
    one_day_rows = []
    for variant, group in turns.groupby("variant", sort=False):
        calls = aggregate_turn_innovations(group, "question")
        calls = calls[["transcriptid", "narrative_innovation"]].rename(
            columns={"narrative_innovation": "question_variant_innovation"}
        )
        work = sample.drop(
            columns=["question_innovation", "question_innovation_z"], errors="ignore"
        ).merge(calls, on="transcriptid", how="left", validate="one_to_one")
        if work["question_variant_innovation"].isna().any():
            raise ValueError(f"{variant} question residuals do not cover fixed response calls")
        work["question_innovation_z"] = zscore(work["question_variant_innovation"])

        system = fit_joint_horizon_system(
            work,
            outcomes=DAILY_OUTCOMES,
            treatments=TREATMENTS,
            controls=controls,
        )
        coefficients = system.coefficients.loc[
            system.coefficients["treatment"].eq("question_innovation_z")
        ].copy()
        coefficients.insert(0, "variant", variant)
        coefficients["p_holm_variant_daily_20"] = holm_adjust(coefficients["p_value"])
        daily_rows.append(coefficients)

        tests = curve_joint_tests(
            system.coefficients, system.covariance, system.coefficient_index
        )
        tests = tests.loc[tests["treatment"].eq("question_innovation_z")].copy()
        tests.insert(0, "variant", variant)
        test_rows.append(tests)

        bands = simultaneous_bands(
            system.coefficients,
            system.covariance,
            system.coefficient_index,
            seed=20260716,
        )
        bands = bands.loc[bands["treatment"].eq("question_innovation_z")].copy()
        bands.insert(0, "variant", variant)
        band_rows.append(bands)

        one_day = fit_joint_horizon_system(
            work,
            outcomes=["CAR_0_1"],
            treatments=TREATMENTS,
            controls=response_controls(work),
        ).coefficients
        one_day = one_day.loc[one_day["treatment"].eq("question_innovation_z")].copy()
        one_day.insert(0, "variant", variant)
        one_day_rows.append(one_day)

    return {
        "daily_coefficients": pd.concat(daily_rows, ignore_index=True),
        "daily_tests": pd.concat(test_rows, ignore_index=True),
        "daily_bands": pd.concat(band_rows, ignore_index=True),
        "one_day": pd.concat(one_day_rows, ignore_index=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run revision-v2 daily pre-event audit.")
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument("--crsp-dir", type=Path, default=DEFAULT_CRSP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run_event_audit(
        revision_dir=args.revision_dir,
        crsp_dir=args.crsp_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
