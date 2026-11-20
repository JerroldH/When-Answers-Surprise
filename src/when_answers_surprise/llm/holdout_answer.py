"""Prospective 2025 response check for LLM-scored answer tone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from when_answers_surprise.empirical.holdout_2025 import HOLDOUT_YEAR, RIDGE_ALPHAS
from when_answers_surprise.empirical.pre_event_audit import prepare_pre_event_sample, safe_pre_event_controls
from when_answers_surprise.llm.answer_pipeline import (
    answer_filtration_features,
    estimate_llm_answer_components,
    merge_scores,
    validate_cached_score_coverage,
)
from when_answers_surprise.sp500_lm import HORIZONS, OOF_TEST_YEARS, fit_response_regression


def analyze_llm_holdout(
    *,
    baseline_dir: Path,
    feature_dir: Path,
    score_cache: Path,
    development_response: Path,
    holdout_response: Path,
    output_dir: Path,
    min_train_rows: int = 120,
    variant_workers: int = 1,
) -> dict[str, object]:
    answer = pd.read_parquet(feature_dir / "answer_first_stage_features.parquet")
    qa_turns = pd.read_parquet(baseline_dir / "qa_turns.parquet")
    scores = pd.read_parquet(score_cache)
    coverage = validate_cached_score_coverage(answer, qa_turns=qa_turns, scores=scores)
    bad = coverage["missing_rows"] + coverage["hash_mismatches"] + coverage["duplicate_rows"]
    if bad:
        raise ValueError(f"LLM holdout score coverage failed: {coverage}")
    non_ok = int((~scores["status"].eq("ok")).sum())
    if non_ok:
        raise ValueError(f"LLM holdout score cache contains {non_ok} non-ok rows")

    development = pd.read_parquet(development_response)
    holdout = pd.read_parquet(holdout_response)
    scored_answer = merge_scores(answer, scores)
    feature_sets = answer_filtration_features(scored_answer)
    development_fit = estimate_llm_answer_components(
        scored_answer,
        min_train_rows=min_train_rows,
        outer_years=OOF_TEST_YEARS,
        variant_workers=variant_workers,
    )
    holdout_fit = estimate_llm_answer_components(
        scored_answer,
        min_train_rows=min_train_rows,
        outer_years=(HOLDOUT_YEAR,),
        variant_workers=variant_workers,
    )
    development_calls = development.assign(
        transcriptid=development["transcriptid"].astype(str)
    ).merge(
        development_fit["call_innovations"],
        on="transcriptid",
        how="inner",
        validate="one_to_one",
    )
    call_2025 = holdout_fit["call_innovations"]
    sample = holdout.assign(transcriptid=holdout["transcriptid"].astype(str)).merge(
        call_2025, on="transcriptid", how="inner", validate="one_to_one"
    )

    standardization = []
    for variant in feature_sets:
        column = f"llm_answer_{variant}"
        values = pd.to_numeric(development_calls[column], errors="coerce").dropna()
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid development standard deviation for {column}")
        sample[f"{column}_z"] = (pd.to_numeric(sample[column], errors="coerce") - mean) / sd
        standardization.append(
            {"variant": variant, "development_n": len(values), "mean": mean, "sd": sd}
        )

    z_columns = [f"llm_answer_{variant}_z" for variant in feature_sets]
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=z_columns)
    if sample["transcriptid"].duplicated().any():
        raise ValueError("LLM holdout response sample has duplicate transcript IDs")

    first_stage = pd.concat(
        [
            development_fit["first_stage"].assign(scope="development_2020_2024"),
            holdout_fit["first_stage"].assign(scope="holdout_2025"),
        ],
        ignore_index=True,
    )
    fold_metrics = pd.concat(
        [
            development_fit["fold_metrics"].assign(scope="development_2020_2024"),
            holdout_fit["fold_metrics"].assign(scope="holdout_2025"),
        ],
        ignore_index=True,
    )
    selections = pd.concat(
        [
            development_fit["selections"].assign(scope="development_2020_2024"),
            holdout_fit["selections"].assign(scope="holdout_2025"),
        ],
        ignore_index=True,
    )
    _assert_nested_time_forward(selections, variants=list(feature_sets))
    response = _holdout_responses(sample, list(feature_sets))
    response["beta_bps"] = 10000.0 * pd.to_numeric(response["beta"], errors="coerce")
    placebo = _holdout_placebos(sample, list(feature_sets))
    placebo["beta_bps"] = 10000.0 * pd.to_numeric(placebo["beta"], errors="coerce")

    output_dir.mkdir(parents=True, exist_ok=True)
    first_stage.to_csv(output_dir / "llm_holdout_first_stage.csv", index=False)
    fold_metrics.to_csv(output_dir / "llm_tuned_ridge_fold_metrics.csv", index=False)
    selections.to_csv(
        output_dir / "llm_tuned_ridge_hyperparameter_selection.csv", index=False
    )
    response.to_csv(output_dir / "llm_holdout_response.csv", index=False)
    placebo.to_csv(output_dir / "llm_holdout_placebo.csv", index=False)
    pd.DataFrame(standardization).to_csv(
        output_dir / "llm_development_standardization.csv", index=False
    )
    sample.to_parquet(output_dir / "llm_holdout_response_sample.parquet", index=False)
    (output_dir / "llm_score_cache_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "contract": "llm-answer-a3-holdout-v2-nested-tuned-ridge",
        "score_cache": str(score_cache),
        "score_rows": len(scores),
        "score_providers": sorted(scores["provider"].dropna().astype(str).unique()),
        "development_calls": int(len(development_calls)),
        "holdout_calls_available": int(len(holdout)),
        "holdout_calls_analyzed": int(len(sample)),
        "variants": list(feature_sets),
        "learner": "time_forward_tuned_ridge",
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "selection_rule": "inner expanding-year MAE",
        "development_outer_years": list(OOF_TEST_YEARS),
        "holdout_outer_year": HOLDOUT_YEAR,
        "nested_time_forward_tuning": True,
        "variant_workers": variant_workers,
        "outcomes_loaded_after_scoring": True,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _assert_nested_time_forward(selections: pd.DataFrame, *, variants: list[str]) -> None:
    selected = selections.loc[selections["selected"].astype(bool)].copy()
    expected = len(variants) * (len(OOF_TEST_YEARS) + 1)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} selected nested models, found {len(selected)}")
    outer = pd.to_numeric(selected["outer_year"], errors="coerce")
    inner_max = pd.to_numeric(selected["max_inner_train_year"], errors="coerce")
    if (inner_max.notna() & inner_max.ge(outer)).any():
        raise ValueError("Nested tuning used an inner training year at or after its outer year")


def _holdout_placebos(sample: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    data = prepare_pre_event_sample(sample)
    controls = safe_pre_event_controls(data)
    rows = []
    for variant in variants:
        treatment = f"llm_answer_{variant}_z"
        for outcome in ("placebo_CAR_m10_m6", "placebo_CAR_m5_m1"):
            result = fit_response_regression(
                data,
                outcome=outcome,
                treatments=["prepared_innovation_z", "question_innovation_z", treatment],
                controls=controls,
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            estimate = result.coefficients.loc[result.coefficients["treatment"].eq(treatment)].iloc[
                0
            ]
            rows.append({"variant": variant, "outcome": outcome, **estimate})
    return pd.DataFrame(rows)


def _holdout_responses(sample: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    controls = [
        column
        for column in (
            "scaled_eps_surprise",
            "eps_surprise",
            "pre_ret_20",
            "pre_vol_20",
            "pre_log_dollar_volume_20",
            "pre_amihud_illiquidity_20",
            "analyst_coverage",
            "forecast_dispersion",
            "forecast_age_days",
            "log_market_cap",
            "match_date_gap_days",
            "call_after_close",
            "match_quality",
        )
        if column in sample
    ]
    rows = []
    for variant in variants:
        treatment = f"llm_answer_{variant}_z"
        for horizon in HORIZONS:
            outcome = f"CAR_0_{horizon}"
            result = fit_response_regression(
                sample,
                outcome=outcome,
                treatments=["prepared_innovation_z", "question_innovation_z", treatment],
                controls=controls,
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            estimate = result.coefficients.loc[result.coefficients["treatment"].eq(treatment)].iloc[
                0
            ]
            rows.append({"variant": variant, "horizon": horizon, "outcome": outcome, **estimate})
    return pd.DataFrame(rows)
