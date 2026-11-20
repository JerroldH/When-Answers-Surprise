"""Diagnostics for existing S&P 500 LM tone baseline outputs.

These routines read frozen baseline artifacts and write diagnostic tables. They
must not change the baseline sample or interpretation without an explicit
empirical-design decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from when_answers_surprise.sp500_lm import (
    HORIZONS,
    OOF_TEST_YEARS,
    PRIMARY_FEATURE,
    coefficient_difference_test,
    fit_response_regression,
    response_controls,
    tabular_regression_pipeline,
    zscore,
)


PRIMARY_COMPONENTS = ("prepared", "question", "answer")
INNOVATION_TREATMENTS = [
    "prepared_innovation_z",
    "question_innovation_z",
    "answer_innovation_z",
]
IBES_CONTROL_COLUMNS = {
    "eps_surprise",
    "scaled_eps_surprise",
    "forecast_dispersion",
    "analyst_coverage",
    "forecast_age_days",
}


@dataclass(frozen=True)
class DiagnosticConfig:
    input_dir: Path
    output_dir: Path
    parity_tolerance: float = 1e-8
    min_train_rows: int = 120


def run_lm_diagnostics(config: DiagnosticConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    response_sample = pd.read_parquet(config.input_dir / "response_model_sample.parquet")
    call_innovations = pd.read_parquet(config.input_dir / "call_level_innovations.parquet")
    turn_innovations = pd.read_parquet(config.input_dir / "turn_level_innovations.parquet")
    qa_turns = pd.read_parquet(
        config.input_dir / "qa_turns.parquet",
        columns=["transcriptid", "turn_index", "question_text", "answer_text"],
    )
    existing_response = pd.read_csv(config.input_dir / "response_coefficients.csv")
    timing_path = config.input_dir / "timing_subsample_responses.csv"
    timing_subsamples = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()

    response_sample["transcriptid"] = response_sample["transcriptid"].astype(str)
    full_controls = response_controls(response_sample)
    manifest = build_diagnostic_common_sample_manifest(response_sample, full_controls)
    frozen = restrict_to_manifest(response_sample, manifest)
    sample_audit = [sample_audit_row("p1_control_stability", frozen, manifest)]

    baseline = estimate_response_curve(
        frozen,
        treatments=INNOVATION_TREATMENTS,
        controls=full_controls,
        model="full_controls_current",
    )
    parity = baseline_parity_table(baseline, existing_response, tolerance=config.parity_tolerance)
    if not bool(parity["passes"].all()):
        write_tables(config.output_dir, manifest=manifest, baseline_parity=parity)
        raise RuntimeError("Diagnostic baseline parity failed; refusing to interpret diagnostics.")

    control_stability = pd.concat(
        [
            baseline,
            estimate_response_curve(
                frozen,
                treatments=INNOVATION_TREATMENTS,
                controls=no_ibes_controls(full_controls),
                model="no_ibes_controls",
            ),
        ],
        ignore_index=True,
    )

    raw_frame, raw_validation = build_raw_innovation_frame(call_innovations)
    raw_data = frozen.merge(raw_frame, on="transcriptid", how="left", validate="one_to_one")
    require_complete_ids(raw_data, manifest, raw_frame.columns.difference(["transcriptid"]), "raw_vs_innovation")
    raw_data = add_raw_innovation_zscores(raw_data)
    sample_audit.append(sample_audit_row("p2_raw_vs_innovation", raw_data, manifest))
    raw_outputs = estimate_raw_innovation_models(raw_data, full_controls)

    residual_outputs = estimate_outcome_residualized_models(frozen, full_controls, config.min_train_rows)

    heterogeneity_outputs = estimate_minimal_heterogeneity(frozen, full_controls)
    sample_audit.append(sample_audit_row("p4_heterogeneity", frozen, manifest))
    example_outputs = build_answer_innovation_examples(
        frozen,
        call_innovations,
        turn_innovations,
        qa_turns,
        n_per_tail=5,
    )

    tables = {
        "manifest": manifest,
        "diagnostic_sample_audit": pd.DataFrame(sample_audit),
        "baseline_parity": parity,
        "control_set_stability": control_stability,
        "raw_decomposition_validation": raw_validation,
        "raw_innovation_response": raw_outputs["response"],
        "raw_innovation_correlations": raw_outputs["correlations"],
        "raw_vs_innovation_descriptive_differences": raw_outputs["descriptive_differences"],
        "outcome_residualized_responses": residual_outputs["responses"],
        "outcome_residualization_diagnostics": residual_outputs["diagnostics"],
        "heterogeneity_minimal_interactions": heterogeneity_outputs["interactions"],
        "heterogeneity_minimal_percentiles": heterogeneity_outputs["percentiles"],
        "answer_innovation_examples_blind_review": example_outputs["blind_review"],
        "answer_innovation_examples_key": example_outputs["key"],
        "timing_subsample_responses": timing_subsamples,
        "timing_exclude_market_hours_answer": timing_subsamples.loc[
            timing_subsamples.get("sample_label", pd.Series(dtype=object)).eq("exclude_market_hours")
            & timing_subsamples.get("treatment", pd.Series(dtype=object)).eq("answer_innovation_z")
        ].copy(),
    }
    write_tables(config.output_dir, **tables)

    summary = {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "frozen_common_sample": int(len(manifest)),
        "full_controls_current": full_controls,
        "baseline_parity_passed": bool(parity["passes"].all()),
        "outputs": sorted(f"{name}.csv" for name in tables),
    }
    (config.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def build_diagnostic_common_sample_manifest(
    response_sample: pd.DataFrame, full_controls: Sequence[str]
) -> pd.DataFrame:
    required = [
        "transcriptid",
        "permno",
        "event_trading_date",
        "event_calendar_quarter",
        "event_year",
        *[f"CAR_0_{horizon}" for horizon in HORIZONS],
        *INNOVATION_TREATMENTS,
        *full_controls,
    ]
    missing = [col for col in required if col not in response_sample.columns]
    if missing:
        raise KeyError(f"Missing required diagnostic sample columns: {missing}")
    clean = response_sample.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    if clean["transcriptid"].duplicated().any():
        dupes = clean.loc[clean["transcriptid"].duplicated(), "transcriptid"].head(5).tolist()
        raise ValueError(f"Duplicate transcript IDs in response sample: {dupes}")
    manifest = clean[["transcriptid", "event_year", "event_trading_date"]].copy()
    manifest["diagnostic_sample_order"] = np.arange(len(manifest), dtype=int)
    return manifest.reset_index(drop=True)


def restrict_to_manifest(data: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest[["transcriptid", "diagnostic_sample_order"]].merge(
        data.assign(transcriptid=data["transcriptid"].astype(str)),
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    return out.sort_values("diagnostic_sample_order").drop(columns=["diagnostic_sample_order"]).reset_index(drop=True)


def sample_audit_row(label: str, data: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    data_ids = data["transcriptid"].astype(str).tolist()
    manifest_ids = manifest["transcriptid"].astype(str).tolist()
    return {
        "diagnostic": label,
        "n_rows": int(len(data)),
        "n_transcripts": int(pd.Series(data_ids).nunique()),
        "matches_manifest_ordered_ids": bool(data_ids == manifest_ids),
    }


def require_complete_ids(
    data: pd.DataFrame, manifest: pd.DataFrame, required_cols: Sequence[str], label: str
) -> None:
    audit = sample_audit_row(label, data, manifest)
    if not audit["matches_manifest_ordered_ids"]:
        raise ValueError(f"{label} does not match frozen manifest IDs.")
    missing = data[list(required_cols)].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    if missing.any():
        missing_ids = data.loc[missing, "transcriptid"].head(5).tolist()
        raise ValueError(f"{label} has missing raw/innovation values for frozen IDs: {missing_ids}")


def no_ibes_controls(full_controls: Sequence[str]) -> list[str]:
    return [col for col in full_controls if col not in IBES_CONTROL_COLUMNS]


def estimate_response_curve(
    data: pd.DataFrame,
    *,
    treatments: Sequence[str],
    controls: Sequence[str],
    model: str,
    outcomes: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows = []
    outcome_names = list(outcomes or [f"CAR_0_{horizon}" for horizon in HORIZONS])
    for outcome in outcome_names:
        horizon: int | str = int(outcome.replace("CAR_0_", "")) if outcome.startswith("CAR_0_") else outcome
        result = fit_response_regression(
            data,
            outcome=outcome,
            treatments=treatments,
            controls=controls,
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        coef = result.coefficients.copy()
        coef.insert(0, "model", model)
        coef.insert(1, "horizon", horizon)
        coef.insert(2, "outcome", outcome)
        rows.append(coef)
    return pd.concat(rows, ignore_index=True)


def baseline_parity_table(
    diagnostic: pd.DataFrame, existing: pd.DataFrame, *, tolerance: float
) -> pd.DataFrame:
    diag = diagnostic.loc[diagnostic["model"].eq("full_controls_current")].copy()
    merged = diag.merge(
        existing,
        on=["horizon", "outcome", "treatment"],
        suffixes=("_diagnostic", "_existing"),
        how="outer",
        indicator=True,
    )
    for col in ["beta", "standard_error"]:
        merged[f"{col}_abs_diff"] = (
            merged[f"{col}_diagnostic"] - merged[f"{col}_existing"]
        ).abs()
    merged["passes"] = (
        merged["_merge"].eq("both")
        & merged["beta_abs_diff"].le(tolerance)
        & merged["standard_error_abs_diff"].le(tolerance)
    )
    return merged[
        [
            "horizon",
            "outcome",
            "treatment",
            "beta_diagnostic",
            "beta_existing",
            "beta_abs_diff",
            "standard_error_diagnostic",
            "standard_error_existing",
            "standard_error_abs_diff",
            "_merge",
            "passes",
        ]
    ]


def build_raw_innovation_frame(call_innovations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = call_innovations.loc[
        call_innovations["semantic_feature"].eq(PRIMARY_FEATURE)
        & call_innovations["component"].isin(PRIMARY_COMPONENTS)
    ].copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    duplicates = data.duplicated(["transcriptid", "component"], keep=False)
    if duplicates.any():
        dupes = data.loc[duplicates, ["transcriptid", "component"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate primary LM call innovations: {dupes}")

    frames = []
    for source_col, suffix in [
        ("raw_score", "raw"),
        ("expected_score_oof", "expected"),
        ("narrative_innovation", "innovation_score"),
    ]:
        wide = data.pivot(index="transcriptid", columns="component", values=source_col)
        wide = wide.rename(columns={component: f"{component}_{suffix}" for component in PRIMARY_COMPONENTS})
        frames.append(wide)
    out = pd.concat(frames, axis=1).reset_index()

    validation_rows = []
    for component in PRIMARY_COMPONENTS:
        diff = out[f"{component}_raw"] - out[f"{component}_expected"] - out[f"{component}_innovation_score"]
        validation_rows.append(
            {
                "component": component,
                "n_obs": int(diff.notna().sum()),
                "max_abs_raw_minus_expected_minus_innovation": float(diff.abs().max()),
            }
        )
    return out, pd.DataFrame(validation_rows)


def add_raw_innovation_zscores(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for component in PRIMARY_COMPONENTS:
        for suffix in ["raw", "expected", "innovation_score"]:
            out[f"{component}_{suffix}_z_diag"] = zscore(out[f"{component}_{suffix}"])
    return out


def estimate_raw_innovation_models(data: pd.DataFrame, full_controls: Sequence[str]) -> dict[str, pd.DataFrame]:
    raw_treatments = [f"{component}_raw_z_diag" for component in PRIMARY_COMPONENTS]
    innovation_treatments = [f"{component}_innovation_score_z_diag" for component in PRIMARY_COMPONENTS]
    joint_treatments = []
    for component in PRIMARY_COMPONENTS:
        joint_treatments.extend([f"{component}_expected_z_diag", f"{component}_innovation_score_z_diag"])

    response = pd.concat(
        [
            estimate_response_curve(
                data, treatments=raw_treatments, controls=full_controls, model="raw_only"
            ),
            estimate_response_curve(
                data,
                treatments=innovation_treatments,
                controls=full_controls,
                model="innovation_only",
            ),
            estimate_response_curve(
                data,
                treatments=joint_treatments,
                controls=full_controls,
                model="joint_expected_innovation",
            ),
        ],
        ignore_index=True,
    )

    corr_rows = []
    for component in PRIMARY_COMPONENTS:
        cols = [
            f"{component}_raw_z_diag",
            f"{component}_expected_z_diag",
            f"{component}_innovation_score_z_diag",
        ]
        corr = data[cols].corr()
        for left in cols:
            for right in cols:
                if left < right:
                    corr_rows.append(
                        {
                            "component": component,
                            "left": left,
                            "right": right,
                            "correlation": float(corr.loc[left, right]),
                        }
                    )

    diff_rows = []
    raw = response.loc[response["model"].eq("raw_only")].copy()
    innov = response.loc[response["model"].eq("innovation_only")].copy()
    for component in PRIMARY_COMPONENTS:
        raw_name = f"{component}_raw_z_diag"
        innov_name = f"{component}_innovation_score_z_diag"
        merged = raw.loc[raw["treatment"].eq(raw_name)].merge(
            innov.loc[innov["treatment"].eq(innov_name)],
            on=["horizon", "outcome"],
            suffixes=("_raw", "_innovation"),
        )
        for row in merged.itertuples(index=False):
            diff_rows.append(
                {
                    "comparison_type": "descriptive_cross_model",
                    "component": component,
                    "horizon": row.horizon,
                    "outcome": row.outcome,
                    "innovation_minus_raw_beta": float(row.beta_innovation - row.beta_raw),
                    "note": "No joint covariance; descriptive only.",
                }
            )

    return {
        "response": response,
        "correlations": pd.DataFrame(corr_rows),
        "descriptive_differences": pd.DataFrame(diff_rows),
    }


def estimate_outcome_residualized_models(
    data: pd.DataFrame, full_controls: Sequence[str], min_train_rows: int
) -> dict[str, pd.DataFrame]:
    work = add_nuisance_time_features(data)
    response_rows = []
    diagnostic_rows = []

    for outcome in [f"CAR_0_{horizon}" for horizon in HORIZONS]:
        residual, diag = oof_outcome_residuals(
            work,
            outcome=outcome,
            controls=full_controls,
            min_train_rows=min_train_rows,
            profile="main_car",
        )
        resid_col = f"{outcome}_residual"
        model_data = work.assign(**{resid_col: residual}).dropna(subset=[resid_col]).copy()
        result = fit_response_regression(
            model_data,
            outcome=resid_col,
            treatments=INNOVATION_TREATMENTS,
            controls=[],
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        out = result.coefficients.copy()
        out.insert(0, "profile", "main_car")
        out.insert(1, "source_outcome", outcome)
        out.insert(2, "second_stage_outcome", resid_col)
        response_rows.append(out)
        diagnostic_rows.append(diag)

    placebo_controls = placebo_nuisance_controls(work, full_controls)
    for outcome in ["placebo_CAR_m5_m1", "placebo_CAR_m10_m1"]:
        residual, diag = oof_outcome_residuals(
            work,
            outcome=outcome,
            controls=placebo_controls,
            min_train_rows=min_train_rows,
            profile="placebo_no_overlap_return_controls",
        )
        resid_col = f"{outcome}_residual"
        model_data = work.assign(**{resid_col: residual}).dropna(subset=[resid_col]).copy()
        result = fit_response_regression(
            model_data,
            outcome=resid_col,
            treatments=INNOVATION_TREATMENTS,
            controls=[],
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        out = result.coefficients.copy()
        out.insert(0, "profile", "placebo_no_overlap_return_controls")
        out.insert(1, "source_outcome", outcome)
        out.insert(2, "second_stage_outcome", resid_col)
        response_rows.append(out)
        diagnostic_rows.append(diag)

    return {
        "responses": pd.concat(response_rows, ignore_index=True),
        "diagnostics": pd.concat(diagnostic_rows, ignore_index=True),
    }


def add_nuisance_time_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    event_date = pd.to_datetime(out["event_trading_date"], errors="coerce")
    out["event_quarter_of_year"] = event_date.dt.quarter.astype("Int64").astype(str)
    out["event_time_trend"] = pd.to_numeric(out["event_year"], errors="coerce") - pd.to_numeric(
        out["event_year"], errors="coerce"
    ).min()
    if {"pre_ret_60", "pre_ret_20"}.issubset(out.columns):
        out["pre_ret_60_minus_20"] = pd.to_numeric(out["pre_ret_60"], errors="coerce") - pd.to_numeric(
            out["pre_ret_20"], errors="coerce"
        )
    return out


def placebo_nuisance_controls(data: pd.DataFrame, full_controls: Sequence[str]) -> list[str]:
    controls = [col for col in full_controls if col != "pre_ret_20"]
    if "pre_ret_60_minus_20" in data.columns:
        controls.append("pre_ret_60_minus_20")
    return controls


def oof_outcome_residuals(
    data: pd.DataFrame,
    *,
    outcome: str,
    controls: Sequence[str],
    min_train_rows: int,
    profile: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    work = data.copy()
    feature_columns = list(dict.fromkeys([*controls, "permno", "event_quarter_of_year", "event_time_trend"]))
    feature_columns = [col for col in feature_columns if col in work.columns and col != "event_calendar_quarter"]
    categorical = [col for col in ["permno", "event_quarter_of_year", "match_quality"] if col in feature_columns]
    values = pd.to_numeric(work[outcome], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(work["event_year"], errors="coerce").to_numpy()
    residual = np.full(len(work), np.nan, dtype=float)
    rows = []
    for year in OOF_TEST_YEARS:
        train_mask = (years < year) & np.isfinite(values)
        test_mask = years == year
        if train_mask.sum() < min_train_rows or test_mask.sum() == 0:
            rows.append(
                {
                    "profile": profile,
                    "outcome": outcome,
                    "fold_year": year,
                    "status": "skipped",
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "feature_columns_json": json.dumps(feature_columns),
                }
            )
            continue
        model = tabular_regression_pipeline("ridge", feature_columns, categorical)
        model.fit(work.loc[train_mask, feature_columns], values[train_mask])
        pred = model.predict(work.loc[test_mask, feature_columns])
        residual[test_mask] = values[test_mask] - pred
        valid = np.isfinite(values[test_mask]) & np.isfinite(pred)
        rows.append(
            {
                "profile": profile,
                "outcome": outcome,
                "fold_year": year,
                "status": "estimated",
                "n_train": int(train_mask.sum()),
                "n_test": int(valid.sum()),
                "r2": float(r2_score(values[test_mask][valid], pred[valid])) if valid.sum() > 2 else np.nan,
                "mae": float(mean_absolute_error(values[test_mask][valid], pred[valid])) if valid.any() else np.nan,
                "feature_columns_json": json.dumps(feature_columns),
            }
        )
    return residual, pd.DataFrame(rows)


def estimate_minimal_heterogeneity(
    data: pd.DataFrame, full_controls: Sequence[str]
) -> dict[str, pd.DataFrame]:
    specs = [
        {
            "moderator": "low_analyst_coverage_z_diag",
            "source_control": "analyst_coverage",
            "transform": lambda frame: zscore(-pd.to_numeric(frame["analyst_coverage"], errors="coerce")),
        },
        {
            "moderator": "low_liquidity_z_diag",
            "source_control": "pre_amihud_illiquidity_20",
            "transform": lambda frame: zscore(
                np.log(pd.to_numeric(frame["pre_amihud_illiquidity_20"], errors="coerce").clip(lower=np.finfo(float).tiny))
            ),
        },
    ]
    interaction_rows = []
    percentile_rows = []
    for spec in specs:
        moderator = str(spec["moderator"])
        interaction = f"answer_x_{moderator}"
        work = data.copy()
        work[moderator] = spec["transform"](work)
        work[interaction] = work["answer_innovation_z"] * work[moderator]
        controls = moderator_controls(full_controls, str(spec["source_control"]))
        treatments = [*INNOVATION_TREATMENTS, moderator, interaction]
        for horizon in HORIZONS:
            result = fit_response_regression(
                work,
                outcome=f"CAR_0_{horizon}",
                treatments=treatments,
                controls=controls,
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            coef = result.coefficients.copy()
            coef.insert(0, "moderator", moderator)
            coef.insert(1, "removed_source_control", spec["source_control"])
            coef.insert(2, "horizon", horizon)
            interaction_rows.append(coef)
            for pct in [10, 50, 90]:
                value = float(np.nanpercentile(work[moderator], pct))
                percentile_rows.append(
                    implied_answer_effect_row(
                        result,
                        moderator=moderator,
                        horizon=horizon,
                        percentile=pct,
                        moderator_value=value,
                        interaction=interaction,
                    )
                )
    return {
        "interactions": pd.concat(interaction_rows, ignore_index=True),
        "percentiles": pd.DataFrame(percentile_rows),
    }


def moderator_controls(full_controls: Sequence[str], source_control: str) -> list[str]:
    return [col for col in full_controls if col != source_control]


def build_answer_innovation_examples(
    response_sample: pd.DataFrame,
    call_innovations: pd.DataFrame,
    turn_innovations: pd.DataFrame,
    qa_turns: pd.DataFrame,
    *,
    n_per_tail: int = 5,
) -> dict[str, pd.DataFrame]:
    calls = call_innovations.loc[
        call_innovations["semantic_feature"].eq(PRIMARY_FEATURE)
        & call_innovations["component"].eq("answer")
    ].copy()
    calls["transcriptid"] = calls["transcriptid"].astype(str)
    calls = calls[["transcriptid", "raw_score", "expected_score_oof", "narrative_innovation"]].rename(
        columns={
            "raw_score": "call_answer_raw_score",
            "expected_score_oof": "call_answer_expected_score",
            "narrative_innovation": "call_answer_innovation",
        }
    )
    calls = calls.merge(
        response_sample[
            ["transcriptid", "ticker", "call_date", "event_trading_date", "permno"]
        ].assign(transcriptid=lambda frame: frame["transcriptid"].astype(str)),
        on="transcriptid",
        how="inner",
        validate="one_to_one",
    )
    if calls.empty:
        return {"blind_review": pd.DataFrame(), "key": pd.DataFrame()}

    answer_turns = turn_innovations.loc[
        turn_innovations["semantic_feature"].eq(PRIMARY_FEATURE)
        & turn_innovations["component"].eq("answer")
    ].copy()
    answer_turns["transcriptid"] = answer_turns["transcriptid"].astype(str)
    turns = answer_turns[
        ["transcriptid", "turn_index", "raw_score", "expected_score_oof", "narrative_innovation"]
    ].rename(
        columns={
            "raw_score": "turn_answer_raw_score",
            "expected_score_oof": "turn_answer_expected_score",
            "narrative_innovation": "turn_answer_innovation",
        }
    )

    text = qa_turns.copy()
    text["transcriptid"] = text["transcriptid"].astype(str)
    text = text[["transcriptid", "turn_index", "question_text", "answer_text"]]
    candidates = (
        turns.merge(calls, on="transcriptid", how="inner", validate="many_to_one")
        .merge(text, on=["transcriptid", "turn_index"], how="left", validate="one_to_one")
        .reset_index(drop=True)
    )
    candidates["question_word_count"] = candidates["question_text"].map(simple_word_count)
    candidates["answer_word_count"] = candidates["answer_text"].map(simple_word_count)
    substantive = candidates.loc[
        candidates["question_word_count"].ge(3)
        & candidates["answer_word_count"].ge(40)
        & ~candidates["answer_text"].map(is_admin_answer_text)
    ].copy()
    if substantive.empty:
        return {"blind_review": pd.DataFrame(), "key": pd.DataFrame()}

    high = substantive.nlargest(n_per_tail, "turn_answer_innovation").assign(tail="high_positive")
    low = substantive.nsmallest(n_per_tail, "turn_answer_innovation").assign(tail="high_negative")
    examples = pd.concat([low, high], ignore_index=True)
    examples = examples.sort_values(
        ["tail", "turn_answer_innovation"],
        ascending=[True, True],
    ).reset_index(drop=True)
    positive = examples["tail"].eq("high_positive")
    examples.loc[positive, "tail_rank"] = (
        examples.loc[positive, "turn_answer_innovation"].rank(method="first", ascending=False).astype(int)
    )
    examples.loc[~positive, "tail_rank"] = (
        examples.loc[~positive, "turn_answer_innovation"].rank(method="first", ascending=True).astype(int)
    )
    examples = examples.sort_values(["tail", "tail_rank"]).reset_index(drop=True)
    examples["blind_id"] = [f"ANS-{idx:02d}" for idx in range(1, len(examples) + 1)]
    examples["question_excerpt"] = examples["question_text"].map(lambda text: excerpt_text(text, 90))
    examples["answer_excerpt"] = examples["answer_text"].map(lambda text: excerpt_text(text, 180))
    examples["heuristic_tags"] = examples.apply(
        lambda row: heuristic_answer_tags(f"{row.get('question_text', '')} {row.get('answer_text', '')}"),
        axis=1,
    )

    blind_review = examples[
        ["blind_id", "question_excerpt", "answer_excerpt", "heuristic_tags"]
    ].copy()
    blind_review["manual_primary_code"] = ""
    blind_review["manual_secondary_code"] = ""
    blind_review["manual_notes"] = ""
    blind_review["allowed_codes"] = (
        "specificity; guidance; margin_pressure; evasiveness; demand_strength; "
        "cost_pressure; pricing; uncertainty; capital_allocation; other"
    )

    key = examples[
        [
            "blind_id",
            "tail",
            "tail_rank",
            "transcriptid",
            "ticker",
            "call_date",
            "event_trading_date",
            "call_answer_raw_score",
            "call_answer_expected_score",
            "call_answer_innovation",
            "turn_answer_raw_score",
            "turn_answer_expected_score",
            "turn_answer_innovation",
            "turn_index",
            "question_word_count",
            "answer_word_count",
            "heuristic_tags",
        ]
    ].copy()
    return {"blind_review": blind_review, "key": key}


def excerpt_text(text: object, max_words: int) -> str:
    if pd.isna(text):
        return ""
    words = str(text).replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def simple_word_count(text: object) -> int:
    if pd.isna(text):
        return 0
    return len(str(text).split())


def is_admin_answer_text(text: object) -> bool:
    haystack = str(text or "").lower()
    admin_phrases = [
        "thank you for participating",
        "thank you for your attendance",
        "no further questions",
        "conclude the presentation",
        "this concludes",
        "see you",
        "have a good day",
        "line just had cut out",
        "line cut out",
        "don't mean to interrupt",
        "can you repeat",
        "can you just start again",
        "could you repeat",
    ]
    return any(phrase in haystack for phrase in admin_phrases)


def heuristic_answer_tags(text: str) -> str:
    haystack = text.lower()
    patterns = {
        "specificity": ["specific", "precisely", "basis points", "percentage", "%", "$", "million", "billion"],
        "guidance": ["guidance", "outlook", "expect", "forecast", "raise", "lower", "range"],
        "margin_pressure": ["margin pressure", "gross margin", "operating margin", "compression", "headwind"],
        "evasiveness": ["not going to", "won't comment", "too early", "not disclose", "can't quantify"],
        "demand_strength": ["strong demand", "demand remains", "backlog", "orders", "pipeline", "momentum"],
        "cost_pressure": ["cost pressure", "inflation", "freight", "labor cost", "input cost", "tariff"],
        "pricing": ["pricing", "price increase", "price realization", "discount", "mix"],
        "uncertainty": ["uncertain", "volatility", "risk", "visibility", "challenging"],
        "capital_allocation": ["buyback", "dividend", "capex", "capital allocation", "free cash flow"],
    }
    tags = [tag for tag, needles in patterns.items() if any(needle in haystack for needle in needles)]
    return ";".join(tags) if tags else "other"


def implied_answer_effect_row(
    result,
    *,
    moderator: str,
    horizon: int,
    percentile: int,
    moderator_value: float,
    interaction: str,
) -> dict[str, object]:
    names = result.coefficient_names
    answer_idx = names.index("answer_innovation_z")
    interaction_idx = names.index(interaction)
    beta = result.coefficients.set_index("treatment")
    effect = float(
        beta.loc["answer_innovation_z", "beta"] + moderator_value * beta.loc[interaction, "beta"]
    )
    var = (
        result.covariance[answer_idx, answer_idx]
        + moderator_value**2 * result.covariance[interaction_idx, interaction_idx]
        + 2 * moderator_value * result.covariance[answer_idx, interaction_idx]
    )
    se = float(np.sqrt(max(var, 0.0)))
    return {
        "moderator": moderator,
        "horizon": horizon,
        "percentile": percentile,
        "moderator_value": moderator_value,
        "implied_answer_beta": effect,
        "standard_error": se,
        "ci95_low": effect - 1.96 * se,
        "ci95_high": effect + 1.96 * se,
    }


def write_tables(output_dir: Path, **tables: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
