"""Revision-v2 empirical layer for the FinNLP paper.

This module consumes the validated codefix2 response sample and first-stage
feature cache. It enriches the cached turn frames with prespecified speaker and
text-content features, estimates Q0-Q2 and A0-A3 projection-residual ladders,
and writes versioned outputs without replacing the frozen baseline artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import mean_absolute_error, r2_score

from when_answers_surprise.empirical.feature_spec import (
    REVISION_CATEGORICAL_FEATURES,
    question_filtration_features,
    revision_answer_filtration_features,
    revision_feature_catalog,
    revision_feature_manifests,
)
from when_answers_surprise.empirical.pipeline import (
    HORIZONS,
    PRIMARY_FEATURE,
    aggregate_turn_innovations,
    base_categorical_features,
    build_qa_turns,
    calibration_slope,
    estimate_all_response_curves,
    fit_response_regression,
    oof_predict_by_year,
    response_controls,
    zscore,
)
from when_answers_surprise.empirical.pre_event_audit import holm_adjust, normal_two_sided_p
from when_answers_surprise.empirical.revision_v2_features import add_strict_analyst_history
from when_answers_surprise.empirical.revision_v2_inference import (
    curve_joint_tests,
    main_joint_system,
    simultaneous_bands,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
ANALYSIS_ROOT = DATA_ROOT / "analysis"
DEFAULT_BASELINE_DIR = ANALYSIS_ROOT / "lm_baseline"
DEFAULT_CACHE_DIR = ANALYSIS_ROOT / "lm_baseline_learner_audit_cache"
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "revision_v2"
DEFAULT_COMPONENTS_PATH = DATA_ROOT / "staging" / "ciq_transcript_components.parquet"

KEY_COLUMNS = ["transcriptid", "turn_index"]
TREATMENTS = ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]
PROXY_TARGETS = ("numeric_density", "quantitative_disclosure", "guidance_density")


@dataclass(frozen=True)
class RevisionV2Config:
    baseline_dir: Path = DEFAULT_BASELINE_DIR
    feature_cache_dir: Path = DEFAULT_CACHE_DIR
    components_path: Path = DEFAULT_COMPONENTS_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_train_rows: int = 120
    max_calls: int | None = None


def run_revision_v2(config: RevisionV2Config) -> dict[str, object]:
    """Build all non-bootstrap revision-v2 analytical artifacts."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    response = pd.read_parquet(config.baseline_dir / "response_model_sample.parquet")
    baseline_turns = pd.read_parquet(config.baseline_dir / "qa_turns.parquet")
    frames = {
        component: pd.read_parquet(
            config.feature_cache_dir / f"{component}_first_stage_features.parquet"
        )
        for component in ("prepared", "question", "answer")
    }
    state = pd.read_parquet(config.feature_cache_dir / "state.parquet")
    if config.max_calls is not None:
        keep_calls = set(
            frames["question"][["event_year", "transcriptid"]]
            .drop_duplicates("transcriptid")
            .sort_values(["event_year", "transcriptid"])
            .groupby("event_year", sort=True)
            .head(config.max_calls)["transcriptid"]
            .astype(str)
        )
        response = _restrict_calls(response, keep_calls)
        baseline_turns = _restrict_calls(baseline_turns, keep_calls)
        frames = {name: _restrict_calls(frame, keep_calls) for name, frame in frames.items()}
        state = _restrict_calls(state, keep_calls)

    components = read_revision_components(config.components_path, baseline_turns["transcriptid"])
    turns = build_revision_turns(baseline_turns, components, frames["question"])
    enriched = enrich_revision_frames(frames["question"], frames["answer"], turns)
    frames["question"] = enriched["question"]
    frames["answer"] = enriched["answer"]

    question_ladder = estimate_turn_ladder(
        frames["question"],
        component="question",
        feature_sets=question_filtration_features(frames["question"]),
        min_train_rows=config.min_train_rows,
    )
    answer_ladder = estimate_turn_ladder(
        frames["answer"],
        component="answer",
        feature_sets=revision_answer_filtration_features(frames["answer"]),
        min_train_rows=config.min_train_rows,
    )
    revision_response = assemble_revision_response(
        response,
        question_ladder["call_innovations"],
        answer_ladder["call_innovations"],
    )

    response_coefficients, response_tests = estimate_all_response_curves(revision_response)
    response_coefficients = add_main_multiplicity(response_coefficients)
    primary_result = build_primary_result(response_coefficients)
    joint_system = main_joint_system(
        revision_response, controls=response_controls(revision_response)
    )
    joint_curve_tests = curve_joint_tests(
        joint_system.coefficients,
        joint_system.covariance,
        joint_system.coefficient_index,
    )
    joint_bands = simultaneous_bands(
        joint_system.coefficients,
        joint_system.covariance,
        joint_system.coefficient_index,
    )
    incremental = estimate_incremental_responses(revision_response)
    answer_year = estimate_answer_year_stability(revision_response)
    answer_year_joint = estimate_answer_year_joint_test(revision_response)
    content = estimate_content_proxy_audit(
        frames["answer"],
        answer_ladder["turn_innovations"],
        revision_response,
        min_train_rows=config.min_train_rows,
    )
    sample_accounting = build_sample_accounting(
        response,
        question_ladder["call_innovations"],
        answer_ladder["call_innovations"],
        frames,
        state,
    )
    speaker_coverage = build_speaker_coverage(turns)
    scale_check = estimate_standardization_invariance(revision_response)

    manifest = revision_feature_manifests()
    manifest.update(
        {
            "primary_estimand": {
                "component": "answer",
                "variant": "A3",
                "outcome": "CAR_0_1",
                "specification": "joint prepared-question-answer response",
            },
            "source_baseline_dir": str(config.baseline_dir),
            "source_feature_cache_dir": str(config.feature_cache_dir),
        }
    )
    add_ladder_fit_to_manifest(
        manifest,
        {
            "question": question_ladder["first_stage"],
            "answer": answer_ladder["first_stage"],
        },
    )

    tables = {
        "question_ladder_first_stage": question_ladder["first_stage"],
        "answer_ladder_first_stage": answer_ladder["first_stage"],
        "question_ladder_responses": estimate_ladder_responses(
            question_ladder, response, component="question"
        ),
        "answer_ladder_responses": estimate_ladder_responses(
            answer_ladder, response, component="answer"
        ),
        "response_coefficients_conditional": response_coefficients,
        "response_coefficient_tests": response_tests,
        "response_joint_coefficient_index": joint_system.coefficient_index,
        "response_curve_joint_tests_conditional": joint_curve_tests,
        "response_simultaneous_bands_conditional": joint_bands,
        "primary_result_conditional": primary_result,
        "incremental_responses": incremental,
        "answer_year_stability": answer_year,
        "answer_year_joint_test": answer_year_joint,
        "content_proxy_first_stage": content["first_stage"],
        "content_proxy_joint_responses": content["joint_responses"],
        "answer_measurement_robustness": content["robustness"],
        "sample_accounting": sample_accounting,
        "speaker_coverage": speaker_coverage,
        "standardization_invariance": scale_check,
        "feature_catalog": revision_feature_catalog(),
    }
    for name, table in tables.items():
        table.to_csv(config.output_dir / f"{name}.csv", index=False)
    np.save(config.output_dir / "response_joint_covariance_conditional.npy", joint_system.covariance)

    turns.to_parquet(config.output_dir / "qa_turns_revision_v2.parquet", index=False)
    frames["question"].to_parquet(
        config.output_dir / "question_first_stage_features_revision_v2.parquet", index=False
    )
    frames["answer"].to_parquet(
        config.output_dir / "answer_first_stage_features_revision_v2.parquet", index=False
    )
    question_ladder["turn_innovations"].to_parquet(
        config.output_dir / "question_ladder_turn_residuals.parquet", index=False
    )
    answer_ladder["turn_innovations"].to_parquet(
        config.output_dir / "answer_ladder_turn_residuals.parquet", index=False
    )
    revision_response.to_parquet(config.output_dir / "response_model_sample_revision_v2.parquet")
    (config.output_dir / "revision_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {
        "status": "complete",
        "estimand_type": "projection_residual",
        "output_dir": str(config.output_dir),
        "response_calls": int(len(revision_response)),
        "question_turns": int(len(frames["question"])),
        "answer_turns": int(len(frames["answer"])),
        "question_variants": list(question_filtration_features()),
        "answer_variants": list(revision_answer_filtration_features()),
        "primary_beta": float(primary_result.iloc[0]["beta"]),
        "primary_conditional_se": float(primary_result.iloc[0]["standard_error"]),
        "outputs": sorted([f"{name}.csv" for name in tables] + [
            "revision_v2_manifest.json",
            "qa_turns_revision_v2.parquet",
            "question_first_stage_features_revision_v2.parquet",
            "answer_first_stage_features_revision_v2.parquet",
            "question_ladder_turn_residuals.parquet",
            "answer_ladder_turn_residuals.parquet",
            "response_model_sample_revision_v2.parquet",
            "response_joint_covariance_conditional.npy",
        ]),
    }
    (config.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def add_ladder_fit_to_manifest(
    manifest: dict[str, object], ladders: dict[str, pd.DataFrame]
) -> None:
    """Attach observed sample and encoded-matrix dimensions to each feature spec."""

    for component, table in ladders.items():
        specifications = manifest.get(component)
        if not isinstance(specifications, dict):
            raise ValueError(f"Feature manifest is missing component: {component}")
        indexed = table.set_index("variant")
        for variant, specification in specifications.items():
            if variant not in indexed.index:
                raise ValueError(f"Feature manifest is missing fitted variant: {component}/{variant}")
            if not isinstance(specification, dict):
                raise TypeError(f"Invalid feature specification: {component}/{variant}")
            row = indexed.loc[variant]
            specification.update(
                {
                    "n_oof_turns": int(row["n_turns"]),
                    "n_oof_calls": int(row["n_calls"]),
                    "min_encoded_features": int(row["min_encoded_features"]),
                    "max_encoded_features": int(row["max_encoded_features"]),
                }
            )


def _restrict_calls(frame: pd.DataFrame, keep_calls: set[str]) -> pd.DataFrame:
    data = frame.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    return data.loc[data["transcriptid"].isin(keep_calls)].reset_index(drop=True)


def read_revision_components(path: Path, transcript_ids: pd.Series) -> pd.DataFrame:
    columns = [
        "transcriptcomponentid",
        "transcriptid",
        "componentorder",
        "componenttypename",
        "transcriptpersonid",
        "personname",
        "proid",
        "speaker_companyname",
        "componenttext",
    ]
    components = pd.read_parquet(path, columns=columns)
    keep = set(transcript_ids.astype(str))
    return components.loc[components["transcriptid"].astype(str).isin(keep)].reset_index(drop=True)


def build_revision_turns(
    baseline_turns: pd.DataFrame,
    components: pd.DataFrame,
    question_frame: pd.DataFrame,
) -> pd.DataFrame:
    events = baseline_turns[
        ["transcriptid", "ticker", "permno", "call_date"]
    ].drop_duplicates("transcriptid")
    turns = build_qa_turns(components, events)
    _assert_unique_turns(turns, "revision-v2 rebuilt turns")
    _assert_unique_turns(baseline_turns, "baseline turns")

    required_keys = question_frame[KEY_COLUMNS].copy()
    required_keys["transcriptid"] = required_keys["transcriptid"].astype(str)
    available = turns[KEY_COLUMNS].copy()
    available["transcriptid"] = available["transcriptid"].astype(str)
    missing = required_keys.merge(available, on=KEY_COLUMNS, how="left", indicator=True)
    missing = missing.loc[missing["_merge"].eq("left_only")]
    if not missing.empty:
        raise ValueError(
            "Rebuilt speaker turns do not cover cached question frame: "
            f"{missing[KEY_COLUMNS].head(5).to_dict('records')}"
        )
    turns = add_strict_analyst_history(turns, question_frame)
    _assert_unique_turns(turns, "revision-v2 history turns")
    return turns


def enrich_revision_frames(
    question_frame: pd.DataFrame,
    answer_frame: pd.DataFrame,
    turns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    analyst_columns = [
        "analyst_key",
        "analyst_brokerage",
        "analyst_prior_question_count",
        "analyst_prior_question_tone_mean",
        "analyst_firm_prior_question_count",
        "analyst_firm_prior_question_tone_mean",
        "analyst_days_since_last_question",
    ]
    question_columns = [
        "question_numeric_density",
        "question_quantitative_disclosure",
        "question_guidance_density",
        "question_topic_guidance",
        "question_topic_margin_pressure",
        "question_topic_demand_strength",
        "question_topic_cost_pressure",
        "question_topic_pricing",
        "question_topic_capital_allocation",
    ]
    answer_proxy_columns = [
        "answer_numeric_density",
        "answer_quantitative_disclosure",
        "answer_guidance_density",
        "answer_administrative_mechanics",
    ]
    meta = turns[KEY_COLUMNS + analyst_columns + question_columns + answer_proxy_columns].copy()
    meta["transcriptid"] = meta["transcriptid"].astype(str)

    question = _merge_turn_meta(question_frame, meta[KEY_COLUMNS + analyst_columns])
    answer = _merge_turn_meta(
        answer_frame,
        meta[KEY_COLUMNS + analyst_columns + question_columns + answer_proxy_columns],
    )
    answer = answer.rename(
        columns={
            name: name.replace("question_", "current_question_", 1)
            for name in question_columns
        }
        | {
            "answer_numeric_density": "proxy_numeric_density",
            "answer_quantitative_disclosure": "proxy_quantitative_disclosure",
            "answer_guidance_density": "proxy_guidance_density",
            "answer_administrative_mechanics": "proxy_administrative_mechanics",
        }
    )
    question["analyst_key"] = question["analyst_key"].fillna("UNKNOWN").astype(str)
    question["analyst_brokerage"] = question["analyst_brokerage"].fillna("UNKNOWN").astype(str)
    answer["analyst_key"] = answer["analyst_key"].fillna("UNKNOWN").astype(str)
    answer["analyst_brokerage"] = answer["analyst_brokerage"].fillna("UNKNOWN").astype(str)
    return {"question": question, "answer": answer}


def _merge_turn_meta(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError("First-stage frame is not unique by transcriptid, turn_index")
    merged = data.merge(meta, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if len(merged) != len(data):
        raise ValueError("Turn metadata merge changed first-stage row count")
    return merged


def estimate_turn_ladder(
    frame: pd.DataFrame,
    *,
    component: str,
    feature_sets: dict[str, list[str]],
    min_train_rows: int,
) -> dict[str, pd.DataFrame]:
    turn_rows = []
    metric_rows = []
    categorical = [*base_categorical_features(), *REVISION_CATEGORICAL_FEATURES]
    for variant, features in feature_sets.items():
        prediction, fold, fold_metrics = oof_predict_by_year(
            frame,
            target="raw_score",
            feature_columns=features,
            categorical_columns=[name for name in categorical if name in features],
            min_train_rows=min_train_rows,
            learner="ridge",
            label=f"revision_v2:{component}:{variant}",
        )
        turns = frame[
            [
                "transcriptid",
                "ticker",
                "permno",
                "event_trading_date",
                "event_year",
                "turn_index",
                "word_count",
            ]
        ].copy()
        turns["component"] = component
        turns["semantic_feature"] = PRIMARY_FEATURE
        turns["raw_score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
        turns["expected_score_oof"] = prediction
        turns["narrative_innovation"] = turns["raw_score"] - turns["expected_score_oof"]
        turns["fold_year"] = fold
        turns["first_stage_estimable"] = np.isfinite(prediction)
        turns["variant"] = variant
        turns = turns.loc[turns["first_stage_estimable"]].copy()
        turn_rows.append(turns)
        metric_rows.append(
            first_stage_summary(
                variant,
                component,
                turns,
                fold_metrics,
                n_features=len(features),
            )
        )

    all_turns = pd.concat(turn_rows, ignore_index=True)
    counts = all_turns.groupby("variant").size()
    if counts.nunique() != 1:
        raise ValueError(f"{component} ladder variants do not use a common turn sample: {counts.to_dict()}")
    calls = []
    for variant, group in all_turns.groupby("variant", sort=False):
        call = aggregate_turn_innovations(group, component)
        call = call[["transcriptid", "narrative_innovation", "turn_count"]].rename(
            columns={
                "narrative_innovation": f"{component}_{variant}",
                "turn_count": f"{component}_{variant}_turn_count",
            }
        )
        calls.append(call)
    call_wide = _merge_on_transcript(calls)
    return {
        "turn_innovations": all_turns,
        "call_innovations": call_wide,
        "first_stage": pd.DataFrame(metric_rows),
    }


def first_stage_summary(
    variant: str,
    component: str,
    turns: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    n_features: int,
) -> dict[str, object]:
    clean = turns.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["raw_score", "expected_score_oof", "narrative_innovation"]
    )
    actual = clean["raw_score"].astype(float).to_numpy()
    predicted = clean["expected_score_oof"].astype(float).to_numpy()
    return {
        "component": component,
        "variant": variant,
        "n_turns": int(len(clean)),
        "n_calls": int(clean["transcriptid"].nunique()),
        "n_features": int(n_features),
        "min_encoded_features": (
            int(fold_metrics["n_encoded_features"].min())
            if "n_encoded_features" in fold_metrics and fold_metrics["n_encoded_features"].notna().any()
            else np.nan
        ),
        "max_encoded_features": (
            int(fold_metrics["n_encoded_features"].max())
            if "n_encoded_features" in fold_metrics and fold_metrics["n_encoded_features"].notna().any()
            else np.nan
        ),
        "oof_r2": float(r2_score(actual, predicted)) if len(clean) > 2 else np.nan,
        "oof_mae": float(mean_absolute_error(actual, predicted)) if len(clean) else np.nan,
        "predicted_actual_correlation": _safe_corr(actual, predicted),
        "calibration_slope": calibration_slope(actual, predicted),
        "turn_residual_sd": float(clean["narrative_innovation"].std(ddof=1)),
        "mean_fold_r2": float(fold_metrics["r2"].mean()) if not fold_metrics.empty else np.nan,
    }


def assemble_revision_response(
    response: pd.DataFrame,
    question_calls: pd.DataFrame,
    answer_calls: pd.DataFrame,
) -> pd.DataFrame:
    data = response.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    data = data.rename(
        columns={
            "question_innovation": "question_innovation_legacy",
            "answer_innovation": "answer_innovation_legacy",
            "question_innovation_z": "question_innovation_z_legacy",
            "answer_innovation_z": "answer_innovation_z_legacy",
        }
    )
    data = data.merge(
        question_calls[["transcriptid", "question_Q2"]],
        on="transcriptid",
        how="left",
        validate="one_to_one",
    ).merge(
        answer_calls[["transcriptid", "answer_A3"]],
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    if data[["question_Q2", "answer_A3"]].isna().any().any():
        missing = data.loc[data[["question_Q2", "answer_A3"]].isna().any(axis=1), "transcriptid"]
        raise ValueError(f"Revision residuals do not cover the fixed response sample: {missing.head().tolist()}")
    data["question_innovation"] = data["question_Q2"]
    data["answer_innovation"] = data["answer_A3"]
    data["question_innovation_z"] = zscore(data["question_innovation"])
    data["answer_innovation_z"] = zscore(data["answer_innovation"])
    return data


def add_main_multiplicity(coefficients: pd.DataFrame) -> pd.DataFrame:
    out = coefficients.copy()
    out["p_value"] = out["t_stat"].map(normal_two_sided_p)
    family = out["treatment"].isin(TREATMENTS) & out["horizon"].isin(HORIZONS)
    out["p_holm_stage_horizon_15"] = np.nan
    out.loc[family, "p_holm_stage_horizon_15"] = holm_adjust(out.loc[family, "p_value"])
    out["beta_bps"] = out["beta"] * 10000
    out["se_bps"] = out["standard_error"] * 10000
    return out


def build_primary_result(coefficients: pd.DataFrame) -> pd.DataFrame:
    primary = coefficients.loc[
        coefficients["horizon"].eq(1)
        & coefficients["treatment"].eq("answer_innovation_z")
    ].copy()
    if len(primary) != 1:
        raise ValueError(f"Expected one primary result, found {len(primary)}")
    primary.insert(0, "estimand_status", "prespecified_primary")
    primary.insert(1, "answer_variant", "A3")
    primary.insert(2, "estimand_type", "projection_residual")
    return primary


def estimate_incremental_responses(data: pd.DataFrame) -> pd.DataFrame:
    increments = [
        ("event_day", "CAR_0_1", None),
        ("days_1_2", "CAR_0_3", "CAR_0_1"),
        ("days_3_4", "CAR_0_5", "CAR_0_3"),
        ("days_5_9", "CAR_0_10", "CAR_0_5"),
        ("days_10_19", "CAR_0_20", "CAR_0_10"),
    ]
    work = data.copy()
    rows = []
    controls = response_controls(work)
    for label, upper, lower in increments:
        outcome = f"increment_{label}"
        work[outcome] = work[upper] if lower is None else work[upper] - work[lower]
        result = fit_response_regression(
            work,
            outcome=outcome,
            treatments=TREATMENTS,
            controls=controls,
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        piece = result.coefficients.copy()
        piece.insert(0, "increment", label)
        piece.insert(1, "outcome", outcome)
        rows.append(piece)
    out = pd.concat(rows, ignore_index=True)
    out["p_value"] = out["t_stat"].map(normal_two_sided_p)
    out["p_holm_increment_15"] = holm_adjust(out["p_value"])
    return out


def estimate_answer_year_stability(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    controls = response_controls(data)
    for sample, subset in [("exclude_2020", data.loc[data["event_year"].ne(2020)])]:
        for horizon in (1, 20):
            result = fit_response_regression(
                subset,
                outcome=f"CAR_0_{horizon}",
                treatments=TREATMENTS,
                controls=controls,
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            row = result.coefficients.loc[
                result.coefficients["treatment"].eq("answer_innovation_z")
            ].iloc[0]
            rows.append({"sample": sample, "horizon": horizon, **row.to_dict()})

    interaction = data.copy()
    interaction_terms = []
    for year in sorted(interaction["event_year"].dropna().astype(int).unique()):
        if year == 2020:
            continue
        name = f"answer_x_year_{year}"
        interaction[name] = interaction["answer_innovation_z"] * interaction["event_year"].eq(year)
        interaction_terms.append(name)
    result = fit_response_regression(
        interaction,
        outcome="CAR_0_1",
        treatments=["answer_innovation_z", *interaction_terms],
        controls=[*controls, "prepared_innovation_z", "question_innovation_z"],
        firm_fe="permno",
        quarter_fe="event_calendar_quarter",
        cluster_cols=("permno", "event_trading_date"),
    )
    for row in result.coefficients.itertuples(index=False):
        rows.append({"sample": "year_interaction", "horizon": 1, **row._asdict()})
    return pd.DataFrame(rows)


def estimate_answer_year_joint_test(data: pd.DataFrame) -> pd.DataFrame:
    """Jointly test the non-2020 A3 answer-by-year interaction terms."""

    interaction = data.copy()
    interaction_terms = []
    for year in sorted(interaction["event_year"].dropna().astype(int).unique()):
        if year == 2020:
            continue
        name = f"answer_x_year_{year}"
        interaction[name] = interaction["answer_innovation_z"] * interaction[
            "event_year"
        ].eq(year)
        interaction_terms.append(name)
    controls = [
        *response_controls(interaction),
        "prepared_innovation_z",
        "question_innovation_z",
    ]
    result = fit_response_regression(
        interaction,
        outcome="CAR_0_1",
        treatments=["answer_innovation_z", *interaction_terms],
        controls=controls,
        firm_fe="permno",
        quarter_fe="event_calendar_quarter",
        cluster_cols=("permno", "event_trading_date"),
    )
    positions = [result.coefficient_names.index(name) for name in interaction_terms]
    beta = (
        result.coefficients.set_index("treatment")
        .loc[interaction_terms, "beta"]
        .to_numpy(dtype=float)
    )
    covariance = result.covariance[np.ix_(positions, positions)]
    statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
    return pd.DataFrame(
        [
            {
                "test": "answer_by_year_interactions_jointly_zero",
                "reference_year": 2020,
                "interaction_years": ",".join(
                    name.rsplit("_", 1)[-1] for name in interaction_terms
                ),
                "wald_chi2": statistic,
                "df": len(interaction_terms),
                "p_value": float(chi2.sf(statistic, len(interaction_terms))),
                "n_obs": len(interaction),
            }
        ]
    )


def estimate_content_proxy_audit(
    answer_frame: pd.DataFrame,
    answer_turn_residuals: pd.DataFrame,
    response: pd.DataFrame,
    *,
    min_train_rows: int,
) -> dict[str, pd.DataFrame]:
    features = revision_answer_filtration_features(answer_frame)["A3"]
    categorical = [*base_categorical_features(), *REVISION_CATEGORICAL_FEATURES]
    proxy_calls = []
    metric_rows = []
    for proxy in PROXY_TARGETS:
        target = f"proxy_{proxy}"
        prediction, fold, fold_metrics = oof_predict_by_year(
            answer_frame,
            target=target,
            feature_columns=features,
            categorical_columns=[name for name in categorical if name in features],
            min_train_rows=min_train_rows,
            learner="ridge",
            label=f"revision_v2:answer_content:{proxy}",
        )
        turns = answer_frame[
            ["transcriptid", "ticker", "permno", "event_trading_date", "event_year", "turn_index", "word_count"]
        ].copy()
        turns["component"] = "answer"
        turns["semantic_feature"] = target
        turns["raw_score"] = pd.to_numeric(answer_frame[target], errors="coerce")
        turns["expected_score_oof"] = prediction
        turns["narrative_innovation"] = turns["raw_score"] - turns["expected_score_oof"]
        turns["fold_year"] = fold
        turns = turns.loc[np.isfinite(prediction)].copy()
        metric_rows.append(
            first_stage_summary(proxy, "answer_content", turns, fold_metrics, n_features=len(features))
        )
        call = aggregate_turn_innovations(turns, "answer")
        proxy_calls.append(
            call[["transcriptid", "narrative_innovation"]].rename(
                columns={"narrative_innovation": f"content_{proxy}"}
            )
        )

    call_content = _merge_on_transcript(proxy_calls)
    joint = response.merge(call_content, on="transcriptid", how="left", validate="one_to_one")
    content_treatments = []
    for proxy in PROXY_TARGETS:
        column = f"content_{proxy}"
        z_column = f"{column}_z"
        joint[z_column] = zscore(joint[column])
        content_treatments.append(z_column)
    result = fit_response_regression(
        joint,
        outcome="CAR_0_1",
        treatments=[*TREATMENTS, *content_treatments],
        controls=response_controls(joint),
        firm_fe="permno",
        quarter_fe="event_calendar_quarter",
        cluster_cols=("permno", "event_trading_date"),
    )
    joint_response = result.coefficients.copy()

    a3_turns = answer_turn_residuals.loc[answer_turn_residuals["variant"].eq("A3")].copy()
    proxy_meta = answer_frame[KEY_COLUMNS + ["proxy_administrative_mechanics"]].copy()
    proxy_meta["transcriptid"] = proxy_meta["transcriptid"].astype(str)
    a3_turns = a3_turns.merge(proxy_meta, on=KEY_COLUMNS, how="left", validate="one_to_one")
    robust_rows = []
    filters = {
        "all_turns": pd.Series(True, index=a3_turns.index),
        "exclude_admin": a3_turns["proxy_administrative_mechanics"].ne(1),
        "minimum_20_words": a3_turns["word_count"].ge(20),
        "minimum_40_words": a3_turns["word_count"].ge(40),
    }
    for label, mask in filters.items():
        call = aggregate_turn_innovations(a3_turns.loc[mask].copy(), "answer")
        sample = response.drop(columns=["answer_innovation", "answer_innovation_z"]).merge(
            call[["transcriptid", "narrative_innovation"]].rename(
                columns={"narrative_innovation": "answer_innovation"}
            ),
            on="transcriptid",
            how="inner",
            validate="one_to_one",
        )
        sample["answer_innovation_z"] = zscore(sample["answer_innovation"])
        estimate = fit_response_regression(
            sample,
            outcome="CAR_0_1",
            treatments=TREATMENTS,
            controls=response_controls(sample),
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        ).coefficients
        answer = estimate.loc[estimate["treatment"].eq("answer_innovation_z")].iloc[0]
        robust_rows.append(
            {
                "filter": label,
                "status": "estimated",
                "n_turns": int(mask.sum()),
                "n_calls": int(len(sample)),
                **answer.to_dict(),
            }
        )

    call_flags = (
        answer_frame.assign(transcriptid=answer_frame["transcriptid"].astype(str))
        .groupby("transcriptid", as_index=False)
        .agg(
            detected_guidance=("proxy_guidance_density", lambda values: int((values > 0).any())),
            detected_quantitative_disclosure=(
                "proxy_quantitative_disclosure",
                lambda values: int((values > 0).any()),
            ),
        )
    )
    call_subsets = response.merge(
        call_flags, on="transcriptid", how="left", validate="one_to_one"
    )
    subset_masks = {
        "calls_without_detected_guidance": call_subsets["detected_guidance"].eq(0),
        "calls_without_major_quantitative_disclosure": call_subsets[
            "detected_quantitative_disclosure"
        ].eq(0),
        "calls_without_guidance_or_quantitative_disclosure": call_subsets[
            "detected_guidance"
        ].eq(0)
        & call_subsets["detected_quantitative_disclosure"].eq(0),
    }
    for label, mask in subset_masks.items():
        sample = call_subsets.loc[mask].copy()
        support = {
            "n_calls": int(len(sample)),
            "n_firms": int(sample["permno"].nunique()),
            "n_event_dates": int(sample["event_trading_date"].nunique()),
        }
        if support["n_calls"] < 200 or support["n_firms"] < 50:
            robust_rows.append(
                {
                    "filter": label,
                    "status": "insufficient_support",
                    "n_turns": np.nan,
                    **support,
                    "treatment": "answer_innovation_z",
                    "beta": np.nan,
                    "standard_error": np.nan,
                    "t_stat": np.nan,
                    "n_obs": np.nan,
                    "standard_error_type": "not_estimated",
                }
            )
            continue
        estimate = fit_response_regression(
            sample,
            outcome="CAR_0_1",
            treatments=TREATMENTS,
            controls=response_controls(sample),
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        ).coefficients
        answer = estimate.loc[estimate["treatment"].eq("answer_innovation_z")].iloc[0]
        robust_rows.append(
            {
                "filter": label,
                "status": "estimated",
                "n_turns": np.nan,
                "n_calls": int(len(sample)),
                **answer.to_dict(),
            }
        )
    return {
        "first_stage": pd.DataFrame(metric_rows),
        "joint_responses": joint_response,
        "robustness": pd.DataFrame(robust_rows),
    }


def estimate_ladder_responses(
    ladder: dict[str, pd.DataFrame],
    response: pd.DataFrame,
    *,
    component: str,
) -> pd.DataFrame:
    data = response.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    data = data.merge(ladder["call_innovations"], on="transcriptid", how="inner", validate="one_to_one")
    rows = []
    variants = ladder["first_stage"]["variant"].tolist()
    for variant in variants:
        treatment = f"{component}_{variant}_z"
        data[treatment] = zscore(data[f"{component}_{variant}"])
        for outcome in ("placebo_CAR_m10_m1", "placebo_CAR_m5_m1", "CAR_0_1", "CAR_0_20"):
            result = fit_response_regression(
                data,
                outcome=outcome,
                treatments=[treatment],
                controls=response_controls(data),
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            row = result.coefficients.iloc[0]
            rows.append({"component": component, "variant": variant, "outcome": outcome, **row.to_dict()})
    return pd.DataFrame(rows)


def build_sample_accounting(
    response: pd.DataFrame,
    question_calls: pd.DataFrame,
    answer_calls: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    state: pd.DataFrame,
) -> pd.DataFrame:
    question_ids = set(question_calls["transcriptid"].astype(str))
    answer_ids = set(answer_calls["transcriptid"].astype(str))
    both = question_ids & answer_ids
    final_ids = set(response["transcriptid"].astype(str))
    rows = [
        {"stage": "cached_question_turns", "unit": "turn", "count": len(frames["question"]), "reason": "time-forward estimable question turns"},
        {"stage": "cached_answer_turns", "unit": "turn", "count": len(frames["answer"]), "reason": "time-forward estimable answer turns"},
        {"stage": "calls_with_q2_and_a3", "unit": "call", "count": len(both), "reason": "both revision residuals available"},
        {"stage": "main_response_sample", "unit": "call", "count": len(final_ids), "reason": "frozen complete outcomes and controls"},
        {"stage": "q2_a3_to_response_attrition", "unit": "call", "count": len(both - final_ids), "reason": "baseline outcome/control completeness restriction"},
    ]
    candidate = state.assign(transcriptid=state["transcriptid"].astype(str))
    candidate = candidate.loc[candidate["transcriptid"].isin(both)].copy()
    candidate["log_market_cap"] = np.log(
        pd.to_numeric(candidate["market_cap"], errors="coerce").clip(lower=1.0)
    )
    reason_groups = [
        ("missing_event_outcome", [f"CAR_0_{horizon}" for horizon in HORIZONS]),
        ("missing_eps_forecast_control", ["scaled_eps_surprise", "eps_surprise"]),
        ("missing_pre_event_market_control", ["pre_ret_20", "pre_vol_20"]),
        (
            "missing_liquidity_control",
            ["pre_log_dollar_volume_20", "pre_amihud_illiquidity_20"],
        ),
        ("missing_analyst_coverage", ["analyst_coverage"]),
        ("missing_forecast_dispersion", ["forecast_dispersion"]),
        ("missing_forecast_timing_control", ["forecast_age_days", "match_date_gap_days"]),
        ("missing_market_cap", ["log_market_cap"]),
        ("missing_call_timing", ["call_after_close"]),
        ("missing_match_quality", ["match_quality"]),
    ]
    remaining = pd.Series(True, index=candidate.index)
    exclusive_total = 0
    for order, (reason, columns) in enumerate(reason_groups, start=1):
        missing = remaining & candidate[columns].replace([np.inf, -np.inf], np.nan).isna().any(
            axis=1
        )
        count = int(missing.sum())
        exclusive_total += count
        rows.append(
            {
                "stage": "q2_a3_to_response_exclusive_reason",
                "unit": "call",
                "count": count,
                "reason": reason,
                "exclusive_order": order,
            }
        )
        remaining &= ~missing
    if int(remaining.sum()) != len(final_ids) or exclusive_total != len(both - final_ids):
        raise ValueError(
            "Exclusive response attrition does not close: "
            f"candidate={len(candidate)}, remaining={int(remaining.sum())}, "
            f"final={len(final_ids)}, excluded={exclusive_total}"
        )
    return pd.DataFrame(rows)


def build_speaker_coverage(turns: pd.DataFrame) -> pd.DataFrame:
    total = max(len(turns), 1)
    counts = turns["analyst_key"].value_counts(dropna=False)
    rows = [
        {"metric": "turns", "count": len(turns), "share": 1.0},
        {"metric": "proid_key", "count": int(turns["analyst_key"].str.startswith("PROID:").sum()), "share": float(turns["analyst_key"].str.startswith("PROID:").sum() / total)},
        {"metric": "hashed_name_key", "count": int(turns["analyst_key"].str.startswith("HASH:").sum()), "share": float(turns["analyst_key"].str.startswith("HASH:").sum() / total)},
        {"metric": "unknown_key", "count": int(counts.get("UNKNOWN", 0)), "share": float(counts.get("UNKNOWN", 0) / total)},
        {"metric": "multi_key", "count": int(counts.get("MULTI", 0)), "share": float(counts.get("MULTI", 0) / total)},
        {"metric": "brokerage_missing", "count": int(turns["analyst_brokerage"].eq("UNKNOWN").sum()), "share": float(turns["analyst_brokerage"].eq("UNKNOWN").sum() / total)},
    ]
    return pd.DataFrame(rows)


def estimate_standardization_invariance(data: pd.DataFrame) -> pd.DataFrame:
    standardized = fit_response_regression(
        data,
        outcome="CAR_0_1",
        treatments=TREATMENTS,
        controls=response_controls(data),
        firm_fe="permno",
        quarter_fe="event_calendar_quarter",
        cluster_cols=("permno", "event_trading_date"),
    ).coefficients
    unscaled_treatments = ["prepared_innovation", "question_innovation", "answer_innovation"]
    unscaled = fit_response_regression(
        data,
        outcome="CAR_0_1",
        treatments=unscaled_treatments,
        controls=response_controls(data),
        firm_fe="permno",
        quarter_fe="event_calendar_quarter",
        cluster_cols=("permno", "event_trading_date"),
    ).coefficients
    std_answer = standardized.loc[standardized["treatment"].eq("answer_innovation_z")].iloc[0]
    raw_answer = unscaled.loc[unscaled["treatment"].eq("answer_innovation")].iloc[0]
    return pd.DataFrame(
        [
            {"scale": "one_sd", **std_answer.to_dict()},
            {"scale": "unstandardized", **raw_answer.to_dict()},
            {"scale": "absolute_t_difference", "t_stat": abs(float(std_answer["t_stat"] - raw_answer["t_stat"]))},
        ]
    )


def _merge_on_transcript(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["transcriptid"])
    out = frames[0].copy()
    out["transcriptid"] = out["transcriptid"].astype(str)
    for frame in frames[1:]:
        right = frame.copy()
        right["transcriptid"] = right["transcriptid"].astype(str)
        out = out.merge(right, on="transcriptid", how="inner", validate="one_to_one")
    return out


def _assert_unique_turns(data: pd.DataFrame, label: str) -> None:
    if data.duplicated(KEY_COLUMNS).any():
        examples = data.loc[data.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head(5)
        raise ValueError(f"{label} contains duplicate turn keys: {examples.to_dict('records')}")


def _safe_corr(left: Sequence[float], right: Sequence[float]) -> float:
    a = pd.Series(left, dtype=float)
    b = pd.Series(right, dtype=float)
    valid = a.notna() & b.notna()
    if valid.sum() < 3 or a.loc[valid].std() == 0 or b.loc[valid].std() == 0:
        return np.nan
    return float(a.loc[valid].corr(b.loc[valid]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the versioned FinNLP revision-v2 analysis.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--components-path", type=Path, default=DEFAULT_COMPONENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args()
    summary = run_revision_v2(
        RevisionV2Config(
            baseline_dir=args.baseline_dir,
            feature_cache_dir=args.feature_cache_dir,
            components_path=args.components_path,
            output_dir=args.output_dir,
            min_train_rows=args.min_train_rows,
            max_calls=args.max_calls,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
