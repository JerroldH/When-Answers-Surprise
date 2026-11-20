"""Focused learner sensitivity for the revision-v2 A3 answer benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from when_answers_surprise.empirical.feature_spec import (
    REVISION_CATEGORICAL_FEATURES,
    revision_answer_filtration_features,
)
from when_answers_surprise.empirical.learner_audit import (
    LearnerSpec,
    fixed_ridge_spec,
    fold_metric_row,
    oof_predict_feature_frame,
    sklearn_warning_filter,
    tuned_elasticnet_spec,
    tuned_ridge_spec_with_alphas,
)
from when_answers_surprise.empirical.pipeline import (
    HORIZONS,
    OOF_TEST_YEARS,
    PRIMARY_FEATURE,
    aggregate_turn_innovations,
    base_categorical_features,
    estimate_all_response_curves,
    zscore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
DEFAULT_REVISION_DIR = DATA_ROOT / "analysis" / "revision_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_REVISION_DIR / "learner_audit"
FROZEN_FIRST_STAGE_FILE = "answer_ladder_first_stage.csv"
FIXED_RIDGE_PLATFORM_REFIT = "fixed_ridge_platform_refit"
EXACT_PARITY_TOLERANCE = 1e-8
A3_RIDGE_ALPHA_GRID = (
    0.01,
    0.1,
    1.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
)
A3_HGB_PARAM_GRID = (
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 80,
        "l2_regularization": 1.0,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 80,
        "l2_regularization": 1.0,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 80,
        "l2_regularization": 10.0,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 80,
        "l2_regularization": 10.0,
    },
)
A3_RANDOM_FOREST_PARAM_GRID = tuple(
    {
        "n_estimators": 300,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "max_depth": 18,
        "max_samples": 0.7,
    }
    for max_features in (0.33, 0.67)
    for min_samples_leaf in (20, 100, 500)
)


@dataclass(frozen=True)
class LearnerAuditV2Config:
    revision_dir: Path = DEFAULT_REVISION_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_train_rows: int = 120
    max_calls: int | None = None
    ridge_alphas: tuple[float, ...] = A3_RIDGE_ALPHA_GRID
    tree_jobs: int = 1
    parity_only: bool = False


def run_a3_learner_audit(config: LearnerAuditV2Config) -> dict[str, object]:
    """Compare linear and outcome-blind tuned tree learners on A3."""

    tuned_ridge = a3_tuned_ridge_spec(config.ridge_alphas)
    if config.tree_jobs < 1:
        raise ValueError("tree_jobs must be at least one")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(
        config.revision_dir / "answer_first_stage_features_revision_v2.parquet"
    )
    response = pd.read_parquet(
        config.revision_dir / "response_model_sample_revision_v2.parquet"
    )
    frame["transcriptid"] = frame["transcriptid"].astype(str)
    response["transcriptid"] = response["transcriptid"].astype(str)
    if config.max_calls is not None:
        keep = set(
            frame[["event_year", "transcriptid"]]
            .drop_duplicates("transcriptid")
            .sort_values(["event_year", "transcriptid"])
            .groupby("event_year", sort=True)
            .head(config.max_calls)["transcriptid"]
        )
        frame = frame.loc[frame["transcriptid"].isin(keep)].reset_index(drop=True)
        response = response.loc[response["transcriptid"].isin(keep)].reset_index(drop=True)

    features = revision_answer_filtration_features(frame)["A3"]
    categorical = [
        name
        for name in [*base_categorical_features(), *REVISION_CATEGORICAL_FEATURES]
        if name in features
    ]
    spec = {
        "component": "answer",
        "grain": "turn",
        "feature_columns": features,
        "categorical_columns": categorical,
    }
    results: dict[str, dict[str, pd.DataFrame]] = {}
    metric_rows = []
    selection_rows = []

    # Refit fixed Ridge first so platform drift is diagnosed before expensive
    # learners. The refit is not allowed to replace the frozen paper benchmark.
    fixed_ridge = fixed_ridge_spec()
    prediction, fold, metrics, selections = oof_predict_feature_frame(
        frame, spec, fixed_ridge, config.min_train_rows
    )
    platform_refit_calls = aggregate_a3_calls(frame, prediction, fold)
    platform_metrics = metrics.assign(learner=FIXED_RIDGE_PLATFORM_REFIT)
    platform_selections = selections.assign(learner=FIXED_RIDGE_PLATFORM_REFIT)

    frozen_calls = response[["transcriptid", "answer_innovation"]].copy()
    results[fixed_ridge.name] = {"calls": frozen_calls}
    results[FIXED_RIDGE_PLATFORM_REFIT] = {"calls": platform_refit_calls}
    metric_rows.append(frozen_fixed_ridge_metrics(config.revision_dir))
    metric_rows.append(platform_metrics)
    selection_rows.append(platform_selections)

    parity, parity_differences = fixed_ridge_parity_details(
        platform_refit_calls, response, tolerance=EXACT_PARITY_TOLERANCE
    )
    parity.to_csv(config.output_dir / "fixed_ridge_a3_parity.csv", index=False)
    parity_differences.to_csv(
        config.output_dir / "fixed_ridge_a3_parity_differences.csv", index=False
    )
    environment = runtime_environment()
    (config.output_dir / "fixed_ridge_a3_runtime.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )
    parity_row = parity.iloc[0]
    exact_parity_passed = bool(parity_row["passes"])
    structural_validation_passed = bool(
        parity_row["structural_validation_passes"]
    )
    if not structural_validation_passed:
        raise ValueError(
            "A3 fixed-Ridge structural validation failed; the platform refit must "
            "cover every frozen response call with finite residuals"
        )

    if config.parity_only:
        metrics_table = pd.concat(metric_rows, ignore_index=True)
        selections_table = pd.concat(selection_rows, ignore_index=True)
        metrics_table.to_csv(
            config.output_dir / "a3_learner_first_stage_metrics.csv", index=False
        )
        selections_table.to_csv(
            config.output_dir / "a3_learner_hyperparameter_selection.csv", index=False
        )
        summary = {
            "status": "complete",
            "mode": "fixed_ridge_parity_only",
            "answer_variant": "A3",
            "fixed_ridge_parity_passed": exact_parity_passed,
            "fixed_ridge_exact_parity_passed": exact_parity_passed,
            "fixed_ridge_exact_parity_is_gate": False,
            "fixed_ridge_structural_validation_passed": (
                structural_validation_passed
            ),
            "fixed_ridge_benchmark_source": "frozen_response_sample",
            "fixed_ridge_platform_refit_role": "diagnostic_only",
            "response_calls": len(response),
            "answer_turns": len(frame),
            "runtime_environment": environment,
            "output_dir": str(config.output_dir),
        }
        (config.output_dir / "build_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    learner_specs: list[LearnerSpec] = [tuned_ridge, tuned_elasticnet_spec()]
    for learner in learner_specs:
        prediction, fold, metrics, selections = oof_predict_feature_frame(
            frame, spec, learner, config.min_train_rows
        )
        calls = aggregate_a3_calls(frame, prediction, fold)
        results[learner.name] = {"calls": calls}
        metric_rows.append(metrics)
        selection_rows.append(selections)

    hgb_prediction, hgb_fold, hgb_metrics, hgb_selection = tuned_hgb_oof_predict(
        frame,
        feature_columns=features,
        categorical_columns=categorical,
        min_train_rows=config.min_train_rows,
    )
    results["tuned_hist_gradient_boosting"] = {
        "calls": aggregate_a3_calls(frame, hgb_prediction, hgb_fold)
    }
    metric_rows.append(hgb_metrics)
    selection_rows.append(hgb_selection)

    rf_prediction, rf_fold, rf_metrics, rf_selection = tuned_random_forest_oof_predict(
        frame,
        feature_columns=features,
        categorical_columns=categorical,
        min_train_rows=config.min_train_rows,
        n_jobs=config.tree_jobs,
    )
    results["tuned_random_forest"] = {
        "calls": aggregate_a3_calls(frame, rf_prediction, rf_fold)
    }
    metric_rows.append(rf_metrics)
    selection_rows.append(rf_selection)

    response_rows = []
    innovation_rows = []
    for learner, result in results.items():
        sample = (
            response.copy()
            if learner == fixed_ridge.name
            else replace_answer_residual(response, result["calls"])
        )
        coefficients, _ = estimate_all_response_curves(sample)
        coefficients.insert(0, "learner", learner)
        response_rows.append(coefficients)
        innovation_rows.append(
            {
                "learner": learner,
                "n_calls": len(result["calls"]),
                "mean": float(result["calls"]["answer_innovation"].mean()),
                "sd": float(result["calls"]["answer_innovation"].std(ddof=1)),
                "correlation_with_fixed_ridge": float(
                    result["calls"]
                    .merge(
                        results["fixed_ridge"]["calls"],
                        on="transcriptid",
                        suffixes=("", "_fixed"),
                        validate="one_to_one",
                    )[["answer_innovation", "answer_innovation_fixed"]]
                    .corr()
                    .iloc[0, 1]
                ),
            }
        )

    metrics = pd.concat(metric_rows, ignore_index=True)
    selections = pd.concat(selection_rows, ignore_index=True)
    responses = pd.concat(response_rows, ignore_index=True)
    innovations = pd.DataFrame(innovation_rows)
    metrics.to_csv(config.output_dir / "a3_learner_first_stage_metrics.csv", index=False)
    selections.to_csv(config.output_dir / "a3_learner_hyperparameter_selection.csv", index=False)
    ridge_diagnostics = ridge_regularization_diagnostics(selections)
    ridge_diagnostics.to_csv(
        config.output_dir / "a3_ridge_regularization_diagnostics.csv", index=False
    )
    tree_diagnostics = tree_tuning_diagnostics(selections)
    tree_diagnostics.to_csv(
        config.output_dir / "a3_tree_tuning_diagnostics.csv", index=False
    )
    responses.to_csv(config.output_dir / "a3_learner_response_coefficients.csv", index=False)
    innovations.to_csv(config.output_dir / "a3_learner_innovation_summary.csv", index=False)

    one_day_answer_all = responses.loc[
        responses["horizon"].eq(1)
        & responses["treatment"].eq("answer_innovation_z")
    ].copy()
    fixed_response_comparison = one_day_answer_all.loc[
        one_day_answer_all["learner"].isin(
            [fixed_ridge.name, FIXED_RIDGE_PLATFORM_REFIT]
        )
    ].copy()
    fixed_response_comparison.to_csv(
        config.output_dir / "fixed_ridge_a3_response_comparison.csv", index=False
    )
    one_day_answer = one_day_answer_all.loc[
        one_day_answer_all["learner"].ne(FIXED_RIDGE_PLATFORM_REFIT)
    ].copy()
    one_day_answer.to_csv(config.output_dir / "a3_learner_primary_comparison.csv", index=False)
    summary = {
        "status": "complete",
        "answer_variant": "A3",
        "tuning_target": "training-year text prediction MAE only",
        "market_outcomes_used_for_tuning": False,
        "ridge_alpha_grid": list(config.ridge_alphas),
        "ridge_grid_min": min(config.ridge_alphas),
        "ridge_grid_max": max(config.ridge_alphas),
        "ridge_grid_candidates": len(config.ridge_alphas),
        "ridge_boundary_selections": int(
            ridge_diagnostics["selected_at_grid_boundary"].fillna(False).sum()
        ),
        "hgb_param_grid": [dict(candidate) for candidate in A3_HGB_PARAM_GRID],
        "hgb_grid_candidates": len(A3_HGB_PARAM_GRID),
        "random_forest_param_grid": [
            dict(candidate) for candidate in A3_RANDOM_FOREST_PARAM_GRID
        ],
        "random_forest_grid_candidates": len(A3_RANDOM_FOREST_PARAM_GRID),
        "random_forest_tree_jobs": config.tree_jobs,
        "standardization_contract": {
            "linear_numeric_features": (
                "training-fold mean imputation and unit-variance scaling without centering"
            ),
            "linear_categorical_features": (
                "training-fold one-hot encoding without variance scaling"
            ),
            "tree_numeric_features": "training-fold median imputation without scaling",
            "tree_categorical_features": "training-fold frequency encoding",
            "target": "raw LM tone units",
            "response_treatment": "learner-specific SD on the fixed response sample",
        },
        "learners": list(results),
        "learner_sensitivity_models": [
            learner for learner in results if learner != FIXED_RIDGE_PLATFORM_REFIT
        ],
        "response_calls": len(response),
        "answer_turns": len(frame),
        "fixed_ridge_parity_passed": exact_parity_passed,
        "fixed_ridge_exact_parity_passed": exact_parity_passed,
        "fixed_ridge_exact_parity_is_gate": False,
        "fixed_ridge_structural_validation_passed": structural_validation_passed,
        "fixed_ridge_benchmark_source": "frozen_response_sample",
        "fixed_ridge_platform_refit_role": "diagnostic_only",
        "runtime_environment": environment,
        "output_dir": str(config.output_dir),
    }
    (config.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def a3_tuned_ridge_spec(
    alphas: Sequence[float] = A3_RIDGE_ALPHA_GRID,
) -> LearnerSpec:
    """Return the outcome-blind A3 Ridge spec with an ordered alpha grid."""

    alpha_grid = tuple(float(alpha) for alpha in alphas)
    if alpha_grid != tuple(sorted(alpha_grid)):
        raise ValueError("A3 Ridge alpha grid must be strictly increasing")
    return tuned_ridge_spec_with_alphas(alpha_grid)


def ridge_regularization_diagnostics(selections: pd.DataFrame) -> pd.DataFrame:
    """Summarize the selected Ridge penalty and its inner-fold fit gap."""

    required = {
        "learner",
        "outer_year",
        "candidate_json",
        "inner_train_mae",
        "inner_mae",
        "inner_generalization_gap",
        "selected",
        "selected_at_grid_boundary",
    }
    missing = sorted(required.difference(selections.columns))
    if missing:
        raise ValueError(f"Ridge selection table is missing columns: {missing}")
    selected = selections.loc[
        selections["learner"].eq("tuned_ridge") & selections["selected"].astype(bool)
    ].copy()
    rows = []
    for row in selected.itertuples(index=False):
        candidate = json.loads(row.candidate_json)
        rows.append(
            {
                "outer_year": row.outer_year,
                "selected_alpha": float(candidate["params"]["alpha"]),
                "inner_train_mae": float(row.inner_train_mae),
                "inner_validation_mae": float(row.inner_mae),
                "inner_generalization_gap": float(row.inner_generalization_gap),
                "selected_at_grid_boundary": bool(row.selected_at_grid_boundary),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "outer_year",
            "selected_alpha",
            "inner_train_mae",
            "inner_validation_mae",
            "inner_generalization_gap",
            "selected_at_grid_boundary",
        ],
    ).sort_values("outer_year", ignore_index=True)


def tree_tuning_diagnostics(selections: pd.DataFrame) -> pd.DataFrame:
    """Keep the selected outcome-blind tree candidate and its fit gap by fold."""

    tree_learners = {"tuned_hist_gradient_boosting", "tuned_random_forest"}
    selected = selections.loc[
        selections["learner"].isin(tree_learners)
        & selections["selected"].fillna(False).astype(bool)
    ].copy()
    columns = [
        "learner",
        "outer_year",
        "candidate_json",
        "n_inner_folds",
        "inner_train_mae",
        "inner_mae",
        "inner_generalization_gap",
        "max_inner_train_year",
        "categorical_encoding",
    ]
    return selected.loc[:, columns].rename(
        columns={
            "candidate_json": "selected_params_json",
            "inner_mae": "inner_validation_mae",
        }
    ).sort_values(["learner", "outer_year"], ignore_index=True)


def aggregate_a3_calls(
    frame: pd.DataFrame, prediction: np.ndarray, fold: np.ndarray
) -> pd.DataFrame:
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
    turns["component"] = "answer"
    turns["semantic_feature"] = PRIMARY_FEATURE
    turns["raw_score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
    turns["expected_score_oof"] = prediction
    turns["narrative_innovation"] = turns["raw_score"] - prediction
    turns["fold_year"] = fold
    turns = turns.loc[np.isfinite(prediction)].copy()
    calls = aggregate_turn_innovations(turns, "answer")
    return calls[["transcriptid", "narrative_innovation"]].rename(
        columns={"narrative_innovation": "answer_innovation"}
    )


def replace_answer_residual(response: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    data = response.drop(columns=["answer_innovation", "answer_innovation_z"]).merge(
        calls,
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    if data["answer_innovation"].isna().any():
        raise ValueError("Learner residuals do not cover the fixed response sample")
    data["answer_innovation_z"] = zscore(data["answer_innovation"])
    return data


def frozen_fixed_ridge_metrics(revision_dir: Path) -> pd.DataFrame:
    """Map the frozen A3 first-stage summary into the learner-audit schema."""

    path = revision_dir / FROZEN_FIRST_STAGE_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen fixed-Ridge summary is required for the canonical benchmark: {path}"
        )
    summary = pd.read_csv(path)
    required = {
        "component",
        "variant",
        "n_turns",
        "n_calls",
        "oof_r2",
        "oof_mae",
        "predicted_actual_correlation",
        "calibration_slope",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    row = summary.loc[
        summary["component"].eq("answer") & summary["variant"].eq("A3")
    ]
    if len(row) != 1:
        raise ValueError(
            f"{path.name} must contain exactly one answer/A3 row; found {len(row)}"
        )
    frozen = row.iloc[0]
    return pd.DataFrame(
        [
            {
                "learner": "fixed_ridge",
                "component": "answer",
                "grain": "turn",
                "fold_year": "overall",
                "status": "frozen_canonical",
                "n_train": np.nan,
                "n_test": int(frozen["n_turns"]),
                "r2": float(frozen["oof_r2"]),
                "mae": float(frozen["oof_mae"]),
                "predicted_actual_corr": float(
                    frozen["predicted_actual_correlation"]
                ),
                "calibration_slope": float(frozen["calibration_slope"]),
                "actual_mean": np.nan,
                "predicted_mean": np.nan,
                "source_n_calls": int(frozen["n_calls"]),
                "source_file": path.name,
            }
        ]
    )


def fixed_ridge_parity(
    calls: pd.DataFrame, response: pd.DataFrame, *, tolerance: float
) -> pd.DataFrame:
    summary, _ = fixed_ridge_parity_details(calls, response, tolerance=tolerance)
    return summary


def fixed_ridge_parity_details(
    calls: pd.DataFrame, response: pd.DataFrame, *, tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report coverage and the full distribution of fixed-Ridge refit drift."""

    merged = response[["transcriptid", "answer_innovation"]].merge(
        calls,
        on="transcriptid",
        how="outer",
        suffixes=("_frozen", "_refit"),
        indicator=True,
        validate="one_to_one",
    )
    matched = merged["_merge"].eq("both")
    finite = matched & np.isfinite(merged["answer_innovation_frozen"]) & np.isfinite(
        merged["answer_innovation_refit"]
    )
    differences = merged.loc[
        finite,
        ["transcriptid", "answer_innovation_frozen", "answer_innovation_refit"],
    ].copy()
    differences["difference"] = (
        differences["answer_innovation_refit"]
        - differences["answer_innovation_frozen"]
    )
    differences["abs_difference"] = differences["difference"].abs()
    differences = differences.sort_values(
        ["abs_difference", "transcriptid"], ascending=[False, True], ignore_index=True
    )

    absolute = differences["abs_difference"]
    signed = differences["difference"]
    max_difference = float(absolute.max()) if not differences.empty else np.nan
    frozen_sd = (
        float(differences["answer_innovation_frozen"].std(ddof=1))
        if len(differences) > 1
        else np.nan
    )
    correlation = (
        float(
            differences[
                ["answer_innovation_frozen", "answer_innovation_refit"]
            ].corr().iloc[0, 1]
        )
        if len(differences) > 1
        else np.nan
    )
    n_left_only = int(merged["_merge"].eq("left_only").sum())
    n_nonfinite = int((matched & ~finite).sum())
    coverage_passes = bool(
        len(response) > 0 and n_left_only == 0 and matched.sum() == len(response)
    )
    structural_validation_passes = bool(coverage_passes and n_nonfinite == 0)

    def scaled(value: float) -> float:
        if np.isfinite(value) and np.isfinite(frozen_sd) and frozen_sd > 0:
            return value / frozen_sd
        return np.nan

    mean_difference = float(absolute.mean()) if not differences.empty else np.nan
    rmse_difference = (
        float(np.sqrt(np.mean(np.square(signed)))) if not differences.empty else np.nan
    )
    p95_difference = (
        float(absolute.quantile(0.95)) if not differences.empty else np.nan
    )
    p99_difference = (
        float(absolute.quantile(0.99)) if not differences.empty else np.nan
    )
    exact_parity_passes = bool(
        structural_validation_passes
        and np.isfinite(max_difference)
        and max_difference <= tolerance
    )
    summary = pd.DataFrame(
        [
            {
                "field": "answer_A3_call_residual",
                "n_frozen": len(response),
                "n_refit": len(calls),
                "n_left_only": n_left_only,
                "n_right_only": int(merged["_merge"].eq("right_only").sum()),
                "n_matched": int(matched.sum()),
                "n_nonfinite_matched": n_nonfinite,
                "coverage_passes": coverage_passes,
                "structural_validation_passes": structural_validation_passes,
                "max_abs_difference": max_difference,
                "mean_abs_difference": mean_difference,
                "rmse_difference": rmse_difference,
                "p95_abs_difference": p95_difference,
                "p99_abs_difference": p99_difference,
                "correlation": correlation,
                "frozen_residual_sd": frozen_sd,
                "max_abs_difference_in_frozen_sd": scaled(max_difference),
                "mean_abs_difference_in_frozen_sd": scaled(mean_difference),
                "rmse_difference_in_frozen_sd": scaled(rmse_difference),
                "p95_abs_difference_in_frozen_sd": scaled(p95_difference),
                "p99_abs_difference_in_frozen_sd": scaled(p99_difference),
                "tolerance": tolerance,
                "exact_parity_is_gate": False,
                "exact_parity_passes": exact_parity_passes,
                # Backward-compatible name: this remains exact numerical
                # parity, but it is no longer used to authorize a run.
                "passes": exact_parity_passes,
            }
        ]
    )
    return summary, differences


def runtime_environment() -> dict[str, object]:
    """Capture the package versions that determine first-stage numerical parity."""

    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        try:
            packages[package] = distribution_version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def tuned_hgb_oof_predict(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    min_train_rows: int,
    candidates: Sequence[dict[str, object]] = A3_HGB_PARAM_GRID,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Tune HGB by inner expanding-year MAE with fold-local frequency encoding."""

    return tuned_frequency_tree_oof_predict(
        frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        min_train_rows=min_train_rows,
        learner_name="tuned_hist_gradient_boosting",
        candidates=candidates,
        model_factory=hgb_model,
    )


def tuned_random_forest_oof_predict(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    min_train_rows: int,
    n_jobs: int,
    candidates: Sequence[dict[str, object]] = A3_RANDOM_FOREST_PARAM_GRID,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Tune a regularized Random Forest using the same nested time split."""

    if n_jobs < 1:
        raise ValueError("Random Forest n_jobs must be at least one")
    return tuned_frequency_tree_oof_predict(
        frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        min_train_rows=min_train_rows,
        learner_name="tuned_random_forest",
        candidates=candidates,
        model_factory=lambda params: random_forest_model(params, n_jobs=n_jobs),
    )


def tuned_frequency_tree_oof_predict(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    min_train_rows: int,
    learner_name: str,
    candidates: Sequence[dict[str, object]],
    model_factory: Callable[[dict[str, object]], object],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Nested time-forward tuning shared by HGB and Random Forest."""

    candidates = tuple(dict(candidate) for candidate in candidates)
    if not candidates:
        raise ValueError(f"{learner_name} requires at least one tuning candidate")
    values = pd.to_numeric(frame["raw_score"], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(frame["event_year"], errors="coerce").to_numpy(dtype=float)
    predictions = np.full(len(frame), np.nan, dtype=float)
    fold = np.full(len(frame), -1, dtype=int)
    metric_rows = []
    selection_rows = []
    for outer_year in OOF_TEST_YEARS:
        outer_train = (years < outer_year) & np.isfinite(values)
        test = years == outer_year
        if outer_train.sum() < min_train_rows or test.sum() == 0:
            metric_rows.append(
                fold_metric_row(
                    learner_name,
                    "answer",
                    "turn",
                    outer_year,
                    "skipped",
                    int(outer_train.sum()),
                    int(test.sum()),
                    values[test],
                    predictions[test],
                )
            )
            continue
        losses: list[list[float]] = [[] for _ in candidates]
        training_losses: list[list[float]] = [[] for _ in candidates]
        validation_years: list[int] = []
        train_max_years: list[int] = []
        for inner_year in sorted(pd.Series(years[outer_train]).dropna().astype(int).unique()):
            inner_train = outer_train & (years < inner_year)
            validation = outer_train & (years == inner_year)
            if inner_train.sum() < min_train_rows or validation.sum() == 0:
                continue
            x_train, x_validation = frequency_encoded_matrices(
                frame.loc[inner_train, feature_columns],
                frame.loc[validation, feature_columns],
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
            )
            for rank, params in enumerate(candidates):
                model = model_factory(params)
                with sklearn_warning_filter():
                    model.fit(x_train, values[inner_train])
                    training_prediction = model.predict(x_train)
                    prediction = model.predict(x_validation)
                training_losses[rank].append(
                    float(mean_absolute_error(values[inner_train], training_prediction))
                )
                losses[rank].append(
                    float(mean_absolute_error(values[validation], prediction))
                )
            validation_years.append(int(inner_year))
            train_max_years.append(int(np.nanmax(years[inner_train])))
        means = [float(np.mean(items)) if items else np.inf for items in losses]
        selected_rank = int(np.argmin(means)) if any(np.isfinite(means)) else 0
        for rank, params in enumerate(candidates):
            inner_train_mae = (
                float(np.mean(training_losses[rank])) if training_losses[rank] else np.nan
            )
            inner_mae = means[rank] if np.isfinite(means[rank]) else np.nan
            selection_rows.append(
                {
                    "learner": learner_name,
                    "component": "answer",
                    "grain": "turn",
                    "outer_year": outer_year,
                    "candidate_rank": rank,
                    "candidate_json": json.dumps(params, sort_keys=True),
                    "selection_source": (
                        "inner_expanding_text_mae"
                        if losses[rank]
                        else "fallback_no_inner_history"
                    ),
                    "n_inner_folds": len(losses[rank]),
                    "inner_train_mae": inner_train_mae,
                    "inner_mae": inner_mae,
                    "inner_generalization_gap": inner_mae - inner_train_mae,
                    "inner_validation_years_json": json.dumps(validation_years),
                    "inner_train_max_years_json": json.dumps(train_max_years),
                    "max_inner_train_year": max(train_max_years) if train_max_years else np.nan,
                    "selected": rank == selected_rank,
                    "categorical_encoding": "fold_local_frequency",
                }
            )
        x_train, x_test = frequency_encoded_matrices(
            frame.loc[outer_train, feature_columns],
            frame.loc[test, feature_columns],
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        )
        model = model_factory(candidates[selected_rank])
        with sklearn_warning_filter():
            model.fit(x_train, values[outer_train])
            predictions[test] = model.predict(x_test)
        fold[test] = outer_year
        metric_rows.append(
            fold_metric_row(
                learner_name,
                "answer",
                "turn",
                outer_year,
                "estimated",
                int(outer_train.sum()),
                int(test.sum()),
                values[test],
                predictions[test],
            )
        )
    metric_rows.append(
        fold_metric_row(
            learner_name,
            "answer",
            "turn",
            "overall",
            "estimated",
            int(np.isfinite(values).sum()),
            int(np.isfinite(predictions).sum()),
            values,
            predictions,
        )
    )
    return predictions, fold, pd.DataFrame(metric_rows), pd.DataFrame(selection_rows)


def frequency_encoded_matrices(
    train: pd.DataFrame,
    other: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build numeric matrices using training-only medians and category frequencies."""

    categorical = set(categorical_columns)
    train_columns = []
    other_columns = []
    for column in feature_columns:
        if column in categorical:
            left = train[column].fillna("UNKNOWN").astype(str)
            right = other[column].fillna("UNKNOWN").astype(str)
            frequency = left.value_counts(normalize=True)
            train_columns.append(left.map(frequency).fillna(0.0).to_numpy(dtype=np.float32))
            other_columns.append(right.map(frequency).fillna(0.0).to_numpy(dtype=np.float32))
        else:
            left = pd.to_numeric(train[column], errors="coerce")
            right = pd.to_numeric(other[column], errors="coerce")
            median = float(left.median()) if left.notna().any() else 0.0
            train_columns.append(left.fillna(median).to_numpy(dtype=np.float32))
            other_columns.append(right.fillna(median).to_numpy(dtype=np.float32))
    return np.column_stack(train_columns), np.column_stack(other_columns)


def hgb_model(params: dict[str, object]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        random_state=0,
    )


def random_forest_model(
    params: dict[str, object], *, n_jobs: int
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_features=float(params["max_features"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_depth=int(params["max_depth"]),
        max_samples=float(params["max_samples"]),
        bootstrap=True,
        random_state=0,
        n_jobs=n_jobs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run revision-v2 focused A3 learner audit.")
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help=(
            "Run only frozen fixed Ridge and the platform-refit drift diagnostics."
        ),
    )
    parser.add_argument(
        "--tree-jobs",
        type=int,
        default=1,
        help="Parallel workers used inside Random Forest fits; local default is one.",
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=list(A3_RIDGE_ALPHA_GRID),
        help="Ordered positive Ridge penalties used only for outcome-blind A3 tuning.",
    )
    args = parser.parse_args()
    summary = run_a3_learner_audit(
        LearnerAuditV2Config(
            revision_dir=args.revision_dir,
            output_dir=args.output_dir,
            min_train_rows=args.min_train_rows,
            max_calls=args.max_calls,
            ridge_alphas=tuple(args.ridge_alphas),
            tree_jobs=args.tree_jobs,
            parity_only=args.parity_only,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
