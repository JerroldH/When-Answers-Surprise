"""Chunked firm-history bootstrap for the tuned-Ridge revision-v2 estimator.

The event sample and all market outcomes stay fixed.  Each replicate samples
firms once, applies those multiplicities to every historical fold, re-selects
Ridge penalties using only resampled pre-test-year text prediction errors, and
then refits P0, Q2, and A0-A3.  The command supports an init/worker/finalize
contract so Slurm arrays can checkpoint 200 replicates safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from when_answers_surprise.empirical.feature_spec import (
    BASE_STATE_FEATURES,
    REVISION_CATEGORICAL_FEATURES,
    question_filtration_features,
    revision_answer_filtration_features,
)
from when_answers_surprise.empirical.learner_audit import (
    fit_first_stage_model,
    select_params_by_inner_time_forward,
    tabular_pipeline,
    tuned_ridge_spec_with_alphas,
)
from when_answers_surprise.empirical.pipeline import (
    HORIZONS,
    OOF_TEST_YEARS,
    PRIMARY_FEATURE,
    aggregate_turn_innovations,
    base_categorical_features,
    fit_response_regression,
    response_controls,
    zscore,
)
from when_answers_surprise.empirical.pre_event_audit import holm_adjust
from when_answers_surprise.empirical.revision_v2_inference import (
    adjusted_inference,
    curve_joint_tests,
    main_joint_system,
    simultaneous_bands,
)
from when_answers_surprise.empirical.revision_v2_tuned import DEFAULT_RIDGE_ALPHAS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025" / "analysis"
DEFAULT_REVISION_DIR = ANALYSIS_ROOT / "revision_v2_tuned_ridge"
DEFAULT_FEATURE_CACHE_DIR = ANALYSIS_ROOT / "lm_baseline_learner_audit_cache"
DEFAULT_OUTPUT_DIR = DEFAULT_REVISION_DIR / "bootstrap"
TREATMENTS = [
    "prepared_innovation_z",
    "question_innovation_z",
    "answer_innovation_z",
]


@dataclass(frozen=True)
class TunedBootstrapConfig:
    revision_dir: Path = DEFAULT_REVISION_DIR
    feature_cache_dir: Path = DEFAULT_FEATURE_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    replicates: int = 200
    seed: int = 20260715
    min_train_rows: int = 120
    jobs: int = 1
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    parity_tolerance: float = 1e-6


_PROCESS_FRAMES: dict[str, pd.DataFrame] | None = None
_PROCESS_RESPONSE: pd.DataFrame | None = None
_PROCESS_FIRMS: list[int] | None = None
_PROCESS_CONFIG: TunedBootstrapConfig | None = None
_PROCESS_THREAD_LIMITER: object | None = None


def initialize_tuned_bootstrap(config: TunedBootstrapConfig) -> dict[str, object]:
    """Validate immutable inputs, write the contract, and run identity refit."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "draws").mkdir(parents=True, exist_ok=True)
    frames, response, firms = load_bootstrap_data(
        config.revision_dir, config.feature_cache_dir
    )
    manifest = bootstrap_manifest(config, frames, response, firms)
    path = config.output_dir / "bootstrap_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        _assert_manifest_compatible(previous, manifest)
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    parity, fold_checks = identity_refit_parity(
        frames, response, firms, config=config
    )
    parity.to_csv(config.output_dir / "identity_refit_parity.csv", index=False)
    fold_checks.to_csv(
        config.output_dir / "identity_refit_fold_checks.csv", index=False
    )
    if not parity["passes"].astype(bool).all():
        raise ValueError("Unit-multiplicity tuned refit failed identity parity")
    summary = {
        "status": "initialized",
        "requested_replicates": config.replicates,
        "fixed_response_calls": len(response),
        "n_firms": len(firms),
        "identity_parity_passed": True,
        "output_dir": str(config.output_dir),
    }
    (config.output_dir / "initialize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def run_tuned_bootstrap_chunk(
    config: TunedBootstrapConfig, *, replicate_start: int, replicate_stop: int
) -> dict[str, object]:
    """Estimate a non-overlapping half-open replicate range and checkpoint it."""

    _validate_config(config)
    if not 0 <= replicate_start < replicate_stop <= config.replicates:
        raise ValueError(
            f"Invalid replicate range [{replicate_start}, {replicate_stop}) for "
            f"{config.replicates} requested replicates"
        )
    _require_initialized(config)
    frames, response, firms = load_bootstrap_data(
        config.revision_dir, config.feature_cache_dir
    )
    draw_dir = config.output_dir / "draws"
    pending = [
        replicate
        for replicate in range(replicate_start, replicate_stop)
        if not valid_replicate_files(draw_dir, replicate)
    ]
    started = time.perf_counter()
    if config.jobs == 1:
        for replicate in pending:
            persist_replicate(
                draw_dir,
                estimate_timed_replicate(
                    frames, response, firms, replicate=replicate, config=config
                ),
            )
    else:
        with ProcessPoolExecutor(
            max_workers=config.jobs,
            initializer=initialize_process_worker,
            initargs=(config,),
        ) as executor:
            futures = {
                executor.submit(run_process_replicate, replicate): replicate
                for replicate in pending
            }
            for future in as_completed(futures):
                persist_replicate(draw_dir, future.result())

    complete = [
        replicate
        for replicate in range(replicate_start, replicate_stop)
        if valid_replicate_files(draw_dir, replicate)
    ]
    if len(complete) != replicate_stop - replicate_start:
        raise ValueError("Bootstrap chunk did not persist every requested replicate")
    summary = {
        "status": "complete",
        "replicate_start": replicate_start,
        "replicate_stop": replicate_stop,
        "valid_replicates": len(complete),
        "new_replicates": len(pending),
        "elapsed_seconds": time.perf_counter() - started,
        "jobs": config.jobs,
    }
    summary_dir = config.output_dir / "chunk_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / f"chunk_{replicate_start:04d}_{replicate_stop:04d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def finalize_tuned_bootstrap(config: TunedBootstrapConfig) -> dict[str, object]:
    """Combine exactly 200 valid checkpoints and add first-stage covariance."""

    _validate_config(config)
    _require_initialized(config)
    _, response, _ = load_bootstrap_data(
        config.revision_dir, config.feature_cache_dir
    )
    draw_dir = config.output_dir / "draws"
    draws, ladder_draws, fold_checks, timing = collect_replicates(
        draw_dir, config.replicates
    )
    if not fold_checks["time_forward_pass"].astype(bool).all():
        raise ValueError("At least one tuned bootstrap fold violates time ordering")
    if not fold_checks["selected_alpha"].isin(config.ridge_alphas).all():
        raise ValueError("A bootstrap fold selected an alpha outside the frozen grid")

    coefficient_index = draw_coefficient_index(draws)
    matrix = (
        draws.pivot(index="replicate", columns="position", values="beta")
        .reindex(columns=coefficient_index["position"])
        .to_numpy(dtype=float)
    )
    if matrix.shape[0] != config.replicates or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap coefficient matrix is incomplete or non-finite")
    first_stage_covariance = np.cov(matrix, rowvar=False, ddof=1)
    conditional = main_joint_system(response, controls=response_controls(response))
    assert_same_coefficient_index(conditional.coefficient_index, coefficient_index)
    adjusted, total_covariance = adjusted_inference(
        conditional, first_stage_covariance
    )
    adjusted["bootstrap_adjusted_p_holm_stage_horizon_15"] = holm_adjust(
        adjusted["bootstrap_adjusted_p_value"]
    )
    adjusted_for_curves = adjusted.rename(
        columns={
            "bootstrap_adjusted_standard_error": "standard_error",
            "bootstrap_adjusted_t_stat": "t_stat",
            "bootstrap_adjusted_p_value": "p_value",
        }
    )
    adjusted_tests = curve_joint_tests(
        adjusted_for_curves, total_covariance, coefficient_index
    )
    adjusted_bands = simultaneous_bands(
        adjusted_for_curves,
        total_covariance,
        coefficient_index,
        seed=config.seed,
    )

    ladder_summary = (
        ladder_draws.groupby("variant", as_index=False)["beta"]
        .agg(
            valid_replicates="count",
            mean_beta="mean",
            bootstrap_sd="std",
            ci_low=lambda values: values.quantile(0.025),
            ci_high=lambda values: values.quantile(0.975),
        )
    )
    draws.to_csv(config.output_dir / "bootstrap_draws.csv", index=False)
    ladder_draws.to_csv(
        config.output_dir / "answer_ladder_bootstrap_draws.csv", index=False
    )
    ladder_summary.to_csv(
        config.output_dir / "answer_ladder_bootstrap_summary.csv", index=False
    )
    fold_checks.to_csv(
        config.output_dir / "bootstrap_fold_checks.csv", index=False
    )
    timing.to_csv(config.output_dir / "replicate_timing.csv", index=False)
    coefficient_index.to_csv(
        config.output_dir / "bootstrap_coefficient_index.csv", index=False
    )
    adjusted.to_csv(
        config.output_dir / "response_coefficients_bootstrap_adjusted.csv",
        index=False,
    )
    adjusted_tests.to_csv(
        config.output_dir / "response_curve_joint_tests_adjusted.csv", index=False
    )
    adjusted_bands.to_csv(
        config.output_dir / "response_simultaneous_bands_adjusted.csv", index=False
    )
    np.save(
        config.output_dir / "first_stage_refit_covariance.npy",
        first_stage_covariance,
    )
    np.save(config.output_dir / "response_total_covariance.npy", total_covariance)

    primary = adjusted.loc[
        adjusted["outcome"].eq("CAR_0_1")
        & adjusted["treatment"].eq("answer_innovation_z")
    ].copy()
    if len(primary) != 1:
        raise ValueError("Expected one bootstrap-adjusted primary coefficient")
    primary.insert(0, "estimand_status", "prespecified_primary")
    primary.insert(1, "answer_variant", "A3")
    primary.insert(2, "uncertainty_label", "first_stage_refit_sensitivity")
    primary.to_csv(
        config.output_dir / "primary_result_bootstrap_adjusted.csv", index=False
    )

    summary = {
        "status": "complete",
        "bootstrap_type": "firm_history_tuned_first_stage_refit_sensitivity",
        "valid_replicates": int(draws["replicate"].nunique()),
        "requested_replicates": config.replicates,
        "fixed_seed": config.seed,
        "fixed_response_calls": len(response),
        "ridge_alpha_grid": list(config.ridge_alphas),
        "retune_inside_each_replicate": True,
        "all_fold_checks_time_forward": True,
        "mean_seconds_per_replicate": float(timing["seconds"].mean()),
        "primary_beta": float(primary.iloc[0]["beta"]),
        "primary_conditional_se": float(
            primary.iloc[0]["conditional_standard_error"]
        ),
        "primary_bootstrap_adjusted_se": float(
            primary.iloc[0]["bootstrap_adjusted_standard_error"]
        ),
        "primary_bootstrap_adjusted_ci_low": float(
            primary.iloc[0]["bootstrap_adjusted_ci_low"]
        ),
        "primary_bootstrap_adjusted_ci_high": float(
            primary.iloc[0]["bootstrap_adjusted_ci_high"]
        ),
        "output_dir": str(config.output_dir),
    }
    (config.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (config.output_dir / "_SUCCESS").write_text(
        "status=complete\n", encoding="utf-8"
    )
    return summary


def run_all_tuned_bootstrap(config: TunedBootstrapConfig) -> dict[str, object]:
    initialize_tuned_bootstrap(config)
    run_tuned_bootstrap_chunk(
        config, replicate_start=0, replicate_stop=config.replicates
    )
    return finalize_tuned_bootstrap(config)


def load_bootstrap_data(
    revision_dir: Path, feature_cache_dir: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[int]]:
    prepared_path = revision_dir / "prepared_first_stage_features.parquet"
    if not prepared_path.is_file():
        prepared_path = feature_cache_dir / "prepared_first_stage_features.parquet"
    frames = {
        "prepared": pd.read_parquet(prepared_path),
        "question": pd.read_parquet(
            revision_dir / "question_first_stage_features_revision_v2.parquet"
        ),
        "answer": pd.read_parquet(
            revision_dir / "answer_first_stage_features_revision_v2.parquet"
        ),
    }
    response = pd.read_parquet(
        revision_dir / "response_model_sample_revision_v2.parquet"
    )
    for frame in [*frames.values(), response]:
        frame["transcriptid"] = frame["transcriptid"].astype(str)
    if response["transcriptid"].duplicated().any():
        raise ValueError("Tuned bootstrap response sample contains duplicate calls")
    if frames["prepared"]["transcriptid"].duplicated().any():
        raise ValueError("Prepared bootstrap frame contains duplicate calls")
    for component in ("question", "answer"):
        if frames[component].duplicated(["transcriptid", "turn_index"]).any():
            raise ValueError(f"{component} bootstrap frame contains duplicate turns")
    firms = sorted(
        pd.concat(
            [
                pd.to_numeric(frame["permno"], errors="coerce")
                for frame in frames.values()
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return frames, response, firms


def draw_firm_multiplicity(
    firms: Sequence[int], rng: np.random.Generator
) -> pd.Series:
    index = pd.Index([int(value) for value in firms], name="permno")
    sampled = rng.choice(index.to_numpy(), size=len(index), replace=True)
    counts = pd.Series(sampled).value_counts().reindex(index, fill_value=0).astype(int)
    counts.index.name = "permno"
    return counts


def tuned_bootstrap_oof_predict(
    frame: pd.DataFrame,
    *,
    component: str,
    variant: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    multiplicity: pd.Series,
    ridge_alphas: Sequence[float],
    min_train_rows: int,
    outer_years: Sequence[int] = OOF_TEST_YEARS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Tune on resampled history and predict each fixed outer test year."""

    features = list(feature_columns)
    values = pd.to_numeric(frame["raw_score"], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(frame["event_year"], errors="coerce").to_numpy(dtype=float)
    firms = pd.to_numeric(frame["permno"], errors="coerce").astype("Int64")
    row_multiplicity = firms.map(multiplicity).fillna(0).astype(int).to_numpy()
    predictions = np.full(len(frame), np.nan, dtype=float)
    fold_year = np.full(len(frame), -1, dtype=int)
    checks = []
    learner = tuned_ridge_spec_with_alphas(ridge_alphas)
    spec = {
        "component": component,
        "grain": "call" if component == "prepared" else "turn",
        "feature_columns": features,
        "categorical_columns": list(categorical_columns),
    }
    for outer_year in outer_years:
        train_mask = (years < outer_year) & np.isfinite(values)
        test_mask = years == outer_year
        train_positions = np.flatnonzero(train_mask)
        expanded_positions = np.repeat(
            train_positions, row_multiplicity[train_positions]
        )
        if test_mask.sum() == 0:
            continue
        if len(expanded_positions) < min_train_rows:
            raise ValueError(
                f"{component}/{variant}/{outer_year} has only "
                f"{len(expanded_positions)} resampled training rows"
            )
        history = frame.iloc[expanded_positions].reset_index(drop=True)
        selected, selection = select_params_by_inner_time_forward(
            history,
            spec,
            learner,
            outer_year=outer_year,
            min_train_rows=min_train_rows,
        )
        selected_row = selection.loc[selection["selected"].astype(bool)]
        if len(selected_row) != 1:
            raise ValueError("Expected one selected tuned-Ridge candidate per fold")
        model = tabular_pipeline(selected, features, categorical_columns)
        history_values = pd.to_numeric(
            history["raw_score"], errors="coerce"
        ).to_numpy(dtype=float)
        fit_first_stage_model(model, history[features], history_values)
        predictions[test_mask] = model.predict(frame.loc[test_mask, features])
        fold_year[test_mask] = outer_year
        valid = np.isfinite(values[test_mask]) & np.isfinite(predictions[test_mask])
        max_train_year = int(np.nanmax(years[expanded_positions]))
        if max_train_year >= outer_year:
            raise ValueError(f"Time-forward leakage in {component}/{variant}/{outer_year}")
        selected_alpha = float(selected.params["alpha"])
        checks.append(
            {
                "component": component,
                "variant": variant,
                "test_year": outer_year,
                "max_train_year": max_train_year,
                "n_original_train_rows": int(train_mask.sum()),
                "n_bootstrap_train_rows": len(expanded_positions),
                "n_test_rows": int(test_mask.sum()),
                "test_mae": float(
                    mean_absolute_error(
                        values[test_mask][valid], predictions[test_mask][valid]
                    )
                ),
                "selected_alpha": selected_alpha,
                "selected_at_grid_boundary": bool(
                    selected_row.iloc[0]["selected_at_grid_boundary"]
                ),
                "n_inner_folds": int(selected_row.iloc[0]["n_inner_folds"]),
                "max_inner_train_year": selected_row.iloc[0][
                    "max_inner_train_year"
                ],
                "time_forward_pass": max_train_year < outer_year,
            }
        )
    return predictions, fold_year, pd.DataFrame(checks)


def estimate_bootstrap_replicate(
    frames: dict[str, pd.DataFrame],
    fixed_response: pd.DataFrame,
    multiplicity: pd.Series,
    *,
    config: TunedBootstrapConfig,
    replicate: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categorical = [*base_categorical_features(), *REVISION_CATEGORICAL_FEATURES]
    checks = []

    prepared_features = list(BASE_STATE_FEATURES)
    prepared_prediction, prepared_fold, table = tuned_bootstrap_oof_predict(
        frames["prepared"],
        component="prepared",
        variant="P0",
        feature_columns=prepared_features,
        categorical_columns=[name for name in categorical if name in prepared_features],
        multiplicity=multiplicity,
        ridge_alphas=config.ridge_alphas,
        min_train_rows=config.min_train_rows,
    )
    checks.append(table)
    prepared_calls = prepared_call_residuals(
        frames["prepared"], prepared_prediction, prepared_fold
    )

    q2_features = question_filtration_features(frames["question"])["Q2"]
    question_prediction, question_fold, table = tuned_bootstrap_oof_predict(
        frames["question"],
        component="question",
        variant="Q2",
        feature_columns=q2_features,
        categorical_columns=[name for name in categorical if name in q2_features],
        multiplicity=multiplicity,
        ridge_alphas=config.ridge_alphas,
        min_train_rows=config.min_train_rows,
    )
    checks.append(table)
    question_calls = turn_call_residuals(
        frames["question"], question_prediction, question_fold, component="question"
    ).rename(columns={"narrative_innovation": "question_innovation"})

    answer_calls = {}
    for variant, features in revision_answer_filtration_features(
        frames["answer"]
    ).items():
        prediction, fold, table = tuned_bootstrap_oof_predict(
            frames["answer"],
            component="answer",
            variant=variant,
            feature_columns=features,
            categorical_columns=[name for name in categorical if name in features],
            multiplicity=multiplicity,
            ridge_alphas=config.ridge_alphas,
            min_train_rows=config.min_train_rows,
        )
        checks.append(table)
        answer_calls[variant] = turn_call_residuals(
            frames["answer"], prediction, fold, component="answer"
        ).rename(columns={"narrative_innovation": f"answer_{variant}"})

    response = assemble_response(
        fixed_response, prepared_calls, question_calls, answer_calls["A3"]
    )
    system = main_joint_system(response, controls=response_controls(response))
    draw = system.coefficients[
        ["position", "outcome", "horizon", "treatment", "beta"]
    ].copy()
    draw.insert(0, "replicate", replicate)

    ladder_rows = []
    base = fixed_response.copy()
    base["transcriptid"] = base["transcriptid"].astype(str)
    for variant, calls in answer_calls.items():
        residual = f"answer_{variant}_bootstrap"
        sample = base.merge(
            calls[["transcriptid", f"answer_{variant}"]].rename(
                columns={f"answer_{variant}": residual}
            ),
            on="transcriptid",
            how="left",
            validate="one_to_one",
        )
        if sample[residual].isna().any():
            raise ValueError(f"Bootstrap {variant} does not cover fixed response calls")
        treatment = f"answer_{variant}_z"
        sample[treatment] = zscore(sample[residual])
        estimate = fit_response_regression(
            sample,
            outcome="CAR_0_1",
            treatments=[treatment],
            controls=response_controls(sample),
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        ).coefficients.iloc[0]
        ladder_rows.append(
            {"replicate": replicate, "variant": variant, **estimate.to_dict()}
        )
    return draw, pd.concat(checks, ignore_index=True), pd.DataFrame(ladder_rows)


def prepared_call_residuals(
    frame: pd.DataFrame, prediction: np.ndarray, fold: np.ndarray
) -> pd.DataFrame:
    out = frame[["transcriptid"]].copy()
    out["transcriptid"] = out["transcriptid"].astype(str)
    out["prepared_innovation"] = (
        pd.to_numeric(frame["raw_score"], errors="coerce") - prediction
    )
    out["fold_year"] = fold
    out = out.loc[np.isfinite(prediction)].copy()
    if out["transcriptid"].duplicated().any():
        raise ValueError("Prepared residuals contain duplicate calls")
    return out


def turn_call_residuals(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    fold: np.ndarray,
    *,
    component: str,
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
    turns["component"] = component
    turns["semantic_feature"] = PRIMARY_FEATURE
    turns["raw_score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
    turns["expected_score_oof"] = prediction
    turns["narrative_innovation"] = turns["raw_score"] - prediction
    turns["fold_year"] = fold
    turns = turns.loc[np.isfinite(prediction)].copy()
    return aggregate_turn_innovations(turns, component)[
        ["transcriptid", "narrative_innovation"]
    ]


def assemble_response(
    fixed: pd.DataFrame,
    prepared: pd.DataFrame,
    question: pd.DataFrame,
    answer_a3: pd.DataFrame,
) -> pd.DataFrame:
    data = fixed.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    data = data.drop(
        columns=[
            "prepared_innovation",
            "prepared_innovation_z",
            "question_innovation",
            "question_innovation_z",
            "answer_innovation",
            "answer_innovation_z",
            "prepared_P0",
            "question_Q2",
            "answer_A3",
        ],
        errors="ignore",
    )
    data = data.merge(
        prepared[["transcriptid", "prepared_innovation"]],
        on="transcriptid",
        how="left",
        validate="one_to_one",
    ).merge(
        question[["transcriptid", "question_innovation"]],
        on="transcriptid",
        how="left",
        validate="one_to_one",
    ).merge(
        answer_a3[["transcriptid", "answer_A3"]].rename(
            columns={"answer_A3": "answer_innovation"}
        ),
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    residuals = [
        "prepared_innovation",
        "question_innovation",
        "answer_innovation",
    ]
    if data[residuals].isna().any().any():
        raise ValueError("Bootstrap residuals do not cover the fixed response sample")
    for name in residuals:
        data[f"{name}_z"] = zscore(data[name])
    if len(data) != len(fixed):
        raise ValueError("Bootstrap assembly changed response row count")
    return data


def identity_refit_parity(
    frames: dict[str, pd.DataFrame],
    response: pd.DataFrame,
    firms: Sequence[int],
    *,
    config: TunedBootstrapConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    multiplicity = pd.Series(1, index=pd.Index(firms, name="permno"), dtype=int)
    draw, checks, ladder = estimate_bootstrap_replicate(
        frames, response, multiplicity, config=config, replicate=-1
    )
    conditional = main_joint_system(
        response, controls=response_controls(response)
    ).coefficients
    comparison = conditional[["position", "outcome", "treatment", "beta"]].merge(
        draw[["position", "outcome", "treatment", "beta"]],
        on=["position", "outcome", "treatment"],
        how="outer",
        suffixes=("_frozen", "_refit"),
        indicator=True,
        validate="one_to_one",
    )
    difference = (comparison["beta_frozen"] - comparison["beta_refit"]).abs()
    rows = [
        {
            "check": "joint_response_coefficients",
            "n_frozen": len(conditional),
            "n_refit": len(draw),
            "max_abs_difference": float(difference.max()),
            "tolerance": config.parity_tolerance,
            "passes": bool(
                comparison["_merge"].eq("both").all()
                and difference.le(config.parity_tolerance).all()
            ),
        }
    ]
    frozen_ladder = pd.read_csv(config.revision_dir / "answer_ladder_responses.csv")
    frozen_ladder = frozen_ladder.loc[frozen_ladder["outcome"].eq("CAR_0_1")]
    ladder_comparison = frozen_ladder[["variant", "beta"]].merge(
        ladder[["variant", "beta"]],
        on="variant",
        how="outer",
        suffixes=("_frozen", "_refit"),
        indicator=True,
        validate="one_to_one",
    )
    ladder_difference = (
        ladder_comparison["beta_frozen"] - ladder_comparison["beta_refit"]
    ).abs()
    rows.append(
        {
            "check": "answer_ladder_one_day_coefficients",
            "n_frozen": len(frozen_ladder),
            "n_refit": len(ladder),
            "max_abs_difference": float(ladder_difference.max()),
            "tolerance": config.parity_tolerance,
            "passes": bool(
                ladder_comparison["_merge"].eq("both").all()
                and ladder_difference.le(config.parity_tolerance).all()
            ),
        }
    )
    return pd.DataFrame(rows), checks


def initialize_process_worker(config: TunedBootstrapConfig) -> None:
    global _PROCESS_FRAMES, _PROCESS_RESPONSE, _PROCESS_FIRMS
    global _PROCESS_CONFIG, _PROCESS_THREAD_LIMITER
    from threadpoolctl import threadpool_limits

    _PROCESS_THREAD_LIMITER = threadpool_limits(limits=1)
    _PROCESS_CONFIG = config
    _PROCESS_FRAMES, _PROCESS_RESPONSE, _PROCESS_FIRMS = load_bootstrap_data(
        config.revision_dir, config.feature_cache_dir
    )


def run_process_replicate(replicate: int) -> dict[str, object]:
    if (
        _PROCESS_FRAMES is None
        or _PROCESS_RESPONSE is None
        or _PROCESS_FIRMS is None
        or _PROCESS_CONFIG is None
    ):
        raise RuntimeError("Bootstrap worker was not initialized")
    return estimate_timed_replicate(
        _PROCESS_FRAMES,
        _PROCESS_RESPONSE,
        _PROCESS_FIRMS,
        replicate=replicate,
        config=_PROCESS_CONFIG,
    )


def estimate_timed_replicate(
    frames: dict[str, pd.DataFrame],
    response: pd.DataFrame,
    firms: Sequence[int],
    *,
    replicate: int,
    config: TunedBootstrapConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    multiplicity = draw_firm_multiplicity(
        firms, np.random.default_rng(config.seed + replicate)
    )
    draw, checks, ladder = estimate_bootstrap_replicate(
        frames, response, multiplicity, config=config, replicate=replicate
    )
    return {
        "replicate": replicate,
        "draw": draw,
        "fold_checks": checks,
        "ladder_draw": ladder,
        "seconds": time.perf_counter() - started,
        "n_sampled_firms": int((multiplicity > 0).sum()),
        "max_multiplicity": int(multiplicity.max()),
    }


def persist_replicate(draw_dir: Path, result: dict[str, object]) -> None:
    replicate = int(result["replicate"])
    atomic_csv(result["draw"], draw_dir / f"replicate_{replicate:04d}.csv")
    atomic_csv(
        result["fold_checks"], draw_dir / f"fold_checks_{replicate:04d}.csv"
    )
    atomic_csv(
        result["ladder_draw"], draw_dir / f"answer_ladder_{replicate:04d}.csv"
    )
    timing = pd.DataFrame(
        [
            {
                "replicate": replicate,
                "seconds": float(result["seconds"]),
                "n_sampled_firms": int(result["n_sampled_firms"]),
                "max_multiplicity": int(result["max_multiplicity"]),
                "status": "complete",
            }
        ]
    )
    atomic_csv(timing, draw_dir / f"timing_{replicate:04d}.csv")
    print(
        f"[tuned-bootstrap] replicate={replicate + 1} "
        f"seconds={float(result['seconds']):.1f}",
        flush=True,
    )


def atomic_csv(value: object, path: Path) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    value.to_csv(temporary, index=False)
    temporary.replace(path)


def valid_replicate_files(draw_dir: Path, replicate: int) -> bool:
    paths = {
        "draw": draw_dir / f"replicate_{replicate:04d}.csv",
        "checks": draw_dir / f"fold_checks_{replicate:04d}.csv",
        "ladder": draw_dir / f"answer_ladder_{replicate:04d}.csv",
        "timing": draw_dir / f"timing_{replicate:04d}.csv",
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        draw = pd.read_csv(paths["draw"])
        checks = pd.read_csv(paths["checks"])
        ladder = pd.read_csv(paths["ladder"])
        timing = pd.read_csv(paths["timing"])
    except Exception:
        return False
    return bool(
        len(draw) == len(HORIZONS) * len(TREATMENTS)
        and set(ladder["variant"]) == {"A0", "A1", "A2", "A3"}
        and not checks.empty
        and checks["time_forward_pass"].astype(bool).all()
        and len(timing) == 1
        and int(timing.iloc[0]["replicate"]) == replicate
    )


def collect_replicates(
    draw_dir: Path, replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    draws = []
    ladders = []
    checks = []
    timing = []
    for replicate in range(replicates):
        if not valid_replicate_files(draw_dir, replicate):
            raise ValueError(f"Missing or invalid tuned bootstrap replicate {replicate}")
        draw = pd.read_csv(draw_dir / f"replicate_{replicate:04d}.csv")
        if draw["replicate"].nunique() != 1 or int(draw["replicate"].iloc[0]) != replicate:
            raise ValueError(f"Bootstrap replicate id mismatch for {replicate}")
        fold = pd.read_csv(draw_dir / f"fold_checks_{replicate:04d}.csv")
        fold.insert(0, "replicate", replicate)
        draws.append(draw)
        checks.append(fold)
        ladders.append(
            pd.read_csv(draw_dir / f"answer_ladder_{replicate:04d}.csv")
        )
        timing.append(pd.read_csv(draw_dir / f"timing_{replicate:04d}.csv"))
    return (
        pd.concat(draws, ignore_index=True),
        pd.concat(ladders, ignore_index=True),
        pd.concat(checks, ignore_index=True),
        pd.concat(timing, ignore_index=True).sort_values("replicate"),
    )


def draw_coefficient_index(draws: pd.DataFrame) -> pd.DataFrame:
    first = int(draws["replicate"].min())
    return (
        draws.loc[
            draws["replicate"].eq(first),
            ["position", "outcome", "horizon", "treatment"],
        ]
        .sort_values("position")
        .reset_index(drop=True)
    )


def assert_same_coefficient_index(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = ["position", "outcome", "horizon", "treatment"]
    if not left[columns].reset_index(drop=True).equals(
        right[columns].reset_index(drop=True)
    ):
        raise ValueError("Bootstrap and conditional coefficient indexes differ")


def bootstrap_manifest(
    config: TunedBootstrapConfig,
    frames: dict[str, pd.DataFrame],
    response: pd.DataFrame,
    firms: Sequence[int],
) -> dict[str, object]:
    del frames
    paths = {
        "response": config.revision_dir / "response_model_sample_revision_v2.parquet",
        "question": config.revision_dir
        / "question_first_stage_features_revision_v2.parquet",
        "answer": config.revision_dir
        / "answer_first_stage_features_revision_v2.parquet",
        "prepared": (
            config.revision_dir / "prepared_first_stage_features.parquet"
            if (config.revision_dir / "prepared_first_stage_features.parquet").is_file()
            else config.feature_cache_dir / "prepared_first_stage_features.parquet"
        ),
    }
    return {
        "bootstrap_contract_version": "revision-v2-tuned-bootstrap-v1",
        "bootstrap_type": "firm_history_tuned_first_stage_refit_sensitivity",
        "estimand_type": "projection_residual",
        "seed": config.seed,
        "requested_replicates": config.replicates,
        "fixed_response_calls": len(response),
        "n_firms": len(firms),
        "test_events_resampled": False,
        "training_history_resampled_by_firm": True,
        "retune_inside_each_replicate": True,
        "market_outcomes_used_for_tuning": False,
        "ridge_alpha_grid": list(config.ridge_alphas),
        "parity_tolerance": config.parity_tolerance,
        "revision_dir": str(config.revision_dir),
        "feature_cache_dir": str(config.feature_cache_dir),
        "source_files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def _assert_manifest_compatible(
    previous: dict[str, object], current: dict[str, object]
) -> None:
    if previous != current:
        changed = {
            key: (previous.get(key), current.get(key))
            for key in sorted(set(previous) | set(current))
            if previous.get(key) != current.get(key)
        }
        raise ValueError(f"Tuned bootstrap resume contract changed: {changed}")


def _require_initialized(config: TunedBootstrapConfig) -> None:
    manifest_path = config.output_dir / "bootstrap_manifest.json"
    parity_path = config.output_dir / "identity_refit_parity.csv"
    if not manifest_path.is_file() or not parity_path.is_file():
        raise ValueError("Run tuned bootstrap initialize before workers or finalization")
    frames, response, firms = load_bootstrap_data(
        config.revision_dir, config.feature_cache_dir
    )
    _assert_manifest_compatible(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        bootstrap_manifest(config, frames, response, firms),
    )
    parity = pd.read_csv(parity_path)
    if len(parity) != 2 or not parity["passes"].astype(bool).all():
        raise ValueError("Tuned bootstrap identity refit has not passed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: TunedBootstrapConfig) -> None:
    if config.replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if config.jobs < 1:
        raise ValueError("Bootstrap jobs must be positive")
    if not config.ridge_alphas:
        raise ValueError("Ridge alpha grid must not be empty")
    if tuple(sorted(config.ridge_alphas)) != config.ridge_alphas:
        raise ValueError("Ridge alpha grid must be strictly increasing")
    if len(set(config.ridge_alphas)) != len(config.ridge_alphas):
        raise ValueError("Ridge alpha grid contains duplicates")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the chunked tuned-Ridge revision-v2 bootstrap."
    )
    parser.add_argument("--mode", choices=("initialize", "worker", "finalize", "all"), default="all")
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--ridge-alphas", nargs="+", type=float, default=list(DEFAULT_RIDGE_ALPHAS)
    )
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--replicate-stop", type=int)
    args = parser.parse_args()
    config = TunedBootstrapConfig(
        revision_dir=args.revision_dir,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=args.output_dir,
        replicates=args.replicates,
        seed=args.seed,
        min_train_rows=args.min_train_rows,
        jobs=args.jobs,
        ridge_alphas=tuple(args.ridge_alphas),
        parity_tolerance=args.parity_tolerance,
    )
    if args.mode == "initialize":
        summary = initialize_tuned_bootstrap(config)
    elif args.mode == "worker":
        stop = args.replicate_stop
        if stop is None:
            raise SystemExit("--replicate-stop is required in worker mode")
        summary = run_tuned_bootstrap_chunk(
            config, replicate_start=args.replicate_start, replicate_stop=stop
        )
    elif args.mode == "finalize":
        summary = finalize_tuned_bootstrap(config)
    else:
        summary = run_all_tuned_bootstrap(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
