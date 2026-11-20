"""Firm-history first-stage refit bootstrap for revision-v2.

The event sample and second-stage outcomes remain fixed.  For each replicate,
firms are sampled once and that multiplicity is applied to every expanding-year
training fold.  Test-year turns are never resampled.  Each completed replicate
is written separately so a 10-replicate benchmark can be resumed to 200.
"""

from __future__ import annotations

import argparse
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
from when_answers_surprise.empirical.pipeline import (
    HORIZONS,
    OOF_TEST_YEARS,
    PRIMARY_FEATURE,
    aggregate_turn_innovations,
    base_categorical_features,
    fit_response_regression,
    response_controls,
    tabular_regression_pipeline,
    zscore,
)
from when_answers_surprise.empirical.revision_v2_inference import (
    adjusted_inference,
    curve_joint_tests,
    main_joint_system,
    simultaneous_bands,
)
from when_answers_surprise.empirical.pre_event_audit import holm_adjust


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
DEFAULT_REVISION_DIR = DATA_ROOT / "analysis" / "revision_v2"
DEFAULT_CACHE_DIR = DATA_ROOT / "analysis" / "lm_baseline_learner_audit_cache"
DEFAULT_OUTPUT_DIR = DEFAULT_REVISION_DIR / "bootstrap"
TREATMENTS = ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]


@dataclass(frozen=True)
class BootstrapConfig:
    revision_dir: Path = DEFAULT_REVISION_DIR
    feature_cache_dir: Path = DEFAULT_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    replicates: int = 200
    seed: int = 20260715
    min_train_rows: int = 120
    jobs: int = 1


_PROCESS_FRAMES: dict[str, pd.DataFrame] | None = None
_PROCESS_RESPONSE: pd.DataFrame | None = None
_PROCESS_FIRMS: list[int] | None = None
_PROCESS_THREAD_LIMITER: object | None = None


def run_first_stage_bootstrap(config: BootstrapConfig) -> dict[str, object]:
    """Run or resume the firm-history first-stage refit bootstrap."""

    if config.replicates < 2:
        raise ValueError("At least two bootstrap replicates are required for covariance")
    if config.jobs < 1:
        raise ValueError("Bootstrap jobs must be at least one")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    draw_dir = config.output_dir / "draws"
    draw_dir.mkdir(parents=True, exist_ok=True)

    frames, response, firms = _load_bootstrap_data(
        config.revision_dir, config.feature_cache_dir
    )
    metadata = {
        "bootstrap_type": "firm_history_first_stage_refit_sensitivity",
        "bootstrap_contract_version": "revision-v2-bootstrap-v2-full-prepared-cache",
        "estimand_type": "projection_residual",
        "seed": config.seed,
        "requested_replicates": config.replicates,
        "n_firms": len(firms),
        "fixed_response_calls": len(response),
        "test_events_resampled": False,
        "training_multiplicity_reused_across_folds": True,
        "revision_dir": str(config.revision_dir),
        "feature_cache_dir": str(config.feature_cache_dir),
        "source_fingerprint": _source_fingerprint(
            config.revision_dir, config.feature_cache_dir
        ),
        "jobs_this_invocation": config.jobs,
        "parallel_backend_this_invocation": (
            "process" if config.jobs > 1 else "serial"
        ),
    }
    _validate_resume_manifest(config.output_dir / "bootstrap_manifest.json", metadata)
    (config.output_dir / "bootstrap_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    parity_path = config.output_dir / "identity_refit_parity.csv"
    if not _valid_identity_parity(parity_path):
        parity, parity_folds = identity_refit_parity(
            frames,
            response,
            firms,
            revision_dir=config.revision_dir,
            min_train_rows=config.min_train_rows,
        )
        parity.to_csv(parity_path, index=False)
        parity_folds.to_csv(
            config.output_dir / "identity_refit_fold_checks.csv", index=False
        )
        if not bool(parity["passes"].all()):
            raise ValueError(
                "Unit-multiplicity first-stage refit does not reproduce frozen coefficients"
            )

    timing_rows = _read_existing_timing(config.output_dir / "replicate_timing.csv")
    started_all = time.perf_counter()
    pending = [
        replicate
        for replicate in range(config.replicates)
        if not _valid_draw_file(draw_dir / f"replicate_{replicate:04d}.csv")
    ]

    def persist(result: dict[str, object]) -> None:
        nonlocal timing_rows
        replicate = int(result["replicate"])
        path = draw_dir / f"replicate_{replicate:04d}.csv"
        draw = result["draw"]
        fold_checks = result["fold_checks"]
        ladder_draw = result["ladder_draw"]
        assert isinstance(draw, pd.DataFrame)
        assert isinstance(fold_checks, pd.DataFrame)
        assert isinstance(ladder_draw, pd.DataFrame)
        draw.to_csv(path, index=False)
        fold_checks.to_csv(draw_dir / f"fold_checks_{replicate:04d}.csv", index=False)
        ladder_draw.to_csv(draw_dir / f"answer_ladder_{replicate:04d}.csv", index=False)
        timing_rows = timing_rows.loc[timing_rows["replicate"].ne(replicate)].copy()
        new_timing = pd.DataFrame(
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
        timing_rows = (
            new_timing
            if timing_rows.empty
            else pd.concat([timing_rows, new_timing], ignore_index=True)
        )
        timing_rows.sort_values("replicate").to_csv(
            config.output_dir / "replicate_timing.csv", index=False
        )
        print(
            f"[revision-v2-bootstrap] replicate={replicate + 1}/{config.replicates} "
            f"seconds={float(result['seconds']):.1f}",
            flush=True,
        )

    if config.jobs == 1:
        for replicate in pending:
            persist(
                _estimate_timed_replicate(
                    frames,
                    response,
                    firms,
                    replicate=replicate,
                    seed=config.seed,
                    min_train_rows=config.min_train_rows,
                )
            )
    else:
        with ProcessPoolExecutor(
            max_workers=config.jobs,
            initializer=_initialize_process_worker,
            initargs=(str(config.revision_dir), str(config.feature_cache_dir)),
        ) as executor:
            futures = {
                executor.submit(
                    _run_process_replicate,
                    replicate,
                    config.seed,
                    config.min_train_rows,
                ): replicate
                for replicate in pending
            }
            for future in as_completed(futures):
                persist(future.result())

    draws = collect_valid_draws(draw_dir, config.replicates)
    ladder_draws, fold_checks = collect_bootstrap_audits(draw_dir, config.replicates)
    if not bool(fold_checks["time_forward_pass"].all()):
        raise ValueError("At least one bootstrap first-stage fold violates time ordering")
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
    coefficient_index = _draw_coefficient_index(draws)
    matrix = (
        draws.pivot(index="replicate", columns="position", values="beta")
        .reindex(columns=coefficient_index["position"])
        .to_numpy(dtype=float)
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Bootstrap coefficient matrix contains missing or infinite values")
    first_stage_covariance = np.cov(matrix, rowvar=False, ddof=1)

    conditional = main_joint_system(response, controls=response_controls(response))
    _assert_same_coefficient_index(conditional.coefficient_index, coefficient_index)
    adjusted, total_covariance = adjusted_inference(conditional, first_stage_covariance)
    adjusted["bootstrap_adjusted_p_holm_stage_horizon_15"] = holm_adjust(
        adjusted["bootstrap_adjusted_p_value"]
    )
    adjusted_tests = curve_joint_tests(
        adjusted.rename(
            columns={
                "bootstrap_adjusted_standard_error": "standard_error",
                "bootstrap_adjusted_t_stat": "t_stat",
                "bootstrap_adjusted_p_value": "p_value",
            }
        ),
        total_covariance,
        coefficient_index,
    )
    adjusted_bands = simultaneous_bands(
        adjusted.rename(
            columns={
                "bootstrap_adjusted_standard_error": "standard_error",
                "bootstrap_adjusted_t_stat": "t_stat",
                "bootstrap_adjusted_p_value": "p_value",
            }
        ),
        total_covariance,
        coefficient_index,
        seed=config.seed,
    )

    draws.to_csv(config.output_dir / "bootstrap_draws.csv", index=False)
    ladder_draws.to_csv(
        config.output_dir / "answer_ladder_bootstrap_draws.csv", index=False
    )
    ladder_summary.to_csv(
        config.output_dir / "answer_ladder_bootstrap_summary.csv", index=False
    )
    fold_checks.to_csv(config.output_dir / "bootstrap_fold_checks.csv", index=False)
    coefficient_index.to_csv(config.output_dir / "bootstrap_coefficient_index.csv", index=False)
    adjusted.to_csv(config.output_dir / "response_coefficients_bootstrap_adjusted.csv", index=False)
    adjusted_tests.to_csv(config.output_dir / "response_curve_joint_tests_adjusted.csv", index=False)
    adjusted_bands.to_csv(config.output_dir / "response_simultaneous_bands_adjusted.csv", index=False)
    np.save(config.output_dir / "first_stage_refit_covariance.npy", first_stage_covariance)
    np.save(config.output_dir / "response_total_covariance.npy", total_covariance)

    primary = adjusted.loc[
        adjusted["outcome"].eq("CAR_0_1")
        & adjusted["treatment"].eq("answer_innovation_z")
    ].copy()
    primary.insert(0, "estimand_status", "prespecified_primary")
    primary.insert(1, "answer_variant", "A3")
    primary.insert(2, "uncertainty_label", "first_stage_refit_sensitivity")
    primary.to_csv(config.output_dir / "primary_result_bootstrap_adjusted.csv", index=False)

    timing_rows = _read_existing_timing(config.output_dir / "replicate_timing.csv")
    summary = {
        "status": "complete",
        "valid_replicates": int(draws["replicate"].nunique()),
        "requested_replicates": config.replicates,
        "fixed_seed": config.seed,
        "fixed_response_calls": len(response),
        "elapsed_seconds_this_invocation": time.perf_counter() - started_all,
        "mean_seconds_per_replicate": float(timing_rows["seconds"].mean()),
        "all_fold_checks_time_forward": bool(fold_checks["time_forward_pass"].all()),
        "primary_beta": float(primary.iloc[0]["beta"]),
        "primary_conditional_se": float(primary.iloc[0]["conditional_standard_error"]),
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
    return summary


def _load_bootstrap_data(
    revision_dir: Path, feature_cache_dir: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[int]]:
    """Load and validate the immutable inputs used by every bootstrap worker."""

    response = pd.read_parquet(
        revision_dir / "response_model_sample_revision_v2.parquet"
    )
    response["transcriptid"] = response["transcriptid"].astype(str)
    frames = {
        "prepared": pd.read_parquet(
            feature_cache_dir / "prepared_first_stage_features.parquet"
        ),
        "question": pd.read_parquet(
            revision_dir / "question_first_stage_features_revision_v2.parquet"
        ),
        "answer": pd.read_parquet(
            revision_dir / "answer_first_stage_features_revision_v2.parquet"
        ),
    }
    validate_bootstrap_inputs(frames, response)
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


def _initialize_process_worker(revision_dir: str, feature_cache_dir: str) -> None:
    """Load one read-only input copy per spawned process and cap BLAS threads."""

    global _PROCESS_FRAMES, _PROCESS_RESPONSE, _PROCESS_FIRMS, _PROCESS_THREAD_LIMITER
    from threadpoolctl import threadpool_limits

    _PROCESS_THREAD_LIMITER = threadpool_limits(limits=1)
    _PROCESS_FRAMES, _PROCESS_RESPONSE, _PROCESS_FIRMS = _load_bootstrap_data(
        Path(revision_dir), Path(feature_cache_dir)
    )


def _run_process_replicate(
    replicate: int, seed: int, min_train_rows: int
) -> dict[str, object]:
    """Process-pool entry point; all large inputs stay resident in the worker."""

    if (
        _PROCESS_FRAMES is None
        or _PROCESS_RESPONSE is None
        or _PROCESS_FIRMS is None
    ):
        raise RuntimeError("Bootstrap process worker was not initialized")
    return _estimate_timed_replicate(
        _PROCESS_FRAMES,
        _PROCESS_RESPONSE,
        _PROCESS_FIRMS,
        replicate=replicate,
        seed=seed,
        min_train_rows=min_train_rows,
    )


def _estimate_timed_replicate(
    frames: dict[str, pd.DataFrame],
    response: pd.DataFrame,
    firms: Sequence[int],
    *,
    replicate: int,
    seed: int,
    min_train_rows: int,
) -> dict[str, object]:
    """Estimate one deterministic replicate and return small persistence payloads."""

    started = time.perf_counter()
    rng = np.random.default_rng(seed + replicate)
    multiplicity = draw_firm_multiplicity(firms, rng)
    draw, fold_checks, ladder_draw = estimate_bootstrap_replicate(
        frames,
        response,
        multiplicity,
        min_train_rows=min_train_rows,
        replicate=replicate,
    )
    return {
        "replicate": replicate,
        "draw": draw,
        "fold_checks": fold_checks,
        "ladder_draw": ladder_draw,
        "seconds": time.perf_counter() - started,
        "n_sampled_firms": int((multiplicity > 0).sum()),
        "max_multiplicity": int(multiplicity.max()),
    }


def draw_firm_multiplicity(firms: Sequence[int], rng: np.random.Generator) -> pd.Series:
    """Sample firms with replacement and return integer multiplicities."""

    firm_index = pd.Index([int(value) for value in firms], name="permno")
    sampled = rng.choice(firm_index.to_numpy(), size=len(firm_index), replace=True)
    counts = pd.Series(sampled).value_counts().reindex(firm_index, fill_value=0).astype(int)
    counts.index.name = "permno"
    return counts


def identity_refit_parity(
    frames: dict[str, pd.DataFrame],
    fixed_response: pd.DataFrame,
    firms: Sequence[int],
    *,
    revision_dir: Path,
    min_train_rows: int,
    tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require a unit-multiplicity refit to reproduce frozen response estimates."""

    multiplicity = pd.Series(1, index=pd.Index(firms, name="permno"), dtype=int)
    draw, folds, ladder = estimate_bootstrap_replicate(
        frames,
        fixed_response,
        multiplicity,
        min_train_rows=min_train_rows,
        replicate=-1,
    )
    conditional = main_joint_system(
        fixed_response, controls=response_controls(fixed_response)
    ).coefficients
    main = conditional[["position", "outcome", "treatment", "beta"]].merge(
        draw[["position", "outcome", "treatment", "beta"]],
        on=["position", "outcome", "treatment"],
        how="outer",
        suffixes=("_frozen", "_refit"),
        indicator=True,
        validate="one_to_one",
    )
    main_difference = (main["beta_frozen"] - main["beta_refit"]).abs()
    rows = [
        {
            "check": "joint_response_coefficients",
            "n_frozen": len(conditional),
            "n_refit": len(draw),
            "max_abs_difference": float(main_difference.max()),
            "tolerance": tolerance,
            "passes": bool(
                main["_merge"].eq("both").all()
                and main_difference.le(tolerance).all()
            ),
        }
    ]

    frozen_ladder = pd.read_csv(revision_dir / "answer_ladder_responses.csv")
    frozen_ladder = frozen_ladder.loc[frozen_ladder["outcome"].eq("CAR_0_1")]
    ladder_check = frozen_ladder[["variant", "beta"]].merge(
        ladder[["variant", "beta"]],
        on="variant",
        how="outer",
        suffixes=("_frozen", "_refit"),
        indicator=True,
        validate="one_to_one",
    )
    ladder_difference = (
        ladder_check["beta_frozen"] - ladder_check["beta_refit"]
    ).abs()
    rows.append(
        {
            "check": "answer_ladder_one_day_coefficients",
            "n_frozen": len(frozen_ladder),
            "n_refit": len(ladder),
            "max_abs_difference": float(ladder_difference.max()),
            "tolerance": tolerance,
            "passes": bool(
                ladder_check["_merge"].eq("both").all()
                and ladder_difference.le(tolerance).all()
            ),
        }
    )
    return pd.DataFrame(rows), folds


def bootstrap_oof_predict(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    multiplicity: pd.Series,
    min_train_rows: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Refit expanding-year Ridge models on multiplicity-replicated histories."""

    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(frame["event_year"], errors="coerce").to_numpy(dtype=float)
    firms = pd.to_numeric(frame["permno"], errors="coerce").astype("Int64")
    row_multiplicity = firms.map(multiplicity).fillna(0).astype(int).to_numpy()
    predictions = np.full(len(frame), np.nan, dtype=float)
    fold_year = np.full(len(frame), -1, dtype=int)
    checks = []
    for test_year in OOF_TEST_YEARS:
        train_mask = (years < test_year) & np.isfinite(values)
        test_mask = years == test_year
        train_positions = np.flatnonzero(train_mask)
        replicated_positions = np.repeat(
            train_positions, row_multiplicity[train_positions]
        )
        if test_mask.sum() == 0:
            continue
        if len(replicated_positions) < min_train_rows:
            raise ValueError(
                f"{label}/{test_year} has only {len(replicated_positions)} "
                "bootstrap training rows"
            )
        max_train_year = int(np.nanmax(years[replicated_positions]))
        if max_train_year >= test_year:
            raise ValueError(f"Time-forward leakage in {label}/{test_year}")
        model = tabular_regression_pipeline(
            "ridge", list(feature_columns), list(categorical_columns)
        )
        model.fit(frame.iloc[replicated_positions][list(feature_columns)], values[replicated_positions])
        predictions[test_mask] = model.predict(frame.loc[test_mask, list(feature_columns)])
        fold_year[test_mask] = test_year
        valid = np.isfinite(values[test_mask]) & np.isfinite(predictions[test_mask])
        checks.append(
            {
                "label": label,
                "test_year": test_year,
                "max_train_year": max_train_year,
                "n_original_train_rows": int(train_mask.sum()),
                "n_bootstrap_train_rows": len(replicated_positions),
                "n_test_rows": int(test_mask.sum()),
                "test_mae": float(
                    mean_absolute_error(values[test_mask][valid], predictions[test_mask][valid])
                ),
                "time_forward_pass": max_train_year < test_year,
            }
        )
    return predictions, fold_year, pd.DataFrame(checks)


def estimate_bootstrap_replicate(
    frames: dict[str, pd.DataFrame],
    fixed_response: pd.DataFrame,
    multiplicity: pd.Series,
    *,
    min_train_rows: int,
    replicate: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit prepared, Q2, and A0-A3 and return response coefficient draws."""

    categorical = [*base_categorical_features(), *REVISION_CATEGORICAL_FEATURES]
    fold_tables = []
    prepared_prediction, prepared_fold, check = bootstrap_oof_predict(
        frames["prepared"],
        target="raw_score",
        feature_columns=BASE_STATE_FEATURES,
        categorical_columns=[name for name in categorical if name in BASE_STATE_FEATURES],
        multiplicity=multiplicity,
        min_train_rows=min_train_rows,
        label="prepared",
    )
    fold_tables.append(check)
    prepared_calls = _prepared_call_residuals(
        frames["prepared"], prepared_prediction, prepared_fold
    )

    q2_features = question_filtration_features(frames["question"])["Q2"]
    question_prediction, question_fold, check = bootstrap_oof_predict(
        frames["question"],
        target="raw_score",
        feature_columns=q2_features,
        categorical_columns=[name for name in categorical if name in q2_features],
        multiplicity=multiplicity,
        min_train_rows=min_train_rows,
        label="question_Q2",
    )
    fold_tables.append(check)
    question_calls = _turn_call_residuals(
        frames["question"], question_prediction, question_fold, component="question"
    ).rename(columns={"narrative_innovation": "question_innovation"})

    answer_call_frames = {}
    for variant, features in revision_answer_filtration_features(frames["answer"]).items():
        prediction, fold, check = bootstrap_oof_predict(
            frames["answer"],
            target="raw_score",
            feature_columns=features,
            categorical_columns=[name for name in categorical if name in features],
            multiplicity=multiplicity,
            min_train_rows=min_train_rows,
            label=f"answer_{variant}",
        )
        fold_tables.append(check)
        answer_call_frames[variant] = _turn_call_residuals(
            frames["answer"], prediction, fold, component="answer"
        ).rename(columns={"narrative_innovation": f"answer_{variant}"})

    response = _assemble_fixed_response(
        fixed_response,
        prepared_calls,
        question_calls,
        answer_call_frames["A3"],
    )
    system = main_joint_system(response, controls=response_controls(response))
    draw = system.coefficients[["position", "outcome", "horizon", "treatment", "beta"]].copy()
    draw.insert(0, "replicate", replicate)

    ladder_rows = []
    base = fixed_response.copy()
    base["transcriptid"] = base["transcriptid"].astype(str)
    for variant, calls in answer_call_frames.items():
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
    return draw, pd.concat(fold_tables, ignore_index=True), pd.DataFrame(ladder_rows)


def _prepared_call_residuals(
    frame: pd.DataFrame, prediction: np.ndarray, fold: np.ndarray
) -> pd.DataFrame:
    out = frame[["transcriptid"]].copy()
    out["transcriptid"] = out["transcriptid"].astype(str)
    out["prepared_innovation"] = pd.to_numeric(frame["raw_score"], errors="coerce") - prediction
    out["fold_year"] = fold
    out = out.loc[np.isfinite(prediction)].copy()
    if out["transcriptid"].duplicated().any():
        raise ValueError("Prepared feature frame is not unique by transcriptid")
    return out


def _turn_call_residuals(
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
    turns["transcriptid"] = turns["transcriptid"].astype(str)
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


def _assemble_fixed_response(
    fixed: pd.DataFrame,
    prepared: pd.DataFrame,
    question: pd.DataFrame,
    answer_a3: pd.DataFrame,
) -> pd.DataFrame:
    data = fixed.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    drop = [
        "prepared_innovation",
        "prepared_innovation_z",
        "question_innovation",
        "question_innovation_z",
        "answer_innovation",
        "answer_innovation_z",
    ]
    data = data.drop(columns=[column for column in drop if column in data])
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
    residuals = ["prepared_innovation", "question_innovation", "answer_innovation"]
    if data[residuals].isna().any().any():
        missing = data.loc[data[residuals].isna().any(axis=1), "transcriptid"]
        raise ValueError(f"Bootstrap residuals do not cover fixed calls: {missing.head().tolist()}")
    for column in residuals:
        data[f"{column}_z"] = zscore(data[column])
    return data


def validate_bootstrap_inputs(
    frames: dict[str, pd.DataFrame], response: pd.DataFrame
) -> None:
    for name, frame in frames.items():
        needed = {"transcriptid", "permno", "event_year", "raw_score"}
        if missing := sorted(needed.difference(frame.columns)):
            raise ValueError(f"Bootstrap {name} frame missing columns: {missing}")
        if name == "prepared" and frame["transcriptid"].astype(str).duplicated().any():
            raise ValueError("Prepared bootstrap frame must be unique by call")
        if name != "prepared" and frame.duplicated(["transcriptid", "turn_index"]).any():
            raise ValueError(f"{name} bootstrap frame has duplicate turn keys")
    if response["transcriptid"].duplicated().any():
        raise ValueError("Bootstrap response sample must be unique by call")


def collect_valid_draws(draw_dir: Path, replicates: int) -> pd.DataFrame:
    tables = []
    for replicate in range(replicates):
        path = draw_dir / f"replicate_{replicate:04d}.csv"
        if not _valid_draw_file(path):
            raise ValueError(f"Missing or invalid bootstrap replicate: {path}")
        table = pd.read_csv(path)
        if table["replicate"].nunique() != 1 or int(table["replicate"].iloc[0]) != replicate:
            raise ValueError(f"Bootstrap replicate id mismatch in {path}")
        tables.append(table)
    draws = pd.concat(tables, ignore_index=True)
    counts = draws.groupby("replicate").size()
    if counts.nunique() != 1:
        raise ValueError(f"Bootstrap replicates have inconsistent coefficient counts: {counts.to_dict()}")
    return draws


def collect_bootstrap_audits(
    draw_dir: Path, replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ladder_tables = []
    fold_tables = []
    for replicate in range(replicates):
        ladder_path = draw_dir / f"answer_ladder_{replicate:04d}.csv"
        fold_path = draw_dir / f"fold_checks_{replicate:04d}.csv"
        if not ladder_path.exists() or not fold_path.exists():
            raise ValueError(f"Bootstrap audit files are incomplete for replicate {replicate}")
        ladder = pd.read_csv(ladder_path)
        if set(ladder["variant"]) != {"A0", "A1", "A2", "A3"}:
            raise ValueError(f"Bootstrap answer ladder is incomplete for replicate {replicate}")
        ladder_tables.append(ladder)
        folds = pd.read_csv(fold_path)
        folds.insert(0, "replicate", replicate)
        fold_tables.append(folds)
    return (
        pd.concat(ladder_tables, ignore_index=True),
        pd.concat(fold_tables, ignore_index=True),
    )


def _valid_draw_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path)
    except Exception:
        return False
    required = {"replicate", "position", "outcome", "treatment", "beta"}
    return required.issubset(table.columns) and len(table) == len(HORIZONS) * len(TREATMENTS)


def _valid_identity_parity(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path)
    except Exception:
        return False
    return bool(
        {"check", "passes", "max_abs_difference", "tolerance"}.issubset(table.columns)
        and len(table) == 2
        and table["passes"].astype(bool).all()
    )


def _draw_coefficient_index(draws: pd.DataFrame) -> pd.DataFrame:
    first = int(draws["replicate"].min())
    index = draws.loc[
        draws["replicate"].eq(first),
        ["position", "outcome", "horizon", "treatment"],
    ].sort_values("position")
    return index.reset_index(drop=True)


def _assert_same_coefficient_index(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = ["position", "outcome", "horizon", "treatment"]
    if not left[columns].reset_index(drop=True).equals(right[columns].reset_index(drop=True)):
        raise ValueError("Bootstrap and conditional coefficient indexes do not match")


def _validate_resume_manifest(path: Path, current: dict[str, object]) -> None:
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    stable = [
        "bootstrap_type",
        "bootstrap_contract_version",
        "seed",
        "n_firms",
        "fixed_response_calls",
        "revision_dir",
        "feature_cache_dir",
        "source_fingerprint",
    ]
    mismatch = {
        key: (previous.get(key), current.get(key))
        for key in stable
        if previous.get(key) != current.get(key)
    }
    if mismatch:
        raise ValueError(f"Bootstrap resume configuration changed: {mismatch}")


def _source_fingerprint(
    revision_dir: Path, feature_cache_dir: Path
) -> dict[str, dict[str, int]]:
    paths = {
        "response_model_sample_revision_v2.parquet": revision_dir
        / "response_model_sample_revision_v2.parquet",
        "question_first_stage_features_revision_v2.parquet": revision_dir
        / "question_first_stage_features_revision_v2.parquet",
        "answer_first_stage_features_revision_v2.parquet": revision_dir
        / "answer_first_stage_features_revision_v2.parquet",
        "prepared_first_stage_features.parquet": feature_cache_dir
        / "prepared_first_stage_features.parquet",
    }
    return {
        name: {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for name, path in paths.items()
    }


def _read_existing_timing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=["replicate", "seconds", "n_sampled_firms", "max_multiplicity", "status"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run revision-v2 first-stage refit bootstrap.")
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    summary = run_first_stage_bootstrap(
        BootstrapConfig(
            revision_dir=args.revision_dir,
            feature_cache_dir=args.feature_cache_dir,
            output_dir=args.output_dir,
            replicates=args.replicates,
            seed=args.seed,
            min_train_rows=args.min_train_rows,
            jobs=args.jobs,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
