"""Learner-invariance audit for the S&P 500 LM tone submission case.

The audit compares first-stage learners on one frozen response sample. Keep this
module path stable during submission cleanup to avoid churn in tests and
paper-package builders.
"""

from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from when_answers_surprise.semantic_scoring import LM_SHA256, load_lm_dictionary
from when_answers_surprise.sp500_lm import (
    HORIZONS,
    LM_FEATURES,
    OOF_TEST_YEARS,
    PRIMARY_FEATURE,
    aggregate_turn_innovations,
    apply_point_in_time_sp500_filter,
    balanced_year_sample,
    base_categorical_features,
    base_state_features,
    build_crsp_event_outcomes,
    build_event_state,
    build_qa_turns,
    build_response_sample,
    calibration_slope,
    estimate_all_response_curves,
    estimate_placebo_responses,
    filter_earnings_calls,
    fit_response_regression,
    pivot_feature_scores,
    residual_predictability_checks,
    response_controls,
    score_lm_sections,
    score_qa_turns,
    zscore,
    _safe_corr,
    merge_prepared_context,
    rename_current_question_features,
)
from when_answers_surprise.sp500_lm_diagnostics import (
    INNOVATION_TREATMENTS,
    add_raw_innovation_zscores,
    baseline_parity_table,
    build_raw_innovation_frame,
    estimate_raw_innovation_models,
)
from when_answers_surprise.wrds_sp500 import read_partitioned_parquet
from when_answers_surprise.empirical.feature_spec import (
    all_feature_manifests,
    answer_filtration_features,
    assert_feature_manifest_closure,
    feature_manifest,
    question_feature_spec,
    require_columns,
)


PRIMARY_COMPONENTS = ("prepared", "question", "answer")
OUTCOME_FEATURE_PREFIXES = (
    "CAR_",
    "postcall_CAR_",
    "placebo_CAR_",
    "abnormal_volume_",
    "realized_volatility_",
)


@dataclass(frozen=True)
class LearnerAuditConfig:
    project_root: Path
    data_root: Path
    input_dir: Path
    output_dir: Path
    cache_dir: Path
    sample_end: str = "2024-12-31"
    min_train_rows: int = 120
    max_events_per_year: int | None = None
    parity_tolerance: float = 1e-8
    refresh_cache: bool = False
    nonlinear: str = "auto"
    nonlinear_smoke_seconds: float = 30.0
    timing_path: Path | None = None
    panel_path: Path | None = None


@dataclass(frozen=True)
class FeatureCache:
    state: pd.DataFrame
    frames: dict[str, pd.DataFrame]
    specs: dict[str, dict[str, object]]
    manifest: dict[str, object]


@dataclass(frozen=True)
class CandidateSpec:
    model_type: str
    params: dict[str, object]


@dataclass(frozen=True)
class LearnerSpec:
    name: str
    candidates: tuple[CandidateSpec, ...]
    tune: bool
    selection_source: str


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, object]] = []

    @contextmanager
    def stage(self, name: str):
        record: dict[str, object] = {"stage": name, "status": "running"}
        started = time.perf_counter()
        print(f"[learner-audit] start {name}", flush=True)
        try:
            yield record
        except Exception as exc:
            record["status"] = "error"
            record["error"] = repr(exc)
            raise
        else:
            record["status"] = "ok"
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            self.records.append(record)
            print(
                f"[learner-audit] {record['status']} {name} {record['seconds']}s",
                flush=True,
            )
            self.write()

    def progress(self, message: str) -> None:
        print(f"[learner-audit] {message}", flush=True)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


def run_lm_learner_invariance_audit(config: LearnerAuditConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(config.timing_path or config.output_dir / "timing.json")

    with logger.stage("feature_cache") as record:
        cache = load_or_build_feature_cache(config, logger)
        baseline_manifest_path = config.input_dir / "baseline_feature_manifest.json"
        if not baseline_manifest_path.exists():
            raise FileNotFoundError(f"Missing baseline feature manifest: {baseline_manifest_path}")
        assert_feature_manifest_closure(
            json.loads(baseline_manifest_path.read_text(encoding="utf-8")), cache.manifest
        )
        record["cache_dir"] = str(config.cache_dir)
        record["frame_rows"] = {name: int(len(frame)) for name, frame in cache.frames.items()}

    with logger.stage("select_nonlinear_learner") as record:
        nonlinear_name, nonlinear_smoke = select_nonlinear_learner(config, cache, logger)
        record["selected_nonlinear"] = nonlinear_name
        record["smoke_rows"] = nonlinear_smoke.to_dict("records")

    fixed_spec = fixed_ridge_spec()
    with logger.stage("fixed_ridge_parity") as record:
        fixed = estimate_learner(cache, fixed_spec, config.min_train_rows, logger)
        parity = fixed_ridge_parity_checks(
            fixed,
            config.input_dir,
            tolerance=config.parity_tolerance,
        )
        write_tables(
            config.output_dir,
            learner_parity_checks=parity,
            learner_first_stage_metrics=fixed["first_stage_metrics"],
            learner_hyperparameter_selection=fixed["hyperparameter_selection"],
        )
        record["parity_passed"] = bool(parity["passes"].all()) if not parity.empty else False
        if not bool(parity["passes"].all()):
            summary = {
                "input_dir": str(config.input_dir),
                "output_dir": str(config.output_dir),
                "cache_dir": str(config.cache_dir),
                "fixed_ridge_parity_passed": False,
                "aborted": "fixed_ridge_parity_failed",
            }
            (config.output_dir / "build_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            raise RuntimeError("Fixed-Ridge parity failed; refusing learner comparisons.")

    learners = [
        fixed_spec,
        tuned_ridge_spec(),
        tuned_elasticnet_spec(),
        nonlinear_spec(nonlinear_name),
    ]
    results = {"fixed_ridge": fixed}
    with logger.stage("learner_first_stages") as record:
        for spec in learners[1:]:
            logger.progress(f"estimate_learner {spec.name}")
            results[spec.name] = estimate_learner(cache, spec, config.min_train_rows, logger)
        record["learners"] = list(results)

    with logger.stage("freeze_common_response_sample") as record:
        full_controls = response_controls(results["fixed_ridge"]["response_sample"])
        manifest = build_learner_common_response_manifest(
            {name: result["response_sample"] for name, result in results.items()},
            full_controls,
        )
        record["common_response_rows"] = int(len(manifest))

    output_tables: dict[str, pd.DataFrame] = {
        "learner_parity_checks": parity,
        "learner_common_response_manifest": manifest,
        "learner_leakage_checks": leakage_checks(cache, results),
        "learner_first_stage_samples": learner_first_stage_samples(cache, results, manifest),
        "learner_first_stage_metrics": concat_result_table(results, "first_stage_metrics"),
        "learner_hyperparameter_selection": concat_result_table(results, "hyperparameter_selection"),
        "learner_calibration": calibration_table(concat_result_table(results, "first_stage_metrics")),
        "learner_residual_predictability": concat_result_table(results, "residual_predictability"),
        "learner_innovation_correlations": innovation_correlations(results, manifest),
        "learner_innovation_distributions": innovation_distributions(results, manifest),
    }

    response_rows = []
    joint_rows = []
    placebo_rows = []
    car_diff_rows = []
    with logger.stage("common_sample_responses") as record:
        for learner, result in results.items():
            common_sample = restrict_to_common_manifest(result["response_sample"], manifest, learner)
            common_sample = restandardize_response_innovations(common_sample)
            full_controls = response_controls(common_sample)

            response_coefficients, _ = estimate_all_response_curves(common_sample)
            response_coefficients.insert(0, "learner", learner)
            response_rows.append(response_coefficients)

            car_diff = estimate_car20_minus_1(common_sample, full_controls)
            car_diff.insert(0, "learner", learner)
            car_diff_rows.append(car_diff)

            placebo = estimate_placebo_responses(common_sample)
            placebo.insert(0, "learner", learner)
            placebo_rows.append(placebo)

            raw_frame, _ = build_raw_innovation_frame(result["call_innovations"])
            raw_data = common_sample.merge(raw_frame, on="transcriptid", how="left", validate="one_to_one")
            raw_data = add_raw_innovation_zscores(raw_data)
            joint = estimate_raw_innovation_models(raw_data, full_controls)["response"]
            joint = joint.loc[joint["model"].eq("joint_expected_innovation")].copy()
            joint.insert(0, "learner", learner)
            joint_rows.append(joint)
        record["learners"] = list(results)

    output_tables["learner_response_coefficients"] = pd.concat(response_rows, ignore_index=True)
    output_tables["learner_joint_expected_innovation_response"] = pd.concat(joint_rows, ignore_index=True)
    output_tables["learner_placebo_responses"] = pd.concat(placebo_rows, ignore_index=True)
    output_tables["learner_car20_minus_1_coefficients"] = pd.concat(car_diff_rows, ignore_index=True)
    output_tables["learner_invariance_summary"] = classify_invariance(
        output_tables["learner_response_coefficients"],
        output_tables["learner_car20_minus_1_coefficients"],
    )

    with logger.stage("write_outputs") as record:
        write_tables(config.output_dir, **output_tables)
        summary = {
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
            "cache_dir": str(config.cache_dir),
            "learners": list(results),
            "selected_nonlinear": nonlinear_name,
            "fixed_ridge_parity_passed": True,
            "common_response_rows": int(len(manifest)),
            "classification": str(
                output_tables["learner_invariance_summary"]["overall_classification"].iloc[0]
            ),
            "outputs": sorted(f"{name}.csv" for name in output_tables),
        }
        (config.output_dir / "build_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        record["outputs"] = summary["outputs"]
    return summary


def load_or_build_feature_cache(config: LearnerAuditConfig, logger: AuditLogger) -> FeatureCache:
    paths = cache_paths(config.cache_dir)
    if not config.refresh_cache and all(path.exists() for path in paths.values()):
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("lm_dictionary_sha256") != LM_SHA256:
            raise ValueError("Unverified LM feature cache; rebuild with --refresh-cache.")
        frames = {
            component: pd.read_parquet(paths[f"{component}_frame"])
            for component in PRIMARY_COMPONENTS
        }
        state = pd.read_parquet(paths["state"])
        return FeatureCache(
            state=state,
            frames=frames,
            specs=manifest["components"],
            manifest=manifest,
        )

    dictionary = load_lm_dictionary(config.project_root / "data" / "external")
    panel_path = config.panel_path or config.data_root / "derived" / "transcript_ibes_panel.parquet"
    panel = pd.read_parquet(panel_path)
    components = pd.read_parquet(config.data_root / "staging" / "ciq_transcript_components.parquet")
    membership = pd.read_parquet(config.data_root / "staging" / "universe_sp500_membership.parquet")
    security_master = pd.read_parquet(config.data_root / "staging" / "security_master.parquet")
    crsp_daily = read_partitioned_parquet(config.data_root / "raw" / "wrds_exports" / "crsp_dsf_sp500")

    earnings = filter_earnings_calls(panel, sample_end=config.sample_end)
    point_in_time = apply_point_in_time_sp500_filter(earnings, membership, security_master)
    point_in_time = balanced_year_sample(point_in_time, config.max_events_per_year)
    event_ids = set(point_in_time["transcriptid"].astype(str))
    components = components.loc[components["transcriptid"].astype(str).isin(event_ids)].copy()

    section_scores = score_lm_sections(point_in_time, dictionary, progress=logger.progress)
    turns = build_qa_turns(components, point_in_time)
    turn_scores = score_qa_turns(turns, dictionary, progress=logger.progress)
    outcomes = build_crsp_event_outcomes(point_in_time, crsp_daily, horizons=HORIZONS)
    state = build_event_state(point_in_time, section_scores, outcomes)

    frames, specs = build_first_stage_feature_frames(state, section_scores, turn_scores)
    manifest = {
        "lm_dictionary_sha256": LM_SHA256,
        "target": PRIMARY_FEATURE,
        "sample_end": config.sample_end,
        "max_events_per_year": config.max_events_per_year,
        "panel_path": str(panel_path),
        "horizons": list(HORIZONS),
        "outer_test_years": list(OOF_TEST_YEARS),
        "components": specs,
        "canonical_feature_manifests": all_feature_manifests(),
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
    }

    state.to_parquet(paths["state"], index=False)
    for component, frame in frames.items():
        frame.to_parquet(paths[f"{component}_frame"], index=False)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return FeatureCache(state=state, frames=frames, specs=specs, manifest=manifest)


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "state": cache_dir / "state.parquet",
        "prepared_frame": cache_dir / "prepared_first_stage_features.parquet",
        "question_frame": cache_dir / "question_first_stage_features.parquet",
        "answer_frame": cache_dir / "answer_first_stage_features.parquet",
        "manifest": cache_dir / "cache_manifest.json",
    }


def build_first_stage_feature_frames(
    state: pd.DataFrame,
    section_scores: pd.DataFrame,
    turn_scores: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    builders = {
        "prepared": build_prepared_feature_frame,
        "question": build_question_feature_frame,
        "answer": build_answer_feature_frame,
    }
    frames = {}
    specs = {}
    for component, builder in builders.items():
        frame, spec = builder(state, section_scores, turn_scores)
        frames[component] = frame
        specs[component] = spec
    return frames, specs


def build_prepared_feature_frame(
    state: pd.DataFrame,
    section_scores: pd.DataFrame,
    turn_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    del turn_scores
    prepared = section_scores.loc[section_scores["section"].eq("prepared")].copy()
    data = state.merge(prepared[["transcriptid", PRIMARY_FEATURE]], on="transcriptid", how="inner")
    feature_columns = base_state_features(data)
    categorical_columns = [col for col in base_categorical_features() if col in feature_columns]
    return make_feature_frame(
        data,
        component="prepared",
        grain="event",
        target_col=PRIMARY_FEATURE,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        id_columns=["transcriptid", "ticker", "permno", "event_trading_date", "event_year"],
    )


def build_question_feature_frame(
    state: pd.DataFrame,
    section_scores: pd.DataFrame,
    turn_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    question = turn_scores.loc[turn_scores["component"].eq("question")].copy()
    data = question.merge(state, on=["transcriptid", "ticker", "permno", "call_date"], how="inner")
    data = merge_prepared_context(data, section_scores)
    feature_columns = question_feature_spec(data, label="learner question first-stage features")
    categorical_columns = [col for col in base_categorical_features() if col in feature_columns]
    return make_feature_frame(
        data,
        component="question",
        grain="turn",
        target_col=PRIMARY_FEATURE,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        id_columns=[
            "transcriptid",
            "ticker",
            "permno",
            "event_trading_date",
            "event_year",
            "turn_index",
            "word_count",
        ],
    )


def build_answer_feature_frame(
    state: pd.DataFrame,
    section_scores: pd.DataFrame,
    turn_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    answer = turn_scores.loc[turn_scores["component"].eq("answer")].copy()
    question_scores = turn_scores.loc[turn_scores["component"].eq("question")].copy()
    question_wide = pivot_feature_scores(
        question_scores,
        index_cols=["transcriptid", "turn_index"],
        prefix_by="component",
    )
    question_wide = rename_current_question_features(question_wide)
    data = answer.merge(state, on=["transcriptid", "ticker", "permno", "call_date"], how="inner")
    data = data.merge(question_wide, on=["transcriptid", "turn_index"], how="left")
    data = merge_prepared_context(data, section_scores)
    feature_sets = answer_filtration_features(data, label_prefix="learner answer filtration")
    feature_columns = feature_sets["A2"]
    categorical_columns = [col for col in base_categorical_features() if col in feature_columns]
    return make_feature_frame(
        data,
        component="answer",
        grain="turn",
        target_col=PRIMARY_FEATURE,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        id_columns=[
            "transcriptid",
            "ticker",
            "permno",
            "event_trading_date",
            "event_year",
            "turn_index",
            "word_count",
        ],
    )


def make_feature_frame(
    data: pd.DataFrame,
    *,
    component: str,
    grain: str,
    target_col: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    id_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_columns = list(dict.fromkeys(feature_columns))
    categorical_columns = [col for col in categorical_columns if col in feature_columns]
    validate_feature_columns(feature_columns)
    frame = data.copy()
    frame["transcriptid"] = frame["transcriptid"].astype(str)
    frame["raw_score"] = pd.to_numeric(frame[target_col], errors="coerce")
    keep = list(dict.fromkeys([*id_columns, "raw_score", *feature_columns]))
    frame = frame[keep].reset_index(drop=True)
    spec = {
        "component": component,
        "grain": grain,
        "target": PRIMARY_FEATURE,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        **feature_manifest(feature_columns),
    }
    return frame, spec


def validate_feature_columns(feature_columns: Sequence[str]) -> None:
    leaked = outcome_like_feature_columns(feature_columns)
    if leaked:
        raise ValueError(f"First-stage feature columns include outcome-like fields: {leaked}")


def outcome_like_feature_columns(feature_columns: Sequence[str]) -> list[str]:
    return [
        col
        for col in feature_columns
        if any(str(col).startswith(prefix) for prefix in OUTCOME_FEATURE_PREFIXES)
    ]


def fixed_ridge_spec() -> LearnerSpec:
    return LearnerSpec(
        name="fixed_ridge",
        candidates=(CandidateSpec("ridge", {"alpha": 1.0}),),
        tune=False,
        selection_source="fixed_existing_ridge",
    )


def tuned_ridge_spec() -> LearnerSpec:
    return tuned_ridge_spec_with_alphas([1.0, 0.1, 10.0, 0.01, 100.0])


def tuned_ridge_spec_with_alphas(alphas: Sequence[float]) -> LearnerSpec:
    """Build a Ridge learner spec from an explicit, positive alpha grid."""

    alpha_grid = tuple(float(alpha) for alpha in alphas)
    if not alpha_grid:
        raise ValueError("Ridge alpha grid must not be empty")
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in alpha_grid):
        raise ValueError("Ridge alpha grid must contain finite positive values")
    if len(set(alpha_grid)) != len(alpha_grid):
        raise ValueError("Ridge alpha grid must not contain duplicates")
    return LearnerSpec(
        name="tuned_ridge",
        candidates=tuple(CandidateSpec("ridge", {"alpha": alpha}) for alpha in alpha_grid),
        tune=True,
        selection_source="inner_expanding_mae",
    )


def tuned_elasticnet_spec() -> LearnerSpec:
    candidates = [
        CandidateSpec("elasticnet", {"alpha": alpha, "l1_ratio": ratio})
        for alpha in [0.01, 0.001, 0.1]
        for ratio in [0.2, 0.5, 0.8]
    ]
    return LearnerSpec(
        name="tuned_elasticnet",
        candidates=tuple(candidates),
        tune=True,
        selection_source="inner_expanding_mae",
    )


def nonlinear_spec(nonlinear_name: str) -> LearnerSpec:
    if nonlinear_name == "hist_gradient_boosting":
        candidate = CandidateSpec(
            "hist_gradient_boosting",
            {
                "max_iter": 120,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 5,
                "l2_regularization": 0.02,
            },
        )
    elif nonlinear_name == "extra_trees":
        candidate = CandidateSpec(
            "extra_trees",
            {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 2},
        )
    else:
        raise ValueError(f"Unknown nonlinear learner: {nonlinear_name}")
    return LearnerSpec(
        name=nonlinear_name,
        candidates=(candidate,),
        tune=False,
        selection_source="nonlinear_smoke_runtime_check",
    )


def select_nonlinear_learner(
    config: LearnerAuditConfig,
    cache: FeatureCache,
    logger: AuditLogger,
) -> tuple[str, pd.DataFrame]:
    if config.nonlinear in {"hist_gradient_boosting", "extra_trees"}:
        return config.nonlinear, pd.DataFrame(
            [{"candidate": config.nonlinear, "status": "forced_by_config", "seconds": 0.0}]
        )
    if config.nonlinear != "auto":
        raise ValueError(f"Unknown nonlinear option: {config.nonlinear}")

    rows = []
    for candidate_name in ["hist_gradient_boosting", "extra_trees"]:
        started = time.perf_counter()
        status = "ok"
        error = ""
        mae = np.nan
        try:
            mae = nonlinear_smoke_loss(cache, candidate_name)
        except Exception as exc:  # pragma: no cover - exact sklearn failures vary.
            status = "error"
            error = repr(exc)
        seconds = time.perf_counter() - started
        if seconds > config.nonlinear_smoke_seconds and status == "ok":
            status = "too_slow"
        rows.append(
            {
                "candidate": candidate_name,
                "status": status,
                "seconds": round(seconds, 3),
                "mae": float(mae) if np.isfinite(mae) else np.nan,
                "error": error,
            }
        )
        logger.progress(f"nonlinear_smoke {candidate_name} status={status} seconds={seconds:.2f}")
        if status == "ok":
            return candidate_name, pd.DataFrame(rows)
    return "extra_trees", pd.DataFrame(rows)


def nonlinear_smoke_loss(cache: FeatureCache, candidate_name: str) -> float:
    frame = cache.frames["prepared"]
    spec = cache.specs["prepared"]
    feature_columns = list(spec["feature_columns"])
    categorical_columns = list(spec["categorical_columns"])
    years = pd.to_numeric(frame["event_year"], errors="coerce").to_numpy()
    values = pd.to_numeric(frame["raw_score"], errors="coerce").to_numpy(dtype=float)
    for year in OOF_TEST_YEARS:
        train_idx = np.flatnonzero((years < year) & np.isfinite(values))
        test_idx = np.flatnonzero(years == year)
        if len(train_idx) >= 10 and len(test_idx) > 0:
            train_idx = train_idx[-min(len(train_idx), 400) :]
            test_idx = test_idx[: min(len(test_idx), 200)]
            model = tabular_pipeline(
                nonlinear_spec(candidate_name).candidates[0],
                feature_columns,
                categorical_columns,
            )
            fit_first_stage_model(model, frame.iloc[train_idx][feature_columns], values[train_idx])
            pred = model.predict(frame.iloc[test_idx][feature_columns])
            valid = np.isfinite(values[test_idx]) & np.isfinite(pred)
            return float(mean_absolute_error(values[test_idx][valid], pred[valid]))
    raise ValueError("No feasible prepared-section smoke fold for nonlinear learner selection.")


def estimate_learner(
    cache: FeatureCache,
    learner: LearnerSpec,
    min_train_rows: int,
    logger: AuditLogger,
) -> dict[str, pd.DataFrame]:
    innovation_rows = {}
    metric_rows = []
    selection_rows = []
    for component in PRIMARY_COMPONENTS:
        frame = cache.frames[component]
        spec = cache.specs[component]
        logger.progress(f"first_stage {learner.name} {component}")
        predictions, fold_year, metrics, selections = oof_predict_feature_frame(
            frame,
            spec,
            learner,
            min_train_rows,
        )
        innovations = feature_frame_to_innovations(
            frame,
            component=component,
            grain=str(spec["grain"]),
            predictions=predictions,
            fold_year=fold_year,
        )
        innovation_rows[component] = innovations
        metric_rows.append(metrics)
        selection_rows.append(selections)

    turn_innovations = pd.concat(
        [innovation_rows["question"], innovation_rows["answer"]],
        ignore_index=True,
    )
    call_innovations = pd.concat(
        [
            innovation_rows["prepared"],
            aggregate_turn_innovations(innovation_rows["question"], "question"),
            aggregate_turn_innovations(innovation_rows["answer"], "answer"),
        ],
        ignore_index=True,
    )
    response_sample = build_response_sample(cache.state, call_innovations)
    residual_predictability = residual_predictability_checks(call_innovations, cache.state)
    residual_predictability.insert(0, "learner", learner.name)
    return {
        "turn_innovations": turn_innovations,
        "call_innovations": call_innovations,
        "response_sample": response_sample,
        "first_stage_metrics": pd.concat(metric_rows, ignore_index=True),
        "hyperparameter_selection": pd.concat(selection_rows, ignore_index=True),
        "residual_predictability": residual_predictability,
    }


def oof_predict_feature_frame(
    frame: pd.DataFrame,
    spec: dict[str, object],
    learner: LearnerSpec,
    min_train_rows: int,
    *,
    outer_years: Sequence[int] = OOF_TEST_YEARS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    feature_columns = list(spec["feature_columns"])
    categorical_columns = list(spec["categorical_columns"])
    validate_feature_columns(feature_columns)
    values = pd.to_numeric(frame["raw_score"], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(frame["event_year"], errors="coerce").to_numpy()
    predictions = np.full(len(frame), np.nan, dtype=float)
    fold_year = np.full(len(frame), -1, dtype=int)
    metric_rows = []
    selection_rows = []

    for outer_year in outer_years:
        train_mask = (years < outer_year) & np.isfinite(values)
        test_mask = years == outer_year
        if train_mask.sum() < min_train_rows or test_mask.sum() == 0:
            metric_rows.append(
                fold_metric_row(
                    learner.name,
                    str(spec["component"]),
                    str(spec["grain"]),
                    outer_year,
                    "skipped",
                    int(train_mask.sum()),
                    int(test_mask.sum()),
                    values[test_mask],
                    predictions[test_mask],
                )
            )
            continue

        if learner.tune:
            selected, selection = select_params_by_inner_time_forward(
                frame,
                spec,
                learner,
                outer_year=outer_year,
                min_train_rows=min_train_rows,
            )
            selection_rows.append(selection)
        else:
            selected = learner.candidates[0]
            selection_rows.append(
                pd.DataFrame(
                    [
                        {
                            "learner": learner.name,
                            "component": spec["component"],
                            "grain": spec["grain"],
                            "outer_year": outer_year,
                            "candidate_rank": 0,
                            "candidate_json": candidate_json(selected),
                            "selection_source": learner.selection_source,
                            "n_inner_folds": 0,
                            "inner_train_mae": np.nan,
                            "inner_mae": np.nan,
                            "inner_generalization_gap": np.nan,
                            "inner_validation_years_json": "[]",
                            "inner_train_max_years_json": "[]",
                            "max_inner_train_year": np.nan,
                            "candidate_is_grid_boundary": False,
                            "selected": True,
                            "selected_at_grid_boundary": False,
                        }
                    ]
                )
            )

        model = tabular_pipeline(selected, feature_columns, categorical_columns)
        fit_first_stage_model(model, frame.loc[train_mask, feature_columns], values[train_mask])
        pred = model.predict(frame.loc[test_mask, feature_columns])
        predictions[test_mask] = pred
        fold_year[test_mask] = outer_year
        metric_rows.append(
            fold_metric_row(
                learner.name,
                str(spec["component"]),
                str(spec["grain"]),
                outer_year,
                "estimated",
                int(train_mask.sum()),
                int(test_mask.sum()),
                values[test_mask],
                predictions[test_mask],
            )
        )

    overall = fold_metric_row(
        learner.name,
        str(spec["component"]),
        str(spec["grain"]),
        "overall",
        "estimated",
        int(np.isfinite(values).sum()),
        int(np.isfinite(predictions).sum()),
        values,
        predictions,
    )
    metric_rows.append(overall)
    selection_table = (
        pd.concat(selection_rows, ignore_index=True)
        if selection_rows
        else pd.DataFrame(
            columns=[
                "learner",
                "component",
                "grain",
                "outer_year",
                "candidate_rank",
                "candidate_json",
                "selection_source",
                "n_inner_folds",
                "inner_train_mae",
                "inner_mae",
                "inner_generalization_gap",
                "inner_validation_years_json",
                "inner_train_max_years_json",
                "max_inner_train_year",
                "candidate_is_grid_boundary",
                "selected",
                "selected_at_grid_boundary",
            ]
        )
    )
    return predictions, fold_year, pd.DataFrame(metric_rows), selection_table


def select_params_by_inner_time_forward(
    frame: pd.DataFrame,
    spec: dict[str, object],
    learner: LearnerSpec,
    *,
    outer_year: int,
    min_train_rows: int,
) -> tuple[CandidateSpec, pd.DataFrame]:
    feature_columns = list(spec["feature_columns"])
    categorical_columns = list(spec["categorical_columns"])
    values = pd.to_numeric(frame["raw_score"], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(frame["event_year"], errors="coerce").to_numpy()
    outer_train = (years < outer_year) & np.isfinite(values)
    inner_years = [int(year) for year in sorted(pd.Series(years[outer_train]).dropna().unique())]
    rows = []
    candidate_scores: list[tuple[float, int, CandidateSpec]] = []
    ridge_alphas = [
        float(candidate.params["alpha"])
        for candidate in learner.candidates
        if candidate.model_type == "ridge" and "alpha" in candidate.params
    ]
    ridge_boundary = (
        (min(ridge_alphas), max(ridge_alphas))
        if len(ridge_alphas) == len(learner.candidates) and ridge_alphas
        else None
    )

    for rank, candidate in enumerate(learner.candidates):
        losses = []
        training_losses = []
        validation_years = []
        train_max_years = []
        for inner_year in inner_years:
            inner_train = outer_train & (years < inner_year)
            inner_validation = outer_train & (years == inner_year)
            if inner_train.sum() < min_train_rows or inner_validation.sum() == 0:
                continue
            model = tabular_pipeline(candidate, feature_columns, categorical_columns)
            fit_first_stage_model(
                model,
                frame.loc[inner_train, feature_columns],
                values[inner_train],
            )
            train_pred = model.predict(frame.loc[inner_train, feature_columns])
            pred = model.predict(frame.loc[inner_validation, feature_columns])
            train_valid = np.isfinite(values[inner_train]) & np.isfinite(train_pred)
            valid = np.isfinite(values[inner_validation]) & np.isfinite(pred)
            if not train_valid.any() or not valid.any():
                continue
            training_losses.append(
                float(
                    mean_absolute_error(
                        values[inner_train][train_valid],
                        train_pred[train_valid],
                    )
                )
            )
            losses.append(float(mean_absolute_error(values[inner_validation][valid], pred[valid])))
            validation_years.append(int(inner_year))
            train_max_years.append(int(np.nanmax(years[inner_train])))
        if losses:
            inner_train_mae = float(np.mean(training_losses))
            inner_mae = float(np.mean(losses))
            inner_generalization_gap = inner_mae - inner_train_mae
            candidate_scores.append((inner_mae, rank, candidate))
            selection_source = learner.selection_source
        else:
            inner_train_mae = np.nan
            inner_mae = np.nan
            inner_generalization_gap = np.nan
            selection_source = "fallback_no_inner_history"
        alpha = float(candidate.params["alpha"]) if ridge_boundary is not None else np.nan
        candidate_is_grid_boundary = bool(
            ridge_boundary is not None
            and (np.isclose(alpha, ridge_boundary[0]) or np.isclose(alpha, ridge_boundary[1]))
        )
        rows.append(
            {
                "learner": learner.name,
                "component": spec["component"],
                "grain": spec["grain"],
                "outer_year": outer_year,
                "candidate_rank": rank,
                "candidate_json": candidate_json(candidate),
                "selection_source": selection_source,
                "n_inner_folds": len(losses),
                "inner_train_mae": inner_train_mae,
                "inner_mae": inner_mae,
                "inner_generalization_gap": inner_generalization_gap,
                "inner_validation_years_json": json.dumps(validation_years),
                "inner_train_max_years_json": json.dumps(train_max_years),
                "max_inner_train_year": max(train_max_years) if train_max_years else np.nan,
                "candidate_is_grid_boundary": candidate_is_grid_boundary,
                "selected": False,
                "selected_at_grid_boundary": False,
            }
        )

    if candidate_scores:
        _, selected_rank, selected = min(candidate_scores, key=lambda item: (item[0], item[1]))
    else:
        selected_rank, selected = 0, learner.candidates[0]
    rows[selected_rank]["selected"] = True
    rows[selected_rank]["selected_at_grid_boundary"] = bool(
        rows[selected_rank]["candidate_is_grid_boundary"]
    )
    if not candidate_scores:
        rows[selected_rank]["selection_source"] = "fallback_no_inner_history"
    return selected, pd.DataFrame(rows)


def tabular_pipeline(
    candidate: CandidateSpec,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Pipeline:
    feature_columns = list(feature_columns)
    categorical_columns = [col for col in categorical_columns if col in feature_columns]
    numeric = [col for col in feature_columns if col not in set(categorical_columns)]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler(with_mean=False))]),
                numeric,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No first-stage feature columns supplied.")
    sparse_threshold = 0.0 if candidate.model_type in {"hist_gradient_boosting", "extra_trees"} else 0.3
    preprocess = ColumnTransformer(transformers, sparse_threshold=sparse_threshold)
    params = candidate.params
    if candidate.model_type == "ridge":
        model = Ridge(alpha=float(params["alpha"]), solver="lsqr")
    elif candidate.model_type == "elasticnet":
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=4000,
            random_state=0,
        )
    elif candidate.model_type == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=int(params["max_iter"]),
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=0,
        )
    elif candidate.model_type == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=0,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown first-stage model type: {candidate.model_type}")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_first_stage_model(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> None:
    with sklearn_warning_filter():
        model.fit(x, y)


@contextmanager
def sklearn_warning_filter():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
            category=UserWarning,
        )
        yield


def feature_frame_to_innovations(
    frame: pd.DataFrame,
    *,
    component: str,
    grain: str,
    predictions: np.ndarray,
    fold_year: np.ndarray,
) -> pd.DataFrame:
    id_columns = ["transcriptid", "ticker", "permno", "event_trading_date", "event_year"]
    if grain == "turn":
        id_columns.extend(["turn_index", "word_count"])
    out = frame[id_columns].copy()
    out["component"] = component
    out["semantic_feature"] = PRIMARY_FEATURE
    out["raw_score"] = pd.to_numeric(frame["raw_score"], errors="coerce").astype(float)
    out["expected_score_oof"] = predictions
    out["narrative_innovation"] = out["raw_score"] - out["expected_score_oof"]
    out["fold_year"] = fold_year
    out["first_stage_estimable"] = np.isfinite(predictions)
    return out.loc[out["first_stage_estimable"]].reset_index(drop=True)


def fold_metric_row(
    learner: str,
    component: str,
    grain: str,
    fold_year: int | str,
    status: str,
    n_train: int,
    n_test: int,
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, object]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    return {
        "learner": learner,
        "component": component,
        "grain": grain,
        "fold_year": fold_year,
        "status": status,
        "n_train": n_train,
        "n_test": int(valid.sum()) if status == "estimated" else n_test,
        "r2": float(r2_score(actual[valid], predicted[valid])) if valid.sum() > 2 else np.nan,
        "mae": float(mean_absolute_error(actual[valid], predicted[valid])) if valid.any() else np.nan,
        "predicted_actual_corr": _safe_corr(actual[valid], predicted[valid]),
        "calibration_slope": calibration_slope(actual[valid], predicted[valid]),
        "actual_mean": float(np.nanmean(actual[valid])) if valid.any() else np.nan,
        "predicted_mean": float(np.nanmean(predicted[valid])) if valid.any() else np.nan,
    }


def fixed_ridge_parity_checks(
    fixed: dict[str, pd.DataFrame],
    input_dir: Path,
    *,
    tolerance: float,
) -> pd.DataFrame:
    existing_call = pd.read_parquet(input_dir / "call_level_innovations.parquet")
    existing_turn = pd.read_parquet(input_dir / "turn_level_innovations.parquet")
    existing_response = pd.read_csv(input_dir / "response_coefficients.csv")
    rows = []
    rows.extend(compare_primary_innovations(fixed["call_innovations"], existing_call, "call", tolerance))
    rows.extend(compare_primary_innovations(fixed["turn_innovations"], existing_turn, "turn", tolerance))
    response_coefficients, _ = estimate_all_response_curves(fixed["response_sample"])
    diagnostic = response_coefficients.copy()
    diagnostic.insert(0, "model", "full_controls_current")
    response_parity = baseline_parity_table(diagnostic, existing_response, tolerance=tolerance)
    rows.extend(response_parity_rows(response_parity))
    return pd.DataFrame(rows)


def compare_primary_innovations(
    generated: pd.DataFrame,
    existing: pd.DataFrame,
    grain: str,
    tolerance: float,
) -> list[dict[str, object]]:
    generated = generated.loc[generated["semantic_feature"].eq(PRIMARY_FEATURE)].copy()
    existing = existing.loc[existing["semantic_feature"].eq(PRIMARY_FEATURE)].copy()
    key_cols = ["transcriptid", "component", "semantic_feature"]
    if grain == "turn":
        key_cols.insert(1, "turn_index")
    generated["transcriptid"] = generated["transcriptid"].astype(str)
    existing["transcriptid"] = existing["transcriptid"].astype(str)
    merged = generated.merge(
        existing,
        on=key_cols,
        how="outer",
        suffixes=("_generated", "_existing"),
        indicator=True,
    )
    rows = []
    for col in ["raw_score", "expected_score_oof", "narrative_innovation"]:
        diff = (merged[f"{col}_generated"] - merged[f"{col}_existing"]).abs()
        max_abs_diff = float(diff.max()) if diff.notna().any() else np.nan
        rows.append(
            {
                "check_type": "innovation_parity",
                "grain": grain,
                "field": col,
                "n_generated": int(len(generated)),
                "n_existing": int(len(existing)),
                "n_merged": int(len(merged)),
                "n_left_only": int(merged["_merge"].eq("left_only").sum()),
                "n_right_only": int(merged["_merge"].eq("right_only").sum()),
                "max_abs_diff": max_abs_diff,
                "tolerance": tolerance,
                "passes": bool(
                    merged["_merge"].eq("both").all()
                    and pd.notna(max_abs_diff)
                    and max_abs_diff <= tolerance
                ),
            }
        )
    return rows


def response_parity_rows(response_parity: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for _, item in response_parity.iterrows():
        rows.append(
            {
                "check_type": "response_parity",
                "grain": "response",
                "field": f"{item['outcome']}:{item['treatment']}",
                "n_generated": np.nan,
                "n_existing": np.nan,
                "n_merged": 1,
                "n_left_only": int(item["_merge"] == "left_only"),
                "n_right_only": int(item["_merge"] == "right_only"),
                "max_abs_diff": max(float(item["beta_abs_diff"]), float(item["standard_error_abs_diff"])),
                "tolerance": np.nan,
                "passes": bool(item["passes"]),
            }
        )
    return rows


def build_learner_common_response_manifest(
    response_samples: dict[str, pd.DataFrame],
    full_controls: Sequence[str],
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
    clean_by_learner = {}
    for learner, sample in response_samples.items():
        missing = [col for col in required if col not in sample.columns]
        if missing:
            raise KeyError(f"{learner} response sample missing required columns: {missing}")
        data = sample.assign(transcriptid=sample["transcriptid"].astype(str))
        if data["transcriptid"].duplicated().any():
            dupes = data.loc[data["transcriptid"].duplicated(), "transcriptid"].head(5).tolist()
            raise ValueError(f"{learner} response sample has duplicate transcript IDs: {dupes}")
        clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
        clean_by_learner[learner] = clean
    fixed = clean_by_learner["fixed_ridge"]
    common_ids = set(fixed["transcriptid"])
    for clean in clean_by_learner.values():
        common_ids &= set(clean["transcriptid"])
    manifest = fixed.loc[fixed["transcriptid"].isin(common_ids), ["transcriptid", "event_year", "event_trading_date"]].copy()
    manifest["diagnostic_sample_order"] = np.arange(len(manifest), dtype=int)
    manifest["n_learners"] = len(response_samples)
    return manifest.reset_index(drop=True)


def restrict_to_common_manifest(data: pd.DataFrame, manifest: pd.DataFrame, learner: str) -> pd.DataFrame:
    out = manifest[["transcriptid", "diagnostic_sample_order"]].merge(
        data.assign(transcriptid=data["transcriptid"].astype(str)),
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    out = out.sort_values("diagnostic_sample_order").drop(columns=["diagnostic_sample_order"]).reset_index(drop=True)
    assert_matches_common_manifest(out, manifest, learner)
    return out


def assert_matches_common_manifest(data: pd.DataFrame, manifest: pd.DataFrame, label: str) -> None:
    data_ids = data["transcriptid"].astype(str).tolist()
    manifest_ids = manifest["transcriptid"].astype(str).tolist()
    if len(data_ids) != len(set(data_ids)):
        raise ValueError(f"{label} has duplicate transcript IDs in common sample.")
    if data_ids != manifest_ids:
        raise ValueError(f"{label} does not match frozen learner-common response sample order.")


def restandardize_response_innovations(response_sample: pd.DataFrame) -> pd.DataFrame:
    out = response_sample.copy()
    for col in ["prepared_innovation", "question_innovation", "answer_innovation"]:
        out[f"{col}_z"] = zscore(out[col])
    return out


def estimate_car20_minus_1(response_sample: pd.DataFrame, full_controls: Sequence[str]) -> pd.DataFrame:
    data = response_sample.assign(CAR_20_minus_1=response_sample["CAR_0_20"] - response_sample["CAR_0_1"])
    result = fit_response_regression(
        data,
        outcome="CAR_20_minus_1",
        treatments=INNOVATION_TREATMENTS,
        controls=full_controls,
        firm_fe="permno",
        quarter_fe="event_calendar_quarter",
        cluster_cols=("permno", "event_trading_date"),
    )
    out = result.coefficients.copy()
    out.insert(0, "horizon", "20_minus_1")
    out.insert(1, "outcome", "CAR_20_minus_1")
    return out


def leakage_checks(cache: FeatureCache, results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for component, spec in cache.specs.items():
        leaked = outcome_like_feature_columns(list(spec["feature_columns"]))
        rows.append(
            {
                "check": "no_outcome_first_stage_features",
                "learner": "",
                "component": component,
                "passes": not leaked,
                "details": json.dumps(leaked),
            }
        )
    selections = concat_result_table(results, "hyperparameter_selection")
    estimated = selections.loc[selections["n_inner_folds"].gt(0)].copy()
    if not estimated.empty:
        estimated["inner_validation_years"] = estimated["inner_validation_years_json"].map(json.loads)
        estimated["inner_train_max_years"] = estimated["inner_train_max_years_json"].map(json.loads)
        ok = True
        for row in estimated.itertuples(index=False):
            for train_year, inner_year in zip(row.inner_train_max_years, row.inner_validation_years, strict=True):
                if not (train_year < inner_year < row.outer_year):
                    ok = False
        rows.append(
            {
                "check": "nested_inner_years_strictly_precede_outer_year",
                "learner": "tuned",
                "component": "",
                "passes": ok,
                "details": "",
            }
        )
    return pd.DataFrame(rows)


def learner_first_stage_samples(
    cache: FeatureCache,
    results: dict[str, dict[str, pd.DataFrame]],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    common_ids = set(manifest["transcriptid"].astype(str))
    for learner, result in results.items():
        for component in PRIMARY_COMPONENTS:
            grain = str(cache.specs[component]["grain"])
            source = result["call_innovations"] if grain == "event" else result["turn_innovations"]
            component_rows = source.loc[source["component"].eq(component)].copy()
            call_rows = result["call_innovations"].loc[result["call_innovations"]["component"].eq(component)].copy()
            rows.append(
                {
                    "learner": learner,
                    "component": component,
                    "grain": grain,
                    "n_feature_rows": int(len(cache.frames[component])),
                    "n_first_stage_estimable_rows": int(len(component_rows)),
                    "n_call_level_rows": int(len(call_rows)),
                    "n_common_response_rows": int(call_rows["transcriptid"].astype(str).isin(common_ids).sum()),
                }
            )
    return pd.DataFrame(rows)


def innovation_correlations(
    results: dict[str, dict[str, pd.DataFrame]],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    common_ids = set(manifest["transcriptid"].astype(str))
    pieces = []
    for learner, result in results.items():
        data = result["call_innovations"].loc[
            result["call_innovations"]["semantic_feature"].eq(PRIMARY_FEATURE)
        ].copy()
        data = data.loc[data["transcriptid"].astype(str).isin(common_ids)]
        data["learner"] = learner
        pieces.append(data[["transcriptid", "component", "learner", "narrative_innovation"]])
    all_data = pd.concat(pieces, ignore_index=True)
    for component, group in all_data.groupby("component", sort=True):
        wide = group.pivot_table(
            index="transcriptid",
            columns="learner",
            values="narrative_innovation",
            aggfunc="first",
        )
        corr = wide.corr()
        for left in corr.columns:
            for right in corr.columns:
                if left < right:
                    rows.append(
                        {
                            "component": component,
                            "left_learner": left,
                            "right_learner": right,
                            "n_obs": int(wide[[left, right]].dropna().shape[0]),
                            "correlation": float(corr.loc[left, right]),
                        }
                    )
    return pd.DataFrame(rows)


def innovation_distributions(
    results: dict[str, dict[str, pd.DataFrame]],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    common_ids = set(manifest["transcriptid"].astype(str))
    for learner, result in results.items():
        data = result["call_innovations"].loc[
            result["call_innovations"]["semantic_feature"].eq(PRIMARY_FEATURE)
        ].copy()
        data = data.loc[data["transcriptid"].astype(str).isin(common_ids)]
        for component, group in data.groupby("component", sort=True):
            values = pd.to_numeric(group["narrative_innovation"], errors="coerce").dropna()
            rows.append(
                {
                    "learner": learner,
                    "component": component,
                    "n_obs": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std()) if len(values) else np.nan,
                    "p01": float(values.quantile(0.01)) if len(values) else np.nan,
                    "p05": float(values.quantile(0.05)) if len(values) else np.nan,
                    "p50": float(values.quantile(0.50)) if len(values) else np.nan,
                    "p95": float(values.quantile(0.95)) if len(values) else np.nan,
                    "p99": float(values.quantile(0.99)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def calibration_table(first_stage_metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "learner",
        "component",
        "grain",
        "fold_year",
        "n_test",
        "predicted_actual_corr",
        "calibration_slope",
        "actual_mean",
        "predicted_mean",
    ]
    return first_stage_metrics[cols].copy()


def classify_invariance(
    response_coefficients: pd.DataFrame,
    car20_minus_1: pd.DataFrame,
) -> pd.DataFrame:
    fixed = response_coefficients.loc[
        response_coefficients["learner"].eq("fixed_ridge")
        & response_coefficients["treatment"].eq("answer_innovation_z")
    ].copy()
    fixed_car = car20_minus_1.loc[
        car20_minus_1["learner"].eq("fixed_ridge")
        & car20_minus_1["treatment"].eq("answer_innovation_z")
    ].copy()
    rows = []
    for learner in sorted(response_coefficients["learner"].unique()):
        if learner == "fixed_ridge":
            continue
        current = response_coefficients.loc[
            response_coefficients["learner"].eq(learner)
            & response_coefficients["treatment"].eq("answer_innovation_z")
        ].copy()
        merged = fixed.merge(current, on=["horizon", "outcome", "treatment"], suffixes=("_fixed", "_learner"))
        main_sign = bool(
            (
                np.sign(merged["beta_fixed"].round(14))
                == np.sign(merged["beta_learner"].round(14))
            ).all()
        )
        main_ci = bool(all(ci_overlap(row) for row in merged.itertuples(index=False)))
        current_car = car20_minus_1.loc[
            car20_minus_1["learner"].eq(learner)
            & car20_minus_1["treatment"].eq("answer_innovation_z")
        ].copy()
        car = fixed_car.merge(current_car, on=["horizon", "outcome", "treatment"], suffixes=("_fixed", "_learner"))
        car_sign = bool(
            not car.empty
            and (
                np.sign(car["beta_fixed"].round(14))
                == np.sign(car["beta_learner"].round(14))
            ).all()
        )
        car_ci = bool(not car.empty and all(ci_overlap(row) for row in car.itertuples(index=False)))
        preserves = bool(main_sign and main_ci and car_sign and car_ci)
        rows.append(
            {
                "learner": learner,
                "main_answer_curve_same_sign": main_sign,
                "main_answer_curve_ci_overlap": main_ci,
                "car20_minus_1_same_sign": car_sign,
                "car20_minus_1_ci_overlap": car_ci,
                "preserves_fixed_ridge_result": preserves,
            }
        )
    if rows and all(row["preserves_fixed_ridge_result"] for row in rows):
        classification = "invariant"
    elif rows and any(row["preserves_fixed_ridge_result"] for row in rows):
        classification = "partially_invariant"
    else:
        classification = "learner_dependent"
    for row in rows:
        row["overall_classification"] = classification
    return pd.DataFrame(rows)


def ci_overlap(row) -> bool:
    fixed_low = row.beta_fixed - 1.96 * row.standard_error_fixed
    fixed_high = row.beta_fixed + 1.96 * row.standard_error_fixed
    learner_low = row.beta_learner - 1.96 * row.standard_error_learner
    learner_high = row.beta_learner + 1.96 * row.standard_error_learner
    return bool(max(fixed_low, learner_low) <= min(fixed_high, learner_high))


def concat_result_table(results: dict[str, dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    return pd.concat([result[key] for result in results.values()], ignore_index=True)


def candidate_json(candidate: CandidateSpec) -> str:
    return json.dumps(
        {"model_type": candidate.model_type, "params": candidate.params},
        sort_keys=True,
    )


def write_tables(output_dir: Path, **tables: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
