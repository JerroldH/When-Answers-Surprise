"""Outcome-blind tuned-Ridge re-estimation of the revision-v2 sequence.

This task consumes the already enriched, frozen feature frames.  It does not
rebuild transcript text features and it never uses market outcomes while
selecting Ridge penalties.  Every prepared/question/answer specification uses
the same inner expanding-year MAE rule and the same outer time-forward folds.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from when_answers_surprise.empirical.feature_spec import (
    BASE_STATE_FEATURES,
    REVISION_CATEGORICAL_FEATURES,
    question_filtration_features,
    revision_answer_filtration_features,
)
from when_answers_surprise.empirical.learner_audit import (
    feature_frame_to_innovations,
    oof_predict_feature_frame,
    tuned_ridge_spec_with_alphas,
)
from when_answers_surprise.empirical.pipeline import (
    OOF_TEST_YEARS,
    PRIMARY_FEATURE,
    aggregate_turn_innovations,
    base_categorical_features,
    estimate_all_response_curves,
    response_controls,
    zscore,
)
from when_answers_surprise.empirical.revision_v2 import (
    add_main_multiplicity,
    build_primary_result,
    estimate_answer_year_joint_test,
    estimate_answer_year_stability,
    estimate_incremental_responses,
    estimate_ladder_responses,
    estimate_standardization_invariance,
)
from when_answers_surprise.empirical.revision_v2_inference import (
    curve_joint_tests,
    main_joint_system,
    simultaneous_bands,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025" / "analysis"
DEFAULT_REVISION_DIR = ANALYSIS_ROOT / "revision_v2"
DEFAULT_FEATURE_CACHE_DIR = ANALYSIS_ROOT / "lm_baseline_learner_audit_cache"
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "revision_v2_tuned_ridge"
DEFAULT_RIDGE_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
)


@dataclass(frozen=True)
class TunedSequentialConfig:
    revision_dir: Path = DEFAULT_REVISION_DIR
    feature_cache_dir: Path = DEFAULT_FEATURE_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    min_train_rows: int = 120
    max_calls_per_year: int | None = None


def run_tuned_sequential(config: TunedSequentialConfig) -> dict[str, object]:
    """Re-estimate P0, Q0-Q2, and A0-A3 with one tuned-Ridge protocol."""

    _require_empty_output(config.output_dir)
    frames, fixed_response = load_tuned_inputs(
        config.revision_dir, config.feature_cache_dir
    )
    if config.max_calls_per_year is not None:
        frames, fixed_response = restrict_smoke_sample(
            frames, fixed_response, config.max_calls_per_year
        )

    learner = tuned_ridge_spec_with_alphas(config.ridge_alphas)
    prepared = estimate_component_ladder(
        frames["prepared"],
        component="prepared",
        feature_sets={"P0": list(BASE_STATE_FEATURES)},
        learner=learner,
        min_train_rows=config.min_train_rows,
    )
    question = estimate_component_ladder(
        frames["question"],
        component="question",
        feature_sets=question_filtration_features(frames["question"]),
        learner=learner,
        min_train_rows=config.min_train_rows,
    )
    answer = estimate_component_ladder(
        frames["answer"],
        component="answer",
        feature_sets=revision_answer_filtration_features(frames["answer"]),
        learner=learner,
        min_train_rows=config.min_train_rows,
    )

    response = assemble_tuned_response(
        fixed_response,
        prepared["call_innovations"],
        question["call_innovations"],
        answer["call_innovations"],
    )
    coefficients, coefficient_tests = estimate_all_response_curves(response)
    coefficients = add_main_multiplicity(coefficients)
    primary = build_primary_result(coefficients)
    joint = main_joint_system(response, controls=response_controls(response))

    first_stage_metrics = pd.concat(
        [
            prepared["fold_metrics"],
            question["fold_metrics"],
            answer["fold_metrics"],
        ],
        ignore_index=True,
    )
    selections = pd.concat(
        [prepared["selections"], question["selections"], answer["selections"]],
        ignore_index=True,
    )
    selected = selections.loc[selections["selected"].astype(bool)].copy()
    if selected.empty:
        raise ValueError("Tuned Ridge did not record any selected hyperparameters")
    ladder_response_base = fixed_response.drop(
        columns=[
            *[f"question_{variant}" for variant in ("Q0", "Q1", "Q2")],
            *[f"answer_{variant}" for variant in ("A0", "A1", "A2", "A3")],
        ],
        errors="ignore",
    )

    tables = {
        "prepared_ladder_first_stage": prepared["first_stage"],
        "question_ladder_first_stage": question["first_stage"],
        "answer_ladder_first_stage": answer["first_stage"],
        "tuned_ridge_fold_metrics": first_stage_metrics,
        "tuned_ridge_hyperparameter_selection": selections,
        "question_ladder_responses": estimate_ladder_responses(
            question, ladder_response_base, component="question"
        ),
        "answer_ladder_responses": estimate_ladder_responses(
            answer, ladder_response_base, component="answer"
        ),
        "response_coefficients_conditional": coefficients,
        "response_coefficient_tests": coefficient_tests,
        "response_joint_coefficient_index": joint.coefficient_index,
        "response_curve_joint_tests_conditional": curve_joint_tests(
            joint.coefficients, joint.covariance, joint.coefficient_index
        ),
        "response_simultaneous_bands_conditional": simultaneous_bands(
            joint.coefficients,
            joint.covariance,
            joint.coefficient_index,
            seed=20260716,
        ),
        "primary_result_conditional": primary,
        "incremental_responses": estimate_incremental_responses(response),
        "answer_year_stability": estimate_answer_year_stability(response),
        "answer_year_joint_test": estimate_answer_year_joint_test(response),
        "standardization_invariance": estimate_standardization_invariance(response),
        "sample_accounting": tuned_sample_accounting(frames, fixed_response, response),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(config.output_dir / f"{name}.csv", index=False)
    np.save(
        config.output_dir / "response_joint_covariance_conditional.npy",
        joint.covariance,
    )

    prepared["turn_innovations"].to_parquet(
        config.output_dir / "prepared_ladder_call_residuals.parquet", index=False
    )
    question["turn_innovations"].to_parquet(
        config.output_dir / "question_ladder_turn_residuals.parquet", index=False
    )
    answer["turn_innovations"].to_parquet(
        config.output_dir / "answer_ladder_turn_residuals.parquet", index=False
    )
    response.to_parquet(
        config.output_dir / "response_model_sample_revision_v2.parquet", index=False
    )

    # Bootstrap workers consume the exact feature frames used here.  A physical
    # copy keeps every versioned run self-contained; smoke runs write their
    # restricted frames so identity refits cannot silently use the full sample.
    destinations = {
        "prepared": config.output_dir / "prepared_first_stage_features.parquet",
        "question": config.output_dir
        / "question_first_stage_features_revision_v2.parquet",
        "answer": config.output_dir / "answer_first_stage_features_revision_v2.parquet",
    }
    if config.max_calls_per_year is None:
        sources = {
            "prepared": config.feature_cache_dir
            / "prepared_first_stage_features.parquet",
            "question": config.revision_dir
            / "question_first_stage_features_revision_v2.parquet",
            "answer": config.revision_dir
            / "answer_first_stage_features_revision_v2.parquet",
        }
        for component, source in sources.items():
            shutil.copy2(source, destinations[component])
    else:
        for component, destination in destinations.items():
            frames[component].to_parquet(destination, index=False)

    manifest = {
        "status": "complete",
        "estimand_type": "projection_residual",
        "learner": "time_forward_tuned_ridge",
        "selection_loss": "inner_expanding_year_mae",
        "market_outcomes_used_for_tuning": False,
        "ridge_alpha_grid": list(config.ridge_alphas),
        "prepared_variants": ["P0"],
        "question_variants": ["Q0", "Q1", "Q2"],
        "answer_variants": ["A0", "A1", "A2", "A3"],
        "fixed_response_sample": True,
        "source_revision_dir": str(config.revision_dir),
        "source_feature_cache_dir": str(config.feature_cache_dir),
    }
    (config.output_dir / "revision_v2_tuned_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {
        **manifest,
        "output_dir": str(config.output_dir),
        "response_calls": int(len(response)),
        "primary_beta": float(primary.iloc[0]["beta"]),
        "primary_conditional_se": float(primary.iloc[0]["standard_error"]),
        "selected_grid_boundary_count": int(
            selected["selected_at_grid_boundary"].astype(bool).sum()
        ),
        "selected_grid_boundary_rows": selected.loc[
            selected["selected_at_grid_boundary"].astype(bool),
            ["component", "variant", "outer_year", "candidate_json"],
        ].to_dict("records"),
        "smoke_sample": config.max_calls_per_year is not None,
    }
    (config.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def load_tuned_inputs(
    revision_dir: Path, feature_cache_dir: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
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
    response = pd.read_parquet(
        revision_dir / "response_model_sample_revision_v2.parquet"
    )
    for frame in [*frames.values(), response]:
        frame["transcriptid"] = frame["transcriptid"].astype(str)
    if frames["prepared"]["transcriptid"].duplicated().any():
        raise ValueError("Prepared frame must be unique by transcriptid")
    for component in ("question", "answer"):
        if frames[component].duplicated(["transcriptid", "turn_index"]).any():
            raise ValueError(
                f"{component.title()} frame must be unique by transcriptid, turn_index"
            )
    if response["transcriptid"].duplicated().any():
        raise ValueError("Response sample must be unique by transcriptid")
    return frames, response


def estimate_component_ladder(
    frame: pd.DataFrame,
    *,
    component: str,
    feature_sets: dict[str, list[str]],
    learner: object,
    min_train_rows: int,
    outer_years: Sequence[int] | None = None,
) -> dict[str, pd.DataFrame]:
    grain = "call" if component == "prepared" else "turn"
    categorical = [*base_categorical_features(), *REVISION_CATEGORICAL_FEATURES]
    innovations = []
    first_stage_rows = []
    metric_tables = []
    selection_tables = []
    call_tables = []

    for variant, features in feature_sets.items():
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError(f"{component}/{variant} is missing features: {missing}")
        spec = {
            "component": component,
            "grain": grain,
            "target": PRIMARY_FEATURE,
            "feature_columns": features,
            "categorical_columns": [name for name in categorical if name in features],
        }
        prediction, fold, metrics, selections = oof_predict_feature_frame(
            frame,
            spec,
            learner,
            min_train_rows,
            outer_years=tuple(outer_years) if outer_years is not None else OOF_TEST_YEARS,
        )
        turns = feature_frame_to_innovations(
            frame,
            component=component,
            grain=grain,
            predictions=prediction,
            fold_year=fold,
        )
        turns["variant"] = variant
        innovations.append(turns)
        metrics.insert(1, "variant", variant)
        selections.insert(2, "variant", variant)
        metric_tables.append(metrics)
        selection_tables.append(selections)
        first_stage_rows.append(
            summarize_first_stage(component, variant, features, turns, metrics)
        )

        if component == "prepared":
            calls = turns[["transcriptid", "narrative_innovation"]].copy()
        else:
            calls = aggregate_turn_innovations(turns, component)[
                ["transcriptid", "narrative_innovation"]
            ]
        calls = calls.rename(
            columns={"narrative_innovation": f"{component}_{variant}"}
        )
        call_tables.append(calls)

    all_innovations = pd.concat(innovations, ignore_index=True)
    counts = all_innovations.groupby("variant").size()
    if counts.nunique() != 1:
        raise ValueError(
            f"{component} variants do not share a common estimable sample: {counts.to_dict()}"
        )
    call_wide = merge_call_tables(call_tables)
    call_counts = [len(table) for table in call_tables]
    if len(set(call_counts)) != 1:
        raise ValueError(
            f"{component} variants do not share a common call sample: {call_counts}"
        )
    return {
        "turn_innovations": all_innovations,
        "call_innovations": call_wide,
        "first_stage": pd.DataFrame(first_stage_rows),
        "fold_metrics": pd.concat(metric_tables, ignore_index=True),
        "selections": pd.concat(selection_tables, ignore_index=True),
    }


def summarize_first_stage(
    component: str,
    variant: str,
    features: Sequence[str],
    innovations: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, object]:
    overall = metrics.loc[metrics["fold_year"].astype(str).eq("overall")]
    if len(overall) != 1:
        raise ValueError(f"Expected one overall first-stage row for {component}/{variant}")
    row = overall.iloc[0]
    estimated_folds = metrics.loc[
        metrics["fold_year"].astype(str).ne("overall")
        & metrics["status"].eq("estimated")
    ]
    return {
        "component": component,
        "variant": variant,
        "n_turns": int(len(innovations)),
        "n_calls": int(innovations["transcriptid"].nunique()),
        "n_features": int(len(features)),
        "min_encoded_features": np.nan,
        "max_encoded_features": np.nan,
        "oof_r2": float(row["r2"]),
        "oof_mae": float(row["mae"]),
        "predicted_actual_correlation": float(row["predicted_actual_corr"]),
        "calibration_slope": float(row["calibration_slope"]),
        "turn_residual_sd": float(innovations["narrative_innovation"].std(ddof=1)),
        "mean_fold_r2": float(estimated_folds["r2"].mean()),
        "learner": "time_forward_tuned_ridge",
    }


def merge_call_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        raise ValueError("No call tables supplied")
    merged = tables[0].copy()
    for table in tables[1:]:
        merged = merged.merge(table, on="transcriptid", how="inner", validate="one_to_one")
    return merged


def assemble_tuned_response(
    fixed_response: pd.DataFrame,
    prepared_calls: pd.DataFrame,
    question_calls: pd.DataFrame,
    answer_calls: pd.DataFrame,
) -> pd.DataFrame:
    response = fixed_response.copy()
    response["transcriptid"] = response["transcriptid"].astype(str)
    legacy = [
        "prepared_innovation",
        "prepared_innovation_z",
        "question_innovation",
        "question_innovation_z",
        "answer_innovation",
        "answer_innovation_z",
    ]
    response = response.rename(
        columns={name: f"{name}_legacy_fixed_ridge" for name in legacy if name in response}
    )
    variant_legacy = {
        name: f"{name}_legacy_fixed_ridge"
        for name in ("prepared_P0", "question_Q2", "answer_A3")
        if name in response
    }
    response = response.rename(columns=variant_legacy)
    response = response.merge(
        prepared_calls[["transcriptid", "prepared_P0"]],
        on="transcriptid",
        how="left",
        validate="one_to_one",
    ).merge(
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
    source = {
        "prepared_innovation": "prepared_P0",
        "question_innovation": "question_Q2",
        "answer_innovation": "answer_A3",
    }
    if response[list(source.values())].isna().any().any():
        missing = response.loc[
            response[list(source.values())].isna().any(axis=1), "transcriptid"
        ]
        raise ValueError(
            f"Tuned residuals do not cover the fixed response sample: {missing.head().tolist()}"
        )
    for destination, origin in source.items():
        response[destination] = response[origin]
        response[f"{destination}_z"] = zscore(response[destination])
    if len(response) != len(fixed_response) or set(response["transcriptid"]) != set(
        fixed_response["transcriptid"].astype(str)
    ):
        raise ValueError("Tuned re-estimation changed the frozen response sample")
    return response


def tuned_sample_accounting(
    frames: dict[str, pd.DataFrame],
    fixed_response: pd.DataFrame,
    response: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "prepared_feature_frame",
                "unit": "call",
                "count": len(frames["prepared"]),
                "reason": "immutable first-stage input",
            },
            {
                "stage": "question_feature_frame",
                "unit": "turn",
                "count": len(frames["question"]),
                "reason": "immutable enriched first-stage input",
            },
            {
                "stage": "answer_feature_frame",
                "unit": "turn",
                "count": len(frames["answer"]),
                "reason": "immutable enriched first-stage input",
            },
            {
                "stage": "frozen_response_input",
                "unit": "call",
                "count": len(fixed_response),
                "reason": "outcomes and controls fixed before learner tuning",
            },
            {
                "stage": "tuned_response_output",
                "unit": "call",
                "count": len(response),
                "reason": "same transcript ids as frozen response input",
            },
        ]
    )


def restrict_smoke_sample(
    frames: dict[str, pd.DataFrame],
    response: pd.DataFrame,
    calls_per_year: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if calls_per_year < 1:
        raise ValueError("max_calls_per_year must be positive")
    calls = (
        frames["prepared"][["event_year", "transcriptid"]]
        .sort_values(["event_year", "transcriptid"])
        .groupby("event_year", sort=True)
        .head(calls_per_year)
    )
    keep = set(calls["transcriptid"].astype(str))
    restricted = {
        name: frame.loc[frame["transcriptid"].isin(keep)].copy()
        for name, frame in frames.items()
    }
    restricted_response = response.loc[response["transcriptid"].isin(keep)].copy()
    return restricted, restricted_response


def _require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-estimate the complete revision-v2 sequence with tuned Ridge."
    )
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ridge-alphas", nargs="+", type=float, default=list(DEFAULT_RIDGE_ALPHAS)
    )
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--max-calls-per-year", type=int)
    args = parser.parse_args()
    summary = run_tuned_sequential(
        TunedSequentialConfig(
            revision_dir=args.revision_dir,
            feature_cache_dir=args.feature_cache_dir,
            output_dir=args.output_dir,
            ridge_alphas=tuple(args.ridge_alphas),
            min_train_rows=args.min_train_rows,
            max_calls_per_year=args.max_calls_per_year,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
