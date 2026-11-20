"""Deterministic supplement audits derived from the frozen revision-v2 evidence.

This module never refits the first-stage models and never writes into the
revision-v2 freeze.  It re-aggregates the frozen A3 turn residuals, summarizes
the frozen pre-event systems, and checks convergence of nested subsets of the
completed 200-draw first-stage refit sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from when_answers_surprise.empirical.pipeline import HORIZONS, response_controls, zscore
from when_answers_surprise.empirical.pre_event_audit import normal_two_sided_p
from when_answers_surprise.empirical.revision_v2_inference import (
    fit_joint_horizon_system,
    simultaneous_bands,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_DIR = (
    PROJECT_ROOT / "outputs" / "frozen" / "revision_v2_tuned" / "20260716_widegrid_v2"
)
DEFAULT_TUNED_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "sockeye"
    / "results"
    / "runs"
    / "20260716_tuned_seq_boot200_widegrid"
    / "revision_v2_tuned_ridge"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "frozen"
    / "revision_v2_supplement"
    / "20260731_local_audits"
)
TREATMENTS = ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]
AGGREGATION_METHODS = ("word_weighted_mean", "equal_turn_mean", "within_call_median")
BOOTSTRAP_CHECKPOINTS = (25, 50, 100, 150, 200)
PRIMARY_BETA_BPS = 34.319678512412025
COMPONENT_IDENTITY_TOLERANCE = 1e-10


def aggregate_answer_turns(
    turn_residuals: pd.DataFrame,
    *,
    method: str,
    variant: str = "A3",
    semantic_feature: str = "lm_net_tone",
) -> pd.DataFrame:
    """Aggregate one frozen answer residual per call under a named rule."""

    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method: {method}")
    required = {
        "transcriptid",
        "variant",
        "semantic_feature",
        "narrative_innovation",
        "word_count",
    }
    if missing := sorted(required.difference(turn_residuals.columns)):
        raise ValueError(f"Answer turn residuals are missing columns: {missing}")
    work = turn_residuals.loc[
        turn_residuals["variant"].eq(variant)
        & turn_residuals["semantic_feature"].eq(semantic_feature)
    ].copy()
    work["narrative_innovation"] = pd.to_numeric(
        work["narrative_innovation"], errors="coerce"
    )
    work["word_count"] = pd.to_numeric(work["word_count"], errors="coerce")
    work = work.dropna(subset=["transcriptid", "narrative_innovation"])
    if work.empty:
        raise ValueError(f"No {variant}/{semantic_feature} answer turn residuals were found")

    if method == "equal_turn_mean":
        values = work.groupby("transcriptid", sort=True)["narrative_innovation"].mean()
    elif method == "within_call_median":
        values = work.groupby("transcriptid", sort=True)["narrative_innovation"].median()
    else:
        values = work.groupby("transcriptid", sort=True).apply(
            _word_weighted_residual, include_groups=False
        )
    result = values.rename("answer_innovation").reset_index()
    if result["transcriptid"].duplicated().any():
        raise ValueError("Answer aggregation produced duplicate transcript ids")
    return result


def estimate_aggregation_robustness(
    response: pd.DataFrame,
    turn_residuals: pd.DataFrame,
    frozen_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate all response horizons for three A3 aggregation rules."""

    if response["transcriptid"].duplicated().any():
        raise ValueError("Frozen response sample must have unique transcript ids")
    baseline = aggregate_answer_turns(turn_residuals, method="word_weighted_mean")
    baseline = baseline.rename(columns={"answer_innovation": "baseline_answer_innovation"})
    rows: list[dict[str, object]] = []
    for method in AGGREGATION_METHODS:
        aggregated = aggregate_answer_turns(turn_residuals, method=method)
        work = response.drop(
            columns=["answer_innovation", "answer_innovation_z"], errors="ignore"
        ).merge(aggregated, on="transcriptid", how="left", validate="one_to_one")
        work = work.merge(baseline, on="transcriptid", how="left", validate="one_to_one")
        if work["answer_innovation"].isna().any():
            missing = int(work["answer_innovation"].isna().sum())
            raise ValueError(f"{method} is missing {missing} frozen response calls")
        work["answer_innovation_z"] = zscore(work["answer_innovation"])
        correlation = float(
            work[["answer_innovation", "baseline_answer_innovation"]].corr().iloc[0, 1]
        )
        residual_sd = float(work["answer_innovation"].std(ddof=1))
        outcomes = [f"CAR_0_{horizon}" for horizon in HORIZONS]
        system = fit_joint_horizon_system(
            work,
            outcomes=outcomes,
            treatments=TREATMENTS,
            controls=response_controls(work),
        )
        answer_coefficients = system.coefficients.loc[
            system.coefficients["treatment"].eq("answer_innovation_z")
        ].set_index("outcome")
        clean_keys = work[
            [
                *outcomes,
                *TREATMENTS,
                *response_controls(work),
                "permno",
                "event_calendar_quarter",
                "event_trading_date",
            ]
        ].dropna()
        for horizon, outcome in zip(HORIZONS, outcomes, strict=True):
            coefficient = answer_coefficients.loc[outcome]
            beta = float(coefficient["beta"])
            standard_error = float(coefficient["standard_error"])
            rows.append(
                {
                    "aggregation": method,
                    "horizon": int(horizon),
                    "outcome": outcome,
                    "treatment": "answer_innovation_z",
                    "n_obs": int(coefficient["n_obs"]),
                    "n_firms": int(clean_keys["permno"].nunique()),
                    "n_event_dates": int(clean_keys["event_trading_date"].nunique()),
                    "answer_residual_sd": residual_sd,
                    "correlation_with_word_weighted_baseline": correlation,
                    "beta": beta,
                    "standard_error": standard_error,
                    "ci_low": beta - 1.96 * standard_error,
                    "ci_high": beta + 1.96 * standard_error,
                    "p_value": normal_two_sided_p(beta / standard_error),
                    "beta_bps": 10_000 * beta,
                    "standard_error_bps": 10_000 * standard_error,
                    "ci_low_bps": 10_000 * (beta - 1.96 * standard_error),
                    "ci_high_bps": 10_000 * (beta + 1.96 * standard_error),
                }
            )
    result = pd.DataFrame(rows)
    _assert_aggregation_parity(result, frozen_coefficients)
    return result


def aggregate_answer_components(
    turn_residuals: pd.DataFrame,
    *,
    variants: tuple[str, ...] = ("A0", "A3"),
    semantic_feature: str = "lm_net_tone",
) -> pd.DataFrame:
    """Aggregate frozen raw, expected, and residual answer tone by call."""

    required = {
        "transcriptid",
        "variant",
        "semantic_feature",
        "raw_score",
        "expected_score_oof",
        "narrative_innovation",
        "word_count",
    }
    if missing := sorted(required.difference(turn_residuals.columns)):
        raise ValueError(f"Answer turn residuals are missing columns: {missing}")
    work = turn_residuals.loc[
        turn_residuals["variant"].isin(variants)
        & turn_residuals["semantic_feature"].eq(semantic_feature)
    ].copy()
    for column in (
        "raw_score",
        "expected_score_oof",
        "narrative_innovation",
        "word_count",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(
        subset=[
            "transcriptid",
            "variant",
            "raw_score",
            "expected_score_oof",
            "narrative_innovation",
        ]
    )
    observed_variants = set(work["variant"].astype(str))
    if observed_variants != set(variants):
        raise ValueError(
            "Answer component audit requires exactly the declared variants: "
            f"expected={sorted(variants)}, observed={sorted(observed_variants)}"
        )

    rows: list[dict[str, object]] = []
    for (variant, transcriptid), group in work.groupby(
        ["variant", "transcriptid"], sort=True
    ):
        weights = group["word_count"].clip(lower=0).fillna(0).astype(float)
        if float(weights.sum()) <= 0:
            weights = pd.Series(1.0, index=group.index)
        rows.append(
            {
                "transcriptid": transcriptid,
                "variant": str(variant),
                "turn_count": int(len(group)),
                "raw_answer_tone": float(
                    np.average(group["raw_score"].astype(float), weights=weights)
                ),
                "expected_answer_tone": float(
                    np.average(
                        group["expected_score_oof"].astype(float), weights=weights
                    )
                ),
                "answer_residual": float(
                    np.average(
                        group["narrative_innovation"].astype(float), weights=weights
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.duplicated(["variant", "transcriptid"]).any():
        raise ValueError("Answer component aggregation produced duplicate call keys")
    counts = result.groupby("variant")["transcriptid"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"A0/A3 component calls differ: {counts.to_dict()}")
    identity_error = (
        result["raw_answer_tone"]
        - result["expected_answer_tone"]
        - result["answer_residual"]
    ).abs().max()
    if not np.isfinite(identity_error) or identity_error > COMPONENT_IDENTITY_TOLERANCE:
        raise ValueError(
            "Frozen answer components fail raw = expected + residual: "
            f"max error={identity_error:.3e}"
        )
    return result


def context_answer_pairing_diagnostic(
    turn_residuals: pd.DataFrame,
    *,
    variants: tuple[str, ...] = ("A0", "A3"),
    semantic_feature: str = "lm_net_tone",
) -> pd.DataFrame:
    """Compare matched contexts with the exact within-call derangement expectation.

    The diagnostic does not refit the benchmark on swapped text.  For each
    multi-turn call, it averages every pairing between an answer and a
    different turn's frozen prediction.  This is the exact expectation of a
    uniformly drawn within-call derangement.
    """

    required = {
        "transcriptid",
        "variant",
        "semantic_feature",
        "raw_score",
        "expected_score_oof",
    }
    if missing := sorted(required.difference(turn_residuals.columns)):
        raise ValueError(f"Pairing diagnostic is missing columns: {missing}")
    work = turn_residuals.loc[
        turn_residuals["variant"].isin(variants)
        & turn_residuals["semantic_feature"].eq(semantic_feature)
    ].copy()
    work["raw_score"] = pd.to_numeric(work["raw_score"], errors="coerce")
    work["expected_score_oof"] = pd.to_numeric(
        work["expected_score_oof"], errors="coerce"
    )
    work = work.dropna(
        subset=["transcriptid", "variant", "raw_score", "expected_score_oof"]
    )
    call_sizes = work.groupby(["variant", "transcriptid"]).size()
    valid_calls = set(
        call_sizes.loc[call_sizes.ge(2)].index.get_level_values("transcriptid")
    )
    work = work.loc[work["transcriptid"].isin(valid_calls)].copy()

    rows: list[dict[str, object]] = []
    for variant, group in work.groupby("variant", sort=False):
        actual = group["raw_score"].to_numpy(dtype=float)
        predicted = group["expected_score_oof"].to_numpy(dtype=float)
        matched_sse = float(np.square(actual - predicted).sum())
        total_sst = float(np.square(actual - actual.mean()).sum())
        expected_deranged_sse = 0.0
        expected_cross_product = 0.0
        call_gains: list[float] = []
        max_within_call_prediction_sd = 0.0
        for _, call in group.groupby("transcriptid", sort=False):
            call_actual = call["raw_score"].to_numpy(dtype=float)
            call_predicted = call["expected_score_oof"].to_numpy(dtype=float)
            turn_count = len(call)
            squared_errors = np.square(
                call_actual[:, None] - call_predicted[None, :]
            )
            matched_call_mse = float(np.trace(squared_errors) / turn_count)
            deranged_call_mse = float(
                (squared_errors.sum() - np.trace(squared_errors))
                / (turn_count * (turn_count - 1))
            )
            call_gains.append(deranged_call_mse - matched_call_mse)
            expected_deranged_sse += float(
                (squared_errors.sum() - np.trace(squared_errors))
                / (turn_count - 1)
            )
            expected_cross_product += float(
                (
                    call_actual.sum() * call_predicted.sum()
                    - np.dot(call_actual, call_predicted)
                )
                / (turn_count - 1)
            )
            max_within_call_prediction_sd = max(
                max_within_call_prediction_sd,
                float(np.std(call_predicted, ddof=0)),
            )

        n_turns = len(group)
        matched_mse = matched_sse / n_turns
        deranged_mse = expected_deranged_sse / n_turns
        expected_covariance = (
            expected_cross_product / n_turns - actual.mean() * predicted.mean()
        )
        denominator = float(actual.std(ddof=0) * predicted.std(ddof=0))
        expected_deranged_correlation = (
            expected_covariance / denominator if denominator > 0 else np.nan
        )
        tolerance = 1e-12
        if max_within_call_prediction_sd <= tolerance:
            positive = 0
            negative = 0
        else:
            positive = int(sum(gain > tolerance for gain in call_gains))
            negative = int(sum(gain < -tolerance for gain in call_gains))
        non_ties = positive + negative
        sign_p = (
            float(binomtest(positive, non_ties, 0.5, alternative="greater").pvalue)
            if non_ties
            else 1.0
        )
        matched_denominator = float(actual.std(ddof=0) * predicted.std(ddof=0))
        matched_correlation = (
            float(np.corrcoef(actual, predicted)[0, 1])
            if matched_denominator > 0
            else np.nan
        )
        rows.append(
            {
                "variant": str(variant),
                "n_calls": int(group["transcriptid"].nunique()),
                "n_turns": int(n_turns),
                "matched_r2": 1.0 - matched_sse / total_sst,
                "expected_deranged_r2": 1.0 - expected_deranged_sse / total_sst,
                "matched_mse": matched_mse,
                "expected_deranged_mse": deranged_mse,
                "deranged_mse_increase_percent": (
                    100.0 * (deranged_mse / matched_mse - 1.0)
                    if matched_mse > 0
                    else np.inf
                ),
                "matched_correlation": matched_correlation,
                "expected_deranged_correlation": expected_deranged_correlation,
                "positive_pairing_gain_calls": positive,
                "negative_pairing_gain_calls": negative,
                "tied_pairing_gain_calls": int(len(call_gains) - non_ties),
                "one_sided_sign_p_value": sign_p,
                "max_within_call_prediction_sd": max_within_call_prediction_sd,
                "diagnostic_scope": (
                    "exact expectation over frozen within-call prediction derangements; "
                    "no first-stage refit"
                ),
            }
        )
    return pd.DataFrame(rows)


def development_answer_decomposition(
    response: pd.DataFrame,
    turn_residuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose raw answer tone and directly test A0-to-A3 reclassification."""

    components = aggregate_answer_components(turn_residuals)
    wide = components.pivot(
        index="transcriptid",
        columns="variant",
        values=["raw_answer_tone", "expected_answer_tone", "answer_residual"],
    )
    wide.columns = [f"{field}_{variant}" for field, variant in wide.columns]
    wide = wide.reset_index()
    work = response.merge(wide, on="transcriptid", how="left", validate="one_to_one")
    required = [
        "raw_answer_tone_A0",
        "raw_answer_tone_A3",
        "expected_answer_tone_A0",
        "expected_answer_tone_A3",
        "answer_residual_A0",
        "answer_residual_A3",
    ]
    if work[required].isna().any().any():
        raise ValueError("Development decomposition is missing frozen answer components")
    raw_variant_error = float(
        (work["raw_answer_tone_A0"] - work["raw_answer_tone_A3"]).abs().max()
    )
    frozen_a3_error = float(
        (work["answer_residual_A3"] - work["answer_innovation"]).abs().max()
    )
    work["answer_D03"] = (
        work["answer_residual_A0"] - work["answer_residual_A3"]
    )
    expected_difference = (
        work["expected_answer_tone_A3"] - work["expected_answer_tone_A0"]
    )
    d03_identity_error = float((work["answer_D03"] - expected_difference).abs().max())
    maximum_identity_error = max(
        raw_variant_error, frozen_a3_error, d03_identity_error
    )
    if maximum_identity_error > COMPONENT_IDENTITY_TOLERANCE:
        raise ValueError(
            "Development answer decomposition fails a frozen identity: "
            f"max error={maximum_identity_error:.3e}"
        )

    raw = work["raw_answer_tone_A3"].astype(float)
    common_sd = float(raw.std(ddof=1))
    if not np.isfinite(common_sd) or common_sd <= 0:
        raise ValueError("Raw answer tone has no usable development-sample scale")
    component_columns = {
        "raw_answer_tone_common_z": "raw_answer_tone_A3",
        "expected_A3_common_z": "expected_answer_tone_A3",
        "residual_A3_common_z": "answer_residual_A3",
        "D03_common_z": "answer_D03",
    }
    for standardized, source in component_columns.items():
        values = work[source].astype(float)
        work[standardized] = (values - values.mean()) / common_sd

    base = ["prepared_innovation_z", "question_innovation_z"]
    specifications = {
        "A3_own_residual_sd": [*base, "answer_innovation_z"],
        "raw_tone_common_sd": [*base, "raw_answer_tone_common_z"],
        "expected_A3_common_sd": [*base, "expected_A3_common_z"],
        "residual_A3_common_sd": [*base, "residual_A3_common_z"],
        "expected_plus_residual_common_sd": [
            *base,
            "expected_A3_common_z",
            "residual_A3_common_z",
        ],
        "U3_plus_D03_common_sd": [*base, "residual_A3_common_z", "D03_common_z"],
    }
    focal_terms = {
        "A3_own_residual_sd": ["answer_innovation_z"],
        "raw_tone_common_sd": ["raw_answer_tone_common_z"],
        "expected_A3_common_sd": ["expected_A3_common_z"],
        "residual_A3_common_sd": ["residual_A3_common_z"],
        "expected_plus_residual_common_sd": [
            "expected_A3_common_z",
            "residual_A3_common_z",
        ],
        "U3_plus_D03_common_sd": ["residual_A3_common_z", "D03_common_z"],
    }
    rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    for label, treatments in specifications.items():
        system = fit_joint_horizon_system(
            work,
            outcomes=["CAR_0_1"],
            treatments=treatments,
            controls=response_controls(work),
        )
        fitted[label] = system
        coefficients = system.coefficients.set_index("treatment")
        for term in focal_terms[label]:
            row = coefficients.loc[term]
            rows.append(
                {
                    "specification": label,
                    "term": term,
                    "beta": float(row["beta"]),
                    "standard_error": float(row["standard_error"]),
                    "p_value": float(row["p_value"]),
                    "beta_bps": 10_000 * float(row["beta"]),
                    "standard_error_bps": 10_000 * float(row["standard_error"]),
                    "ci_low_bps": 10_000
                    * (float(row["beta"]) - 1.96 * float(row["standard_error"])),
                    "ci_high_bps": 10_000
                    * (float(row["beta"]) + 1.96 * float(row["standard_error"])),
                    "n_obs": int(row["n_obs"]),
                    "common_raw_answer_tone_sd": common_sd,
                    "uncertainty": "conditional firm/date two-way clustered",
                }
            )

    joint = fitted["expected_plus_residual_common_sd"]
    index = joint.coefficient_index.set_index("treatment")
    expected_position = int(index.loc["expected_A3_common_z", "position"])
    residual_position = int(index.loc["residual_A3_common_z", "position"])
    coefficients = joint.coefficients.set_index("treatment")
    difference = float(
        coefficients.loc["residual_A3_common_z", "beta"]
        - coefficients.loc["expected_A3_common_z", "beta"]
    )
    variance = float(
        joint.covariance[residual_position, residual_position]
        + joint.covariance[expected_position, expected_position]
        - 2.0 * joint.covariance[residual_position, expected_position]
    )
    difference_se = float(np.sqrt(max(variance, 0.0)))
    rows.append(
        {
            "specification": "expected_plus_residual_common_sd",
            "term": "residual_minus_expected",
            "beta": difference,
            "standard_error": difference_se,
            "p_value": normal_two_sided_p(difference / difference_se),
            "beta_bps": 10_000 * difference,
            "standard_error_bps": 10_000 * difference_se,
            "ci_low_bps": 10_000 * (difference - 1.96 * difference_se),
            "ci_high_bps": 10_000 * (difference + 1.96 * difference_se),
            "n_obs": int(joint.n_obs),
            "common_raw_answer_tone_sd": common_sd,
            "uncertainty": "conditional firm/date two-way clustered Wald contrast",
        }
    )
    identities = pd.DataFrame(
        [
            {
                "check": "raw_tone_invariant_A0_A3",
                "max_abs_error": raw_variant_error,
            },
            {
                "check": "aggregated_A3_matches_frozen_response",
                "max_abs_error": frozen_a3_error,
            },
            {
                "check": "D03_equals_expected_A3_minus_expected_A0",
                "max_abs_error": d03_identity_error,
            },
        ]
    )
    identities["tolerance"] = COMPONENT_IDENTITY_TOLERANCE
    identities["n_calls"] = len(work)
    identities["common_raw_answer_tone_sd"] = common_sd
    return pd.DataFrame(rows), identities


def pre_event_precision_summary(
    coefficients: pd.DataFrame,
    bands: pd.DataFrame,
    joint_tests: pd.DataFrame,
    *,
    window_family: str,
) -> pd.DataFrame:
    """Summarize what a simultaneous pre-event family can and cannot exclude."""

    rows = []
    for treatment, group in coefficients.groupby("treatment", sort=False):
        family_bands = bands.loc[bands["treatment"].eq(treatment)].copy()
        joint = joint_tests.loc[joint_tests["treatment"].eq(treatment)]
        if len(family_bands) != len(group) or len(joint) != 1:
            raise ValueError(f"Pre-event inputs do not align for {window_family}/{treatment}")
        allowed = np.maximum(
            family_bands["simultaneous_ci_low"].abs(),
            family_bands["simultaneous_ci_high"].abs(),
        )
        rows.append(
            {
                "window_family": window_family,
                "treatment": treatment,
                "family_size": int(len(group)),
                "joint_wald_chi2": float(joint.iloc[0]["wald_chi2"]),
                "joint_df": int(joint.iloc[0]["df"]),
                "joint_p_value": float(joint.iloc[0]["p_value"]),
                "max_abs_point_estimate": float(group["beta"].abs().max()),
                "max_abs_effect_allowed_by_simultaneous_ci": float(allowed.max()),
                "max_abs_point_estimate_bps": float(10_000 * group["beta"].abs().max()),
                "max_abs_effect_allowed_by_simultaneous_ci_bps": float(
                    10_000 * allowed.max()
                ),
                "interpretation_constraint": (
                    "precision_bound_only; non-rejection is not evidence of a zero effect"
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_convergence(
    draws: pd.DataFrame,
    coefficient_index: pd.DataFrame,
    conditional_coefficients: pd.DataFrame,
    conditional_covariance: np.ndarray,
    *,
    checkpoints: tuple[int, ...] = BOOTSTRAP_CHECKPOINTS,
    primary_outcome: str = "CAR_0_1",
    primary_treatment: str = "answer_innovation_z",
) -> pd.DataFrame:
    """Recompute adjusted primary uncertainty from nested replicate-ID subsets."""

    if "position" not in conditional_coefficients:
        conditional_coefficients = coefficient_index.merge(
            conditional_coefficients,
            on=["outcome", "treatment"],
            how="left",
            validate="one_to_one",
        )
    _assert_index_alignment(coefficient_index, conditional_coefficients)
    positions = coefficient_index.sort_values("position")["position"].astype(int).to_list()
    replicate_ids = sorted(pd.to_numeric(draws["replicate"], errors="raise").unique().tolist())
    primary = coefficient_index.loc[
        coefficient_index["outcome"].eq(primary_outcome)
        & coefficient_index["treatment"].eq(primary_treatment)
    ]
    if len(primary) != 1:
        raise ValueError("Primary bootstrap coefficient is not uniquely indexed")
    primary_position = int(primary.iloc[0]["position"])
    primary_offset = positions.index(primary_position)
    conditional_row = conditional_coefficients.loc[
        conditional_coefficients["outcome"].eq(primary_outcome)
        & conditional_coefficients["treatment"].eq(primary_treatment)
    ]
    if len(conditional_row) != 1:
        raise ValueError("Conditional primary coefficient is not unique")
    beta = float(conditional_row.iloc[0]["beta"])
    conditional_variance = float(conditional_covariance[primary_offset, primary_offset])
    rows = []
    for draw_count in checkpoints:
        if len(replicate_ids) < draw_count:
            raise ValueError(
                f"Bootstrap convergence needs {draw_count} draws but found {len(replicate_ids)}"
            )
        selected_ids = replicate_ids[:draw_count]
        matrix = (
            draws.loc[draws["replicate"].isin(selected_ids)]
            .pivot(index="replicate", columns="position", values="beta")
            .reindex(index=selected_ids, columns=positions)
            .to_numpy(dtype=float)
        )
        if matrix.shape != (draw_count, len(positions)) or not np.isfinite(matrix).all():
            raise ValueError(f"Bootstrap matrix is incomplete at {draw_count} draws")
        first_stage_covariance = np.cov(matrix, rowvar=False, ddof=1)
        contribution = float(first_stage_covariance[primary_offset, primary_offset])
        adjusted_se = float(np.sqrt(max(conditional_variance + contribution, 0.0)))
        rows.append(
            {
                "draws": int(draw_count),
                "replicate_id_min": int(selected_ids[0]),
                "replicate_id_max": int(selected_ids[-1]),
                "primary_outcome": primary_outcome,
                "primary_treatment": primary_treatment,
                "beta": beta,
                "conditional_standard_error": float(np.sqrt(max(conditional_variance, 0.0))),
                "first_stage_variance_contribution": contribution,
                "first_stage_standard_error_contribution": float(
                    np.sqrt(max(contribution, 0.0))
                ),
                "bootstrap_adjusted_standard_error": adjusted_se,
                "bootstrap_adjusted_ci_low": beta - 1.96 * adjusted_se,
                "bootstrap_adjusted_ci_high": beta + 1.96 * adjusted_se,
                "beta_bps": 10_000 * beta,
                "bootstrap_adjusted_standard_error_bps": 10_000 * adjusted_se,
                "bootstrap_adjusted_ci_low_bps": 10_000 * (beta - 1.96 * adjusted_se),
                "bootstrap_adjusted_ci_high_bps": 10_000 * (beta + 1.96 * adjusted_se),
            }
        )
    result = pd.DataFrame(rows)
    reference = result.loc[result["draws"].eq(max(checkpoints))].iloc[0]
    result["adjusted_se_difference_from_200"] = (
        result["bootstrap_adjusted_standard_error"]
        - float(reference["bootstrap_adjusted_standard_error"])
    )
    result["ci_low_difference_from_200"] = (
        result["bootstrap_adjusted_ci_low"] - float(reference["bootstrap_adjusted_ci_low"])
    )
    result["ci_high_difference_from_200"] = (
        result["bootstrap_adjusted_ci_high"] - float(reference["bootstrap_adjusted_ci_high"])
    )
    return result


def build_first_stage_fold_summary(
    fold_metrics: pd.DataFrame, hyperparameter_selection: pd.DataFrame
) -> pd.DataFrame:
    """Attach the selected Ridge alpha and boundary flag to each outer fold."""

    fold_metrics = fold_metrics.loc[
        fold_metrics["fold_year"].astype(str).str.fullmatch(r"\d{4}")
    ].copy()
    fold_metrics["fold_year"] = pd.to_numeric(
        fold_metrics["fold_year"], errors="raise"
    ).astype(int)
    selected = hyperparameter_selection.loc[
        hyperparameter_selection["selected"].map(_as_bool)
    ].copy()
    if selected.empty:
        raise ValueError("No selected tuned-Ridge hyperparameters were found")
    selected["selected_alpha"] = selected["candidate_json"].map(_read_ridge_alpha)
    keys = ["component", "variant", "grain", "outer_year"]
    if selected.duplicated(keys).any():
        raise ValueError("Selected tuned-Ridge hyperparameters are not unique by outer fold")
    selected = selected[
        [
            *keys,
            "selected_alpha",
            "selected_at_grid_boundary",
            "n_inner_folds",
            "max_inner_train_year",
        ]
    ].rename(columns={"outer_year": "fold_year"})
    summary = fold_metrics.merge(
        selected,
        on=["component", "variant", "grain", "fold_year"],
        how="left",
        validate="one_to_one",
    )
    if summary["selected_alpha"].isna().any():
        raise ValueError("At least one first-stage fold lacks a selected Ridge alpha")
    summary["selected_at_grid_boundary"] = summary["selected_at_grid_boundary"].map(_as_bool)
    columns = [
        "component",
        "variant",
        "grain",
        "fold_year",
        "status",
        "n_train",
        "n_test",
        "r2",
        "mae",
        "predicted_actual_corr",
        "calibration_slope",
        "selected_alpha",
        "selected_at_grid_boundary",
        "n_inner_folds",
        "max_inner_train_year",
    ]
    return summary[columns].sort_values(["component", "variant", "fold_year"])


def run_supplement_audits(
    *,
    frozen_dir: Path,
    tuned_run_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run all deterministic local audits and freeze them with file checksums."""

    frozen_dir = frozen_dir.resolve()
    tuned_run_dir = tuned_run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Supplement audit output directory is not empty: {output_dir}")
    required_sources = _required_source_paths(frozen_dir, tuned_run_dir)
    for path in required_sources.values():
        if not path.exists():
            raise FileNotFoundError(path)

    response = pd.read_parquet(required_sources["response_sample"])
    turn_residuals = pd.read_parquet(required_sources["answer_turn_residuals"])
    frozen_coefficients = pd.read_csv(required_sources["response_coefficients"])
    aggregation = estimate_aggregation_robustness(
        response, turn_residuals, frozen_coefficients
    )
    context_pairing = context_answer_pairing_diagnostic(turn_residuals)
    answer_decomposition, answer_component_identities = (
        development_answer_decomposition(response, turn_residuals)
    )

    daily_coefficients = pd.read_csv(required_sources["daily_pre_event_coefficients"])
    daily_bands = pd.read_csv(required_sources["daily_pre_event_bands"])
    daily_tests = pd.read_csv(required_sources["daily_pre_event_joint_tests"])
    five_day_coefficients = pd.read_csv(required_sources["five_day_coefficients"])
    five_day_index = pd.read_csv(required_sources["five_day_index"])
    five_day_covariance = np.load(required_sources["five_day_covariance"])
    five_day_bands = simultaneous_bands(
        five_day_coefficients,
        five_day_covariance,
        five_day_index,
        seed=20260716,
    )
    five_day_tests = pd.read_csv(required_sources["five_day_joint_tests"])
    pre_event_precision = pd.concat(
        [
            pre_event_precision_summary(
                daily_coefficients,
                daily_bands,
                daily_tests,
                window_family="daily_AR_m20_m1",
            ),
            pre_event_precision_summary(
                five_day_coefficients,
                five_day_bands,
                five_day_tests,
                window_family="nonoverlapping_five_day_bins",
            ),
        ],
        ignore_index=True,
    )

    draws = pd.read_csv(required_sources["bootstrap_draws"])
    coefficient_index = pd.read_csv(required_sources["bootstrap_index"])
    conditional_covariance = np.load(required_sources["conditional_covariance"])
    convergence = bootstrap_convergence(
        draws,
        coefficient_index,
        frozen_coefficients,
        conditional_covariance,
    )
    fold_summary = build_first_stage_fold_summary(
        pd.read_csv(required_sources["fold_metrics"]),
        pd.read_csv(required_sources["hyperparameter_selection"]),
    )
    boundary_summary = (
        fold_summary.groupby(["component", "variant"], as_index=False)
        .agg(
            folds=("fold_year", "count"),
            boundary_selections=("selected_at_grid_boundary", "sum"),
            min_selected_alpha=("selected_alpha", "min"),
            max_selected_alpha=("selected_alpha", "max"),
        )
        .sort_values(["component", "variant"])
    )
    inventory = _diagnostic_inventory(frozen_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "aggregation_robustness.csv": aggregation,
        "context_answer_pairing.csv": context_pairing,
        "development_answer_decomposition.csv": answer_decomposition,
        "development_answer_component_identities.csv": answer_component_identities,
        "pre_event_precision.csv": pre_event_precision,
        "five_day_pre_event_simultaneous_bands.csv": five_day_bands,
        "bootstrap_convergence.csv": convergence,
        "first_stage_fold_summary.csv": fold_summary,
        "ridge_grid_boundary_summary.csv": boundary_summary,
        "existing_diagnostic_inventory.csv": inventory,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    primary = aggregation.loc[
        aggregation["aggregation"].eq("word_weighted_mean")
        & aggregation["horizon"].eq(1)
    ].iloc[0]
    manifest = {
        "status": "complete",
        "contract": "revision-v2-supplement-audits-v1",
        "derivation_only": True,
        "frozen_source_modified": False,
        "frozen_source": str(frozen_dir),
        "tuned_turn_residual_source": str(tuned_run_dir),
        "response_calls": int(len(response)),
        "aggregation_methods": list(AGGREGATION_METHODS),
        "bootstrap_checkpoints": list(BOOTSTRAP_CHECKPOINTS),
        "pre_event_band_seed": 20260716,
        "primary_word_weighted_beta_bps": float(primary["beta_bps"]),
        "primary_parity_target_bps": PRIMARY_BETA_BPS,
        "context_pairing_multi_turn_calls": int(
            context_pairing["n_calls"].max()
        ),
        "answer_component_identity_tolerance": COMPONENT_IDENTITY_TOLERANCE,
        "source_files": [
            _file_record(path, root=PROJECT_ROOT)
            for path in required_sources.values()
        ],
        "output_files": [
            _file_record(output_dir / filename, root=output_dir)
            for filename in outputs
        ],
        "deferred": [
            "independent human annotation",
            "new external LLM calls",
            "additional learner comparisons",
        ],
    }
    (output_dir / "supplement_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _word_weighted_residual(group: pd.DataFrame) -> float:
    weights = group["word_count"].clip(lower=0).fillna(0).astype(float)
    if float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=group.index)
    return float(np.average(group["narrative_innovation"].astype(float), weights=weights))


def _assert_aggregation_parity(
    aggregation: pd.DataFrame, frozen_coefficients: pd.DataFrame
) -> None:
    recalculated = aggregation.loc[
        aggregation["aggregation"].eq("word_weighted_mean")
        & aggregation["horizon"].eq(1)
    ]
    frozen = frozen_coefficients.loc[
        frozen_coefficients["outcome"].eq("CAR_0_1")
        & frozen_coefficients["treatment"].eq("answer_innovation_z")
    ]
    if len(recalculated) != 1 or len(frozen) != 1:
        raise ValueError("Aggregation parity primary result is not uniquely defined")
    recalculated_beta = float(recalculated.iloc[0]["beta"])
    frozen_beta = float(frozen.iloc[0]["beta"])
    if not np.isclose(recalculated_beta, frozen_beta, rtol=0.0, atol=1e-12):
        raise ValueError(
            "Word-weighted aggregation does not recover the frozen A3 one-day coefficient: "
            f"{10_000 * recalculated_beta:.10f} != {10_000 * frozen_beta:.10f} bps"
        )
    if not np.isclose(10_000 * recalculated_beta, PRIMARY_BETA_BPS, rtol=0.0, atol=1e-7):
        raise ValueError("Frozen A3 one-day coefficient no longer matches the declared parity target")


def _assert_index_alignment(
    coefficient_index: pd.DataFrame, conditional_coefficients: pd.DataFrame
) -> None:
    expected = coefficient_index.sort_values("position")[
        ["position", "outcome", "treatment"]
    ].reset_index(drop=True)
    observed = conditional_coefficients.sort_values("position")[
        ["position", "outcome", "treatment"]
    ].reset_index(drop=True)
    if not expected.equals(observed):
        raise ValueError("Conditional covariance and bootstrap coefficient index do not align")


def _read_ridge_alpha(candidate_json: str) -> float:
    payload = json.loads(candidate_json)
    if payload.get("model_type") != "ridge":
        raise ValueError(f"Selected first-stage candidate is not Ridge: {payload}")
    return float(payload["params"]["alpha"])


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _required_source_paths(frozen_dir: Path, tuned_run_dir: Path) -> dict[str, Path]:
    event = frozen_dir / "event_audit"
    bootstrap = frozen_dir / "bootstrap"
    return {
        "response_sample": frozen_dir / "response_model_sample_revision_v2.parquet",
        "answer_turn_residuals": tuned_run_dir / "answer_ladder_turn_residuals.parquet",
        "response_coefficients": frozen_dir / "response_coefficients_conditional.csv",
        "conditional_covariance": frozen_dir / "response_joint_covariance_conditional.npy",
        "fold_metrics": frozen_dir / "tuned_ridge_fold_metrics.csv",
        "hyperparameter_selection": frozen_dir / "tuned_ridge_hyperparameter_selection.csv",
        "daily_pre_event_coefficients": (
            event / "daily_pre_event_coefficients_conditional.csv"
        ),
        "daily_pre_event_bands": (
            event / "daily_pre_event_simultaneous_bands_conditional.csv"
        ),
        "daily_pre_event_joint_tests": (
            event / "daily_pre_event_joint_tests_conditional.csv"
        ),
        "five_day_coefficients": (
            event / "five_day_pre_event_coefficients_conditional.csv"
        ),
        "five_day_covariance": (
            event / "five_day_pre_event_covariance_conditional.npy"
        ),
        "five_day_index": event / "five_day_pre_event_coefficient_index.csv",
        "five_day_joint_tests": (
            event / "five_day_pre_event_joint_tests_conditional.csv"
        ),
        "bootstrap_draws": bootstrap / "bootstrap_draws.csv",
        "bootstrap_index": bootstrap / "bootstrap_coefficient_index.csv",
    }


def _diagnostic_inventory(frozen_dir: Path) -> pd.DataFrame:
    entries = [
        ("sample_attrition", "sample_accounting.csv"),
        ("speaker_coverage", "speaker_coverage.csv"),
        ("nonoverlapping_post_event_increments", "incremental_responses.csv"),
        ("exclude_2020_and_year_interactions", "answer_year_stability.csv"),
        ("content_proxies", "content_proxy_joint_responses.csv"),
        ("turn_filters", "answer_measurement_robustness.csv"),
        ("learner_sensitivity", "learner_audit/a3_learner_primary_comparison.csv"),
        ("multiplicity", "response_simultaneous_bands_conditional.csv"),
    ]
    rows = []
    for topic, relative in entries:
        path = frozen_dir / relative
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "topic": topic,
                "source_file": relative,
                "rows": int(len(pd.read_csv(path))),
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(path.resolve())
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic supplement audits from the frozen revision-v2 results."
    )
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--tuned-run-dir", type=Path, default=DEFAULT_TUNED_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_supplement_audits(
        frozen_dir=args.frozen_dir,
        tuned_run_dir=args.tuned_run_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "response_calls": manifest["response_calls"],
                "primary_word_weighted_beta_bps": manifest[
                    "primary_word_weighted_beta_bps"
                ],
                "output": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
