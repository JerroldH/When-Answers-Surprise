"""Canonical first-stage feature specifications for the empirical pipeline.

The ordered lists in this module are the data contract shared by the baseline,
learner audit, sequential validation, and cached-LLM analysis.  A declared
feature is required at the point where a model frame is built; it is never
silently removed because a merge or a stale cache omitted it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import pandas as pd


BASE_STATE_FEATURES = (
    "scaled_eps_surprise",
    "eps_surprise",
    "forecast_dispersion",
    "analyst_coverage",
    "forecast_age_days",
    "match_date_gap_days",
    "pre_ret_5",
    "pre_ret_20",
    "pre_ret_60",
    "pre_vol_20",
    "pre_vol_60",
    "pre_log_volume_20",
    "pre_log_dollar_volume_20",
    "pre_amihud_illiquidity_20",
    "pre_market_ret_20",
    "market_cap",
    "call_after_close",
    "firm_events_prior",
    "prepared_lm_net_tone_lag1",
    "prepared_lm_net_tone_mean4",
    "prepared_lm_negative_per_1k_lag1",
    "prepared_lm_negative_per_1k_mean4",
    "prepared_lm_uncertainty_per_1k_lag1",
    "prepared_lm_uncertainty_per_1k_mean4",
    "answer_lm_net_tone_lag1",
    "answer_lm_net_tone_mean4",
    "ticker",
    "match_quality",
    "calendar_quarter",
)

TURN_POSITION_FEATURES = ("turn_index",)
PREPARED_CONTEXT_FEATURES = (
    "prepared_lm_net_tone",
    "prepared_lm_negative_per_1k",
    "prepared_lm_uncertainty_per_1k",
    "prepared_word_count",
)
PRIOR_DIALOGUE_FEATURES = (
    "prior_answer_lm_net_tone_mean",
    "prior_answer_word_count_mean",
    "prior_question_lm_net_tone_mean",
    "prior_question_word_count_mean",
)
CURRENT_QUESTION_FEATURES = (
    "question_lm_net_tone",
    "question_lm_negative_per_1k",
    "question_lm_uncertainty_per_1k",
    "current_question_word_count",
)

ANALYST_CONTEXT_FEATURES = (
    "analyst_key",
    "analyst_brokerage",
    "analyst_prior_question_count",
    "analyst_prior_question_tone_mean",
    "analyst_firm_prior_question_count",
    "analyst_firm_prior_question_tone_mean",
    "analyst_days_since_last_question",
)

CURRENT_QUESTION_CONTENT_FEATURES = (
    "current_question_numeric_density",
    "current_question_quantitative_disclosure",
    "current_question_guidance_density",
    "current_question_topic_guidance",
    "current_question_topic_margin_pressure",
    "current_question_topic_demand_strength",
    "current_question_topic_cost_pressure",
    "current_question_topic_pricing",
    "current_question_topic_capital_allocation",
)

REVISION_CATEGORICAL_FEATURES = (
    "analyst_key",
    "analyst_brokerage",
)

FEATURE_SET_LABELS = {
    "A0": "pre-call state",
    "A1": "pre-call state plus prepared/prior dialogue",
    "A2": "pre-call state plus prepared/prior dialogue plus current question",
}


def require_columns(frame: pd.DataFrame, names: Sequence[str], *, label: str) -> list[str]:
    """Return declared columns or fail with the exact missing names."""

    ordered = list(dict.fromkeys(names))
    missing = [name for name in ordered if name not in frame.columns]
    if missing:
        raise ValueError(f"{label} requires declared columns missing from frame: {missing}")
    return ordered


def reject_merge_suffix_features(names: Sequence[str], *, label: str = "model features") -> None:
    """Reject pandas merge suffixes before they can become model inputs."""

    invalid = [name for name in names if str(name).endswith(("_x", "_y"))]
    if invalid:
        raise ValueError(f"{label} contains merge-suffix columns that are not valid features: {invalid}")


def base_state_feature_spec(frame: pd.DataFrame | None = None, *, label: str = "base state") -> list[str]:
    names = list(BASE_STATE_FEATURES)
    if frame is not None:
        require_columns(frame, names, label=label)
    return names


def question_feature_spec(
    frame: pd.DataFrame | None = None, *, label: str = "question first-stage features"
) -> list[str]:
    names = [
        *BASE_STATE_FEATURES,
        *TURN_POSITION_FEATURES,
        *PREPARED_CONTEXT_FEATURES,
        *PRIOR_DIALOGUE_FEATURES,
    ]
    return _resolve(names, frame, label)


def answer_filtration_features(
    frame: pd.DataFrame | None = None, *, label_prefix: str = "answer filtration"
) -> dict[str, list[str]]:
    """Return the canonical A0/A1/A2 ladder in stable order."""

    a0 = list(BASE_STATE_FEATURES)
    a1 = [*a0, *TURN_POSITION_FEATURES, *PREPARED_CONTEXT_FEATURES, *PRIOR_DIALOGUE_FEATURES]
    a2 = [*a1, *CURRENT_QUESTION_FEATURES]
    result = {"A0": a0, "A1": a1, "A2": a2}
    if frame is not None:
        for variant, names in result.items():
            _resolve(names, frame, f"{label_prefix} {variant}")
    return result


def question_filtration_features(
    frame: pd.DataFrame | None = None, *, label_prefix: str = "question filtration"
) -> dict[str, list[str]]:
    """Return the revision-v2 Q0/Q1/Q2 ladder in stable order.

    Q2 uses only information known before the question text is produced. Current
    question text and topic features are deliberately excluded.
    """

    q0 = list(BASE_STATE_FEATURES)
    q1 = [*q0, *TURN_POSITION_FEATURES, *PREPARED_CONTEXT_FEATURES, *PRIOR_DIALOGUE_FEATURES]
    q2 = [*q1, *ANALYST_CONTEXT_FEATURES]
    result = {"Q0": q0, "Q1": q1, "Q2": q2}
    if frame is not None:
        for variant, names in result.items():
            _resolve(names, frame, f"{label_prefix} {variant}")
    return result


def revision_answer_filtration_features(
    frame: pd.DataFrame | None = None, *, label_prefix: str = "revision answer filtration"
) -> dict[str, list[str]]:
    """Return A0/A1/A2 plus the prespecified revision-v2 A3 benchmark."""

    legacy = answer_filtration_features()
    a3 = [
        *legacy["A2"],
        *ANALYST_CONTEXT_FEATURES,
        *CURRENT_QUESTION_CONTENT_FEATURES,
    ]
    result = {**legacy, "A3": a3}
    if frame is not None:
        for variant, names in result.items():
            _resolve(names, frame, f"{label_prefix} {variant}")
    return result


def revision_feature_manifests(frame: pd.DataFrame | None = None) -> dict[str, object]:
    """Return the versioned projection-residual feature contract."""

    question = question_filtration_features(frame)
    answer = revision_answer_filtration_features(frame)
    return {
        "contract_version": "revision-v2",
        "estimand_type": "projection_residual",
        "question": {name: feature_manifest(columns) for name, columns in question.items()},
        "answer": {name: feature_manifest(columns) for name, columns in answer.items()},
    }


def revision_feature_catalog() -> pd.DataFrame:
    """Build the paper-facing feature map from the canonical feature groups."""

    groups = [
        ("pre_call_state", BASE_STATE_FEATURES, "before_call", "numeric/categorical pipeline"),
        ("turn_position", TURN_POSITION_FEATURES, "before_current_text", "numeric"),
        ("prepared_context", PREPARED_CONTEXT_FEATURES, "before_qa", "numeric"),
        ("prior_dialogue", PRIOR_DIALOGUE_FEATURES, "before_current_turn", "strict prior-turn mean"),
        ("current_question_lm", CURRENT_QUESTION_FEATURES, "before_answer_only", "numeric"),
        (
            "analyst_context",
            ANALYST_CONTEXT_FEATURES,
            "before_question_and_answer",
            "strict prior-call history",
        ),
        (
            "current_question_content",
            CURRENT_QUESTION_CONTENT_FEATURES,
            "before_answer_only",
            "frozen lexicon/regex proxy",
        ),
    ]
    rows = []
    categorical = set(REVISION_CATEGORICAL_FEATURES) | {
        "ticker",
        "match_quality",
        "calendar_quarter",
    }
    for group, features, availability, transformation in groups:
        for feature in features:
            rows.append(
                {
                    "feature": feature,
                    "feature_group": group,
                    "availability": availability,
                    "transformation": transformation,
                    "categorical": feature in categorical,
                    "missing_handling": (
                        "explicit UNKNOWN" if feature in categorical else "training-fold mean"
                    ),
                }
            )
    return pd.DataFrame(rows)


def feature_manifest(names: Sequence[str]) -> dict[str, object]:
    """Return the ordered feature list, count, and stable SHA-256 digest."""

    ordered = list(dict.fromkeys(names))
    reject_merge_suffix_features(ordered)
    encoded = json.dumps(ordered, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {
        "feature_columns": ordered,
        "n_features": len(ordered),
        "feature_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def all_feature_manifests(frame: pd.DataFrame | None = None) -> dict[str, dict[str, object]]:
    """Resolve the shared question and answer specifications for a manifest."""

    answer = answer_filtration_features(frame)
    specs = {
        "question": question_feature_spec(frame),
        **answer,
    }
    return {name: feature_manifest(columns) for name, columns in specs.items()}


def assert_feature_manifest_closure(
    baseline_manifest: dict[str, object], cache_manifest: dict[str, object]
) -> None:
    """Assert that baseline and learner/cache manifests resolve to one contract."""

    left = baseline_manifest.get("specifications", baseline_manifest.get("canonical_feature_manifests"))
    right = cache_manifest.get("canonical_feature_manifests", cache_manifest.get("specifications"))
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("Feature manifest closure requires specifications in both manifests")
    for name in ("question", "A0", "A1", "A2"):
        left_columns = left.get(name, {}).get("feature_columns")
        right_columns = right.get(name, {}).get("feature_columns")
        if left_columns != right_columns:
            raise ValueError(
                f"Feature manifest closure failed for {name}: baseline={left_columns}, cache={right_columns}"
            )


def _resolve(names: Sequence[str], frame: pd.DataFrame | None, label: str) -> list[str]:
    ordered = list(dict.fromkeys(names))
    reject_merge_suffix_features(ordered, label=label)
    if frame is not None:
        require_columns(frame, ordered, label=label)
    return ordered
