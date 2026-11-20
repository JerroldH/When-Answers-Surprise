"""Text scoring and dialogue-history feature helpers."""

from when_answers_surprise.empirical.pipeline import (
    add_prior_turn_history,
    base_categorical_features,
    base_state_features,
    build_qa_turns,
    pivot_feature_scores,
    score_lm_sections,
    score_operator_placebo,
    score_qa_turns,
)

__all__ = [
    "add_prior_turn_history",
    "base_categorical_features",
    "base_state_features",
    "build_qa_turns",
    "pivot_feature_scores",
    "score_lm_sections",
    "score_operator_placebo",
    "score_qa_turns",
]

