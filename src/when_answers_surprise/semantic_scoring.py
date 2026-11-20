from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LM_FILENAME = "Loughran-McDonald_MasterDictionary_1993-2025.csv"
# Digest of the study CSV with CRLF normalized to LF.
LM_SHA256 = "941bfcf097cdcff42d5a1a8bae6e5e39bb07386aaf64939fe6a63a00ee33153a"

LM_CATEGORIES = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
    "complexity",
]

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z']+")
SENTENCE_PATTERN = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class LMDictionary:
    category_words: dict[str, set[str]]
    source_path: Path | None
    backend: str


def load_lm_dictionary(external_dir: Path) -> LMDictionary:
    """Load the complete study dictionary without downloading or substituting words."""
    path = external_dir / LM_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Place the official 1993-2025 LM master CSV at {path}.")
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != LM_SHA256:
        raise ValueError(
            f"Incomplete or different LM dictionary: {path}. "
            f"Expected the original 1993-2025 CSV (LF SHA-256: {LM_SHA256})."
        )
    frame = pd.read_csv(io.BytesIO(data))
    frame.columns = frame.columns.str.strip().str.lower().str.replace(" ", "_")
    words = frame["word"].astype(str).str.upper().str.strip()
    categories = {k: set(words[frame[k].gt(0)]) for k in LM_CATEGORIES}
    return LMDictionary(categories, path, "loughran_mcdonald_1993_2025")


def score_lm_text(text: str, dictionary: LMDictionary) -> dict[str, float | int | str]:
    tokens = [token.upper().strip("'") for token in TOKEN_PATTERN.findall(str(text))]
    word_count = len(tokens)
    token_set = dictionary.category_words
    counts = {category: 0 for category in LM_CATEGORIES}
    if word_count:
        for token in tokens:
            for category, words in token_set.items():
                if token in words:
                    counts[category] += 1

    scale = word_count / 1000 if word_count else np.nan
    rates = {
        f"lm_{category}_per_1k": (counts[category] / scale if word_count else 0.0)
        for category in LM_CATEGORIES
    }
    sentence_count = max(len(SENTENCE_PATTERN.findall(str(text))), 1)
    avg_word_length = float(np.mean([len(token) for token in tokens])) if tokens else 0.0
    positive = rates["lm_positive_per_1k"]
    negative = rates["lm_negative_per_1k"]

    return {
        "word_count": int(word_count),
        "sentence_count": int(sentence_count),
        "avg_word_length": avg_word_length,
        "lm_net_tone": float(positive - negative),
        **{f"lm_{category}_count": int(counts[category]) for category in LM_CATEGORIES},
        **rates,
        "scoring_backend": dictionary.backend,
    }


def semantic_feature_columns() -> list[str]:
    return [
        "lm_net_tone",
        *(f"lm_{category}_per_1k" for category in LM_CATEGORIES),
        "word_count",
        "sentence_count",
        "avg_word_length",
        "text_length",
    ]


def scores_to_long(features: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "accession_number", "scoring_backend"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"semantic feature table missing columns: {sorted(missing)}")

    factors = ["lm_net_tone", *(f"lm_{category}_per_1k" for category in LM_CATEGORIES)]
    rows = []
    id_columns = [
        "ticker",
        "accession_number",
        "event_date_trading",
        "sample_strict",
        "sample_moderate",
        "sample_broad",
        "scoring_backend",
        "text_length",
        "word_count",
    ]
    available_ids = [column for column in id_columns if column in features.columns]
    for factor in factors:
        if factor not in features.columns:
            continue
        out = features[available_ids].copy()
        out["semantic_factor"] = factor
        out["raw_score"] = features[factor].astype(float)
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score_events(events: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible synthetic scorer.

    The real-data pipeline uses LM scoring. This small wrapper lets the old
    synthetic notebook path continue to run if someone imports it directly.
    """

    required = {"ticker", "accession_number", "release_text", "text_length"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")

    dictionary = LMDictionary(
        category_words=_compact_lm_fallback(),
        source_path=None,
        backend="lm_compact_fallback_for_synthetic",
    )
    rows = []
    for event in events.itertuples(index=False):
        scores = score_lm_text(event.release_text, dictionary)
        for factor in ["lm_net_tone", "lm_uncertainty_per_1k", "lm_litigious_per_1k"]:
            rows.append(
                {
                    "ticker": event.ticker,
                    "accession_number": event.accession_number,
                    "semantic_factor": factor,
                    "raw_score": scores[factor],
                    "scoring_backend": scores["scoring_backend"],
                    "text_length": event.text_length,
                }
            )
    return pd.DataFrame(rows)


def _compact_lm_fallback() -> dict[str, set[str]]:
    return {
        "negative": {
            "ADVERSE",
            "DECLINE",
            "DECREASE",
            "LOSS",
            "LOSSES",
            "RISK",
            "RISKS",
            "WEAK",
            "WEAKER",
            "UNCERTAIN",
            "IMPAIRMENT",
            "CHARGE",
            "CHARGES",
            "LITIGATION",
            "DEFAULT",
        },
        "positive": {
            "BENEFIT",
            "BENEFITS",
            "FAVORABLE",
            "GROWTH",
            "IMPROVED",
            "IMPROVEMENT",
            "POSITIVE",
            "PROFIT",
            "PROFITS",
            "STRONG",
            "STRENGTH",
            "SUCCESS",
            "RECORD",
        },
        "uncertainty": {"UNCERTAIN", "UNCERTAINTY", "VOLATILE", "VOLATILITY", "CONTINGENT"},
        "litigious": {"CLAIM", "CLAIMS", "LITIGATION", "LAWSUIT", "LEGAL", "SETTLEMENT"},
        "strong_modal": {"ALWAYS", "BEST", "CLEARLY", "DEFINITELY", "STRONGLY"},
        "weak_modal": {"COULD", "MAY", "MIGHT", "POSSIBLE", "POSSIBLY"},
        "constraining": {"COVENANT", "COVENANTS", "RESTRICT", "RESTRICTED", "RESTRICTION"},
        "complexity": {"COMPLEX", "COMPLEXITY", "MULTIFACETED", "INTERDEPENDENT"},
    }
