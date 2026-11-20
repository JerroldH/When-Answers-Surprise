"""Prespecified speaker, history, and content features for revision v2.

These transformations use transcript text and speaker metadata only. They never
read market outcomes. Analyst histories exclude the current call and every call
on the same calendar date because intraday speaker ordering is unavailable.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd


UNKNOWN_ANALYST = "UNKNOWN"
MULTI_ANALYST = "MULTI"
UNKNOWN_BROKERAGE = "UNKNOWN"

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[+-]?(?:[$€£])?\d[\d,]*(?:\.\d+)?(?:%|x)?(?![A-Za-z])",
    flags=re.IGNORECASE,
)
QUANTITATIVE_CONTEXT_PATTERN = re.compile(
    r"[$€£]|%|\bbasis\s+points?\b|\bbps\b|\bmillion\b|\bbillion\b|"
    r"\b\d[\d,.]*\s*(?:-|to)\s*\d[\d,.]*\b",
    flags=re.IGNORECASE,
)
GUIDANCE_TERMS = ("guidance", "outlook", "expect", "forecast", "raise", "lower", "range")
TOPIC_TERMS = {
    "guidance": GUIDANCE_TERMS,
    "margin_pressure": (
        "margin pressure",
        "gross margin",
        "operating margin",
        "compression",
        "headwind",
    ),
    "demand_strength": (
        "strong demand",
        "demand remains",
        "backlog",
        "orders",
        "pipeline",
        "momentum",
    ),
    "cost_pressure": (
        "cost pressure",
        "inflation",
        "freight",
        "labor cost",
        "input cost",
        "tariff",
    ),
    "pricing": ("pricing", "price increase", "price realization", "discount", "mix"),
    "capital_allocation": (
        "buyback",
        "dividend",
        "capex",
        "capital allocation",
        "free cash flow",
    ),
}
ADMIN_PHRASES = (
    "thank you for participating",
    "thank you for your attendance",
    "no further questions",
    "conclude the presentation",
    "this concludes",
    "line just had cut out",
    "line cut out",
    "can you repeat",
    "can you just start again",
    "could you repeat",
    "missed the first part",
    "technical difficulty",
)
TOKEN_PATTERN = re.compile(r"\b[\w]+(?:['-][\w]+)*\b", flags=re.UNICODE)


def _clean_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", " ", text)


def stable_analyst_key(proid: object, person_name: object, brokerage: object) -> str:
    """Return a stable internal key without exposing fallback speaker names."""

    if proid is not None and not pd.isna(proid):
        numeric = pd.to_numeric(pd.Series([proid]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return f"PROID:{int(numeric)}" if float(numeric).is_integer() else f"PROID:{numeric}"
    name = _clean_identifier(person_name)
    company = _clean_identifier(brokerage)
    if not name:
        return UNKNOWN_ANALYST
    digest = hashlib.sha256(f"{name}|{company}".encode()).hexdigest()[:20]
    return f"HASH:{digest}"


def resolve_turn_category(values: Iterable[object], *, unknown: str, multiple: str) -> str:
    clean = {str(value).strip() for value in values if value is not None and str(value).strip()}
    clean.discard(unknown)
    if not clean:
        return unknown
    if len(clean) > 1:
        return multiple
    return next(iter(clean))


def text_content_proxies(text: object) -> dict[str, float | int]:
    """Return frozen, outcome-independent content and call-mechanics proxies."""

    value = "" if text is None or pd.isna(text) else str(text)
    lower = value.lower()
    tokens = TOKEN_PATTERN.findall(lower)
    denominator = max(len(tokens), 1)
    numbers = NUMBER_PATTERN.findall(value)
    guidance_count = sum(
        len(re.findall(rf"\b{re.escape(term)}\w*\b", lower)) for term in GUIDANCE_TERMS
    )
    result: dict[str, float | int] = {
        "numeric_density": 100.0 * len(numbers) / denominator,
        "quantitative_disclosure": int(
            bool(numbers) and bool(QUANTITATIVE_CONTEXT_PATTERN.search(value))
        ),
        "guidance_density": 100.0 * guidance_count / denominator,
        "administrative_mechanics": int(any(phrase in lower for phrase in ADMIN_PHRASES)),
    }
    for topic, terms in TOPIC_TERMS.items():
        result[f"topic_{topic}"] = int(any(term in lower for term in terms))
    return result


def add_turn_text_proxies(turns: pd.DataFrame) -> pd.DataFrame:
    """Add question and answer proxy columns without changing turn keys."""

    out = turns.copy()
    for component in ("question", "answer"):
        text_col = f"{component}_text"
        if text_col not in out:
            raise ValueError(f"Turn frame is missing {text_col}")
        proxy = pd.DataFrame(out[text_col].map(text_content_proxies).tolist(), index=out.index)
        proxy = proxy.rename(columns={name: f"{component}_{name}" for name in proxy.columns})
        out = pd.concat([out, proxy], axis=1)
    return out


def add_strict_analyst_history(turns: pd.DataFrame, question_scores: pd.DataFrame) -> pd.DataFrame:
    """Attach histories built only from dates strictly before the current call."""

    required_turn = {"transcriptid", "turn_index", "ticker", "call_date", "analyst_key"}
    required_score = {"transcriptid", "turn_index", "raw_score"}
    if missing := sorted(required_turn.difference(turns.columns)):
        raise ValueError(f"Analyst history turns missing columns: {missing}")
    if missing := sorted(required_score.difference(question_scores.columns)):
        raise ValueError(f"Analyst history question scores missing columns: {missing}")

    key = ["transcriptid", "turn_index"]
    score = question_scores[key + ["raw_score"]].copy()
    score["transcriptid"] = score["transcriptid"].astype(str)
    if score.duplicated(key).any():
        raise ValueError("Question score frame must be unique by transcriptid, turn_index")
    data = turns.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    data = data.merge(score, on=key, how="left", validate="one_to_one")
    data["call_date"] = pd.to_datetime(data["call_date"], errors="coerce").dt.normalize()
    data["analyst_key"] = data["analyst_key"].fillna(UNKNOWN_ANALYST).astype(str)
    data["ticker"] = data["ticker"].fillna("").astype(str)

    valid = ~data["analyst_key"].isin([UNKNOWN_ANALYST, MULTI_ANALYST]) & data[
        "call_date"
    ].notna()
    initial = {
        "analyst_prior_question_count": 0.0,
        "analyst_prior_question_tone_mean": np.nan,
        "analyst_firm_prior_question_count": 0.0,
        "analyst_firm_prior_question_tone_mean": np.nan,
        "analyst_days_since_last_question": np.nan,
    }
    for column, value in initial.items():
        data[column] = value

    if valid.any():
        daily = (
            data.loc[valid]
            .groupby(["analyst_key", "call_date"], as_index=False)
            .agg(day_count=("raw_score", "count"), day_sum=("raw_score", "sum"))
            .sort_values(["analyst_key", "call_date"])
        )
        grouped = daily.groupby("analyst_key", sort=False)
        daily["analyst_prior_question_count"] = grouped["day_count"].cumsum() - daily[
            "day_count"
        ]
        prior_sum = grouped["day_sum"].cumsum() - daily["day_sum"]
        daily["analyst_prior_question_tone_mean"] = prior_sum.div(
            daily["analyst_prior_question_count"].replace(0, np.nan)
        )
        daily["analyst_days_since_last_question"] = (
            daily["call_date"] - grouped["call_date"].shift(1)
        ).dt.days
        data = _replace_history_columns(
            data,
            daily,
            ["analyst_key", "call_date"],
            [
                "analyst_prior_question_count",
                "analyst_prior_question_tone_mean",
                "analyst_days_since_last_question",
            ],
        )

        firm_daily = (
            data.loc[valid]
            .groupby(["analyst_key", "ticker", "call_date"], as_index=False)
            .agg(day_count=("raw_score", "count"), day_sum=("raw_score", "sum"))
            .sort_values(["analyst_key", "ticker", "call_date"])
        )
        firm_group = firm_daily.groupby(["analyst_key", "ticker"], sort=False)
        firm_daily["analyst_firm_prior_question_count"] = (
            firm_group["day_count"].cumsum() - firm_daily["day_count"]
        )
        firm_prior_sum = firm_group["day_sum"].cumsum() - firm_daily["day_sum"]
        firm_daily["analyst_firm_prior_question_tone_mean"] = firm_prior_sum.div(
            firm_daily["analyst_firm_prior_question_count"].replace(0, np.nan)
        )
        data = _replace_history_columns(
            data,
            firm_daily,
            ["analyst_key", "ticker", "call_date"],
            ["analyst_firm_prior_question_count", "analyst_firm_prior_question_tone_mean"],
        )

    if data.duplicated(key).any():
        raise ValueError("Analyst history merge changed turn uniqueness")
    count_columns = [
        "analyst_prior_question_count",
        "analyst_firm_prior_question_count",
    ]
    data[count_columns] = data[count_columns].fillna(0.0)
    return data.drop(columns="raw_score")


def _replace_history_columns(
    data: pd.DataFrame, history: pd.DataFrame, keys: list[str], columns: list[str]
) -> pd.DataFrame:
    return data.drop(columns=columns).merge(
        history[keys + columns], on=keys, how="left", validate="many_to_one"
    )
