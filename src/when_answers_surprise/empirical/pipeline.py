"""S&P 500 earnings-call LM tone case-study implementation.

This module intentionally keeps the submission pipeline's stable import path.
It is not part of the small public 8-K demo API, and edits here should preserve
the frozen empirical design unless an empirical rerun is explicitly approved.
"""

from __future__ import annotations

import json
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from when_answers_surprise.semantic_scoring import LM_CATEGORIES, LMDictionary, load_lm_dictionary, score_lm_text
from when_answers_surprise.wrds_sp500 import read_partitioned_parquet
from when_answers_surprise.empirical.feature_spec import (
    BASE_STATE_FEATURES,
    all_feature_manifests,
    answer_filtration_features,
    base_state_feature_spec,
    feature_manifest,
    question_feature_spec,
    reject_merge_suffix_features,
    require_columns,
)
from when_answers_surprise.empirical.revision_v2_features import (
    MULTI_ANALYST,
    UNKNOWN_ANALYST,
    UNKNOWN_BROKERAGE,
    add_turn_text_proxies,
    resolve_turn_category,
    stable_analyst_key,
)


EARNINGS_CALL_TYPE = "Earnings Calls"
HORIZONS = (1, 3, 5, 10, 20)
OOF_TEST_YEARS = (2020, 2021, 2022, 2023, 2024)
TIMING_MIN_SAMPLE_ROWS = 120
TIMING_MIN_FIRMS = 30
TIMING_MIN_EVENT_DATES = 30
SECTION_TEXT_COLUMNS = {
    "prepared": "prepared_text",
    "question": "question_text",
    "answer": "answer_text",
    "qa": "qa_text",
    "full": "full_text",
}
PRIMARY_FEATURE = "lm_net_tone"
LM_FEATURES = [
    "lm_net_tone",
    "lm_positive_per_1k",
    "lm_negative_per_1k",
    "lm_uncertainty_per_1k",
    "lm_litigious_per_1k",
    "lm_weak_modal_per_1k",
    "lm_strong_modal_per_1k",
    "lm_constraining_per_1k",
    "word_count",
]
SECONDARY_FEATURES = [
    "lm_negative_per_1k",
    "lm_uncertainty_per_1k",
    "lm_weak_modal_per_1k",
    "word_count",
]
SAFE_HARBOR_PATTERN = re.compile(
    r"\bsafe\s+harbou?r\b|\bforward[-\s]+looking\s+statements?\b|"
    r"\brisk[s]?\s+and\s+uncertainties\b|\bnon[-\s]+gaap\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LMBaselineConfig:
    project_root: Path
    data_root: Path
    output_dir: Path
    figures_dir: Path
    sample_end: str = "2024-12-31"
    min_train_rows: int = 120
    max_events_per_year: int | None = None
    run_model_robustness: bool = True
    timing_path: Path | None = None
    panel_path: Path | None = None


@dataclass(frozen=True)
class RegressionResult:
    coefficients: pd.DataFrame
    covariance: np.ndarray
    coefficient_names: list[str]
    n_obs: int
    n_firms: int
    n_event_dates: int


class StageLogger:
    """Tiny stdout/json stage logger for long local pipeline runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, object]] = []

    @contextmanager
    def stage(self, name: str):
        record: dict[str, object] = {"stage": name, "status": "running"}
        started = time.perf_counter()
        handle = StageHandle(record)
        print(f"[lm-tone] start {name}", flush=True)
        try:
            yield handle
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
                f"[lm-tone] {record['status']} {name} {record['seconds']}s",
                flush=True,
            )
            self.write()

    def progress(self, message: str) -> None:
        print(f"[lm-tone] {message}", flush=True)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


class StageHandle:
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record

    def observe(self, name: str, table: pd.DataFrame | pd.Series | None) -> None:
        observations = list(self.record.get("observations", []))
        if table is None:
            observations.append({"name": name, "rows": 0, "columns": 0, "deep_memory_bytes": 0})
        elif isinstance(table, pd.Series):
            observations.append(
                {
                    "name": name,
                    "rows": int(len(table)),
                    "columns": 1,
                    "deep_memory_bytes": int(table.memory_usage(deep=True)),
                }
            )
        else:
            observations.append(
                {
                    "name": name,
                    "rows": int(len(table)),
                    "columns": int(len(table.columns)),
                    "deep_memory_bytes": int(table.memory_usage(deep=True).sum()),
                }
            )
        self.record["observations"] = observations


def run_lm_pipeline(config: LMBaselineConfig) -> dict[str, object]:
    """Run the WRDS S&P 500 LM tone baseline and write analysis outputs."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    timing_path = config.timing_path or config.output_dir / "timing.json"
    logger = StageLogger(timing_path)

    with logger.stage("read_inputs") as stage:
        dictionary = load_lm_dictionary(config.project_root / "data" / "external")
        panel_path = config.panel_path or config.data_root / "derived" / "transcript_ibes_panel.parquet"
        panel = pd.read_parquet(panel_path)
        components = pd.read_parquet(config.data_root / "staging" / "ciq_transcript_components.parquet")
        membership = pd.read_parquet(config.data_root / "staging" / "universe_sp500_membership.parquet")
        security_master = pd.read_parquet(config.data_root / "staging" / "security_master.parquet")
        crsp_daily = read_partitioned_parquet(config.data_root / "raw" / "wrds_exports" / "crsp_dsf_sp500")
        for name, table in [
            ("panel", panel),
            ("components", components),
            ("membership", membership),
            ("security_master", security_master),
            ("crsp_daily", crsp_daily),
        ]:
            stage.observe(name, table)
        stage.record["panel_path"] = str(panel_path)

    with logger.stage("filter_event_sample") as stage:
        earnings = filter_earnings_calls(panel, sample_end=config.sample_end)
        point_in_time = apply_point_in_time_sp500_filter(earnings, membership, security_master)
        point_in_time = balanced_year_sample(point_in_time, config.max_events_per_year)
        event_ids = set(point_in_time["transcriptid"].astype(str))
        components = components.loc[components["transcriptid"].astype(str).isin(event_ids)].copy()
        stage.observe("earnings", earnings)
        stage.observe("point_in_time", point_in_time)
        stage.observe("components_filtered", components)

    with logger.stage("score_sections") as stage:
        section_scores = score_lm_sections(point_in_time, dictionary, progress=logger.progress)
        operator_scores = score_operator_placebo(components, point_in_time, dictionary)
        stage.observe("section_scores", section_scores)
        stage.observe("operator_scores", operator_scores)

    with logger.stage("build_turns") as stage:
        turns = build_qa_turns(components, point_in_time)
        stage.observe("turns", turns)

    with logger.stage("score_turns") as stage:
        turn_scores = score_qa_turns(turns, dictionary, progress=logger.progress)
        stage.observe("turn_scores", turn_scores)

    with logger.stage("build_outcomes_state") as stage:
        outcomes = build_crsp_event_outcomes(point_in_time, crsp_daily, horizons=HORIZONS)
        state = build_event_state(point_in_time, section_scores, outcomes)
        stage.observe("outcomes", outcomes)
        stage.observe("state", state)

    with logger.stage("first_stage_innovations") as stage:
        prepared_innovations, prepared_diag = construct_prepared_innovations(
            state, section_scores, config, progress=logger.progress
        )
        question_innovations, question_diag = construct_question_innovations(
            state, turn_scores, section_scores, config, progress=logger.progress
        )
        answer_innovations, answer_diag = construct_answer_innovations(
            state, turn_scores, section_scores, config, progress=logger.progress
        )
        turn_innovations = pd.concat([question_innovations, answer_innovations], ignore_index=True)
        call_innovations = pd.concat(
            [
                prepared_innovations,
                aggregate_turn_innovations(question_innovations, "question"),
                aggregate_turn_innovations(answer_innovations, "answer"),
            ],
            ignore_index=True,
        )
        for name, table in [
            ("prepared_innovations", prepared_innovations),
            ("question_innovations", question_innovations),
            ("answer_innovations", answer_innovations),
            ("turn_innovations", turn_innovations),
            ("call_innovations", call_innovations),
        ]:
            stage.observe(name, table)

    with logger.stage("first_stage_diagnostics") as stage:
        first_stage = pd.concat([prepared_diag, question_diag, answer_diag], ignore_index=True)
        by_year = first_stage_by_year(call_innovations, turn_innovations)
        residual_time = residual_time_summary(call_innovations)
        residual_predictability = residual_predictability_checks(call_innovations, state)
        model_comparison = (
            first_stage_model_comparison(state, section_scores, config)
            if config.run_model_robustness
            else pd.DataFrame()
        )
        for name, table in [
            ("first_stage", first_stage),
            ("by_year", by_year),
            ("residual_time", residual_time),
            ("residual_predictability", residual_predictability),
            ("model_comparison", model_comparison),
        ]:
            stage.observe(name, table)

    with logger.stage("response_regressions") as stage:
        response_sample = build_response_sample(state, call_innovations)
        response_coefficients, coefficient_tests = estimate_all_response_curves(response_sample)
        timing_subsample_responses = estimate_timing_subsample_responses(response_sample)
        heterogeneity, heterogeneity_percentiles = estimate_heterogeneity(response_sample)
        validation_placebos = estimate_placebo_responses(response_sample)
        validation_orthogonality = innovation_orthogonality_checks(response_sample)
        validation_parsing = parsing_quality_validation(response_sample)
        attrition = sample_attrition(
            panel,
            earnings,
            point_in_time,
            outcomes,
            response_sample,
            call_innovations=call_innovations,
        )
        for name, table in [
            ("response_sample", response_sample),
            ("response_coefficients", response_coefficients),
            ("coefficient_tests", coefficient_tests),
            ("heterogeneity", heterogeneity),
            ("validation_placebos", validation_placebos),
            ("timing_subsample_responses", timing_subsample_responses),
        ]:
            stage.observe(name, table)

    with logger.stage("write_outputs") as stage:
        write_lm_outputs(
            config=config,
            attrition=attrition,
            section_scores=section_scores,
            operator_scores=operator_scores,
            turns=turns,
            turn_innovations=turn_innovations,
            call_innovations=call_innovations,
            first_stage=first_stage,
            by_year=by_year,
            residual_time=residual_time,
            residual_predictability=residual_predictability,
            model_comparison=model_comparison,
            outcomes=outcomes,
            response_sample=response_sample,
            response_coefficients=response_coefficients,
            coefficient_tests=coefficient_tests,
            heterogeneity=heterogeneity,
            heterogeneity_percentiles=heterogeneity_percentiles,
            validation_placebos=validation_placebos,
            validation_orthogonality=validation_orthogonality,
            validation_parsing=validation_parsing,
            timing_subsample_responses=timing_subsample_responses,
        )
        stage.observe("response_sample", response_sample)

    with logger.stage("figure") as stage:
        figure_main_response(response_coefficients, config.figures_dir / "sp500_lm_main_response.png")
        stage.observe("response_coefficients", response_coefficients)

    summary = {
        "dictionary_backend": dictionary.backend,
        "sample_end": config.sample_end,
        "max_events_per_year": config.max_events_per_year,
        "run_model_robustness": config.run_model_robustness,
        "panel_path": str(config.panel_path) if config.panel_path else None,
        **{row["metric"]: row["value"] for row in attrition.to_dict("records")},
    }
    (config.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.write()
    return summary


def filter_earnings_calls(panel: pd.DataFrame, *, sample_end: str = "2024-12-31") -> pd.DataFrame:
    data = panel.copy()
    data["transcriptid"] = data["transcriptid"].astype(str)
    data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()
    data["call_date"] = pd.to_datetime(
        data.get("call_date", data["mostimportantdateutc"]), errors="coerce"
    ).dt.normalize()
    mask = data["keydeveventtypename"].eq(EARNINGS_CALL_TYPE) & data["call_date"].le(
        pd.Timestamp(sample_end)
    )
    return data.loc[mask].sort_values(["call_date", "ticker", "transcriptid"]).reset_index(drop=True)


def select_transcript_version_rows(events: pd.DataFrame, *, version: str) -> pd.DataFrame:
    """Select first or latest transcript version within each Capital IQ key development event."""

    if version not in {"first", "latest"}:
        raise ValueError("version must be 'first' or 'latest'.")
    data = events.copy()
    data["keydevid"] = data["keydevid"].astype(str)
    data["transcriptid"] = data["transcriptid"].astype(str)
    data["_creation_date"] = pd.to_datetime(
        data.get("transcriptcreationdate_utc"), errors="coerce"
    )
    if "transcriptcreationtime_utc" in data:
        data["_creation_time"] = data["transcriptcreationtime_utc"].astype("string").fillna("")
    else:
        data["_creation_time"] = ""
    data["_transcriptid_numeric"] = pd.to_numeric(data["transcriptid"], errors="coerce")
    ordered = data.sort_values(
        ["keydevid", "_creation_date", "_creation_time", "_transcriptid_numeric", "transcriptid"],
        na_position="last",
        kind="stable",
    )
    keep = "first" if version == "first" else "last"
    selected = ordered.drop_duplicates("keydevid", keep=keep)
    return selected.drop(columns=["_creation_date", "_creation_time", "_transcriptid_numeric"])


def apply_point_in_time_sp500_filter(
    events: pd.DataFrame, membership: pd.DataFrame, security_master: pd.DataFrame
) -> pd.DataFrame:
    """Keep calls whose ticker/company link was valid on the call date."""

    events = events.copy()
    events["call_date"] = pd.to_datetime(events["call_date"], errors="coerce").dt.normalize()
    events["ticker"] = events["ticker"].astype("string").str.strip().str.upper()
    events["companyid_norm"] = events.get("companyid", "").astype("string").str.replace(r"\.0$", "", regex=True)

    members = membership.copy()
    members["permno"] = pd.to_numeric(members["permno"], errors="coerce").astype("Int64")
    members["mbrstartdt"] = pd.to_datetime(members["mbrstartdt"], errors="coerce").dt.normalize()
    members["mbrenddt"] = pd.to_datetime(members["mbrenddt"], errors="coerce").dt.normalize()

    master = security_master.copy()
    master["permno"] = pd.to_numeric(master["permno"], errors="coerce").astype("Int64")
    ticker_col = "ticker_upper" if "ticker_upper" in master else "ticker"
    master["ticker_match"] = master[ticker_col].astype("string").str.strip().str.upper()
    master["namedt"] = pd.to_datetime(master.get("namedt"), errors="coerce").dt.normalize()
    master["nameendt"] = pd.to_datetime(master.get("nameendt"), errors="coerce").dt.normalize()
    if "ciq_companyid" in master:
        master["ciq_companyid_norm"] = (
            master["ciq_companyid"].astype("string").str.replace(r"\.0$", "", regex=True)
        )
    else:
        master["ciq_companyid_norm"] = pd.NA
    for column in ("ciq_startdate", "ciq_enddate"):
        if column in master:
            master[column] = pd.to_datetime(master[column], errors="coerce").dt.normalize()

    spell_cols = [
        "permno",
        "ticker_match",
        "namedt",
        "nameendt",
        "gvkey",
        "ibes_ticker",
        "ciq_companyid_norm",
        "ciq_startdate",
        "ciq_enddate",
        "exchcd",
        "siccd",
    ]
    spells = master[[col for col in spell_cols if col in master.columns]].drop_duplicates()
    if "ibes_ticker" in spells:
        spells = spells.rename(columns={"ibes_ticker": "ibes_ticker_security_master"})
    spells = spells.merge(
        members[["permno", "mbrstartdt", "mbrenddt"]].drop_duplicates(),
        on="permno",
        how="inner",
    )

    identity_overlaps = [
        column
        for column in ("permno", "gvkey", "exchcd", "siccd")
        if column in events.columns and column in spells.columns
    ]
    candidates = events.merge(
        spells,
        left_on="ticker",
        right_on="ticker_match",
        how="left",
        suffixes=("_event", ""),
    )
    if "ibes_ticker" in candidates and "ibes_ticker_security_master" in candidates:
        candidates["ibes_ticker"] = candidates["ibes_ticker"].fillna(
            candidates["ibes_ticker_security_master"]
        )
    call = candidates["call_date"]
    name_start = candidates["namedt"].fillna(pd.Timestamp("1900-01-01"))
    name_end = candidates["nameendt"].fillna(pd.Timestamp("2262-04-11"))
    member_start = candidates["mbrstartdt"].fillna(pd.Timestamp("1900-01-01"))
    member_end = candidates["mbrenddt"].fillna(pd.Timestamp("2262-04-11"))
    ciq_start = (
        candidates["ciq_startdate"]
        if "ciq_startdate" in candidates
        else pd.Series(pd.NaT, index=candidates.index)
    ).fillna(pd.Timestamp("1900-01-01"))
    ciq_end = (
        candidates["ciq_enddate"]
        if "ciq_enddate" in candidates
        else pd.Series(pd.NaT, index=candidates.index)
    ).fillna(pd.Timestamp("2262-04-11"))
    in_window = (
        call.ge(name_start)
        & call.le(name_end)
        & call.ge(member_start)
        & call.le(member_end)
        & call.ge(ciq_start)
        & call.le(ciq_end)
    )
    companyid_match = candidates["companyid_norm"].eq(candidates["ciq_companyid_norm"])
    candidates = candidates.loc[
        in_window
        & candidates["permno"].notna()
        & candidates["ciq_companyid_norm"].notna()
        & companyid_match
    ].copy()
    if candidates.empty:
        return candidates
    for column in identity_overlaps:
        event_column = f"{column}_event"
        comparable = candidates[event_column].notna() & candidates[column].notna()
        if comparable.any() and not candidates.loc[comparable, event_column].astype(
            "string"
        ).eq(candidates.loc[comparable, column].astype("string")).all():
            raise ValueError(
                f"Event and point-in-time security master disagree on {column}"
            )

    candidates["_name_start_rank"] = candidates["namedt"].fillna(pd.Timestamp("1900-01-01"))
    candidates = candidates.sort_values(
        ["transcriptid", "_name_start_rank", "permno"],
        ascending=[True, False, True],
        kind="stable",
    )
    deduped = candidates.drop_duplicates("transcriptid", keep="first").copy()
    deduped["sp500_point_in_time"] = True
    drop_cols = [
        "companyid_norm",
        "_name_start_rank",
        *[f"{column}_event" for column in identity_overlaps],
    ]
    return deduped.drop(columns=[col for col in drop_cols if col in deduped]).reset_index(drop=True)


def balanced_year_sample(events: pd.DataFrame, max_events_per_year: int | None) -> pd.DataFrame:
    if max_events_per_year is None:
        return events.reset_index(drop=True)
    data = events.copy()
    data["event_year"] = pd.to_datetime(data["call_date"], errors="coerce").dt.year
    sampled = (
        data.sort_values(["event_year", "ticker", "call_date", "transcriptid"])
        .groupby("event_year", group_keys=False)
        .head(max_events_per_year)
    )
    return sampled.drop(columns=["event_year"]).reset_index(drop=True)


def score_lm_sections(
    events: pd.DataFrame,
    dictionary: LMDictionary,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, event in enumerate(events.itertuples(index=False), start=1):
        if progress and (idx == 1 or idx % 1000 == 0 or idx == len(events)):
            progress(f"score_sections {idx}/{len(events)}")
        transcriptid = str(event.transcriptid)
        base = {
            "transcriptid": transcriptid,
            "ticker": str(event.ticker),
            "permno": int(event.permno) if pd.notna(event.permno) else np.nan,
            "call_date": pd.Timestamp(event.call_date),
        }
        prepared = _score_text(getattr(event, "prepared_text", ""), dictionary)
        question = _score_text(getattr(event, "question_text", ""), dictionary)
        answer = _score_text(getattr(event, "answer_text", ""), dictionary)
        qa = _combine_scores([question, answer], dictionary)
        full = _combine_scores([prepared, question, answer], dictionary)
        for section, score in {
            "prepared": prepared,
            "question": question,
            "answer": answer,
            "qa": qa,
            "full": full,
        }.items():
            rows.append(
                {
                    **base,
                    "section": section,
                    "has_safe_harbor": bool(
                        SAFE_HARBOR_PATTERN.search(str(getattr(event, SECTION_TEXT_COLUMNS.get(section, "full_text"), "")))
                    ),
                    **score,
                }
            )
    return pd.DataFrame(rows)


def score_operator_placebo(
    components: pd.DataFrame, events: pd.DataFrame, dictionary: LMDictionary
) -> pd.DataFrame:
    data = _normalize_components(components)
    operator = data.loc[data["componenttypename"].str.contains("Operator Message", na=False)].copy()
    if operator.empty:
        return pd.DataFrame(columns=["transcriptid", "section", *LM_FEATURES])
    text = (
        operator.groupby("transcriptid", sort=True)["componenttext"]
        .apply(lambda values: "\n\n".join(str(value) for value in values if str(value)))
        .reset_index()
    )
    meta = events[["transcriptid", "ticker", "call_date"]].copy()
    meta["transcriptid"] = meta["transcriptid"].astype(str)
    text = text.merge(meta, on="transcriptid", how="left")
    rows = []
    for row in text.itertuples(index=False):
        rows.append(
            {
                "transcriptid": str(row.transcriptid),
                "ticker": row.ticker,
                "call_date": row.call_date,
                "section": "operator_placebo",
                **_score_text(row.componenttext, dictionary),
            }
        )
    return pd.DataFrame(rows)


def build_qa_turns(components: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Collapse Capital IQ components into ordered analyst-question/manager-answer turns."""

    data = _normalize_components(components)
    meta = events[["transcriptid", "ticker", "permno", "call_date"]].copy()
    meta["transcriptid"] = meta["transcriptid"].astype(str)
    data = data.merge(meta, on="transcriptid", how="inner")
    rows: list[dict[str, object]] = []
    for transcriptid, group in data.groupby("transcriptid", sort=False):
        current_questions: list[str] = []
        current_answers: list[str] = []
        current_analyst_keys: list[str] = []
        current_brokerages: list[str] = []
        turn_start_order = np.nan
        turn_index = 0

        def flush_turn() -> None:
            nonlocal current_questions, current_answers
            nonlocal current_analyst_keys, current_brokerages
            nonlocal turn_start_order, turn_index
            if not current_questions and not current_answers:
                return
            turn_index += 1
            first = group.iloc[0]
            rows.append(
                {
                    "transcriptid": str(transcriptid),
                    "ticker": first["ticker"],
                    "permno": first["permno"],
                    "call_date": first["call_date"],
                    "turn_index": turn_index,
                    "turn_start_order": turn_start_order,
                    "question_text": "\n\n".join(current_questions),
                    "answer_text": "\n\n".join(current_answers),
                    "analyst_key": resolve_turn_category(
                        current_analyst_keys,
                        unknown=UNKNOWN_ANALYST,
                        multiple=MULTI_ANALYST,
                    ),
                    "analyst_brokerage": resolve_turn_category(
                        current_brokerages,
                        unknown=UNKNOWN_BROKERAGE,
                        multiple=MULTI_ANALYST,
                    ),
                }
            )
            current_questions = []
            current_answers = []
            current_analyst_keys = []
            current_brokerages = []
            turn_start_order = np.nan

        for component in group.itertuples(index=False):
            ctype = str(component.componenttypename or "")
            text = str(component.componenttext or "").strip()
            if ctype == "Question":
                if current_questions and current_answers:
                    flush_turn()
                if not current_questions:
                    turn_start_order = component.componentorder
                if text:
                    current_questions.append(text)
                    current_analyst_keys.append(
                        stable_analyst_key(
                            getattr(component, "proid", None),
                            getattr(component, "personname", None),
                            getattr(component, "speaker_companyname", None),
                        )
                    )
                    brokerage = getattr(component, "speaker_companyname", None)
                    current_brokerages.append(
                        str(brokerage).strip()
                        if brokerage is not None
                        and not pd.isna(brokerage)
                        and str(brokerage).strip()
                        else UNKNOWN_BROKERAGE
                    )
            elif ctype == "Answer":
                if not current_questions and current_answers:
                    flush_turn()
                if not current_questions:
                    turn_start_order = component.componentorder
                if text:
                    current_answers.append(text)
        flush_turn()
    turns = pd.DataFrame(rows)
    if turns.empty:
        return turns
    turns = turns.sort_values(["call_date", "ticker", "transcriptid", "turn_index"]).reset_index(drop=True)
    turns["question_word_count_raw"] = turns["question_text"].map(_simple_word_count)
    turns["answer_word_count_raw"] = turns["answer_text"].map(_simple_word_count)
    turns["has_question"] = turns["question_word_count_raw"].gt(0)
    turns["has_answer"] = turns["answer_word_count_raw"].gt(0)
    return add_turn_text_proxies(turns)


def score_qa_turns(
    turns: pd.DataFrame,
    dictionary: LMDictionary,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, turn in enumerate(turns.itertuples(index=False), start=1):
        if progress and (idx == 1 or idx % 25000 == 0 or idx == len(turns)):
            progress(f"score_turns {idx}/{len(turns)}")
        base = {
            "transcriptid": str(turn.transcriptid),
            "ticker": turn.ticker,
            "permno": turn.permno,
            "call_date": pd.Timestamp(turn.call_date),
            "turn_index": int(turn.turn_index),
            "turn_start_order": turn.turn_start_order,
        }
        for component, text in [("question", turn.question_text), ("answer", turn.answer_text)]:
            score = _score_text(text, dictionary)
            rows.append({**base, "component": component, **score})
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    return add_prior_turn_history(scores)


def add_prior_turn_history(turn_scores: pd.DataFrame) -> pd.DataFrame:
    data = turn_scores.sort_values(["transcriptid", "turn_index", "component"]).copy()
    wide = pivot_feature_scores(data, index_cols=["transcriptid", "turn_index"], prefix_by="component")
    wide = wide.sort_values(["transcriptid", "turn_index"])
    for feature in ["lm_net_tone", "lm_negative_per_1k", "lm_uncertainty_per_1k", "word_count"]:
        for component in ["question", "answer"]:
            col = f"{component}_{feature}"
            if col not in wide:
                wide[col] = np.nan
            history_col = f"prior_{component}_{feature}_mean"
            wide[history_col] = wide.groupby("transcriptid")[col].transform(
                lambda s: s.shift(1).expanding(min_periods=1).mean()
            )
    keep = [
        "transcriptid",
        "turn_index",
        *[col for col in wide.columns if col.startswith("prior_")],
    ]
    return data.merge(wide[keep], on=["transcriptid", "turn_index"], how="left")


def build_crsp_event_outcomes(
    events: pd.DataFrame,
    crsp_daily: pd.DataFrame,
    *,
    horizons: Sequence[int],
    include_post_event: bool = True,
) -> pd.DataFrame:
    daily = crsp_daily.copy()
    daily["permno"] = pd.to_numeric(daily["permno"], errors="coerce").astype("Int64")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    for col in ["ret", "vwretd", "vol", "prc", "shrout"]:
        if col in daily:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.dropna(subset=["permno", "date", "ret", "vwretd"]).sort_values(["permno", "date"])
    market_days = pd.DatetimeIndex(sorted(daily["date"].dropna().unique()))

    rows: list[dict[str, object]] = []
    events = events.copy()
    events["event_datetime_ny"] = pd.to_datetime(events["event_datetime_utc"], utc=True, errors="coerce").dt.tz_convert(
        "America/New_York"
    )
    for permno, group in events.groupby("permno", sort=True):
        permno_daily = daily.loc[daily["permno"].eq(permno)].reset_index(drop=True)
        if permno_daily.empty:
            continue
        dates = pd.DatetimeIndex(permno_daily["date"])
        ret = permno_daily["ret"].astype(float)
        market = permno_daily["vwretd"].astype(float)
        abnormal = ret - market
        log_volume = np.log1p(permno_daily["vol"].clip(lower=0))
        dollar_volume = permno_daily["prc"].abs() * permno_daily["vol"]
        log_dollar_volume = np.log1p(dollar_volume.clip(lower=0))
        amihud = ret.abs() / dollar_volume.replace(0, np.nan)
        market_cap = permno_daily["prc"].abs() * permno_daily["shrout"] * 1000.0

        for event in group.itertuples(index=False):
            event_day = align_call_to_trading_day(event.event_datetime_ny, market_days)
            if event_day is None:
                continue
            locs = np.flatnonzero(dates == event_day)
            if len(locs) == 0:
                later = np.flatnonzero(dates > event_day)
                if len(later) == 0:
                    continue
                idx = int(later[0])
                event_day = dates[idx]
            else:
                idx = int(locs[0])
            if idx < 60 or (
                include_post_event and idx + max(horizons) > len(permno_daily)
            ):
                continue
            pre20 = slice(idx - 20, idx)
            pre60 = slice(idx - 60, idx)
            row = {
                "transcriptid": str(event.transcriptid),
                "ticker": event.ticker,
                "permno": int(permno),
                "event_trading_date": event_day,
                "event_year": int(event_day.year),
                "event_calendar_quarter": f"{event_day.year}Q{math.ceil(event_day.month / 3)}",
                "pre_ret_5": float(ret.iloc[idx - 5 : idx].sum()),
                "pre_ret_20": float(ret.iloc[pre20].sum()),
                "pre_ret_60": float(ret.iloc[pre60].sum()),
                "pre_vol_20": float(ret.iloc[pre20].std()),
                "pre_vol_60": float(ret.iloc[pre60].std()),
                "pre_log_volume_20": float(log_volume.iloc[pre20].mean()),
                "pre_log_dollar_volume_20": float(log_dollar_volume.iloc[pre20].mean()),
                "pre_amihud_illiquidity_20": float(amihud.iloc[pre20].mean()),
                "pre_market_ret_20": float(market.iloc[pre20].sum()),
                "market_cap": float(market_cap.iloc[idx - 1]),
                "placebo_CAR_m5_m1": float(abnormal.iloc[idx - 5 : idx].sum()),
                "placebo_CAR_m10_m1": float(abnormal.iloc[idx - 10 : idx].sum()),
                "abnormal_volume_0_1": np.nan,
                "realized_volatility_0_1": np.nan,
            }
            if not include_post_event:
                rows.append(row)
                continue
            pre_vol_mean = log_volume.iloc[pre20].mean()
            for horizon in horizons:
                window = aligned_event_window(abnormal, idx, horizon)
                post_start = idx + 1
                post_stop = post_start + horizon
                volume_window = aligned_event_window(log_volume, idx, horizon)
                return_window = aligned_event_window(ret, idx, horizon)
                row[f"CAR_0_{horizon}"] = float(window.sum())
                row[f"postcall_CAR_1_{horizon}"] = (
                    float(aligned_event_window(abnormal, post_start, horizon).sum())
                    if post_stop <= len(abnormal)
                    else np.nan
                )
                row[f"abnormal_volume_0_{horizon}"] = float(volume_window.mean() - pre_vol_mean)
                row[f"realized_volatility_0_{horizon}"] = float(return_window.std()) if horizon > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


CAR_HORIZON_SEMANTICS = "h trading days beginning on the aligned event day"


def aligned_event_window(values: pd.Series, event_index: int, horizon: int) -> pd.Series:
    """Return exactly ``horizon`` trading rows beginning at ``event_index``."""

    if horizon <= 0:
        raise ValueError("horizon must be a positive number of trading rows")
    start = int(event_index)
    stop = start + int(horizon)
    if start < 0 or stop > len(values):
        raise IndexError(
            f"Requested trading window [{start}, {stop}) outside a series of length {len(values)}"
        )
    window = values.iloc[start:stop]
    if len(window) != horizon:
        raise AssertionError("Aligned event window did not contain the requested trading-row count")
    return window


def align_call_to_trading_day(
    event_datetime_ny: pd.Timestamp, trading_days: pd.DatetimeIndex
) -> pd.Timestamp | None:
    if pd.isna(event_datetime_ny) or trading_days.empty:
        return None
    ts = pd.Timestamp(event_datetime_ny)
    event_date = pd.Timestamp(ts.date())
    after_close = classify_call_timing(ts) == "after_close"
    if not after_close and event_date in trading_days:
        return event_date
    later = trading_days[trading_days > event_date]
    return pd.Timestamp(later[0]) if len(later) else None


def classify_call_timing(event_datetime_ny: pd.Timestamp) -> str:
    """Classify a New York call timestamp using the primary daily alignment rule."""

    ts = pd.Timestamp(event_datetime_ny)
    if pd.isna(ts):
        return "unknown"
    minute = ts.hour * 60 + ts.minute
    if minute < 9 * 60 + 30:
        return "premarket"
    if minute < 16 * 60:
        return "market_hours"
    return "after_close"


def build_event_state(
    events: pd.DataFrame, section_scores: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    base_cols = [
        "transcriptid",
        "ticker",
        "permno",
        "gvkey",
        "call_date",
        "event_datetime_utc",
        "keydevid",
        "headline",
        "match_quality",
        "actual_announcement_datetime_utc",
        "actual_known_before_call",
        "actual_rejection_reason",
        "eps_surprise",
        "scaled_eps_surprise",
        "forecast_dispersion",
        "analyst_coverage",
        "forecast_age_days",
        "match_date_gap_days",
        "prepared_word_count",
        "question_word_count",
        "answer_word_count",
        "operator_component_count",
        "component_count",
    ]
    event_base = events[[col for col in base_cols if col in events.columns]].copy()
    event_base["transcriptid"] = event_base["transcriptid"].astype(str)
    event_base["call_date"] = pd.to_datetime(
        event_base["call_date"], errors="coerce"
    ).dt.normalize()
    outcomes = outcomes.copy()
    outcomes["transcriptid"] = outcomes["transcriptid"].astype(str)
    data = event_base.merge(
        outcomes,
        on=["transcriptid", "ticker", "permno"],
        how="inner",
        validate="one_to_one",
    )
    event_ny = pd.to_datetime(
        data["event_datetime_utc"], utc=True, errors="coerce"
    ).dt.tz_convert("America/New_York")
    data["call_hour_ny"] = event_ny.dt.hour
    data["call_minute_ny"] = event_ny.dt.hour * 60 + event_ny.dt.minute
    data["call_timing_bucket"] = event_ny.map(classify_call_timing)
    data["market_hours_call"] = data["call_timing_bucket"].eq("market_hours")
    data["call_after_close"] = data["call_timing_bucket"].eq("after_close").astype(float)
    data["calendar_quarter"] = data["event_calendar_quarter"].astype(str)
    data["event_year"] = pd.to_datetime(data["event_trading_date"]).dt.year
    data["has_questions"] = data["question_word_count"].fillna(0).gt(0)
    data["has_answers"] = data["answer_word_count"].fillna(0).gt(0)
    data["high_confidence_parse"] = data["has_questions"] & data["has_answers"] & data["prepared_word_count"].fillna(0).gt(0)

    prepared = section_scores.loc[section_scores["section"].eq("prepared")].copy()
    answer = section_scores.loc[section_scores["section"].eq("answer")].copy()
    # Historical information must be built from every call known at the time,
    # not only calls that later obtain a complete CRSP outcome window.
    history_source = event_base[
        ["transcriptid", "ticker", "permno", "call_date", "event_datetime_utc"]
    ].merge(
        prepared[["transcriptid", *LM_FEATURES]].add_prefix("prepared_").rename(
            columns={"prepared_transcriptid": "transcriptid"}
        ),
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    history_source = history_source.merge(
        answer[["transcriptid", PRIMARY_FEATURE, "word_count"]]
        .rename(
            columns={
                PRIMARY_FEATURE: "answer_lm_net_tone",
                "word_count": "answer_word_count_scored",
            }
        ),
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    history_source = add_strictly_prior_call_history(
        history_source,
        value_columns=[
            "prepared_lm_net_tone",
            "prepared_lm_negative_per_1k",
            "prepared_lm_uncertainty_per_1k",
            "answer_lm_net_tone",
        ],
    )
    history_cols = [
        "transcriptid",
        "firm_events_prior",
        *[col for col in history_source.columns if col.endswith("_lag1") or col.endswith("_mean4")],
    ]
    data = data.merge(
        history_source[history_cols],
        on="transcriptid",
        how="left",
        validate="one_to_one",
    )
    return data.sort_values(["event_trading_date", "ticker", "transcriptid"]).reset_index(drop=True)


def add_strictly_prior_call_history(
    calls: pd.DataFrame, *, value_columns: Sequence[str]
) -> pd.DataFrame:
    """Add firm history using only calls with a strictly earlier timestamp."""

    data = calls.copy()
    event_time = pd.to_datetime(data["event_datetime_utc"], utc=True, errors="coerce")
    call_day_end = (
        pd.to_datetime(data["call_date"], errors="coerce")
        .dt.tz_localize("UTC")
        .add(pd.Timedelta(days=1))
        .sub(pd.Timedelta(nanoseconds=1))
    )
    data["_history_time"] = event_time.fillna(call_day_end)
    data = data.sort_values(
        ["permno", "_history_time", "transcriptid"], kind="stable"
    )
    data["firm_events_prior"] = 0
    for column in value_columns:
        data[f"{column}_lag1"] = np.nan
        data[f"{column}_mean4"] = np.nan

    for _, firm_calls in data.groupby("permno", sort=False):
        prior_values = {column: [] for column in value_columns}
        prior_count = 0
        for _, simultaneous_calls in firm_calls.groupby("_history_time", sort=False):
            indices = simultaneous_calls.index
            data.loc[indices, "firm_events_prior"] = prior_count
            for column in value_columns:
                values = prior_values[column]
                if values:
                    data.loc[indices, f"{column}_lag1"] = values[-1]
                    recent = pd.to_numeric(
                        pd.Series(values[-4:]), errors="coerce"
                    ).dropna()
                    if not recent.empty:
                        data.loc[indices, f"{column}_mean4"] = float(recent.mean())
                values.extend(simultaneous_calls[column].tolist())
            prior_count += len(simultaneous_calls)

    data["firm_events_prior"] = data["firm_events_prior"].astype(int)
    return data.drop(columns="_history_time")


def construct_prepared_innovations(
    state: pd.DataFrame,
    section_scores: pd.DataFrame,
    config: LMBaselineConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = section_scores.loc[section_scores["section"].eq("prepared")].copy()
    data = state.merge(prepared[["transcriptid", *LM_FEATURES]], on="transcriptid", how="inner")
    return construct_event_level_innovations(
        data=data,
        component="prepared",
        features=LM_FEATURES,
        feature_prefix="",
        model_features=base_state_features(data),
        categorical_features=base_categorical_features(),
        config=config,
        progress=progress,
    )


def merge_prepared_context(data: pd.DataFrame, section_scores: pd.DataFrame) -> pd.DataFrame:
    """Merge prepared tone fields while reserving event-state word count.

    The event-state count uses the staging tokenizer (letters, numbers, and
    apostrophe/hyphen words), whereas LM scoring counts alphabetic tokens for
    dictionary rates.  They are intentionally different measurements.  The
    event-state measurement is the declared model feature, so the scorer's
    same-looking ``word_count`` is dropped before the merge rather than
    creating ``prepared_word_count_x/y`` columns.
    """

    prepared = section_scores.loc[section_scores["section"].eq("prepared")][
        ["transcriptid", *LM_FEATURES]
    ].rename(columns={feature: f"prepared_{feature}" for feature in LM_FEATURES})
    if "prepared_word_count" in data.columns and "prepared_word_count" in prepared.columns:
        # Keep this diagnostic available to callers without making it a model
        # feature; equality is not required because the two tokenizers differ.
        data.attrs["prepared_score_word_count_available"] = True
    prepared = prepared.drop(columns=["prepared_word_count"])
    merged = data.merge(prepared.reset_index(drop=True), on="transcriptid", how="left", validate="many_to_one")
    reject_merge_suffix_features(merged.columns, label="prepared-context merge")
    require_columns(merged, ["prepared_word_count"], label="event-state prepared context")
    return merged


def rename_current_question_features(question_wide: pd.DataFrame) -> pd.DataFrame:
    """Give the current question count an unambiguous name before answer merges."""

    return question_wide.rename(columns={"question_word_count": "current_question_word_count"})


def construct_question_innovations(
    state: pd.DataFrame,
    turn_scores: pd.DataFrame,
    section_scores: pd.DataFrame,
    config: LMBaselineConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    question = turn_scores.loc[turn_scores["component"].eq("question")].copy()
    if question.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = question.merge(state, on=["transcriptid", "ticker", "permno", "call_date"], how="inner")
    data = merge_prepared_context(data, section_scores)
    model_features = question_feature_spec(data)
    return construct_turn_level_innovations(
        data=data,
        component="question",
        features=LM_FEATURES,
        model_features=model_features,
        categorical_features=base_categorical_features(),
        config=config,
        progress=progress,
    )


def construct_answer_innovations(
    state: pd.DataFrame,
    turn_scores: pd.DataFrame,
    section_scores: pd.DataFrame,
    config: LMBaselineConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate answer surprises relative to pre-call state, prior dialogue, and current question."""

    answer = turn_scores.loc[turn_scores["component"].eq("answer")].copy()
    question_scores = turn_scores.loc[turn_scores["component"].eq("question")].copy()
    if answer.empty or question_scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    question_wide = pivot_feature_scores(
        question_scores,
        index_cols=["transcriptid", "turn_index"],
        prefix_by="component",
    )

    data = answer.merge(state, on=["transcriptid", "ticker", "permno", "call_date"], how="inner")
    data = data.merge(
        rename_current_question_features(question_wide.reset_index()),
        on=["transcriptid", "turn_index"],
        how="left",
        validate="many_to_one",
    )
    data = merge_prepared_context(data, section_scores)
    model_features = answer_filtration_features(data, label_prefix="baseline answer filtration")["A2"]
    return construct_turn_level_innovations(
        data=data,
        component="answer",
        features=LM_FEATURES,
        model_features=model_features,
        categorical_features=base_categorical_features(),
        config=config,
        progress=progress,
    )


def construct_event_level_innovations(
    *,
    data: pd.DataFrame,
    component: str,
    features: Sequence[str],
    feature_prefix: str,
    model_features: Sequence[str],
    categorical_features: Sequence[str],
    config: LMBaselineConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    for feature in features:
        if progress:
            progress(f"first_stage {component} {feature}")
        target = f"{feature_prefix}{feature}"
        pred, fold, diag = oof_predict_by_year(
            data,
            target=target,
            feature_columns=model_features,
            categorical_columns=categorical_features,
            min_train_rows=config.min_train_rows,
            learner="ridge",
            progress=progress,
            label=f"{component}:{feature}",
        )
        out = data[["transcriptid", "ticker", "permno", "event_trading_date", "event_year"]].copy()
        out["component"] = component
        out["semantic_feature"] = feature
        out["raw_score"] = data[target].astype(float)
        out["expected_score_oof"] = pred
        out["narrative_innovation"] = out["raw_score"] - out["expected_score_oof"]
        out["fold_year"] = fold
        out["first_stage_estimable"] = np.isfinite(pred)
        rows.append(out.loc[out["first_stage_estimable"]].copy())
        diagnostics.append(
            diagnostic_row(
                out,
                component=component,
                semantic_feature=feature,
                model="ridge",
                grain="event",
                fold_diagnostics=diag,
            )
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diagnostics)


def construct_turn_level_innovations(
    *,
    data: pd.DataFrame,
    component: str,
    features: Sequence[str],
    model_features: Sequence[str],
    categorical_features: Sequence[str],
    config: LMBaselineConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    for feature in features:
        if progress:
            progress(f"first_stage {component} {feature}")
        pred, fold, diag = oof_predict_by_year(
            data,
            target=feature,
            feature_columns=model_features,
            categorical_columns=categorical_features,
            min_train_rows=config.min_train_rows,
            learner="ridge",
            progress=progress,
            label=f"{component}:{feature}",
        )
        out = data[
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
        out["component"] = component
        out["semantic_feature"] = feature
        out["raw_score"] = data[feature].astype(float)
        out["expected_score_oof"] = pred
        out["narrative_innovation"] = out["raw_score"] - out["expected_score_oof"]
        out["fold_year"] = fold
        out["first_stage_estimable"] = np.isfinite(pred)
        rows.append(out.loc[out["first_stage_estimable"]].copy())
        diagnostics.append(
            diagnostic_row(
                out,
                component=component,
                semantic_feature=feature,
                model="ridge",
                grain="turn",
                fold_diagnostics=diag,
            )
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diagnostics)


def oof_predict_by_year(
    data: pd.DataFrame,
    *,
    target: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    min_train_rows: int,
    learner: str,
    progress: Callable[[str], None] | None = None,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Predict each test year using only earlier years."""

    feature_columns = require_columns(
        data,
        feature_columns,
        label=f"{label or 'first-stage'} features",
    )
    reject_merge_suffix_features(feature_columns, label=f"{label or 'first-stage'} features")
    categorical_columns = [col for col in categorical_columns if col in feature_columns]
    values = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(data["event_year"], errors="coerce").to_numpy()
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_year = np.full(len(data), -1, dtype=int)
    diag_rows = []
    for year in OOF_TEST_YEARS:
        train_mask = (years < year) & np.isfinite(values)
        test_mask = years == year
        if train_mask.sum() < min_train_rows or test_mask.sum() == 0:
            continue
        if progress:
            prefix = f" {label}" if label else ""
            progress(
                f"first_stage_fold{prefix} year={year} train={int(train_mask.sum())} test={int(test_mask.sum())}"
            )
        model = tabular_regression_pipeline(learner, list(feature_columns), list(categorical_columns))
        model.fit(data.loc[train_mask, feature_columns], values[train_mask])
        predictions[test_mask] = model.predict(data.loc[test_mask, feature_columns])
        fold_year[test_mask] = year
        try:
            n_encoded_features = len(
                model.named_steps["preprocess"].get_feature_names_out()
            )
        except (AttributeError, KeyError):
            n_encoded_features = np.nan
        test_values = values[test_mask]
        pred_values = predictions[test_mask]
        valid = np.isfinite(test_values) & np.isfinite(pred_values)
        diag_rows.append(
            {
                "fold_year": year,
                "n_train": int(train_mask.sum()),
                "n_test": int(valid.sum()),
                "r2": float(r2_score(test_values[valid], pred_values[valid])) if valid.sum() > 2 else np.nan,
                "mae": float(mean_absolute_error(test_values[valid], pred_values[valid])) if valid.any() else np.nan,
                "predicted_actual_corr": _safe_corr(test_values[valid], pred_values[valid]),
                "calibration_slope": calibration_slope(test_values[valid], pred_values[valid]),
                "n_encoded_features": n_encoded_features,
            }
        )
    return predictions, fold_year, pd.DataFrame(diag_rows)


def tabular_regression_pipeline(
    learner: str, feature_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
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
    preprocess = ColumnTransformer(transformers, sparse_threshold=0.3)
    if learner == "ridge":
        model = Ridge(alpha=1.0, solver="lsqr")
    elif learner == "elasticnet":
        model = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=4000, random_state=0)
    elif learner == "hist_gradient_boosting":
        preprocess = ColumnTransformer(transformers, sparse_threshold=0.0)
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.02,
            random_state=0,
        )
    else:
        raise ValueError(f"Unknown first-stage learner: {learner}")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def diagnostic_row(
    innovations: pd.DataFrame,
    *,
    component: str,
    semantic_feature: str,
    model: str,
    grain: str,
    fold_diagnostics: pd.DataFrame,
) -> dict[str, object]:
    clean = innovations.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["raw_score", "expected_score_oof", "narrative_innovation"]
    )
    raw = clean["raw_score"].astype(float).to_numpy()
    pred = clean["expected_score_oof"].astype(float).to_numpy()
    resid = clean["narrative_innovation"].astype(float).to_numpy()
    return {
        "component": component,
        "semantic_feature": semantic_feature,
        "model": model,
        "grain": grain,
        "n_obs": int(len(clean)),
        "first_stage_r2": float(r2_score(raw, pred)) if len(clean) > 2 else np.nan,
        "mae": float(mean_absolute_error(raw, pred)) if len(clean) else np.nan,
        "predicted_actual_corr": _safe_corr(raw, pred),
        "calibration_slope": calibration_slope(raw, pred),
        "residual_mean": float(np.nanmean(resid)) if len(resid) else np.nan,
        "residual_variance": float(np.nanvar(resid)) if len(resid) else np.nan,
        "fold_diagnostics_json": fold_diagnostics.to_json(orient="records"),
    }


def aggregate_turn_innovations(turn_innovations: pd.DataFrame, component: str) -> pd.DataFrame:
    if turn_innovations.empty:
        return pd.DataFrame()
    data = turn_innovations.loc[turn_innovations["component"].eq(component)].copy()
    rows = []
    for (transcriptid, feature), group in data.groupby(["transcriptid", "semantic_feature"], sort=True):
        first = group.iloc[0]
        if feature == "word_count":
            raw = group["raw_score"].sum()
            expected = group["expected_score_oof"].sum()
            innovation = group["narrative_innovation"].sum()
        else:
            weights = group["word_count"].clip(lower=0).astype(float)
            if weights.sum() <= 0:
                weights = pd.Series(1.0, index=group.index)
            raw = np.average(group["raw_score"], weights=weights)
            expected = np.average(group["expected_score_oof"], weights=weights)
            innovation = np.average(group["narrative_innovation"], weights=weights)
        rows.append(
            {
                "transcriptid": transcriptid,
                "ticker": first["ticker"],
                "permno": first["permno"],
                "event_trading_date": first["event_trading_date"],
                "event_year": first["event_year"],
                "component": component,
                "semantic_feature": feature,
                "raw_score": float(raw),
                "expected_score_oof": float(expected),
                "narrative_innovation": float(innovation),
                "fold_year": first["fold_year"],
                "first_stage_estimable": True,
                "turn_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_response_sample(state: pd.DataFrame, call_innovations: pd.DataFrame) -> pd.DataFrame:
    """Create the call-level regression sample after all required outcomes and controls exist."""

    primary = call_innovations.loc[call_innovations["semantic_feature"].eq(PRIMARY_FEATURE)].copy()
    innovation_key = ["transcriptid", "component", "semantic_feature"]
    duplicate_keys = primary.duplicated(innovation_key, keep=False)
    if duplicate_keys.any():
        examples = primary.loc[duplicate_keys, innovation_key].drop_duplicates().head(5)
        raise ValueError(
            "Primary innovation data contain duplicate call-innovation keys: "
            f"{examples.to_dict('records')}"
        )
    wide = primary.pivot(
        index="transcriptid", columns="component", values="narrative_innovation"
    ).rename(
        columns={
            "prepared": "prepared_innovation",
            "question": "question_innovation",
            "answer": "answer_innovation",
        }
    )
    sample = state.merge(
        wide.reset_index(),
        on="transcriptid",
        how="inner",
        validate="one_to_one",
    )
    required_cars = [f"CAR_0_{h}" for h in HORIZONS]
    required = [
        "prepared_innovation",
        "question_innovation",
        "answer_innovation",
        *required_cars,
        "scaled_eps_surprise",
        "pre_ret_20",
        "pre_vol_20",
        "analyst_coverage",
        "forecast_dispersion",
    ]
    sample = sample.replace([np.inf, -np.inf], np.nan).copy()
    sample["log_market_cap"] = np.log(pd.to_numeric(sample["market_cap"], errors="coerce").clip(lower=1.0))
    common_required = list(dict.fromkeys([*required, *response_controls(sample)]))
    sample = sample.dropna(subset=common_required).copy()
    for col in ["prepared_innovation", "question_innovation", "answer_innovation"]:
        sample[f"{col}_z"] = zscore(sample[col])
    sample["low_analyst_coverage_z"] = zscore(
        -pd.to_numeric(sample["analyst_coverage"], errors="coerce")
    )
    sample["high_forecast_dispersion_z"] = zscore(
        pd.to_numeric(sample["forecast_dispersion"], errors="coerce")
    )
    return sample.reset_index(drop=True)


def estimate_all_response_curves(response_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tests = []
    treatments = ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]
    controls = response_controls(response_sample)
    for horizon in HORIZONS:
        outcome = f"CAR_0_{horizon}"
        result = fit_response_regression(
            response_sample,
            outcome=outcome,
            treatments=treatments,
            controls=controls,
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        coef = result.coefficients.copy()
        coef.insert(0, "horizon", horizon)
        coef.insert(1, "outcome", outcome)
        rows.append(coef)
        tests.append(coefficient_difference_test(result, horizon, "answer_innovation_z", "prepared_innovation_z"))

    diff_result = fit_response_regression(
        response_sample.assign(CAR_20_minus_1=response_sample["CAR_0_20"] - response_sample["CAR_0_1"]),
        outcome="CAR_20_minus_1",
        treatments=treatments,
        controls=controls,
        firm_fe="permno",
        quarter_fe="event_calendar_quarter",
        cluster_cols=("permno", "event_trading_date"),
    )
    tests.append(single_coefficient_test(diff_result, "answer_innovation_z", test_name="answer_20_minus_1"))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(tests)


def estimate_timing_subsample_responses(response_sample: pd.DataFrame) -> pd.DataFrame:
    """Estimate the joint response specification across call-timing subsamples."""

    if response_sample.empty:
        return pd.DataFrame()
    data = response_sample.copy()
    if "call_timing_bucket" not in data.columns:
        data["call_timing_bucket"] = "unknown"
    if "market_hours_call" not in data.columns:
        data["market_hours_call"] = data["call_timing_bucket"].eq("market_hours")

    samples = [("full_sample", data, "all")]
    samples.append(("exclude_market_hours", data.loc[~data["market_hours_call"]], "excluding_market_hours"))
    for bucket in ("premarket", "market_hours", "after_close"):
        samples.append((f"{bucket}_only", data.loc[data["call_timing_bucket"].eq(bucket)], bucket))

    rows = []
    for sample_label, subset, timing_bucket in samples:
        if subset.empty:
            continue
        support, insufficiency_reasons = timing_sample_support(subset)
        sample_metadata = {
            "sample_label": sample_label,
            "timing_bucket": timing_bucket,
            "exclude_market_hours": sample_label == "exclude_market_hours",
            "innovation_standardization": "full_sample",
            **support,
        }
        if insufficiency_reasons:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            **sample_metadata,
                            "status": "insufficient_sample",
                            "insufficient_reason": ";".join(insufficiency_reasons),
                            "horizon": pd.NA,
                            "outcome": pd.NA,
                            "treatment": pd.NA,
                            "beta": np.nan,
                            "standard_error": np.nan,
                            "t_stat": np.nan,
                            "n_obs": pd.NA,
                            "n_firms": pd.NA,
                            "n_event_dates": pd.NA,
                            "standard_error_type": pd.NA,
                        }
                    ]
                )
            )
            continue
        coefficients, _ = estimate_all_response_curves(subset)
        coefficients.insert(0, "sample_label", sample_label)
        coefficients.insert(1, "timing_bucket", timing_bucket)
        coefficients["status"] = "estimated"
        coefficients["insufficient_reason"] = ""
        for column, value in sample_metadata.items():
            if column not in {"sample_label", "timing_bucket"}:
                coefficients[column] = value
        rows.append(coefficients)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def timing_sample_support(subset: pd.DataFrame) -> tuple[dict[str, int], list[str]]:
    """Measure two-way-cluster support and explain why a subgroup is skipped."""

    n_rows = int(len(subset))
    n_firms = int(subset["permno"].nunique()) if "permno" in subset else 0
    n_event_dates = (
        int(subset["event_trading_date"].nunique())
        if "event_trading_date" in subset
        else 0
    )
    support = {
        "n_sample_rows": n_rows,
        "n_sample_firms": n_firms,
        "n_sample_event_dates": n_event_dates,
    }
    reasons = []
    if n_rows < TIMING_MIN_SAMPLE_ROWS:
        reasons.append(f"rows={n_rows}<{TIMING_MIN_SAMPLE_ROWS}")
    if n_firms < TIMING_MIN_FIRMS:
        reasons.append(f"firms={n_firms}<{TIMING_MIN_FIRMS}")
    if n_event_dates < TIMING_MIN_EVENT_DATES:
        reasons.append(f"event_dates={n_event_dates}<{TIMING_MIN_EVENT_DATES}")
    return support, reasons


def fit_response_regression(
    data: pd.DataFrame,
    *,
    outcome: str,
    treatments: Sequence[str],
    controls: Sequence[str],
    firm_fe: str,
    quarter_fe: str,
    cluster_cols: tuple[str, str],
) -> RegressionResult:
    """Fit the paper response model with firm and quarter fixed effects."""

    regressors = list(dict.fromkeys([*treatments, *controls]))
    needed = [outcome, *regressors, firm_fe, quarter_fe, *cluster_cols]
    needed = list(dict.fromkeys(needed))
    clean = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    design = clean[regressors].copy()
    categorical = [col for col in ["match_quality"] if col in design]
    if categorical:
        for col in categorical:
            design[col] = design[col].astype(str)
        design = pd.get_dummies(design, columns=categorical, drop_first=True, dtype=float)
    firm_dummies = pd.get_dummies(clean[firm_fe].astype(str), prefix="firm", drop_first=True, dtype=float)
    quarter_dummies = pd.get_dummies(clean[quarter_fe].astype(str), prefix="qtr", drop_first=True, dtype=float)
    x = pd.concat([design.astype(float), firm_dummies, quarter_dummies], axis=1)
    x.insert(0, "intercept", 1.0)
    y = clean[outcome].astype(float).to_numpy()
    x_values = x.to_numpy(dtype=float)
    xtx_inv = np.linalg.pinv(x_values.T @ x_values)
    beta = xtx_inv @ x_values.T @ y
    resid = y - x_values @ beta
    cov = two_way_cluster_covariance(
        x_values,
        resid,
        xtx_inv,
        clean[cluster_cols[0]],
        clean[cluster_cols[1]],
    )
    coef_rows = []
    for treatment in treatments:
        idx = list(x.columns).index(treatment)
        se = float(np.sqrt(max(cov[idx, idx], 0.0)))
        coef_rows.append(
            {
                "treatment": treatment,
                "beta": float(beta[idx]),
                "standard_error": se,
                "t_stat": float(beta[idx] / se) if se > 0 else np.nan,
                "n_obs": int(len(clean)),
                "n_firms": int(clean[cluster_cols[0]].nunique()),
                "n_event_dates": int(clean[cluster_cols[1]].nunique()),
                "standard_error_type": "two_way_cluster_permno_event_trading_date",
            }
        )
    return RegressionResult(
        coefficients=pd.DataFrame(coef_rows),
        covariance=cov,
        coefficient_names=list(x.columns),
        n_obs=int(len(clean)),
        n_firms=int(clean[cluster_cols[0]].nunique()),
        n_event_dates=int(clean[cluster_cols[1]].nunique()),
    )


def two_way_cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    xtx_inv: np.ndarray,
    cluster_a: pd.Series,
    cluster_b: pd.Series,
) -> np.ndarray:
    """Cameron-Gelbach-Miller style two-way cluster sandwich: firm + date - intersection."""

    cov_a = one_way_cluster_covariance(x, residuals, xtx_inv, cluster_a)
    cov_b = one_way_cluster_covariance(x, residuals, xtx_inv, cluster_b)
    pair = cluster_a.astype(str).reset_index(drop=True) + "||" + cluster_b.astype(str).reset_index(drop=True)
    cov_ab = one_way_cluster_covariance(x, residuals, xtx_inv, pair)
    return cov_a + cov_b - cov_ab


def one_way_cluster_covariance(
    x: np.ndarray, residuals: np.ndarray, xtx_inv: np.ndarray, clusters: pd.Series
) -> np.ndarray:
    cluster_values = pd.Series(clusters).reset_index(drop=True)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for value in cluster_values.dropna().unique():
        mask = cluster_values.eq(value).to_numpy()
        score = x[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    n_obs, n_params = x.shape
    n_clusters = max(cluster_values.nunique(dropna=True), 1)
    finite_sample = (n_clusters / max(n_clusters - 1, 1)) * (
        (n_obs - 1) / max(n_obs - n_params, 1)
    )
    return finite_sample * xtx_inv @ meat @ xtx_inv


def coefficient_difference_test(
    result: RegressionResult, horizon: int, left: str, right: str
) -> dict[str, object]:
    names = result.coefficient_names
    i = names.index(left)
    j = names.index(right)
    beta = result.coefficients.set_index("treatment")
    diff = float(beta.loc[left, "beta"] - beta.loc[right, "beta"])
    var = result.covariance[i, i] + result.covariance[j, j] - 2 * result.covariance[i, j]
    se = float(np.sqrt(max(var, 0.0)))
    return {
        "test": "answer_equals_prepared",
        "horizon": horizon,
        "left": left,
        "right": right,
        "difference": diff,
        "standard_error": se,
        "z_stat": float(diff / se) if se > 0 else np.nan,
    }


def single_coefficient_test(result: RegressionResult, treatment: str, *, test_name: str) -> dict[str, object]:
    row = result.coefficients.loc[result.coefficients["treatment"].eq(treatment)].iloc[0]
    return {
        "test": test_name,
        "horizon": "20_minus_1",
        "left": treatment,
        "right": "0",
        "difference": float(row["beta"]),
        "standard_error": float(row["standard_error"]),
        "z_stat": float(row["t_stat"]),
    }


def estimate_heterogeneity(response_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pct_rows = []
    controls = response_controls(response_sample)
    for moderator in ["low_analyst_coverage_z", "high_forecast_dispersion_z"]:
        data = response_sample.copy()
        interaction = f"answer_x_{moderator}"
        data[interaction] = data["answer_innovation_z"] * data[moderator]
        for horizon in HORIZONS:
            result = fit_response_regression(
                data,
                outcome=f"CAR_0_{horizon}",
                treatments=["answer_innovation_z", moderator, interaction],
                controls=[col for col in controls if col != moderator],
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            coef = result.coefficients.copy()
            coef.insert(0, "moderator", moderator)
            coef.insert(1, "horizon", horizon)
            rows.append(coef)
            beta = coef.set_index("treatment")
            for pct in [10, 50, 90]:
                value = float(np.nanpercentile(data[moderator], pct))
                pct_rows.append(
                    {
                        "moderator": moderator,
                        "horizon": horizon,
                        "percentile": pct,
                        "moderator_value": value,
                        "implied_answer_beta": float(
                            beta.loc["answer_innovation_z", "beta"] + beta.loc[interaction, "beta"] * value
                        ),
                    }
                )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(pct_rows)


def estimate_placebo_responses(response_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    controls = response_controls(response_sample)
    treatments = ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]
    for outcome in ["placebo_CAR_m5_m1", "placebo_CAR_m10_m1"]:
        result = fit_response_regression(
            response_sample,
            outcome=outcome,
            treatments=treatments,
            controls=controls,
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        out = result.coefficients.copy()
        out.insert(0, "outcome", outcome)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def innovation_orthogonality_checks(response_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [
        "scaled_eps_surprise",
        "eps_surprise",
        "pre_ret_20",
        "pre_vol_20",
        "analyst_coverage",
        "forecast_dispersion",
        "forecast_age_days",
    ]
    for innovation in ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]:
        for predictor in predictors:
            clean = response_sample[[innovation, predictor]].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "innovation": innovation,
                    "predictor": predictor,
                    "n_obs": int(len(clean)),
                    "correlation": _safe_corr(clean[innovation], clean[predictor]),
                }
            )
    return pd.DataFrame(rows)


def parsing_quality_validation(response_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flag_value, label in [(True, "high_confidence_parse"), (False, "lower_confidence_parse")]:
        group = response_sample.loc[response_sample["high_confidence_parse"].eq(flag_value)].copy()
        if len(group) < 100:
            continue
        result = fit_response_regression(
            group,
            outcome="CAR_0_20",
            treatments=["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"],
            controls=response_controls(group),
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        out = result.coefficients.copy()
        out.insert(0, "subsample", label)
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def first_stage_model_comparison(
    state: pd.DataFrame, section_scores: pd.DataFrame, config: LMBaselineConfig
) -> pd.DataFrame:
    prepared = section_scores.loc[section_scores["section"].eq("prepared")].copy()
    data = state.merge(prepared[["transcriptid", PRIMARY_FEATURE]], on="transcriptid", how="inner")
    rows = []
    for learner in ["ridge", "elasticnet", "hist_gradient_boosting"]:
        pred, _, _ = oof_predict_by_year(
            data,
            target=PRIMARY_FEATURE,
            feature_columns=base_state_features(data),
            categorical_columns=base_categorical_features(),
            min_train_rows=config.min_train_rows,
            learner=learner,
        )
        clean = pd.DataFrame({"raw": data[PRIMARY_FEATURE].astype(float), "pred": pred}).dropna()
        rows.append(
            {
                "component": "prepared",
                "semantic_feature": PRIMARY_FEATURE,
                "model": learner,
                "n_obs": int(len(clean)),
                "first_stage_r2": float(r2_score(clean["raw"], clean["pred"])) if len(clean) > 2 else np.nan,
                "mae": float(mean_absolute_error(clean["raw"], clean["pred"])) if len(clean) else np.nan,
                "predicted_actual_corr": _safe_corr(clean["raw"], clean["pred"]),
            }
        )
    return pd.DataFrame(rows)


def first_stage_by_year(call_innovations: pd.DataFrame, turn_innovations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, data in [("call", call_innovations), ("turn", turn_innovations)]:
        if data.empty:
            continue
        for keys, group in data.groupby(["component", "semantic_feature", "event_year"], sort=True):
            component, feature, year = keys
            clean = group.dropna(subset=["raw_score", "expected_score_oof", "narrative_innovation"])
            rows.append(
                {
                    "source": source,
                    "component": component,
                    "semantic_feature": feature,
                    "event_year": int(year),
                    "n_obs": int(len(clean)),
                    "r2": float(r2_score(clean["raw_score"], clean["expected_score_oof"])) if len(clean) > 2 else np.nan,
                    "mae": float(mean_absolute_error(clean["raw_score"], clean["expected_score_oof"])) if len(clean) else np.nan,
                    "residual_mean": float(clean["narrative_innovation"].mean()) if len(clean) else np.nan,
                    "residual_variance": float(clean["narrative_innovation"].var()) if len(clean) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def residual_time_summary(call_innovations: pd.DataFrame) -> pd.DataFrame:
    if call_innovations.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in call_innovations.groupby(["component", "semantic_feature", "event_year"], sort=True):
        component, feature, year = keys
        rows.append(
            {
                "component": component,
                "semantic_feature": feature,
                "event_year": int(year),
                "mean": float(group["narrative_innovation"].mean()),
                "variance": float(group["narrative_innovation"].var()),
                "p05": float(group["narrative_innovation"].quantile(0.05)),
                "p95": float(group["narrative_innovation"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def residual_predictability_checks(call_innovations: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [
        col
        for col in base_state_features(state)
        if col in state.columns and col not in call_innovations.columns
    ]
    merged = call_innovations.merge(state[["transcriptid", *predictors]], on="transcriptid", how="left")
    for keys, group in merged.groupby(["component", "semantic_feature"], sort=True):
        component, feature = keys
        candidates = []
        for predictor in predictors:
            if not pd.api.types.is_numeric_dtype(group[predictor]):
                continue
            clean = group[["narrative_innovation", predictor]].replace([np.inf, -np.inf], np.nan).dropna()
            candidates.append(
                {
                    "predictor": predictor,
                    "n_obs": int(len(clean)),
                    "abs_correlation": abs(_safe_corr(clean["narrative_innovation"], clean[predictor])),
                }
            )
        best = max(candidates, key=lambda row: row["abs_correlation"]) if candidates else None
        rows.append(
            {
                "component": component,
                "semantic_feature": feature,
                "n_obs": int(best["n_obs"]) if best else 0,
                "strongest_predictor": best["predictor"] if best else "",
                "max_abs_residual_predictor_corr": float(best["abs_correlation"]) if best else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_lm_outputs(config: LMBaselineConfig, **tables: pd.DataFrame) -> None:
    parquet_names = {
        "section_scores": "section_lm_scores.parquet",
        "operator_scores": "boilerplate_placebo_scores.parquet",
        "turns": "qa_turns.parquet",
        "turn_innovations": "turn_level_innovations.parquet",
        "call_innovations": "call_level_innovations.parquet",
        "outcomes": "event_outcomes.parquet",
        "response_sample": "response_model_sample.parquet",
    }
    csv_names = {
        "attrition": "attrition.csv",
        "first_stage": "first_stage_diagnostics.csv",
        "by_year": "first_stage_diagnostics_by_year.csv",
        "residual_time": "first_stage_residual_time_summary.csv",
        "residual_predictability": "first_stage_residual_predictability.csv",
        "model_comparison": "first_stage_model_comparison.csv",
        "response_coefficients": "response_coefficients.csv",
        "coefficient_tests": "coefficient_tests.csv",
        "heterogeneity": "heterogeneity_interactions.csv",
        "heterogeneity_percentiles": "heterogeneity_percentile_effects.csv",
        "validation_placebos": "validation_placebos.csv",
        "validation_orthogonality": "validation_orthogonality.csv",
        "validation_parsing": "validation_parsing_quality.csv",
        "timing_subsample_responses": "timing_subsample_responses.csv",
    }
    for key, name in parquet_names.items():
        table = tables.get(key)
        if table is not None and not table.empty:
            table.to_parquet(config.output_dir / name, index=False)
    for key, name in csv_names.items():
        table = tables.get(key)
        if table is not None:
            table.to_csv(config.output_dir / name, index=False)

    feature_payload = {
        "car_horizon_semantics": CAR_HORIZON_SEMANTICS,
        "specifications": all_feature_manifests(),
    }
    (config.output_dir / "baseline_feature_manifest.json").write_text(
        json.dumps(feature_payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def figure_main_response(response_coefficients: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = response_coefficients.loc[
        response_coefficients["treatment"].isin(["prepared_innovation_z", "answer_innovation_z"])
    ].copy()
    if plot.empty:
        return
    plot["label"] = plot["treatment"].map(
        {
            "prepared_innovation_z": "Prepared narrative innovation",
            "answer_innovation_z": "Elicited answer innovation",
        }
    )
    plot["beta_bps"] = plot["beta"] * 10000
    plot["se_bps"] = plot["standard_error"] * 10000
    fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
    for label, group in plot.groupby("label", sort=False):
        group = group.sort_values("horizon")
        ax.errorbar(
            group["horizon"],
            group["beta_bps"],
            yerr=1.96 * group["se_bps"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
        )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Trading-day CAR horizon")
    ax.set_ylabel("Basis points per one-SD LM net-tone innovation")
    ax.set_title("S&P 500 Earnings-Call Narrative Innovation Response")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sample_attrition(
    original_panel: pd.DataFrame,
    earnings: pd.DataFrame,
    point_in_time: pd.DataFrame,
    outcomes: pd.DataFrame,
    response_sample: pd.DataFrame,
    *,
    call_innovations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    outcomes_year = pd.to_numeric(
        pd.to_datetime(outcomes.get("event_trading_date"), errors="coerce").dt.year,
        errors="coerce",
    ) if "event_trading_date" in outcomes else pd.Series(dtype=float)
    innovations = call_innovations if call_innovations is not None else pd.DataFrame()
    by_component = {
        component: set(
            innovations.loc[innovations.get("component", pd.Series(dtype=object)).eq(component), "transcriptid"].astype(str)
        )
        for component in ("prepared", "question", "answer")
    }
    rows = [
        {"metric": "all_staged_transcript_events", "value": int(len(original_panel)), "stage": "staged", "reason": "all staged panel rows"},
        {"metric": "earnings_call_events_through_sample_end", "value": int(len(earnings)), "stage": "event_filter", "reason": "earnings calls through sample end"},
        {"metric": "point_in_time_sp500_earnings_calls", "value": int(len(point_in_time)), "stage": "point_in_time", "reason": "valid ticker/company membership on call date"},
        {"metric": "all_crsp_ready_calls", "value": int(len(outcomes)), "stage": "crsp_ready", "reason": "all calls with all requested CRSP windows"},
        {"metric": "all_crsp_ready_years", "value": int(outcomes_year.nunique()), "stage": "crsp_ready", "reason": "unique event years represented by CRSP-ready calls"},
        {"metric": "training_only_calls_2018_2019", "value": int(outcomes_year.isin([2018, 2019]).sum()), "stage": "crsp_ready", "reason": "expanding first-stage training years"},
        {"metric": "response_period_calls_2020_2024", "value": int(outcomes_year.isin([2020, 2021, 2022, 2023, 2024]).sum()), "stage": "crsp_ready", "reason": "response-period test years"},
        {"metric": "calls_with_prepared_innovation", "value": len(by_component["prepared"]), "stage": "innovation", "reason": "prepared first-stage innovation available"},
        {"metric": "calls_with_question_and_answer_innovations", "value": len(by_component["question"] & by_component["answer"]), "stage": "innovation", "reason": "both question and answer innovations available"},
        {"metric": "crsp_event_window_ready", "value": int(len(outcomes)), "stage": "crsp_ready", "reason": "compatibility alias for prior reports"},
        {"metric": "main_common_response_sample", "value": int(len(response_sample)), "stage": "response_sample", "reason": "complete outcomes, controls, and three innovations"},
        {"metric": "main_unique_firms", "value": int(response_sample["permno"].nunique()) if not response_sample.empty else 0, "stage": "response_sample", "reason": "unique firms in final common sample"},
    ]
    return pd.DataFrame(rows)


def response_controls(data: pd.DataFrame) -> list[str]:
    candidates = [
        "scaled_eps_surprise",
        "eps_surprise",
        "pre_ret_20",
        "pre_vol_20",
        "pre_log_dollar_volume_20",
        "pre_amihud_illiquidity_20",
        "analyst_coverage",
        "forecast_dispersion",
        "forecast_age_days",
        "log_market_cap",
        "match_date_gap_days",
        "call_after_close",
        "match_quality",
    ]
    return [col for col in candidates if col in data.columns]


def base_state_features(data: pd.DataFrame) -> list[str]:
    return base_state_feature_spec(data)


def base_categorical_features() -> list[str]:
    return ["ticker", "match_quality", "calendar_quarter"]


def pivot_feature_scores(data: pd.DataFrame, *, index_cols: list[str], prefix_by: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=[*index_cols])
    values = data.pivot_table(
        index=index_cols,
        columns=prefix_by,
        values=LM_FEATURES,
        aggfunc="last",
        sort=False,
    )
    prefixes = list(dict.fromkeys(data[prefix_by].dropna().tolist()))
    out = values.copy()
    out.columns = [
        f"{prefix}_{feature}"
        for feature, prefix in out.columns.to_list()
    ]
    expected = [f"{prefix}_{feature}" for prefix in prefixes for feature in LM_FEATURES]
    for column in expected:
        if column not in out.columns:
            out[column] = np.nan
    return out.reindex(columns=expected).reset_index()


def _normalize_components(components: pd.DataFrame) -> pd.DataFrame:
    data = components.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]
    if "transcriptcomponenttypename" in data and "componenttypename" not in data:
        data = data.rename(columns={"transcriptcomponenttypename": "componenttypename"})
    data["transcriptid"] = data["transcriptid"].astype(str)
    data["componentorder"] = pd.to_numeric(data["componentorder"], errors="coerce")
    data["componenttypename"] = data["componenttypename"].astype("string").str.strip()
    data["componenttext"] = data["componenttext"].fillna("").astype(str)
    sort_cols = [
        col for col in ["transcriptid", "componentorder", "transcriptcomponentid"] if col in data
    ]
    return data.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _score_text(text: str, dictionary: LMDictionary) -> dict[str, float | int | str]:
    score = score_lm_text(text or "", dictionary)
    score["text_length"] = len(str(text or ""))
    return score


def _combine_scores(
    scores: Iterable[dict[str, float | int | str]], dictionary: LMDictionary
) -> dict[str, float | int | str]:
    parts = list(scores)
    word_count = int(sum(int(part.get("word_count", 0) or 0) for part in parts))
    sentence_count = int(sum(int(part.get("sentence_count", 0) or 0) for part in parts))
    text_length = int(sum(int(part.get("text_length", 0) or 0) for part in parts))
    counts = {
        category: int(sum(int(part.get(f"lm_{category}_count", 0) or 0) for part in parts))
        for category in LM_CATEGORIES
    }
    scale = word_count / 1000 if word_count else np.nan
    rates = {
        f"lm_{category}_per_1k": (counts[category] / scale if word_count else 0.0)
        for category in LM_CATEGORIES
    }
    return {
        "word_count": word_count,
        "sentence_count": max(sentence_count, 1),
        "avg_word_length": np.nan,
        "lm_net_tone": float(rates["lm_positive_per_1k"] - rates["lm_negative_per_1k"]),
        **{f"lm_{category}_count": counts[category] for category in LM_CATEGORIES},
        **rates,
        "scoring_backend": dictionary.backend,
        "text_length": text_length,
    }


def _simple_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w]+(?:['-][\w]+)*\b", str(text or "")))


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return values * np.nan
    return (values - values.mean(skipna=True)) / std


def calibration_slope(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 3 or np.nanvar(predicted[valid]) == 0:
        return np.nan
    x = np.column_stack([np.ones(valid.sum()), predicted[valid]])
    coef = np.linalg.pinv(x.T @ x) @ x.T @ actual[valid]
    return float(coef[1])


def _safe_corr(left: Sequence[float], right: Sequence[float]) -> float:
    left = pd.Series(left, dtype=float)
    right = pd.Series(right, dtype=float)
    valid = left.notna() & right.notna()
    if valid.sum() < 3:
        return np.nan
    if left.loc[valid].std() == 0 or right.loc[valid].std() == 0:
        return np.nan
    value = left.loc[valid].corr(right.loc[valid])
    return float(value) if pd.notna(value) else np.nan
