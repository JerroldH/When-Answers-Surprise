"""Outcome-blind 2025 holdout build, scoring, response, and freeze workflow."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
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
    build_first_stage_feature_frames,
    make_feature_frame,
    tuned_ridge_spec_with_alphas,
)
from when_answers_surprise.empirical.pipeline import (
    HORIZONS,
    PRIMARY_FEATURE,
    apply_point_in_time_sp500_filter,
    base_categorical_features,
    build_crsp_event_outcomes,
    build_event_state,
    build_qa_turns,
    filter_earnings_calls,
    response_controls,
    score_lm_sections,
    score_qa_turns,
)
from when_answers_surprise.empirical.pre_event_audit import (
    NON_OVERLAP_OUTCOMES,
    add_full_sample_multiplicity,
    estimate_pre_event_cells,
    joint_pre_period_tests,
    prepare_pre_event_sample,
    safe_pre_event_controls,
)
from when_answers_surprise.empirical.revision_v2_inference import (
    JointSystemResult,
    adjusted_inference,
    fit_joint_horizon_system,
)
from when_answers_surprise.empirical.revision_v2 import enrich_revision_frames
from when_answers_surprise.empirical.revision_v2_features import add_strict_analyst_history
from when_answers_surprise.empirical.revision_v2_tuned import estimate_component_ladder
from when_answers_surprise.empirical.revision_v2_tuned_bootstrap import (
    draw_firm_multiplicity,
    prepared_call_residuals,
    tuned_bootstrap_oof_predict,
    turn_call_residuals,
)
from when_answers_surprise.ibes_events import build_ibes_event_dataset
from when_answers_surprise.ibes_staging import merge_summary_with_actuals
from when_answers_surprise.semantic_scoring import load_lm_dictionary
from when_answers_surprise.wrds_sp500 import (
    build_ciq_transcript_text,
    build_security_master,
    normalize_sp500_membership,
    read_partitioned_parquet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "wrds_holdout_2025_ciz"
DEFAULT_RAW_ROOT = DEFAULT_DATA_ROOT / "raw" / "wrds_exports"
DEFAULT_BUILD_DIR = DEFAULT_DATA_ROOT / "build"
DEFAULT_FEATURE_DIR = DEFAULT_BUILD_DIR / "outcome_blind_features"
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "wrds_holdout_2025" / "holdout_protocol.json"
)
DEFAULT_RESIDUAL_DIR = DEFAULT_DATA_ROOT / "analysis" / "residual_freeze"
DEFAULT_RESPONSE_DIR = DEFAULT_DATA_ROOT / "analysis" / "response"
DEFAULT_BOOTSTRAP_DIR = DEFAULT_DATA_ROOT / "analysis" / "bootstrap"
DEFAULT_FREEZE_DIR = (
    PROJECT_ROOT / "outputs" / "frozen" / "holdout_2025" / "20260731_protocol_v1"
)
DEFAULT_LM_HOLDOUT_LADDER_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "frozen"
    / "holdout_2025_lm_ladder"
    / "20260809_icaif_figure3"
)
DEFAULT_HISTORICAL_DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
DEFAULT_FROZEN_REVISION = (
    PROJECT_ROOT / "outputs" / "frozen" / "revision_v2_tuned" / "20260716_widegrid_v2"
)
HOLDOUT_YEAR = 2025
RIDGE_ALPHAS = (
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
    100000.0,
    300000.0,
    1000000.0,
    3000000.0,
    10000000.0,
    30000000.0,
    100000000.0,
)
RAW_TREATMENTS = {
    "prepared_innovation": "prepared_P0",
    "question_innovation": "question_Q2",
    "answer_innovation": "answer_A3",
}
STANDARDIZED_TREATMENTS = [
    "prepared_innovation_z",
    "question_innovation_z",
    "answer_innovation_z",
]
BOOTSTRAP_CONVERGENCE_DRAWS = (25, 50, 100, 150, 200)
FORBIDDEN_FIRST_STAGE_PREFIXES = (
    "car_",
    "postcall_",
    "abnormal_volume_",
    "realized_volatility_",
)


def map_ciz_daily_to_legacy(
    ciz_daily: pd.DataFrame,
    ciz_index: pd.DataFrame,
    security_history: pd.DataFrame,
) -> pd.DataFrame:
    """Map the CRSP CIZ daily/index/security-history inputs to pipeline fields."""

    daily = _lower_columns(ciz_daily)
    index = _lower_columns(ciz_index)
    history = _lower_columns(security_history)
    required_daily = {
        "permno",
        "dlycaldt",
        "dlyret",
        "dlyprc",
        "dlyvol",
        "dlycap",
        "dlycumfacpr",
        "dlycumfacshr",
    }
    required_index = {"dlycaldt", "vwretd"}
    required_history = {"permno", "secinfostartdt", "secinfoenddt"}
    for label, data, required in (
        ("CIZ daily", daily, required_daily),
        ("CIZ daily index", index, required_index),
        ("CIZ security history", history, required_history),
    ):
        if missing := sorted(required.difference(data.columns)):
            raise ValueError(f"{label} is missing columns: {missing}")

    daily["permno"] = pd.to_numeric(daily["permno"], errors="coerce").astype("Int64")
    daily["dlycaldt"] = pd.to_datetime(daily["dlycaldt"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["permno", "dlycaldt", "dlyret"]).copy()
    if daily.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("CIZ daily data contain duplicate permno-date rows")
    index["dlycaldt"] = pd.to_datetime(index["dlycaldt"], errors="coerce").dt.normalize()
    if index["dlycaldt"].duplicated().any():
        raise ValueError("CIZ daily index contains duplicate calendar dates")

    history["permno"] = pd.to_numeric(history["permno"], errors="coerce").astype("Int64")
    history["secinfostartdt"] = pd.to_datetime(
        history["secinfostartdt"], errors="coerce"
    ).dt.normalize()
    history["secinfoenddt"] = pd.to_datetime(
        history["secinfoenddt"], errors="coerce"
    ).dt.normalize()
    daily = daily.reset_index(drop=True)
    daily["_daily_row"] = np.arange(len(daily))
    expanded = daily.merge(history, on="permno", how="left", suffixes=("", "_history"))
    starts = expanded["secinfostartdt"].fillna(pd.Timestamp("1900-01-01"))
    ends = expanded["secinfoenddt"].fillna(pd.Timestamp("2262-04-11"))
    valid = expanded["dlycaldt"].between(starts, ends, inclusive="both")
    matched = expanded.loc[valid].sort_values(
        ["_daily_row", "secinfostartdt"], na_position="first"
    )
    matched = matched.drop_duplicates("_daily_row", keep="last")
    unmatched_ids = sorted(set(daily["_daily_row"]).difference(matched["_daily_row"]))
    if unmatched_ids:
        unmatched = daily.loc[daily["_daily_row"].isin(unmatched_ids)].copy()
        for column in history.columns:
            if column != "permno" and column not in unmatched:
                unmatched[column] = pd.NA
        matched = pd.concat([matched, unmatched], ignore_index=True, sort=False)
    merged = matched.merge(
        index[["dlycaldt", "vwretd"]],
        on="dlycaldt",
        how="left",
        validate="many_to_one",
    )
    if merged["vwretd"].isna().any():
        raise ValueError("CIZ daily rows are missing the daily value-weighted index return")

    price = pd.to_numeric(merged["dlyprc"], errors="coerce")
    market_cap = pd.to_numeric(merged["dlycap"], errors="coerce")
    legacy = pd.DataFrame(
        {
            "permno": merged["permno"],
            "permco": pd.to_numeric(merged.get("permco"), errors="coerce"),
            "date": merged["dlycaldt"],
            "ret": pd.to_numeric(merged["dlyret"], errors="coerce"),
            "retx": pd.to_numeric(merged.get("dlyretx"), errors="coerce"),
            "prc": price,
            "vol": pd.to_numeric(merged["dlyvol"], errors="coerce"),
            "shrout": np.divide(
                market_cap,
                price.abs(),
                out=np.full(len(merged), np.nan),
                where=price.abs().to_numpy(dtype=float) > 0,
            ),
            "cfacpr": pd.to_numeric(merged["dlycumfacpr"], errors="coerce"),
            "cfacshr": pd.to_numeric(merged["dlycumfacshr"], errors="coerce"),
            "vwretd": pd.to_numeric(merged["vwretd"], errors="coerce"),
            "cusip": merged.get("cusip"),
            "ncusip": merged.get("cusip_history", merged.get("cusip")),
            "ticker": merged.get("ticker"),
            "comnam": merged.get("issuernm"),
            "primexch": merged.get("primaryexch"),
            "trdstat": merged.get("tradingstatusflg"),
        }
    )
    legacy["shrcd"] = _ciz_share_code(merged)
    legacy["exchcd"] = _ciz_exchange_code(merged)
    legacy = legacy.sort_values(["permno", "date"]).reset_index(drop=True)
    if legacy.duplicated(["permno", "date"]).any():
        raise ValueError("CIZ legacy mapping is not unique by permno-date")
    return legacy


def bridge_metrics(old_events: pd.DataFrame, ciz_events: pd.DataFrame) -> dict[str, object]:
    """Evaluate the prespecified 2024 CIZ--SIZ comparability gate."""

    for label, frame in (("SIZ", old_events), ("CIZ", ciz_events)):
        if "transcriptid" not in frame or "CAR_0_1" not in frame:
            raise ValueError(f"{label} bridge events need transcriptid and CAR_0_1")
        if frame["transcriptid"].astype(str).duplicated().any():
            raise ValueError(f"{label} bridge events contain duplicate transcript ids")
    old = old_events.copy()
    new = ciz_events.copy()
    old["transcriptid"] = old["transcriptid"].astype(str)
    new["transcriptid"] = new["transcriptid"].astype(str)
    comparison = old.merge(
        new,
        on="transcriptid",
        how="left",
        suffixes=("_siz", "_ciz"),
        validate="one_to_one",
        indicator=True,
    )
    complete = comparison.dropna(subset=["CAR_0_1_siz", "CAR_0_1_ciz"])
    coverage = float(len(complete) / max(len(old), 1))
    correlation = (
        float(complete[["CAR_0_1_siz", "CAR_0_1_ciz"]].corr().iloc[0, 1])
        if len(complete) >= 2
        else np.nan
    )
    median_abs_bps = (
        float(
            10_000
            * (complete["CAR_0_1_siz"] - complete["CAR_0_1_ciz"]).abs().median()
        )
        if len(complete)
        else np.nan
    )
    gate = bool(
        coverage >= 0.99
        and np.isfinite(correlation)
        and correlation >= 0.99
        and np.isfinite(median_abs_bps)
        and median_abs_bps <= 5.0
    )
    return {
        "status": "complete",
        "eligible_siz_events": int(len(old)),
        "comparable_events": int(len(complete)),
        "eligible_event_coverage": coverage,
        "car_0_1_pearson_correlation": correlation,
        "car_0_1_median_absolute_difference_bps": median_abs_bps,
        "coverage_threshold": 0.99,
        "correlation_threshold": 0.99,
        "median_absolute_difference_threshold_bps": 5.0,
        "comparability_gate_passed": gate,
        "pooled_comparison_allowed": gate,
    }


def freeze_holdout_protocol(
    *,
    frozen_revision_dir: Path = DEFAULT_FROZEN_REVISION,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, object]:
    """Write the holdout contract before any 2025 response is estimated."""

    if protocol_path.exists():
        raise FileExistsError(f"Holdout protocol already exists: {protocol_path}")
    response_path = frozen_revision_dir / "response_model_sample_revision_v2.parquet"
    response = pd.read_parquet(response_path)
    scalers = {}
    for standardized, raw in (
        ("prepared_innovation_z", "prepared_innovation"),
        ("question_innovation_z", "question_innovation"),
        ("answer_innovation_z", "answer_innovation"),
    ):
        values = pd.to_numeric(response[raw], errors="coerce")
        scalers[standardized] = {
            "raw_column": raw,
            "mean_2020_2024": float(values.mean()),
            "sd_2020_2024": float(values.std(ddof=1)),
            "n_2020_2024": int(values.notna().sum()),
        }
    protocol = {
        "status": "frozen_before_2025_response",
        "contract": "finnlp-2025-holdout-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "population": "2025 point-in-time S&P 500 earnings calls",
        "training_years": list(range(2018, 2025)),
        "test_year": 2025,
        "feature_contracts": ["P0", "Q2", "A3"],
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "selection_rule": "inner expanding-year MAE",
        "aggregation": "word-count weighted mean",
        "training_standardization": scalers,
        "primary_outcome": "CAR_0_1",
        "primary_model": "P0/Q2/A3 joint response model",
        "controls": response_controls(response),
        "fixed_effects": ["permno", "event_calendar_quarter"],
        "clustering": ["permno", "event_trading_date"],
        "secondary_analyses": [
            "other CAR horizons",
            "2025 pooled interaction if the CIZ-SIZ bridge passes",
            "pre-event checks",
        ],
        "report_all_results_regardless_of_direction_or_significance": True,
        "frozen_training_response_source": str(response_path.resolve()),
        "frozen_training_response_sha256": _sha256(response_path),
        "frozen_revision_manifest_sha256": _sha256(
            frozen_revision_dir / "freeze_manifest.json"
        ),
    }
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    checksum = _sha256(protocol_path)
    protocol_path.with_suffix(".sha256").write_text(f"{checksum}\n", encoding="utf-8")
    return protocol


def _read_holdout_ibes(raw_root, historical_data_root, table, partition_date, history_year):
    data = read_partitioned_parquet(raw_root / table)
    if history_year is None:
        return data
    if (raw_root / table / f"year={history_year}").is_dir():
        return data
    history = read_partitioned_parquet(
        historical_data_root / "raw" / "wrds_exports" / table / f"year={history_year}"
    )
    dates = pd.to_datetime(history[partition_date], errors="raise")
    if history.empty or not dates.dt.year.eq(history_year).all():
        raise ValueError(f"Invalid {table} partition for {history_year}")
    if not pd.to_datetime(data[partition_date], errors="raise").dt.year.gt(history_year).all():
        raise ValueError(f"Overlapping {table} history and holdout partitions")
    if set(history.columns) != set(data.columns):
        raise ValueError(f"Incompatible history schema: {table}")
    return pd.concat([history[data.columns], data], ignore_index=True)


def build_holdout_2025(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    build_dir: Path = DEFAULT_BUILD_DIR,
    historical_data_root: Path = DEFAULT_HISTORICAL_DATA_ROOT,
    frozen_revision_dir: Path | None = DEFAULT_FROZEN_REVISION,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    ibes_history_year: int | None = None,
    dictionary_dir: Path | None = None,
) -> dict[str, object]:
    """Build the CIZ bridge and outcome-blind 2018--2025 feature frames."""

    _require_empty(build_dir, "holdout build")
    ciz_daily = read_partitioned_parquet(raw_root / "crsp_ciz_daily_holdout")
    ciz_index = read_partitioned_parquet(raw_root / "crsp_ciz_daily_index_holdout")
    security_history = read_partitioned_parquet(
        raw_root / "crsp_ciz_security_history_holdout"
    )
    legacy_ciz = map_ciz_daily_to_legacy(ciz_daily, ciz_index, security_history)

    membership_raw = read_partitioned_parquet(raw_root / "crsp_sp500_membership")
    membership_new = normalize_sp500_membership(membership_raw)
    names_new = _security_history_as_names(security_history)
    security_new = build_security_master(
        membership_new,
        names_new,
        _read_optional_export(raw_root / "crsp_ccmxpf_linktable_sp500"),
        _read_optional_export(raw_root / "wrdsapps_ibcrsphist_sp500"),
        _read_optional_export(raw_root / "ciq_gvkey_sp500"),
    )
    events_raw = read_partitioned_parquet(raw_root / "ciq_transcript_events_holdout")
    components_raw = read_partitioned_parquet(
        raw_root / "ciq_transcript_components_holdout"
    )
    transcript_new, _ = build_ciq_transcript_text(events_raw, components_raw)
    ibes_new, _ = merge_summary_with_actuals(
        _read_holdout_ibes(raw_root, historical_data_root, "ibes_statsum_epsus_sp500", "statpers", ibes_history_year),
        _read_holdout_ibes(raw_root, historical_data_root, "ibes_act_epsus_sp500", "fpedats", ibes_history_year),
    )
    panel_new = build_ibes_event_dataset(transcript_new, ibes_new).merged_transcripts

    panel_old = pd.read_parquet(
        historical_data_root / "derived" / "transcript_ibes_panel.parquet"
    )
    components_old = pd.read_parquet(
        historical_data_root / "staging" / "ciq_transcript_components.parquet"
    )
    membership_old = pd.read_parquet(
        historical_data_root / "staging" / "universe_sp500_membership.parquet"
    )
    security_old = pd.read_parquet(
        historical_data_root / "staging" / "security_master.parquet"
    )
    old_daily = read_partitioned_parquet(
        historical_data_root / "raw" / "wrds_exports" / "crsp_dsf_sp500"
    )
    panel = _prefer_new_years(panel_old, panel_new, "call_date", cutoff="2024-01-01")
    new_ids = set(panel_new["transcriptid"].astype(str))
    components = pd.concat(
        [
            components_old.loc[
                ~components_old["transcriptid"].astype(str).isin(new_ids)
            ],
            components_raw,
        ],
        ignore_index=True,
        sort=False,
    )
    membership = _deduplicate_rows(
        pd.concat([membership_old, membership_new], ignore_index=True, sort=False)
    )
    security = _deduplicate_rows(
        pd.concat([security_old, security_new], ignore_index=True, sort=False)
    )
    daily = _prefer_new_years(old_daily, legacy_ciz, "date", cutoff="2024-01-01")

    earnings = filter_earnings_calls(panel, sample_end="2025-12-31")
    point_in_time = apply_point_in_time_sp500_filter(earnings, membership, security)
    # A new unified-panel fit has no earlier development regression to use for the bridge.
    if frozen_revision_dir is None:
        bridge_events = point_in_time.loc[pd.to_datetime(point_in_time["call_date"]).dt.year.eq(2024)]
        before_2025 = pd.Timestamp("2025-01-01")
        siz = build_crsp_event_outcomes(
            bridge_events, old_daily.loc[pd.to_datetime(old_daily["date"]).lt(before_2025)], horizons=HORIZONS
        )
        bridge_daily = legacy_ciz.loc[pd.to_datetime(legacy_ciz["date"]).lt(before_2025)]
    else:
        frozen_response = pd.read_parquet(frozen_revision_dir / "response_model_sample_revision_v2.parquet")
        bridge_events = frozen_response.loc[frozen_response["event_year"].eq(2024)]
        siz = bridge_events
        bridge_daily = legacy_ciz
    ciz = build_crsp_event_outcomes(bridge_events, bridge_daily, horizons=HORIZONS)
    bridge = bridge_metrics(siz[["transcriptid", "CAR_0_1"]], ciz[["transcriptid", "CAR_0_1"]])
    event_ids = set(point_in_time["transcriptid"].astype(str))
    components = components.loc[
        components["transcriptid"].astype(str).isin(event_ids)
    ].copy()
    dictionary = load_lm_dictionary(dictionary_dir or PROJECT_ROOT / "data" / "external")
    section_scores = score_lm_sections(point_in_time, dictionary)
    turns = build_qa_turns(components, point_in_time)
    turn_scores = score_qa_turns(turns, dictionary)
    pre_event = build_crsp_event_outcomes(
        point_in_time,
        daily,
        horizons=HORIZONS,
        include_post_event=False,
    )
    pre_event = pre_event.drop(
        columns=[
            column
            for column in pre_event.columns
            if str(column).lower().startswith(
                ("abnormal_volume_", "realized_volatility_")
            )
        ]
    )
    state = build_event_state(point_in_time, section_scores, pre_event)
    _assert_outcome_blind(state)
    frames, specs = build_first_stage_feature_frames(
        state, section_scores, turn_scores
    )
    frames, revision_specs = build_holdout_revision_feature_frames(frames, turns)
    specs.update(revision_specs)
    for frame in frames.values():
        _assert_outcome_blind(frame)

    qa_turns = turns[
        [
            "transcriptid",
            "turn_index",
            "call_date",
            "question_text",
            "answer_text",
        ]
    ].copy()
    if qa_turns.duplicated(["transcriptid", "turn_index"]).any():
        raise ValueError("Holdout Q&A turns contain duplicate transcript/turn keys")

    del components, section_scores, turn_scores, turns
    gc.collect()

    feature_dir = build_dir / "outcome_blind_features"
    feature_dir.mkdir(parents=True, exist_ok=False)
    for component, frame in frames.items():
        frame.to_parquet(
            feature_dir / f"{component}_first_stage_features.parquet", index=False
        )
    qa_turns.to_parquet(build_dir / "qa_turns.parquet", index=False)
    state.to_parquet(build_dir / "outcome_blind_event_state.parquet", index=False)
    point_in_time.to_parquet(build_dir / "point_in_time_events.parquet", index=False)
    legacy_ciz.to_parquet(build_dir / "ciz_legacy_daily.parquet", index=False)
    pd.DataFrame([bridge]).to_csv(build_dir / "ciz_siz_bridge_metrics.csv", index=False)
    (build_dir / "ciz_siz_bridge_metrics.json").write_text(
        json.dumps(bridge, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "contract": "finnlp-2025-holdout-build-v2",
        "outcome_blind": True,
        "training_years_present": sorted(
            pd.to_numeric(frames["prepared"]["event_year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "point_in_time_calls": int(len(point_in_time)),
        "holdout_calls": int(point_in_time["call_date"].astype(str).str[:4].eq("2025").sum()),
        "feature_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "feature_specs": specs,
        "bridge": bridge,
        "files": _records_under(build_dir),
    }
    (build_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if frozen_revision_dir is not None and not protocol_path.exists():
        freeze_holdout_protocol(
            frozen_revision_dir=frozen_revision_dir,
            protocol_path=protocol_path,
        )
    return manifest


def build_holdout_revision_feature_frames(
    frames: dict[str, pd.DataFrame],
    turns: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    """Upgrade baseline Q/A frames to the frozen Q2/A3 feature contracts."""

    required = {"prepared", "question", "answer"}
    if missing := sorted(required.difference(frames)):
        raise ValueError(f"Holdout feature frames missing components: {missing}")
    history_turns = add_strict_analyst_history(turns, frames["question"])
    enriched = enrich_revision_frames(
        frames["question"], frames["answer"], history_turns
    )
    feature_sets = {
        "question": question_filtration_features(enriched["question"])["Q2"],
        "answer": revision_answer_filtration_features(enriched["answer"])["A3"],
    }
    categorical = [
        *base_categorical_features(),
        *REVISION_CATEGORICAL_FEATURES,
    ]
    updated = dict(frames)
    specs: dict[str, dict[str, object]] = {}
    for component in ("question", "answer"):
        frame, spec = make_feature_frame(
            enriched[component],
            component=component,
            grain="turn",
            target_col="raw_score",
            feature_columns=feature_sets[component],
            categorical_columns=categorical,
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
        updated[component] = frame
        specs[component] = spec
    return updated, specs


def estimate_lm_holdout_answer_ladder(
    *,
    feature_path: Path = DEFAULT_FEATURE_DIR / "answer_first_stage_features.parquet",
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    output_dir: Path = DEFAULT_LM_HOLDOUT_LADDER_DIR,
    min_train_rows: int = 120,
) -> dict[str, object]:
    """Estimate the outcome-blind LM A0--A3 ladder on the 2025 test sample."""

    _require_empty(output_dir, "LM holdout answer ladder")
    protocol = _read_and_verify_protocol(protocol_path)
    frame = pd.read_parquet(feature_path)
    _assert_outcome_blind(frame)

    years = pd.to_numeric(frame["event_year"], errors="raise").astype(int)
    training_years = sorted(years.loc[years.lt(HOLDOUT_YEAR)].unique().tolist())
    if training_years != list(range(2018, HOLDOUT_YEAR)):
        raise ValueError(
            "LM holdout ladder requires uninterrupted 2018--2024 training history"
        )
    test = frame.loc[years.eq(HOLDOUT_YEAR)]
    if len(test) != 34_001 or test["transcriptid"].astype(str).nunique() != 1_954:
        raise ValueError(
            "LM holdout ladder must use the fixed 34,001-turn, 1,954-call test sample"
        )
    if test.duplicated(["transcriptid", "turn_index"]).any():
        raise ValueError("LM holdout ladder test rows are not unique by call and turn")

    feature_sets = revision_answer_filtration_features(frame)
    expected_features = {"A0": 29, "A1": 38, "A2": 42, "A3": 58}
    if set(feature_sets) != set(expected_features):
        raise ValueError("LM holdout ladder must contain exactly A0--A3")
    observed_features = {name: len(columns) for name, columns in feature_sets.items()}
    if observed_features != expected_features:
        raise ValueError(
            f"LM holdout ladder feature counts changed: {observed_features}"
        )

    learner = tuned_ridge_spec_with_alphas(RIDGE_ALPHAS)
    result = estimate_component_ladder(
        frame,
        component="answer",
        feature_sets=feature_sets,
        learner=learner,
        min_train_rows=min_train_rows,
        outer_years=(HOLDOUT_YEAR,),
    )
    first_stage = result["first_stage"].copy()
    if set(first_stage["variant"]) != set(expected_features) or len(first_stage) != 4:
        raise ValueError("LM holdout ladder did not return one row for each A0--A3 model")
    if set(first_stage["n_turns"].astype(int)) != {34_001}:
        raise ValueError("LM holdout ladder variants do not share 34,001 test turns")
    if set(first_stage["n_calls"].astype(int)) != {1_954}:
        raise ValueError("LM holdout ladder variants do not share 1,954 test calls")
    reported_features = dict(
        zip(first_stage["variant"], first_stage["n_features"], strict=True)
    )
    if reported_features != expected_features:
        raise ValueError("LM holdout ladder reported unexpected feature counts")

    selections = result["selections"].copy()
    selected = selections.loc[selections["selected"].astype(bool)]
    if len(selected) != 4 or set(selected["variant"]) != set(expected_features):
        raise ValueError("LM holdout ladder requires one selected model per variant")
    outer_year = pd.to_numeric(selected["outer_year"], errors="raise").astype(int)
    inner_max = pd.to_numeric(selected["max_inner_train_year"], errors="raise").astype(int)
    if not outer_year.eq(HOLDOUT_YEAR).all() or inner_max.ge(outer_year).any():
        raise ValueError("LM holdout ladder violated nested time-forward tuning")

    first_stage.insert(0, "scorer", "LM dictionary")
    first_stage.insert(1, "sample", "holdout_2025")
    first_stage["scope"] = "holdout_2025"
    fold_metrics = result["fold_metrics"].assign(scope="holdout_2025")
    selections = selections.assign(scope="holdout_2025")

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "first_stage": output_dir / "lm_holdout_answer_ladder.csv",
        "fold_metrics": output_dir / "lm_holdout_answer_fold_metrics.csv",
        "selections": output_dir / "lm_holdout_answer_hyperparameter_selection.csv",
        "turn_innovations": output_dir / "lm_holdout_answer_turn_innovations.parquet",
    }
    first_stage.to_csv(outputs["first_stage"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    selections.to_csv(outputs["selections"], index=False)
    result["turn_innovations"].to_parquet(outputs["turn_innovations"], index=False)

    def source_label(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    manifest = {
        "status": "complete",
        "contract": "icaif-lm-answer-holdout-ladder-v1",
        "outcomes_loaded": False,
        "sample": "holdout_2025",
        "test_year": HOLDOUT_YEAR,
        "test_turns": 34_001,
        "test_calls": 1_954,
        "training_years": training_years,
        "variants": list(feature_sets),
        "feature_counts": expected_features,
        "learner": "time_forward_tuned_ridge",
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "selection_rule": "inner expanding-year MAE",
        "nested_time_forward_tuning": True,
        "input": {
            "path": source_label(feature_path),
            "bytes": feature_path.stat().st_size,
            "sha256": _sha256(feature_path),
        },
        "protocol": {
            "path": source_label(protocol_path),
            "status": protocol["status"],
            "sha256": _sha256(protocol_path),
        },
        "outputs": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs.values()
        ],
    }
    (output_dir / "lm_holdout_answer_ladder_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def score_holdout_2025(
    *,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    residual_dir: Path = DEFAULT_RESIDUAL_DIR,
    min_train_rows: int = 120,
) -> dict[str, object]:
    """Score P0/Q2/A3 without loading any post-event outcome columns."""

    _require_empty(residual_dir, "holdout residual")
    protocol = _read_and_verify_protocol(protocol_path)
    frames = {
        component: pd.read_parquet(
            feature_dir / f"{component}_first_stage_features.parquet"
        )
        for component in ("prepared", "question", "answer")
    }
    for component, frame in frames.items():
        _assert_outcome_blind(frame)
        years = pd.to_numeric(frame["event_year"], errors="coerce")
        if not years.eq(HOLDOUT_YEAR).any() or years.min() > 2018:
            raise ValueError(
                f"{component} feature frame must contain 2018--2024 history and 2025 tests"
            )
    learner = tuned_ridge_spec_with_alphas(RIDGE_ALPHAS)
    prepared = estimate_component_ladder(
        frames["prepared"],
        component="prepared",
        feature_sets={"P0": list(BASE_STATE_FEATURES)},
        learner=learner,
        min_train_rows=min_train_rows,
        outer_years=(HOLDOUT_YEAR,),
    )
    question_features = question_filtration_features(frames["question"])["Q2"]
    question = estimate_component_ladder(
        frames["question"],
        component="question",
        feature_sets={"Q2": question_features},
        learner=learner,
        min_train_rows=min_train_rows,
        outer_years=(HOLDOUT_YEAR,),
    )
    answer_features = revision_answer_filtration_features(frames["answer"])["A3"]
    answer = estimate_component_ladder(
        frames["answer"],
        component="answer",
        feature_sets={"A3": answer_features},
        learner=learner,
        min_train_rows=min_train_rows,
        outer_years=(HOLDOUT_YEAR,),
    )
    metadata = frames["prepared"].loc[
        pd.to_numeric(frames["prepared"]["event_year"], errors="coerce").eq(HOLDOUT_YEAR),
        ["transcriptid", "ticker", "permno", "event_trading_date", "event_year"],
    ].copy()
    residuals = (
        metadata.merge(
            prepared["call_innovations"][["transcriptid", "prepared_P0"]],
            on="transcriptid",
            validate="one_to_one",
        )
        .merge(
            question["call_innovations"][["transcriptid", "question_Q2"]],
            on="transcriptid",
            validate="one_to_one",
        )
        .merge(
            answer["call_innovations"][["transcriptid", "answer_A3"]],
            on="transcriptid",
            validate="one_to_one",
        )
    )
    residuals["training_max_year"] = max(
        int(
            pd.to_numeric(frame["event_year"], errors="coerce")
            .loc[lambda values: values.lt(HOLDOUT_YEAR)]
            .max()
        )
        for frame in frames.values()
    )
    if residuals["training_max_year"].ge(HOLDOUT_YEAR).any():
        raise ValueError("Holdout first-stage training includes 2025")
    if residuals["transcriptid"].duplicated().any():
        raise ValueError("Holdout residuals are not unique by transcriptid")

    residual_dir.mkdir(parents=True, exist_ok=False)
    residual_path = residual_dir / "holdout_2025_residuals.parquet"
    residuals.to_parquet(residual_path, index=False)
    diagnostics = {
        "prepared": prepared,
        "question": question,
        "answer": answer,
    }
    diagnostic_files = []
    for component, result in diagnostics.items():
        for name in ("first_stage", "fold_metrics", "selections"):
            path = residual_dir / f"{component}_{name}.csv"
            result[name].to_csv(path, index=False)
            diagnostic_files.append(path)
    manifest = {
        "status": "residuals_frozen",
        "contract": "finnlp-2025-holdout-residual-freeze-v1",
        "protocol_sha256": _sha256(protocol_path),
        "protocol_status": protocol["status"],
        "outcome_columns_loaded": False,
        "test_year": HOLDOUT_YEAR,
        "training_max_year": int(residuals["training_max_year"].max()),
        "test_calls": int(len(residuals)),
        "residual_file": _file_record(residual_path, residual_dir),
        "diagnostic_files": [
            _file_record(path, residual_dir) for path in diagnostic_files
        ],
    }
    (residual_dir / "residual_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def estimate_holdout_response(
    *,
    build_dir: Path = DEFAULT_BUILD_DIR,
    residual_dir: Path = DEFAULT_RESIDUAL_DIR,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    response_dir: Path = DEFAULT_RESPONSE_DIR,
    frozen_revision_dir: Path = DEFAULT_FROZEN_REVISION,
) -> dict[str, object]:
    """Verify the residual freeze, then and only then load 2025 outcomes."""

    _require_empty(response_dir, "holdout response")
    protocol = _read_and_verify_protocol(protocol_path)
    residuals = _read_verified_residuals(residual_dir, protocol_path)
    state = pd.read_parquet(build_dir / "outcome_blind_event_state.parquet")
    _assert_outcome_blind(state)

    # Post-event returns are first loaded after both checks above have passed.
    daily = pd.read_parquet(build_dir / "ciz_legacy_daily.parquet")
    events = pd.read_parquet(build_dir / "point_in_time_events.parquet")
    events = events.loc[
        pd.to_datetime(events["call_date"], errors="coerce").dt.year.eq(HOLDOUT_YEAR)
    ].copy()
    outcomes = build_crsp_event_outcomes(events, daily, horizons=HORIZONS)
    outcome_columns = ["transcriptid", *[f"CAR_0_{horizon}" for horizon in HORIZONS]]
    outcome_merged = (
        state.loc[pd.to_numeric(state["event_year"], errors="coerce").eq(HOLDOUT_YEAR)]
        .merge(outcomes[outcome_columns], on="transcriptid", validate="one_to_one")
        .merge(residuals, on="transcriptid", validate="one_to_one", suffixes=("", "_residual"))
    )
    outcome_merged["log_market_cap"] = np.log(
        pd.to_numeric(outcome_merged["market_cap"], errors="coerce").clip(lower=1.0)
    )
    outcome_merged = _apply_training_standardization(outcome_merged, protocol)
    controls = _frozen_response_controls(outcome_merged, protocol)
    sample, missingness = _complete_holdout_response_sample(
        outcome_merged,
        controls=controls,
    )
    attrition = pd.DataFrame(
        [
            {
                "stage": "frozen_2025_residuals",
                "calls": int(len(residuals)),
                "dropped_from_previous": 0,
                "reason": "outcome-blind P0/Q2/A3 residuals",
            },
            {
                "stage": "post_event_outcomes_merged",
                "calls": int(len(outcome_merged)),
                "dropped_from_previous": int(len(residuals) - len(outcome_merged)),
                "reason": "matched CIZ event-outcome row",
            },
            {
                "stage": "protocol_complete_response_sample",
                "calls": int(len(sample)),
                "dropped_from_previous": int(len(outcome_merged) - len(sample)),
                "reason": "complete frozen treatments, controls, outcomes, FE, and clusters",
            },
        ]
    )
    primary = fit_joint_horizon_system(
        sample,
        outcomes=["CAR_0_1"],
        treatments=STANDARDIZED_TREATMENTS,
        controls=controls,
    )
    secondary = fit_joint_horizon_system(
        sample,
        outcomes=[f"CAR_0_{horizon}" for horizon in HORIZONS],
        treatments=STANDARDIZED_TREATMENTS,
        controls=controls,
    )
    if primary.n_obs != len(sample) or secondary.n_obs != len(sample):
        raise ValueError("Holdout response regression changed the protocol-complete sample")
    placebo_cells, placebo_joint, placebo_controls = _holdout_pre_event_placebos(sample)
    bridge = json.loads(
        (build_dir / "ciz_siz_bridge_metrics.json").read_text(encoding="utf-8")
    )
    interaction = _pooled_holdout_interaction(
        sample,
        frozen_revision_dir=frozen_revision_dir,
        allowed=bool(bridge["pooled_comparison_allowed"]),
        controls=controls,
    )
    response_dir.mkdir(parents=True, exist_ok=False)
    sample.to_parquet(response_dir / "holdout_2025_response_sample.parquet", index=False)
    primary.coefficients.to_csv(
        response_dir / "holdout_2025_primary_coefficients.csv", index=False
    )
    secondary.coefficients.to_csv(
        response_dir / "holdout_2025_secondary_horizons.csv", index=False
    )
    interaction.to_csv(response_dir / "holdout_2025_pooled_interaction.csv", index=False)
    placebo_cells.to_csv(
        response_dir / "holdout_2025_pre_event_placebos.csv", index=False
    )
    placebo_joint.to_csv(
        response_dir / "holdout_2025_pre_event_joint_tests.csv", index=False
    )
    attrition.to_csv(response_dir / "holdout_2025_response_attrition.csv", index=False)
    missingness.to_csv(
        response_dir / "holdout_2025_response_missingness.csv", index=False
    )
    np.save(response_dir / "holdout_2025_primary_covariance.npy", primary.covariance)
    primary_answer = primary.coefficients.loc[
        primary.coefficients["treatment"].eq("answer_innovation_z")
    ].iloc[0]
    manifest = {
        "status": "complete",
        "contract": "finnlp-2025-holdout-response-v2",
        "protocol_sha256": _sha256(protocol_path),
        "residual_manifest_sha256": _sha256(
            residual_dir / "residual_freeze_manifest.json"
        ),
        "residual_checksum_verified_before_outcome_load": True,
        "control_contract_verified": True,
        "controls": controls,
        "fixed_effects": ["permno", "event_calendar_quarter"],
        "clustering": ["permno", "event_trading_date"],
        "log_market_cap_definition": "log(max(market_cap, 1.0))",
        "residual_calls": int(len(residuals)),
        "outcome_merged_calls": int(len(outcome_merged)),
        "test_calls": int(primary.n_obs),
        "primary_beta": float(primary_answer["beta"]),
        "primary_standard_error": float(primary_answer["standard_error"]),
        "primary_ci_low": float(
            primary_answer["beta"] - 1.96 * primary_answer["standard_error"]
        ),
        "primary_ci_high": float(
            primary_answer["beta"] + 1.96 * primary_answer["standard_error"]
        ),
        "bridge_gate_passed": bool(bridge["comparability_gate_passed"]),
        "pooled_interaction_estimated": bool(bridge["pooled_comparison_allowed"]),
        "pre_event_placebo_reported": True,
        "pre_event_outcomes": list(NON_OVERLAP_OUTCOMES),
        "pre_event_controls": placebo_controls,
        "results_reported_without_significance_filter": True,
    }
    (response_dir / "holdout_response_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def run_holdout_bootstrap(
    *,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    response_dir: Path = DEFAULT_RESPONSE_DIR,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    output_dir: Path = DEFAULT_BOOTSTRAP_DIR,
    replicates: int = 200,
    seed: int = 20260731,
    min_train_rows: int = 120,
    mode: str = "all",
    replicate_start: int | None = None,
    replicate_stop: int | None = None,
) -> dict[str, object]:
    """Run checkpointed firm-history refits, optionally as Slurm-safe shards."""

    if replicates < 2:
        raise ValueError("Holdout bootstrap needs at least two replicates")
    if mode not in {"all", "initialize", "worker", "finalize"}:
        raise ValueError(f"Unknown holdout bootstrap mode: {mode}")
    if mode in {"all", "initialize"}:
        plan = _initialize_holdout_bootstrap(
            feature_dir=feature_dir,
            response_dir=response_dir,
            protocol_path=protocol_path,
            output_dir=output_dir,
            replicates=replicates,
            seed=seed,
        )
    else:
        plan = _verify_holdout_bootstrap_plan(
            feature_dir=feature_dir,
            response_dir=response_dir,
            protocol_path=protocol_path,
            output_dir=output_dir,
            replicates=replicates,
            seed=seed,
        )
    if mode == "initialize":
        return plan
    if mode == "worker":
        if replicate_start is None or replicate_stop is None:
            raise ValueError("Worker mode requires replicate_start and replicate_stop")
        return _run_holdout_bootstrap_draws(
            feature_dir=feature_dir,
            response_dir=response_dir,
            protocol_path=protocol_path,
            output_dir=output_dir,
            replicates=replicates,
            seed=seed,
            min_train_rows=min_train_rows,
            replicate_start=replicate_start,
            replicate_stop=replicate_stop,
        )
    if mode == "all":
        _run_holdout_bootstrap_draws(
            feature_dir=feature_dir,
            response_dir=response_dir,
            protocol_path=protocol_path,
            output_dir=output_dir,
            replicates=replicates,
            seed=seed,
            min_train_rows=min_train_rows,
            replicate_start=0,
            replicate_stop=replicates,
        )
    return _finalize_holdout_bootstrap(
        feature_dir=feature_dir,
        response_dir=response_dir,
        protocol_path=protocol_path,
        output_dir=output_dir,
        replicates=replicates,
        seed=seed,
    )


def _initialize_holdout_bootstrap(
    *,
    feature_dir: Path,
    response_dir: Path,
    protocol_path: Path,
    output_dir: Path,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Freeze bootstrap input identity before any checkpoint is computed."""

    protocol = _read_and_verify_protocol(protocol_path)
    response_manifest_path = response_dir / "holdout_response_manifest.json"
    response_manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
    if (
        response_manifest.get("status") != "complete"
        or response_manifest.get("contract") != "finnlp-2025-holdout-response-v2"
        or response_manifest.get("control_contract_verified") is not True
    ):
        raise ValueError("Holdout response must be complete before bootstrap")
    feature_paths = {
        component: feature_dir / f"{component}_first_stage_features.parquet"
        for component in ("prepared", "question", "answer")
    }
    for path in [*feature_paths.values(), protocol_path, response_manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    response_sample_path = response_dir / "holdout_2025_response_sample.parquet"
    if not response_sample_path.is_file():
        raise FileNotFoundError(response_sample_path)
    identity = {
        "contract": "finnlp-2025-holdout-bootstrap-plan-v2",
        "replicates": int(replicates),
        "seed": int(seed),
        "protocol_sha256": _sha256(protocol_path),
        "protocol_status": protocol["status"],
        "controls": _frozen_response_controls(
            pd.read_parquet(response_sample_path), protocol
        ),
        "response_manifest_sha256": _sha256(response_manifest_path),
        "response_sample_bytes": response_sample_path.stat().st_size,
        "response_sample_sha256": _sha256(response_sample_path),
        "feature_files": {
            component: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for component, path in feature_paths.items()
        },
    }
    plan = {
        "status": "initialized",
        **identity,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "draws").mkdir(parents=True, exist_ok=True)
    (output_dir / "workers").mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "bootstrap_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        mismatches = [
            key for key, value in identity.items() if existing.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Existing holdout bootstrap plan has different inputs: "
                + ", ".join(mismatches)
            )
        return existing
    orphaned = [
        path
        for directory in (output_dir / "draws", output_dir / "workers")
        for path in directory.iterdir()
    ]
    if orphaned:
        raise FileExistsError(
            "Bootstrap checkpoints exist without a frozen bootstrap_plan.json"
        )
    _write_json_atomic(plan, plan_path)
    return plan


def _verify_holdout_bootstrap_plan(
    *,
    feature_dir: Path,
    response_dir: Path,
    protocol_path: Path,
    output_dir: Path,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Verify a frozen plan without rehashing large feature files per worker."""

    plan_path = output_dir / "bootstrap_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = {
        "status": "initialized",
        "contract": "finnlp-2025-holdout-bootstrap-plan-v2",
        "replicates": int(replicates),
        "seed": int(seed),
        "protocol_sha256": _sha256(protocol_path),
        "response_manifest_sha256": _sha256(
            response_dir / "holdout_response_manifest.json"
        ),
    }
    mismatches = [key for key, value in expected.items() if plan.get(key) != value]
    response_sample_path = response_dir / "holdout_2025_response_sample.parquet"
    if response_sample_path.stat().st_size != plan.get("response_sample_bytes"):
        mismatches.append("response_sample_bytes")
    if _sha256(response_sample_path) != plan.get("response_sample_sha256"):
        mismatches.append("response_sample_sha256")
    feature_records = plan.get("feature_files", {})
    for component in ("prepared", "question", "answer"):
        path = feature_dir / f"{component}_first_stage_features.parquet"
        record = feature_records.get(component, {})
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            mismatches.append(f"feature_files.{component}.bytes")
    if mismatches:
        raise ValueError(
            "Holdout bootstrap plan verification failed: "
            + ", ".join(sorted(set(mismatches)))
        )
    return plan


def _run_holdout_bootstrap_draws(
    *,
    feature_dir: Path,
    response_dir: Path,
    protocol_path: Path,
    output_dir: Path,
    replicates: int,
    seed: int,
    min_train_rows: int,
    replicate_start: int,
    replicate_stop: int,
) -> dict[str, object]:
    """Compute one non-overlapping range of bootstrap checkpoints."""

    if not 0 <= replicate_start < replicate_stop <= replicates:
        raise ValueError(
            "Bootstrap worker range must satisfy "
            f"0 <= start < stop <= {replicates}; got "
            f"{replicate_start}:{replicate_stop}"
        )
    protocol = _read_and_verify_protocol(protocol_path)
    frames = {
        component: pd.read_parquet(
            feature_dir / f"{component}_first_stage_features.parquet"
        )
        for component in ("prepared", "question", "answer")
    }
    for frame in frames.values():
        _assert_outcome_blind(frame)
    fixed_response = pd.read_parquet(
        response_dir / "holdout_2025_response_sample.parquet"
    )
    controls = _frozen_response_controls(fixed_response, protocol)
    history_firms = sorted(
        pd.to_numeric(
            frames["prepared"].loc[
                pd.to_numeric(frames["prepared"]["event_year"], errors="coerce").lt(
                    HOLDOUT_YEAR
                ),
                "permno",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    draw_dir = output_dir / "draws"
    categorical = [*base_categorical_features(), *REVISION_CATEGORICAL_FEATURES]
    completed = []
    for replicate in range(replicate_start, replicate_stop):
        path = draw_dir / f"replicate_{replicate:04d}.csv"
        check_path = draw_dir / f"fold_checks_{replicate:04d}.csv"
        if path.exists() and check_path.exists():
            _read_validated_bootstrap_checkpoint(path, check_path, replicate)
            completed.append(replicate)
            continue
        if path.exists() != check_path.exists():
            raise ValueError(
                f"Incomplete bootstrap checkpoint pair for replicate {replicate}"
            )
        multiplicity = draw_firm_multiplicity(
            history_firms, np.random.default_rng(seed + replicate)
        )
        calls = {}
        checks = []
        specifications = {
            "prepared": ("P0", list(BASE_STATE_FEATURES)),
            "question": (
                "Q2",
                question_filtration_features(frames["question"])["Q2"],
            ),
            "answer": (
                "A3",
                revision_answer_filtration_features(frames["answer"])["A3"],
            ),
        }
        for component, (variant, features) in specifications.items():
            prediction, fold, check = tuned_bootstrap_oof_predict(
                frames[component],
                component=component,
                variant=variant,
                feature_columns=features,
                categorical_columns=[
                    name for name in categorical if name in features
                ],
                multiplicity=multiplicity,
                ridge_alphas=RIDGE_ALPHAS,
                min_train_rows=min_train_rows,
                outer_years=(HOLDOUT_YEAR,),
            )
            checks.append(check)
            if component == "prepared":
                call = prepared_call_residuals(
                    frames[component], prediction, fold
                ).rename(columns={"prepared_innovation": "prepared_P0"})
            else:
                call = turn_call_residuals(
                    frames[component], prediction, fold, component=component
                ).rename(
                    columns={
                        "narrative_innovation": (
                            "question_Q2" if component == "question" else "answer_A3"
                        )
                    }
                )
            calls[component] = call
        draw_sample = fixed_response.drop(
            columns=[*RAW_TREATMENTS.values(), *STANDARDIZED_TREATMENTS],
            errors="ignore",
        )
        for component, raw_column in (
            ("prepared", "prepared_P0"),
            ("question", "question_Q2"),
            ("answer", "answer_A3"),
        ):
            draw_sample = draw_sample.merge(
                calls[component][["transcriptid", raw_column]],
                on="transcriptid",
                validate="one_to_one",
            )
        draw_sample = _apply_training_standardization(draw_sample, protocol)
        system = fit_joint_horizon_system(
            draw_sample,
            outcomes=["CAR_0_1"],
            treatments=STANDARDIZED_TREATMENTS,
            controls=controls,
        )
        if system.n_obs != len(fixed_response):
            raise ValueError(
                f"Bootstrap replicate {replicate} changed the fixed 2025 test sample"
            )
        draw = system.coefficients[
            ["position", "outcome", "horizon", "treatment", "beta"]
        ].copy()
        draw.insert(0, "replicate", replicate)
        fold_checks = pd.concat(checks, ignore_index=True)
        fold_checks.insert(0, "replicate", replicate)
        _write_csv_atomic(fold_checks, check_path)
        _write_csv_atomic(draw, path)
        _read_validated_bootstrap_checkpoint(path, check_path, replicate)
        completed.append(replicate)

    summary = {
        "status": "complete",
        "contract": "finnlp-2025-holdout-bootstrap-worker-v2",
        "replicate_start": int(replicate_start),
        "replicate_stop": int(replicate_stop),
        "completed_replicates": completed,
        "protocol_sha256": _sha256(protocol_path),
    }
    summary_path = (
        output_dir
        / "workers"
        / f"worker_{replicate_start:04d}_{replicate_stop:04d}.json"
    )
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise ValueError(
                "Existing bootstrap worker summary does not match validated checkpoints"
            )
        return existing
    _write_json_atomic(summary, summary_path)
    return summary


def _finalize_holdout_bootstrap(
    *,
    feature_dir: Path,
    response_dir: Path,
    protocol_path: Path,
    output_dir: Path,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Combine a complete checkpoint set and compute adjusted inference."""

    protocol = _read_and_verify_protocol(protocol_path)
    plan = _verify_holdout_bootstrap_plan(
        feature_dir=feature_dir,
        response_dir=response_dir,
        protocol_path=protocol_path,
        output_dir=output_dir,
        replicates=replicates,
        seed=seed,
    )
    plan_path = output_dir / "bootstrap_plan.json"
    draw_dir = output_dir / "draws"
    missing = [
        replicate
        for replicate in range(replicates)
        if not (draw_dir / f"replicate_{replicate:04d}.csv").is_file()
        or not (draw_dir / f"fold_checks_{replicate:04d}.csv").is_file()
    ]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(f"Missing bootstrap checkpoints: {preview}")

    draws = pd.concat(
        [
            _read_validated_bootstrap_checkpoint(
                draw_dir / f"replicate_{replicate:04d}.csv",
                draw_dir / f"fold_checks_{replicate:04d}.csv",
                replicate,
            )
            for replicate in range(replicates)
        ],
        ignore_index=True,
    )
    coefficient_index = (
        draws[["position", "outcome", "horizon", "treatment"]]
        .drop_duplicates()
        .sort_values("position")
    )
    matrix = (
        draws.pivot(index="replicate", columns="position", values="beta")
        .reindex(index=range(replicates), columns=coefficient_index["position"])
        .to_numpy(dtype=float)
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Holdout bootstrap draw matrix is incomplete")
    fixed_response = pd.read_parquet(
        response_dir / "holdout_2025_response_sample.parquet"
    )
    controls = _frozen_response_controls(fixed_response, protocol)
    first_stage_covariance = np.cov(matrix, rowvar=False, ddof=1)
    conditional = fit_joint_horizon_system(
        fixed_response,
        outcomes=["CAR_0_1"],
        treatments=STANDARDIZED_TREATMENTS,
        controls=controls,
    )
    if conditional.n_obs != len(fixed_response):
        raise ValueError("Bootstrap finalization changed the fixed 2025 test sample")
    adjusted, total_covariance = adjusted_inference(
        conditional, first_stage_covariance
    )
    convergence = _holdout_bootstrap_convergence(conditional, matrix)
    _write_csv_atomic(draws, output_dir / "bootstrap_draws.csv")
    _write_csv_atomic(
        adjusted,
        output_dir / "holdout_2025_bootstrap_adjusted_coefficients.csv",
    )
    _write_npy_atomic(
        first_stage_covariance, output_dir / "first_stage_refit_covariance.npy"
    )
    _write_npy_atomic(total_covariance, output_dir / "response_total_covariance.npy")
    _write_csv_atomic(
        convergence,
        output_dir / "holdout_2025_bootstrap_convergence.csv",
    )
    primary = adjusted.loc[
        adjusted["treatment"].eq("answer_innovation_z")
    ].iloc[0]
    summary = {
        "status": "complete",
        "contract": "finnlp-2025-holdout-bootstrap-v2",
        "valid_replicates": int(draws["replicate"].nunique()),
        "requested_replicates": int(replicates),
        "fixed_test_calls": int(conditional.n_obs),
        "within_history_retuning": True,
        "training_years_strictly_before_2025": True,
        "control_contract_verified": True,
        "controls": controls,
        "convergence_draw_counts": list(BOOTSTRAP_CONVERGENCE_DRAWS),
        "primary_beta": float(primary["beta"]),
        "primary_adjusted_standard_error": float(
            primary["bootstrap_adjusted_standard_error"]
        ),
        "primary_adjusted_ci_low": float(primary["bootstrap_adjusted_ci_low"]),
        "primary_adjusted_ci_high": float(primary["bootstrap_adjusted_ci_high"]),
        "plan_sha256": _sha256(plan_path),
    }
    _write_json_atomic(summary, output_dir / "bootstrap_manifest.json")
    return summary


def _holdout_bootstrap_convergence(
    conditional: JointSystemResult,
    draw_matrix: np.ndarray,
) -> pd.DataFrame:
    """Recompute A3 adjusted inference on nested deterministic draw prefixes."""

    if draw_matrix.ndim != 2 or draw_matrix.shape[1] != len(
        conditional.coefficients
    ):
        raise ValueError("Holdout bootstrap convergence draw matrix has wrong shape")
    if draw_matrix.shape[0] < max(BOOTSTRAP_CONVERGENCE_DRAWS):
        raise ValueError("Holdout bootstrap convergence needs all 200 draws")
    answer_rows = conditional.coefficients.loc[
        conditional.coefficients["treatment"].eq("answer_innovation_z")
    ]
    if len(answer_rows) != 1:
        raise ValueError("Holdout bootstrap convergence needs one A3 coefficient")
    answer_position = int(answer_rows.iloc[0]["position"])
    rows = []
    for draw_count in BOOTSTRAP_CONVERGENCE_DRAWS:
        first_stage_covariance = np.cov(
            draw_matrix[:draw_count], rowvar=False, ddof=1
        )
        adjusted, _ = adjusted_inference(conditional, first_stage_covariance)
        answer = adjusted.loc[
            adjusted["treatment"].eq("answer_innovation_z")
        ].iloc[0]
        rows.append(
            {
                "draws": int(draw_count),
                "treatment": "answer_innovation_z",
                "beta": float(answer["beta"]),
                "conditional_standard_error": float(
                    answer["conditional_standard_error"]
                ),
                "first_stage_standard_deviation": float(
                    np.sqrt(
                        max(
                            float(
                                first_stage_covariance[
                                    answer_position, answer_position
                                ]
                            ),
                            0.0,
                        )
                    )
                ),
                "bootstrap_adjusted_standard_error": float(
                    answer["bootstrap_adjusted_standard_error"]
                ),
                "bootstrap_adjusted_ci_low": float(
                    answer["bootstrap_adjusted_ci_low"]
                ),
                "bootstrap_adjusted_ci_high": float(
                    answer["bootstrap_adjusted_ci_high"]
                ),
            }
        )
    return pd.DataFrame(rows)


def freeze_holdout_2025(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    output_dir: Path = DEFAULT_FREEZE_DIR,
) -> dict[str, object]:
    """Freeze compact holdout evidence only after all gates complete."""

    _require_empty(output_dir, "holdout freeze")
    build = data_root / "build"
    residual = data_root / "analysis" / "residual_freeze"
    response = data_root / "analysis" / "response"
    bootstrap = data_root / "analysis" / "bootstrap"
    required = [
        (protocol_path, Path("protocol/holdout_protocol.json")),
        (protocol_path.with_suffix(".sha256"), Path("protocol/holdout_protocol.sha256")),
        (build / "build_manifest.json", Path("build/build_manifest.json")),
        (
            build / "ciz_siz_bridge_metrics.csv",
            Path("build/ciz_siz_bridge_metrics.csv"),
        ),
        (
            build / "ciz_siz_bridge_metrics.json",
            Path("build/ciz_siz_bridge_metrics.json"),
        ),
        (
            residual / "residual_freeze_manifest.json",
            Path("analysis/residual_freeze/residual_freeze_manifest.json"),
        ),
        (
            response / "holdout_response_manifest.json",
            Path("analysis/response/holdout_response_manifest.json"),
        ),
        (
            response / "holdout_2025_primary_coefficients.csv",
            Path("analysis/response/holdout_2025_primary_coefficients.csv"),
        ),
        (
            response / "holdout_2025_secondary_horizons.csv",
            Path("analysis/response/holdout_2025_secondary_horizons.csv"),
        ),
        (
            response / "holdout_2025_pooled_interaction.csv",
            Path("analysis/response/holdout_2025_pooled_interaction.csv"),
        ),
        (
            response / "holdout_2025_pre_event_placebos.csv",
            Path("analysis/response/holdout_2025_pre_event_placebos.csv"),
        ),
        (
            response / "holdout_2025_pre_event_joint_tests.csv",
            Path("analysis/response/holdout_2025_pre_event_joint_tests.csv"),
        ),
        (
            response / "holdout_2025_response_attrition.csv",
            Path("analysis/response/holdout_2025_response_attrition.csv"),
        ),
        (
            response / "holdout_2025_response_missingness.csv",
            Path("analysis/response/holdout_2025_response_missingness.csv"),
        ),
        (
            response / "holdout_2025_primary_covariance.npy",
            Path("analysis/response/holdout_2025_primary_covariance.npy"),
        ),
        (
            bootstrap / "bootstrap_manifest.json",
            Path("analysis/bootstrap/bootstrap_manifest.json"),
        ),
        (
            bootstrap / "holdout_2025_bootstrap_adjusted_coefficients.csv",
            Path("analysis/bootstrap/holdout_2025_bootstrap_adjusted_coefficients.csv"),
        ),
        (
            bootstrap / "first_stage_refit_covariance.npy",
            Path("analysis/bootstrap/first_stage_refit_covariance.npy"),
        ),
        (
            bootstrap / "response_total_covariance.npy",
            Path("analysis/bootstrap/response_total_covariance.npy"),
        ),
        (
            bootstrap / "holdout_2025_bootstrap_convergence.csv",
            Path("analysis/bootstrap/holdout_2025_bootstrap_convergence.csv"),
        ),
    ]
    for source, _ in required:
        if not source.exists():
            raise FileNotFoundError(source)
    bootstrap_manifest = json.loads(
        (bootstrap / "bootstrap_manifest.json").read_text(encoding="utf-8")
    )
    if (
        bootstrap_manifest.get("status") != "complete"
        or bootstrap_manifest.get("contract")
        != "finnlp-2025-holdout-bootstrap-v2"
        or bootstrap_manifest.get("control_contract_verified") is not True
        or int(bootstrap_manifest.get("valid_replicates", 0)) != 200
    ):
        raise ValueError("Holdout freeze requires 200 complete bootstrap refits")
    output_dir.mkdir(parents=True, exist_ok=False)
    copied = []
    for source, relative in required:
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(_file_record(destination, output_dir))
    manifest = {
        "status": "frozen",
        "contract": "finnlp-2025-holdout-freeze-v2",
        "data_root": str(data_root.resolve()),
        "files": copied,
    }
    (output_dir / "holdout_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def archive_holdout_response_revision(
    *,
    data_root: Path,
    archive_label: str,
    reason: str,
) -> dict[str, object]:
    """Move superseded response/bootstrap artifacts into a recoverable archive."""

    if not archive_label or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in archive_label
    ):
        raise ValueError("Archive label may contain only letters, numbers, dot, dash, underscore")
    if not reason.strip():
        raise ValueError("Archive reason must be non-empty")
    data_root = data_root.resolve()
    response = data_root / "analysis" / "response"
    bootstrap = data_root / "analysis" / "bootstrap"
    response_marker = data_root / "markers" / "_SUCCESS_response"
    required = [response, bootstrap, response_marker]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError(
            "Cannot archive an incomplete accepted response revision: " + ", ".join(missing)
        )
    archive_root = data_root / "archive"
    destination = archive_root / archive_label
    temporary = archive_root / f".{archive_label}.incomplete"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")

    source_records = []
    for path in sorted(response.rglob("*")) + sorted(bootstrap.rglob("*")):
        if path.is_file():
            source_records.append(_file_record(path, data_root))
    source_records.append(_file_record(response_marker, data_root))
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        (temporary / "analysis").mkdir()
        (temporary / "markers").mkdir()
        response.replace(temporary / "analysis" / "response")
        bootstrap.replace(temporary / "analysis" / "bootstrap")
        response_marker.replace(temporary / "markers" / response_marker.name)
        manifest = {
            "status": "archived_superseded_response",
            "contract": "finnlp-2025-holdout-response-archive-v1",
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_label": archive_label,
            "reason": reason.strip(),
            "source_data_root": str(data_root),
            "files_before_archive": source_records,
        }
        _write_json_atomic(manifest, temporary / "archive_manifest.json")
        temporary.replace(destination)
    except Exception:
        raise RuntimeError(
            "Response archive did not complete. Inspect and recover the preserved "
            f"temporary archive at {temporary} before retrying."
        )
    return manifest


def _frozen_response_controls(
    sample: pd.DataFrame, protocol: dict[str, object]
) -> list[str]:
    """Require the response design to match the frozen control list exactly."""

    expected = protocol.get("controls")
    if not isinstance(expected, list) or not expected or not all(
        isinstance(column, str) and column for column in expected
    ):
        raise ValueError("Frozen holdout protocol has an invalid controls contract")
    observed = response_controls(sample)
    if observed != expected:
        missing = [column for column in expected if column not in observed]
        unexpected = [column for column in observed if column not in expected]
        raise ValueError(
            "Holdout response controls do not match the frozen protocol: "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected_order={expected}, observed_order={observed}"
        )
    if protocol.get("fixed_effects") != ["permno", "event_calendar_quarter"]:
        raise ValueError("Frozen holdout fixed-effects contract has changed")
    if protocol.get("clustering") != ["permno", "event_trading_date"]:
        raise ValueError("Frozen holdout clustering contract has changed")
    return list(expected)


def _complete_holdout_response_sample(
    sample: pd.DataFrame, *, controls: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one explicit complete-case restriction for every response analysis."""

    required = list(
        dict.fromkeys(
            [
                *[f"CAR_0_{horizon}" for horizon in HORIZONS],
                *STANDARDIZED_TREATMENTS,
                *controls,
                "transcriptid",
                "permno",
                "event_calendar_quarter",
                "event_trading_date",
            ]
        )
    )
    missing_columns = [column for column in required if column not in sample]
    if missing_columns:
        raise ValueError(
            f"Holdout response sample is missing required columns: {missing_columns}"
        )
    work = sample.copy()
    numeric_columns = work.select_dtypes(include=[np.number]).columns
    work.loc[:, numeric_columns] = work.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    missingness = pd.DataFrame(
        [
            {
                "column": column,
                "missing_calls_before_complete_case": int(work[column].isna().sum()),
            }
            for column in required
        ]
    )
    complete = work.dropna(subset=required).reset_index(drop=True)
    if complete.empty:
        raise ValueError("Frozen holdout response complete-case sample is empty")
    if complete["transcriptid"].duplicated().any():
        raise ValueError("Frozen holdout response sample has duplicate transcript ids")
    return complete, missingness


def _holdout_pre_event_placebos(
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Estimate the frozen compact two-window pre-event audit."""

    prepared = prepare_pre_event_sample(sample)
    controls = safe_pre_event_controls(prepared)
    all_cells = add_full_sample_multiplicity(
        estimate_pre_event_cells(prepared, sample_label="full")
    )
    placebos = all_cells.loc[
        all_cells["outcome"].isin(NON_OVERLAP_OUTCOMES)
    ].reset_index(drop=True)
    joint = joint_pre_period_tests(all_cells).reset_index(drop=True)
    expected_cells = len(NON_OVERLAP_OUTCOMES) * len(STANDARDIZED_TREATMENTS)
    if len(placebos) != expected_cells or len(joint) != len(STANDARDIZED_TREATMENTS):
        raise ValueError("Holdout pre-event placebo output is incomplete")
    if not placebos["n_obs"].eq(len(sample)).all() or not joint["n_obs"].eq(
        len(sample)
    ).all():
        raise ValueError("Holdout pre-event placebo changed the response sample")
    return placebos, joint, controls


def _apply_training_standardization(
    sample: pd.DataFrame, protocol: dict[str, object]
) -> pd.DataFrame:
    result = sample.copy()
    scalers = protocol["training_standardization"]
    for raw, residual in RAW_TREATMENTS.items():
        standardized = f"{raw}_z"
        scaler = scalers[standardized]
        if scaler.get("raw_column") != raw:
            raise ValueError(
                f"Frozen training scaler raw column mismatch for {standardized}"
            )
        mean = float(scaler["mean_2020_2024"])
        sd = float(scaler["sd_2020_2024"])
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid frozen training scaler for {standardized}")
        result[standardized] = (
            pd.to_numeric(result[residual], errors="coerce") - mean
        ) / sd
    return result


def _pooled_holdout_interaction(
    holdout: pd.DataFrame,
    *,
    frozen_revision_dir: Path,
    allowed: bool,
    controls: Sequence[str],
) -> pd.DataFrame:
    if not allowed:
        return pd.DataFrame(
            [
                {
                    "status": "not_estimated_bridge_gate_failed",
                    "treatment": "answer_innovation_z_x_holdout_2025",
                }
            ]
        )
    historical = pd.read_parquet(
        frozen_revision_dir / "response_model_sample_revision_v2.parquet"
    )
    historical = historical.copy()
    historical["holdout_2025"] = 0.0
    current = holdout.copy()
    current["holdout_2025"] = 1.0
    common = sorted(set(historical.columns).intersection(current.columns))
    pooled = pd.concat([historical[common], current[common]], ignore_index=True)
    observed_controls = response_controls(pooled)
    if observed_controls != list(controls):
        raise ValueError(
            "Pooled holdout interaction controls do not match the frozen response: "
            f"expected={list(controls)}, observed={observed_controls}"
        )
    interactions = []
    for treatment in STANDARDIZED_TREATMENTS:
        name = f"{treatment}_x_holdout_2025"
        pooled[name] = pooled[treatment] * pooled["holdout_2025"]
        interactions.append(name)
    estimate = fit_joint_horizon_system(
        pooled,
        outcomes=["CAR_0_1"],
        treatments=[*STANDARDIZED_TREATMENTS, *interactions],
        controls=[*controls, "holdout_2025"],
    )
    result = estimate.coefficients.loc[
        estimate.coefficients["treatment"].isin(interactions)
    ].copy()
    result.insert(0, "status", "estimated_bridge_gate_passed")
    return result


def _read_verified_residuals(
    residual_dir: Path, protocol_path: Path
) -> pd.DataFrame:
    manifest_path = residual_dir / "residual_freeze_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Residual freeze manifest must exist before response estimation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "residuals_frozen":
        raise ValueError("Residual freeze is not complete")
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("Residual freeze was produced under a different holdout protocol")
    residual_path = residual_dir / manifest["residual_file"]["path"]
    if _sha256(residual_path) != manifest["residual_file"]["sha256"]:
        raise ValueError("Frozen holdout residual checksum does not match")
    residuals = pd.read_parquet(residual_path)
    _assert_outcome_blind(residuals)
    if pd.to_numeric(residuals["training_max_year"], errors="coerce").ge(
        HOLDOUT_YEAR
    ).any():
        raise ValueError("Frozen residuals contain non-prior training years")
    return residuals


def _read_and_verify_protocol(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    checksum_path = path.with_suffix(".sha256")
    if not checksum_path.exists():
        raise FileNotFoundError(checksum_path)
    expected = checksum_path.read_text(encoding="utf-8").strip()
    if expected != _sha256(path):
        raise ValueError("Holdout protocol checksum mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_2025_response":
        raise ValueError("Holdout protocol is not frozen")
    return protocol


def _assert_outcome_blind(data: pd.DataFrame) -> None:
    forbidden = [
        column
        for column in data.columns
        if str(column).lower().startswith(FORBIDDEN_FIRST_STAGE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Outcome-blind stage contains forbidden columns: {forbidden}")


def _security_history_as_names(history: pd.DataFrame) -> pd.DataFrame:
    data = _lower_columns(history).rename(
        columns={
            "secinfostartdt": "namedt",
            "secinfoenddt": "nameendt",
            "issuernm": "comnam",
            "primaryexch": "primexch",
            "tradingstatusflg": "trdstat",
        }
    )
    if "cusip" in data:
        data["ncusip"] = data["cusip"]
    return data


def _ciz_share_code(data: pd.DataFrame) -> np.ndarray:
    def col(name: str) -> pd.Series:
        return data.get(name, pd.Series("", index=data.index)).astype(str).str.upper()

    share = col("sharetype")
    security = col("securitytype")
    subtype = col("securitysubtype")
    us = col("usincflg")
    issuer = col("issuertype")
    return np.select(
        [
            (share == "NS")
            & (security == "EQTY")
            & (subtype == "COM")
            & (us == "Y")
            & (issuer == "ACOR"),
            (share == "NS")
            & (security == "EQTY")
            & (subtype == "COM")
            & (us == "Y")
            & (issuer == "CORP"),
            (share == "NS")
            & (security == "EQTY")
            & (subtype == "COM")
            & (us == "N")
            & (issuer == "CORP"),
        ],
        [10, 11, 12],
        default=75,
    )


def _ciz_exchange_code(data: pd.DataFrame) -> np.ndarray:
    primary = data.get(
        "primaryexch", pd.Series("", index=data.index)
    ).astype(str).str.upper()
    condition = data.get(
        "conditionaltype", pd.Series("", index=data.index)
    ).astype(str).str.upper()
    status = data.get(
        "tradingstatusflg", pd.Series("", index=data.index)
    ).astype(str).str.upper()
    regular = condition.eq("RW") & status.eq("A")
    return np.select(
        [
            primary.eq("N") & regular,
            primary.eq("A") & regular,
            primary.eq("Q") & regular,
            primary.eq("R") & regular,
            primary.eq("B") & regular,
            primary.eq("I") & regular,
        ],
        [1, 2, 3, 4, 5, 6],
        default=99,
    )


def _prefer_new_years(
    old: pd.DataFrame, new: pd.DataFrame, date_column: str, *, cutoff: str
) -> pd.DataFrame:
    old_dates = pd.to_datetime(old[date_column], errors="coerce")
    new_dates = pd.to_datetime(new[date_column], errors="coerce")
    boundary = pd.Timestamp(cutoff)
    return pd.concat(
        [old.loc[old_dates.lt(boundary)], new.loc[new_dates.ge(boundary)]],
        ignore_index=True,
        sort=False,
    )


def _deduplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates().reset_index(drop=True)


def _read_optional_export(path: Path) -> pd.DataFrame | None:
    try:
        return read_partitioned_parquet(path)
    except FileNotFoundError:
        return None


def _lower_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).lower() for column in result.columns]
    return result


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {path.resolve()}")


def _records_under(directory: Path) -> list[dict[str, object]]:
    return [
        _file_record(path, directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv_atomic(data: pd.DataFrame, path: Path) -> None:
    """Write a checkpoint only after the complete CSV is durable on disk."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
    data.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_npy_atomic(data: np.ndarray, path: Path) -> None:
    """Write a NumPy artifact atomically without implicit suffix changes."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
    with temporary.open("wb") as handle:
        np.save(handle, data)
    temporary.replace(path)


def _write_json_atomic(data: dict[str, object], path: Path) -> None:
    """Write JSON through a same-directory temporary file and atomic rename."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _read_validated_bootstrap_checkpoint(
    draw_path: Path, check_path: Path, replicate: int
) -> pd.DataFrame:
    """Return one complete draw only after validating its paired diagnostics."""

    if not draw_path.is_file() or not check_path.is_file():
        raise FileNotFoundError(
            f"Missing bootstrap checkpoint pair for replicate {replicate}"
        )
    draw = pd.read_csv(draw_path)
    required = {"replicate", "position", "outcome", "horizon", "treatment", "beta"}
    if missing := sorted(required.difference(draw.columns)):
        raise ValueError(
            f"Bootstrap replicate {replicate} missing draw columns: {missing}"
        )
    if draw.empty or not draw["replicate"].eq(replicate).all():
        raise ValueError(f"Bootstrap draw identity mismatch for replicate {replicate}")
    if set(draw["treatment"].astype(str)) != set(STANDARDIZED_TREATMENTS):
        raise ValueError(
            f"Bootstrap treatment set mismatch for replicate {replicate}"
        )
    if draw["position"].duplicated().any() or not np.isfinite(
        pd.to_numeric(draw["beta"], errors="coerce")
    ).all():
        raise ValueError(f"Invalid bootstrap coefficients for replicate {replicate}")
    checks = pd.read_csv(check_path)
    if (
        checks.empty
        or "replicate" not in checks
        or not checks["replicate"].eq(replicate).all()
    ):
        raise ValueError(
            f"Bootstrap fold-check identity mismatch for replicate {replicate}"
        )
    return draw


def build_main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the CIZ bridge and outcome-blind 2025 holdout inputs."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument(
        "--historical-data-root", type=Path, default=DEFAULT_HISTORICAL_DATA_ROOT
    )
    parser.add_argument(
        "--frozen-revision-dir", type=Path, default=DEFAULT_FROZEN_REVISION
    )
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args()
    result = build_holdout_2025(
        raw_root=args.raw_root,
        build_dir=args.build_dir,
        historical_data_root=args.historical_data_root,
        frozen_revision_dir=args.frozen_revision_dir,
        protocol_path=args.protocol_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_main() -> int:
    parser = argparse.ArgumentParser(
        description="Score, estimate, archive, or bootstrap the frozen 2025 holdout."
    )
    parser.add_argument(
        "stage",
        choices=("score", "lm-ladder", "response", "archive-response", "bootstrap"),
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--residual-dir", type=Path, default=DEFAULT_RESIDUAL_DIR)
    parser.add_argument("--response-dir", type=Path, default=DEFAULT_RESPONSE_DIR)
    parser.add_argument(
        "--frozen-revision-dir", type=Path, default=DEFAULT_FROZEN_REVISION,
        help="Development tuned response directory for the pooled interaction.",
    )
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP_DIR)
    parser.add_argument(
        "--lm-ladder-dir", type=Path, default=DEFAULT_LM_HOLDOUT_LADDER_DIR
    )
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--bootstrap-mode",
        choices=("all", "initialize", "worker", "finalize"),
        default="all",
    )
    parser.add_argument("--replicate-start", type=int)
    parser.add_argument("--replicate-stop", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--archive-label")
    parser.add_argument("--archive-reason")
    args = parser.parse_args()
    if args.stage == "score":
        result = score_holdout_2025(
            feature_dir=args.feature_dir,
            protocol_path=args.protocol_path,
            residual_dir=args.residual_dir,
        )
    elif args.stage == "lm-ladder":
        result = estimate_lm_holdout_answer_ladder(
            feature_path=args.feature_dir / "answer_first_stage_features.parquet",
            protocol_path=args.protocol_path,
            output_dir=args.lm_ladder_dir,
        )
    elif args.stage == "response":
        result = estimate_holdout_response(
            build_dir=args.build_dir,
            residual_dir=args.residual_dir,
            protocol_path=args.protocol_path,
            response_dir=args.response_dir,
            frozen_revision_dir=args.frozen_revision_dir,
        )
    elif args.stage == "archive-response":
        if args.data_root is None or args.archive_label is None or args.archive_reason is None:
            parser.error(
                "archive-response requires --data-root, --archive-label, and --archive-reason"
            )
        result = archive_holdout_response_revision(
            data_root=args.data_root,
            archive_label=args.archive_label,
            reason=args.archive_reason,
        )
    else:
        result = run_holdout_bootstrap(
            feature_dir=args.feature_dir,
            response_dir=args.response_dir,
            protocol_path=args.protocol_path,
            output_dir=args.bootstrap_dir,
            replicates=args.replicates,
            seed=args.seed,
            mode=args.bootstrap_mode,
            replicate_start=args.replicate_start,
            replicate_stop=args.replicate_stop,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def freeze_main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze complete 2025 holdout evidence with checksums."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    args = parser.parse_args()
    result = freeze_holdout_2025(
        data_root=args.data_root,
        protocol_path=args.protocol_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
