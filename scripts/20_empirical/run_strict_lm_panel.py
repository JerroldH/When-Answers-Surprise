"""Run the licensed-data LM analysis on one strict panel, from matching to 2025 returns.

Run from the checkout root:
    python scripts/20_empirical/run_strict_lm_panel.py --historical-root HISTORICAL_DATA \
        --holdout-raw HOLDOUT_WRDS_EXPORTS --dictionary-dir LM_DICTIONARY_DIR --output-dir NEW_DIR

The historical root contains the strict pre-2024 panel, staging tables and raw exports.
Use a private output directory; it contains licensed rows. No network or LLM is used.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from when_answers_surprise.data.wrds_sp500 import CIQ_MIDNIGHT_PLACEHOLDER
from when_answers_surprise.empirical import holdout_2025 as h
from when_answers_surprise.empirical import pipeline as lm

STAGES = {"prepared": "P0", "question": "Q2", "answer": "A3"}
OUTER_YEARS = tuple(range(2020, 2025))


def check_event_timing(build):
    cols = ["transcriptid", "mostimportanttimeutc", "event_datetime_utc", "call_date",
            "STATPERS", "actual_known_before_call", "scaled_eps_surprise", "eps_surprise"]
    e = pd.read_parquet(build / "point_in_time_events.parquet", columns=cols)
    midnight = e.mostimportanttimeutc.astype("string").str.strip().str.fullmatch(CIQ_MIDNIGHT_PLACEHOLDER, na=False)
    snapshot = pd.to_datetime(e.STATPERS)
    if (e.transcriptid.duplicated().any() or e.loc[midnight, "event_datetime_utc"].notna().any()
        or (snapshot.notna() & ~snapshot.lt(pd.to_datetime(e.call_date))).any()
        or (e[["eps_surprise", "scaled_eps_surprise"]].notna().any(axis=1)
            & ~e.actual_known_before_call.eq(True).fillna(False)).any()):
        raise ValueError("Duplicate event or invalid pre-call timing in the unified panel")


def fit_dev(job):
    component, build, dev = job
    variant = STAGES[component]
    frame = pd.read_parquet(build / "outcome_blind_features" / f"{component}_first_stage_features.parquet",
                            filters=[("event_year", "<", 2025)])
    h._assert_outcome_blind(frame)
    keys = ["transcriptid"] if component == "prepared" else ["transcriptid", "turn_index"]
    if frame.duplicated(keys).any() or set(frame.event_year) != set(range(2018, 2025)):
        raise ValueError(f"Invalid development features for {component}")
    if component == "prepared":
        features = h.BASE_STATE_FEATURES
    elif component == "question":
        features = h.question_filtration_features(frame)[variant]
    else:
        features = h.revision_answer_filtration_features(frame)[variant]
    result = h.estimate_component_ladder(frame, component=component, feature_sets={variant: list(features)},
            learner=h.tuned_ridge_spec_with_alphas(h.RIDGE_ALPHAS), min_train_rows=120, outer_years=OUTER_YEARS)
    result["selections"].to_csv(dev / f"{component}_selections.csv", index=False)
    result["fold_metrics"].to_csv(dev / f"{component}_fold_metrics.csv", index=False)
    return result["call_innovations"].rename(columns={f"{component}_{variant}": "narrative_innovation"}).assign(
        component=component, semantic_feature=lm.PRIMARY_FEATURE)


def development_state(hist, build):
    state = pd.read_parquet(build / "outcome_blind_event_state.parquet",
                            filters=[("event_year", ">=", 2020), ("event_year", "<", 2025)])
    h._assert_outcome_blind(state)
    # Use the same CRSP series as the blind panel, but no 2025 dates for development CARs.
    cols = ["permno", "date", "ret", "vwretd", "vol", "prc", "shrout"]
    directory = hist / "raw/wrds_exports/crsp_dsf_sp500"
    parts = []
    for year in range(2018, 2024):
        files = sorted((directory / f"year={year}").glob("part-*.parquet"))
        if not files:
            raise FileNotFoundError(f"Missing legacy CRSP daily partition {year}")
        parts.extend(pd.read_parquet(p, columns=cols) for p in files)
    parts.append(pd.read_parquet(build / "ciz_legacy_daily.parquet", columns=cols,
                 filters=[("date", ">=", pd.Timestamp("2024-01-01")), ("date", "<", pd.Timestamp("2025-01-01"))]))
    daily = pd.concat(parts, ignore_index=True)
    if daily.date.max() >= pd.Timestamp("2025-01-01") or daily.duplicated(["permno", "date"]).any():
        raise ValueError("Invalid development return history")
    outcomes = h.build_crsp_event_outcomes(state, daily, horizons=h.HORIZONS)
    keys = ["transcriptid", "event_trading_date"]
    return state.merge(outcomes[keys + [f"CAR_0_{n}" for n in h.HORIZONS]], on=keys, validate="one_to_one")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("historical-root", "holdout-raw", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dictionary-dir", type=Path, default=ROOT / "data/external")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT):
        parser.error("--output-dir must be outside this checkout; the run writes licensed rows")
    out.mkdir(parents=True, exist_ok=False)
    build, dev = out / "build", out / "development"
    dev.mkdir()
    print("1. Build one strict event panel (including 2023 IBES history)", flush=True)
    h.build_holdout_2025(raw_root=args.holdout_raw, historical_data_root=args.historical_root,
                         build_dir=build, frozen_revision_dir=None, ibes_history_year=2023,
                         dictionary_dir=args.dictionary_dir)
    check_event_timing(build)
    print("2. Fit development folds and freeze development scales", flush=True)
    with ProcessPoolExecutor(max_workers=len(STAGES)) as pool:
        fits = list(pool.map(fit_dev, [(stage, build, dev) for stage in STAGES]))
    sample = lm.build_response_sample(development_state(args.historical_root, build), pd.concat(fits, ignore_index=True))
    if sample.transcriptid.duplicated().any() or set(sample.event_year) != set(OUTER_YEARS):
        raise ValueError("Invalid development response sample")
    sample.to_parquet(dev / "response_model_sample_revision_v2.parquet", index=False)
    (dev / "freeze_manifest.json").write_text(json.dumps({
        "build_manifest_sha256": h._sha256(build / "build_manifest.json"), "n_dev": len(sample),
        "outer_years": OUTER_YEARS, "ibes_history_year": 2023,
    }, indent=2), encoding="utf-8")
    protocol = out / "protocol/holdout_protocol.json"
    h.freeze_holdout_protocol(frozen_revision_dir=dev, protocol_path=protocol)
    print(f"3. Fit and freeze 2025 LM residuals ({len(sample)} development calls)", flush=True)
    residual = out / "residual_freeze"
    frozen = h.score_holdout_2025(feature_dir=build / "outcome_blind_features", residual_dir=residual, protocol_path=protocol)
    if frozen["outcome_columns_loaded"] or frozen["training_max_year"] != 2024 or frozen["status"] != "residuals_frozen":
        raise ValueError("Invalid residual freeze")
    print("4. Estimate 2025 returns on the verified freeze", flush=True)
    r = h.estimate_holdout_response(build_dir=build, residual_dir=residual, response_dir=out / "response",
                                   frozen_revision_dir=dev, protocol_path=protocol)
    print(f"Holdout calls: {r['test_calls']}; answer CAR[0,1]: {10000*r['primary_beta']:.6f} bps", flush=True)


if __name__ == "__main__":
    main()
