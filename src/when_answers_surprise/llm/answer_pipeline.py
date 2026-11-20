"""Appendix-only LLM-scored answer-tone representation check.

The LLM only creates an answer-tone score. It does not see market outcomes,
firm metadata, analyst questions, prepared remarks, prior dialogue, or existing
LM scores. The downstream A0/A1/A2/A3 answer-filtration design is unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from when_answers_surprise.empirical.feature_spec import (
    assert_feature_manifest_closure,
    revision_answer_filtration_features,
)
from when_answers_surprise.empirical.holdout_2025 import RIDGE_ALPHAS
from when_answers_surprise.empirical.learner_audit import tuned_ridge_spec_with_alphas
from when_answers_surprise.empirical.revision_v2_tuned import estimate_component_ladder
from when_answers_surprise.llm.tone_scoring import (
    LLMScoreConfig,
    auth_token,
    manifest_json,
    provider_config,
    score_answer_text,
    sha256_text,
)
from when_answers_surprise.sp500_lm import (
    HORIZONS,
    OOF_TEST_YEARS,
    fit_response_regression,
    response_controls,
    zscore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025" / "analysis"
BASELINE_DIR = ANALYSIS_ROOT / "lm_baseline"
FEATURE_CACHE_DIR = ANALYSIS_ROOT / "lm_baseline_learner_audit_cache"

VARIANT_LABELS = {
    "A0": "A0: pre-call state",
    "A1": "A1: + prepared/prior dialogue",
    "A2": "A2: + current question",
    "A3": "A3: + analyst/content context",
}
OBSERVABLES = [
    "eps_surprise",
    "forecast_dispersion",
    "analyst_coverage",
    "pre_ret_20",
    "pre_vol_20",
]
TERMINAL_SCORE_STATUSES = {"ok", "invalid_output"}


def default_output_dir_for_config(config: LLMScoreConfig) -> Path:
    provider = "gemini" if config.provider in {"gemini", "gemini_adc"} else config.provider
    model = re.sub(r"^publishers/google/models/", "", config.model)
    model = re.sub(r"^google/", "", model)
    model = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
    prefix = f"{provider}_"
    if model.startswith(prefix):
        model = model[len(prefix) :]
    return ANALYSIS_ROOT / f"llm_answer_tone_{provider}_{model}"


def legacy_main() -> None:
    parser = argparse.ArgumentParser(description="Run appendix-only LLM answer-tone check.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=FEATURE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--provider",
        default="fake",
        choices=["fake", "gemini_adc", "gemini", "openai", "deepseek", "qwen"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--duplicate-fraction", type=float, default=0.01)
    parser.add_argument("--max-duplicate-rows", type=int, default=500)
    parser.add_argument("--max-failure-rate", type=float, default=0.02)
    parser.add_argument("--min-duplicate-exact-agreement", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=1000)
    args = parser.parse_args()

    max_rows = None if args.allow_full_run else args.max_rows
    config = provider_config(args.provider, model=args.model)
    output_dir = args.output_dir or default_output_dir_for_config(config)
    run_check(
        baseline_dir=args.baseline_dir,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=output_dir,
        score_config=config,
        max_rows=max_rows,
        min_train_rows=args.min_train_rows,
        duplicate_fraction=args.duplicate_fraction,
        max_duplicate_rows=args.max_duplicate_rows,
        max_failure_rate=args.max_failure_rate,
        min_duplicate_exact_agreement=args.min_duplicate_exact_agreement,
        workers=max(args.workers, 1),
        flush_every=max(args.flush_every, 1),
    )


def run_check(
    *,
    baseline_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    score_config: LLMScoreConfig,
    max_rows: int | None,
    min_train_rows: int,
    duplicate_fraction: float,
    max_duplicate_rows: int,
    max_failure_rate: float,
    min_duplicate_exact_agreement: float,
    workers: int = 1,
    flush_every: int = 1000,
) -> dict[str, object]:
    score_summary = score_online(
        baseline_dir=baseline_dir,
        feature_cache_dir=feature_cache_dir,
        output_dir=output_dir,
        score_config=score_config,
        max_rows=max_rows,
        duplicate_fraction=duplicate_fraction,
        max_duplicate_rows=max_duplicate_rows,
        workers=workers,
        flush_every=flush_every,
        emit=False,
    )
    analysis_summary = analyze_scores(
        baseline_dir=baseline_dir,
        feature_cache_dir=feature_cache_dir,
        output_dir=output_dir,
        min_train_rows=min_train_rows,
        max_failure_rate=max_failure_rate,
        min_duplicate_exact_agreement=min_duplicate_exact_agreement,
        emit=False,
    )
    summary = {"score": score_summary, "analysis": analysis_summary}
    print(json.dumps(summary, indent=2))
    return summary


def score_online(
    *,
    baseline_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    score_config: LLMScoreConfig,
    max_rows: int | None,
    duplicate_fraction: float,
    max_duplicate_rows: int,
    workers: int = 1,
    flush_every: int = 1000,
    historical_score_cache: Path | None = None,
    emit: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_frame = pd.read_parquet(feature_cache_dir / "answer_first_stage_features.parquet")
    qa_turns = pd.read_parquet(baseline_dir / "qa_turns.parquet")

    scoring_input = build_scoring_input(answer_frame, qa_turns, max_rows=max_rows)
    score_path = output_dir / "answer_llm_scores.parquet"
    if historical_score_cache is None:
        scores = score_with_cache(
            scoring_input,
            score_path,
            score_config,
            workers=workers,
            flush_every=flush_every,
        )
    else:
        scores = extend_score_panel(
            scoring_input,
            score_path,
            historical_score_cache,
            score_config,
            workers=workers,
            flush_every=flush_every,
        )
    duplicate_audit = duplicate_score_audit(
        scoring_input,
        score_config,
        duplicate_fraction=duplicate_fraction,
        max_duplicate_rows=max_duplicate_rows,
        workers=workers,
    )
    score_distribution = build_score_distribution(scores)
    duplicate_audit.to_csv(output_dir / "llm_duplicate_score_agreement.csv", index=False)
    score_distribution.to_csv(output_dir / "llm_score_distribution.csv", index=False)
    (output_dir / "answer_llm_scoring_manifest.json").write_text(
        manifest_json(
            score_config,
            rows_requested=len(scoring_input),
            rows_scored=len(scores),
            rows_successful=int(scores["status"].eq("ok").sum()) if "status" in scores else 0,
        ),
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output_dir),
        "provider": score_config.provider,
        "model": score_config.model,
        "rows_requested": int(len(scoring_input)),
        "rows_scored": int(len(scores)),
        "rows_successful": int(scores["status"].eq("ok").sum()) if "status" in scores else 0,
        "historical_score_cache": (
            str(historical_score_cache) if historical_score_cache is not None else None
        ),
        "outputs": [
            "answer_llm_scores.parquet",
            "answer_llm_scoring_manifest.json",
            "llm_duplicate_score_agreement.csv",
            "llm_score_distribution.csv",
        ],
    }
    (output_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if emit:
        print(json.dumps(summary, indent=2))
    return summary


def analyze_scores(
    *,
    baseline_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    min_train_rows: int,
    max_failure_rate: float,
    min_duplicate_exact_agreement: float,
    score_cache: Path | None = None,
    emit: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest_path = baseline_dir / "baseline_feature_manifest.json"
    cache_manifest_path = feature_cache_dir / "cache_manifest.json"
    if not baseline_manifest_path.exists() or not cache_manifest_path.exists():
        raise FileNotFoundError("Cached LLM analysis requires baseline and cache feature manifests")
    assert_feature_manifest_closure(
        json.loads(baseline_manifest_path.read_text(encoding="utf-8")),
        json.loads(cache_manifest_path.read_text(encoding="utf-8")),
    )
    response_sample = pd.read_parquet(baseline_dir / "response_model_sample.parquet")
    answer_frame = pd.read_parquet(feature_cache_dir / "answer_first_stage_features.parquet")
    qa_turns = pd.read_parquet(baseline_dir / "qa_turns.parquet")
    score_path = score_cache or output_dir / "answer_llm_scores.parquet"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing LLM score cache: {score_path}")
    scores = pd.read_parquet(score_path)
    cache_coverage = validate_cached_score_coverage(answer_frame, qa_turns=qa_turns, scores=scores)
    if (
        cache_coverage["missing_rows"]
        or cache_coverage["hash_mismatches"]
        or cache_coverage["duplicate_rows"]
    ):
        raise ValueError(f"LLM score cache coverage failed: {cache_coverage}")
    non_ok = int((~scores["status"].eq("ok")).sum())
    if non_ok:
        raise ValueError(f"LLM score cache contains {non_ok} non-ok rows")
    duplicate_path = output_dir / "llm_duplicate_score_agreement.csv"
    duplicate_audit = pd.read_csv(duplicate_path) if duplicate_path.exists() else pd.DataFrame()
    llm_frame = merge_scores(answer_frame, scores)
    ladder = estimate_llm_answer_ladder(llm_frame, response_sample, min_train_rows=min_train_rows)
    score_distribution = build_score_distribution(scores)
    diagnostics = build_diagnostics(
        scores,
        duplicate_audit,
        ladder,
        response_sample,
        max_failure_rate=max_failure_rate,
        min_duplicate_exact_agreement=min_duplicate_exact_agreement,
    )

    outputs = {
        "table_llm_answer_filtration_ladder": build_ladder_table(ladder),
        "table_llm_representation_check": diagnostics,
        "llm_score_distribution": score_distribution,
        "llm_duplicate_score_agreement": duplicate_audit,
        "llm_filtration_first_stage_detail": ladder["first_stage"],
        "llm_filtration_correlations": ladder["correlations"],
        "llm_filtration_placebo_response": ladder["placebo"],
        "llm_filtration_main_response": ladder["response"],
        "llm_tuned_ridge_fold_metrics": ladder["fold_metrics"],
        "llm_tuned_ridge_hyperparameter_selection": ladder["selections"],
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        if name.startswith("table_"):
            write_latex(table, output_dir / f"{name}.tex")
    (output_dir / "llm_score_cache_coverage.json").write_text(
        json.dumps(cache_coverage, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {
        "output_dir": str(output_dir),
        "rows_scored": int(len(scores)),
        "rows_successful": int(scores["status"].eq("ok").sum()) if "status" in scores else 0,
        "cache_path": str(score_path),
        "cache_coverage": cache_coverage,
        "diagnostics": diagnostics.to_dict("records"),
        "outputs": sorted(f"{name}.csv" for name in outputs),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if emit:
        print(json.dumps(summary, indent=2))
    return summary


def build_scoring_input(
    answer_frame: pd.DataFrame, qa_turns: pd.DataFrame, *, max_rows: int | None
) -> pd.DataFrame:
    left_columns = ["transcriptid", "turn_index"]
    if "event_year" in answer_frame.columns:
        left_columns.append("event_year")
    left = answer_frame[left_columns].copy()
    left["transcriptid"] = left["transcriptid"].astype(str)
    qa = qa_turns[["transcriptid", "turn_index", "answer_text"]].copy()
    qa["transcriptid"] = qa["transcriptid"].astype(str)
    data = left.merge(qa, on=["transcriptid", "turn_index"], how="inner", validate="many_to_one")
    data["text_sha256"] = data["answer_text"].map(sha256_text)
    data = data.drop_duplicates(["transcriptid", "turn_index"]).sort_values(
        ["transcriptid", "turn_index"]
    )
    if max_rows is not None and len(data) > max_rows:
        data = stratified_scoring_sample(data, max_rows=max_rows)
    return data.reset_index(drop=True)


def validate_cached_score_coverage(
    answer_frame: pd.DataFrame,
    *,
    qa_turns: pd.DataFrame,
    scores: pd.DataFrame,
) -> dict[str, object]:
    """Validate transcript/turn/text-hash and cache-key coverage without scoring."""

    required = build_scoring_input(answer_frame, qa_turns, max_rows=None)
    required_keys = {
        f"{row.transcriptid}||{int(row.turn_index)}" for row in required.itertuples(index=False)
    }
    required_hashes = {
        f"{row.transcriptid}||{int(row.turn_index)}": row.text_sha256
        for row in required.itertuples(index=False)
    }
    fields = ["transcriptid", "turn_index", "text_sha256", "provider", "model", "prompt_version"]
    missing_fields = [field for field in fields if field not in scores.columns]
    if missing_fields:
        raise ValueError(f"LLM score cache missing required audit fields: {missing_fields}")
    cache = scores.copy()
    cache["transcriptid"] = cache["transcriptid"].astype(str)
    cache["turn_index"] = pd.to_numeric(cache["turn_index"], errors="coerce")
    cache_keys = cache.apply(
        lambda row: (
            f"{row.transcriptid}||{int(row.turn_index)}" if pd.notna(row.turn_index) else ""
        ),
        axis=1,
    )
    duplicate_rows = int(cache_keys[cache_keys.isin(required_keys)].duplicated().sum())
    covered = set(cache_keys) & required_keys
    missing_keys = required_keys - covered
    hash_mismatches = int(
        sum(
            row.text_sha256 != required_hashes.get(key)
            for key, row in zip(cache_keys, cache.itertuples(index=False), strict=False)
            if key in required_hashes
        )
    )
    nonmissing_audit_fields = int(cache[fields].notna().all(axis=1).sum()) if not cache.empty else 0
    identities = cache[["provider", "model", "prompt_version"]].drop_duplicates()
    if len(identities) != 1:
        raise ValueError(
            "LLM score cache must contain exactly one provider/model/prompt identity; "
            f"found {identities.to_dict(orient='records')}"
        )
    return {
        "required_rows": len(required_keys),
        "cached_rows": len(cache),
        "covered_rows": len(covered),
        "missing_rows": len(missing_keys),
        "missing_keys": sorted(missing_keys),
        "hash_mismatches": hash_mismatches,
        "duplicate_rows": duplicate_rows,
        "rows_with_all_audit_fields": nonmissing_audit_fields,
        "providers": sorted(cache["provider"].dropna().astype(str).unique()),
        "models": sorted(cache["model"].dropna().astype(str).unique()),
        "prompt_versions": sorted(cache["prompt_version"].dropna().astype(str).unique()),
    }


def stratified_scoring_sample(data: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if "event_year" not in data.columns or data["event_year"].nunique(dropna=True) <= 1:
        return data.head(max_rows)
    years = sorted(data["event_year"].dropna().unique())
    per_year = max(1, max_rows // len(years))
    pieces = []
    for year in years:
        group = data.loc[data["event_year"].eq(year)]
        pieces.append(group.sample(n=min(len(group), per_year), random_state=1729))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) < max_rows:
        remainder = data.drop(index=sampled.index, errors="ignore")
        if not remainder.empty:
            sampled = pd.concat(
                [
                    sampled,
                    remainder.sample(
                        n=min(len(remainder), max_rows - len(sampled)),
                        random_state=1730,
                    ),
                ],
                ignore_index=False,
            )
    return sampled.sort_values(["event_year", "transcriptid", "turn_index"]).head(max_rows)


def score_with_cache(
    input_rows: pd.DataFrame,
    score_path: Path,
    config: LLMScoreConfig,
    *,
    workers: int = 1,
    flush_every: int = 1000,
) -> pd.DataFrame:
    requested_keys = {
        cache_key_for_values(
            row.transcriptid,
            row.turn_index,
            row.text_sha256,
            config.provider,
            config.model,
            config.prompt_version,
        )
        for row in input_rows.itertuples(index=False)
    }
    existing = pd.read_parquet(score_path) if score_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing["cache_key"] = cache_key(existing)
        current = existing["cache_key"].map(lambda key: key.endswith(config_suffix(config)))
        terminal = (
            existing["status"].isin(TERMINAL_SCORE_STATUSES) if "status" in existing else False
        )
        cached_keys = set(existing.loc[current & terminal, "cache_key"])
    else:
        cached_keys = set()

    pending = []
    for row in input_rows.itertuples(index=False):
        key = cache_key_for_values(
            row.transcriptid,
            row.turn_index,
            row.text_sha256,
            config.provider,
            config.model,
            config.prompt_version,
        )
        if key not in cached_keys:
            pending.append(row)

    combined = existing.drop(columns=["cache_key"], errors="ignore")
    total_pending = len(pending)
    for start in range(0, total_pending, flush_every):
        batch = pending[start : start + flush_every]
        rows = score_pending_rows(batch, config, workers=workers)
        if rows:
            combined = append_score_rows(combined, pd.DataFrame(rows), score_path)
        done = min(start + len(batch), total_pending)
        print(f"[llm-score] cached {done}/{total_pending} pending rows", flush=True)
    if not combined.empty:
        combined["cache_key"] = cache_key(combined)
        current = combined["cache_key"].isin(requested_keys)
        return combined.loc[current].drop(columns=["cache_key"])
    return combined


def extend_score_panel(
    input_rows: pd.DataFrame,
    score_path: Path,
    historical_score_cache: Path,
    config: LLMScoreConfig,
    *,
    workers: int = 1,
    flush_every: int = 1000,
) -> pd.DataFrame:
    """Extend a scorer's historical panel without changing provider provenance."""

    if not historical_score_cache.exists():
        raise FileNotFoundError(historical_score_cache)
    sources = [pd.read_parquet(historical_score_cache)]
    if score_path.exists() and score_path.resolve() != historical_score_cache.resolve():
        sources.append(pd.read_parquet(score_path))
    combined = pd.concat(sources, ignore_index=True, sort=False)
    combined = deduplicate_score_cache(combined)

    required = input_rows.copy()
    required["cache_key"] = [
        cache_key_for_values(
            row.transcriptid,
            row.turn_index,
            row.text_sha256,
            config.provider,
            config.model,
            config.prompt_version,
        )
        for row in required.itertuples(index=False)
    ]
    required_keys = set(required["cache_key"])
    covered = set(combined.loc[combined["status"].eq("ok"), "cache_key"]) & required_keys
    pending = [row for row in required.itertuples(index=False) if row.cache_key not in covered]

    for start in range(0, len(pending), flush_every):
        batch = pending[start : start + flush_every]
        rows = score_pending_rows(batch, config, workers=workers)
        if rows:
            combined = pd.concat([combined.drop(columns=["cache_key"]), pd.DataFrame(rows)])
            combined = deduplicate_score_cache(combined)
            combined.drop(columns=["cache_key"]).to_parquet(score_path, index=False)
        done = min(start + len(batch), len(pending))
        print(f"[llm-score] extended {done}/{len(pending)} missing rows", flush=True)

    panel = combined.loc[combined["cache_key"].isin(required_keys)].copy()
    if panel["cache_key"].duplicated().any():
        raise ValueError("Extended LLM score panel has duplicate scorer cache keys")
    panel = panel.drop(columns=["cache_key"])
    panel.to_parquet(score_path, index=False)
    return panel


def deduplicate_score_cache(scores: pd.DataFrame) -> pd.DataFrame:
    data = scores.copy()
    data["cache_key"] = cache_key(data)
    data["_status_rank"] = (
        data["status"].map({"api_error": 0, "invalid_output": 1, "ok": 2}).fillna(-1)
    )
    if "scored_at_utc" not in data:
        data["scored_at_utc"] = ""
    data = data.sort_values(["cache_key", "_status_rank", "scored_at_utc"])
    return data.drop_duplicates("cache_key", keep="last").drop(columns=["_status_rank"])


def append_score_rows(
    existing: pd.DataFrame, new_scores: pd.DataFrame, score_path: Path
) -> pd.DataFrame:
    if existing.empty:
        combined = new_scores.copy()
    else:
        combined = pd.concat([existing, new_scores], ignore_index=True)
    combined["cache_key"] = cache_key(combined)
    combined = combined.sort_values("scored_at_utc").drop_duplicates("cache_key", keep="last")
    combined = combined.drop(columns=["cache_key"])
    combined.to_parquet(score_path, index=False)
    return combined


def score_pending_rows(
    rows: list[object], config: LLMScoreConfig, *, workers: int
) -> list[dict[str, object]]:
    def score_row(row: object) -> dict[str, object]:
        result = score_answer_text(row.answer_text, config)
        return {
            "transcriptid": str(row.transcriptid),
            "turn_index": int(row.turn_index),
            "text_sha256": row.text_sha256,
            **result,
        }

    if workers <= 1 or len(rows) <= 1:
        return [score_row(row) for row in rows]

    if config.provider == "gemini_adc":
        auth_token(config)

    scored = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_row, row) for row in rows]
        for future in as_completed(futures):
            scored.append(future.result())
    return scored


def duplicate_score_audit(
    input_rows: pd.DataFrame,
    config: LLMScoreConfig,
    *,
    duplicate_fraction: float,
    max_duplicate_rows: int,
    workers: int,
) -> pd.DataFrame:
    if input_rows.empty or duplicate_fraction <= 0:
        return pd.DataFrame()
    n_rows = max(1, min(max_duplicate_rows, int(np.ceil(len(input_rows) * duplicate_fraction))))
    sample = input_rows.sample(n=n_rows, random_state=1729)

    def score_duplicate_pair(row: object) -> dict[str, object]:
        first = score_answer_text(row.answer_text, config)
        second = score_answer_text(row.answer_text, config)
        return {
            "transcriptid": str(row.transcriptid),
            "turn_index": int(row.turn_index),
            "first_status": first["status"],
            "second_status": second["status"],
            "first_score": first["score"],
            "second_score": second["score"],
            "exact_agreement": (
                first["score"] == second["score"] and first["status"] == second["status"] == "ok"
            ),
            "within_one_agreement": within_one(first["score"], second["score"]),
        }

    sampled_rows = list(sample.itertuples(index=False))
    if workers <= 1 or len(sampled_rows) <= 1:
        rows = [score_duplicate_pair(row) for row in sampled_rows]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_duplicate_pair, row) for row in sampled_rows]
            for future in as_completed(futures):
                rows.append(future.result())
    return pd.DataFrame(rows)


def merge_scores(answer_frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    ok = scores.loc[scores["status"].eq("ok")].copy()
    if ok.duplicated(["transcriptid", "turn_index"]).any():
        raise ValueError("LLM score cache has duplicate transcriptid, turn_index rows.")
    keep = ok[["transcriptid", "turn_index", "score"]].rename(columns={"score": "llm_tone_score"})
    keep["transcriptid"] = keep["transcriptid"].astype(str)
    frame = answer_frame.copy()
    frame["transcriptid"] = frame["transcriptid"].astype(str)
    merged = frame.merge(
        keep,
        on=["transcriptid", "turn_index"],
        how="inner",
        validate="one_to_one",
    )
    merged["raw_score"] = pd.to_numeric(merged["llm_tone_score"], errors="coerce")
    return merged


def estimate_llm_answer_ladder(
    answer_frame: pd.DataFrame,
    response_sample: pd.DataFrame,
    *,
    min_train_rows: int,
    outer_years: Sequence[int] = OOF_TEST_YEARS,
    variant_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    fitted = estimate_llm_answer_components(
        answer_frame,
        min_train_rows=min_train_rows,
        outer_years=outer_years,
        variant_workers=variant_workers,
    )
    feature_sets = answer_filtration_features(answer_frame)
    call_wide = fitted["call_innovations"]
    if call_wide.empty:
        return {
            **fitted,
            "call_sample": pd.DataFrame(),
            "correlations": pd.DataFrame(),
            "placebo": pd.DataFrame(),
            "response": pd.DataFrame(),
        }
    data = response_sample.assign(transcriptid=response_sample["transcriptid"].astype(str)).merge(
        call_wide, on="transcriptid", how="inner", validate="one_to_one"
    )
    for variant in feature_sets:
        col = f"llm_answer_{variant}"
        if col in data:
            data[f"{col}_z"] = zscore(data[col])

    return {
        **fitted,
        "call_sample": data,
        "correlations": observable_correlations(data, feature_sets),
        "placebo": placebo_responses(data, feature_sets),
        "response": main_responses(data, feature_sets),
    }


def estimate_llm_answer_components(
    answer_frame: pd.DataFrame,
    *,
    min_train_rows: int,
    outer_years: Sequence[int],
    variant_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Run the LM pipeline's nested time-forward tuned Ridge on LLM scores."""

    feature_sets = answer_filtration_features(answer_frame)
    if variant_workers > 1:
        with ProcessPoolExecutor(max_workers=min(variant_workers, len(feature_sets))) as pool:
            futures = [
                pool.submit(
                    _estimate_llm_answer_variant,
                    answer_frame,
                    variant,
                    features,
                    min_train_rows,
                    tuple(outer_years),
                )
                for variant, features in feature_sets.items()
            ]
            result = _merge_llm_variant_results([future.result() for future in futures])
    else:
        result = estimate_component_ladder(
            answer_frame,
            component="answer",
            feature_sets=feature_sets,
            learner=tuned_ridge_spec_with_alphas(RIDGE_ALPHAS),
            min_train_rows=min_train_rows,
            outer_years=tuple(outer_years),
        )
    result["turn_innovations"] = result["turn_innovations"].assign(
        semantic_feature="llm_tone_score"
    )
    result["call_innovations"] = result["call_innovations"].rename(
        columns=lambda column: (
            f"llm_{column}" if str(column).startswith("answer_A") else column
        )
    )
    first_stage = result["first_stage"].copy()
    first_stage["variant_label"] = first_stage["variant"].map(VARIANT_LABELS)
    result["first_stage"] = first_stage
    return result


def _estimate_llm_answer_variant(
    answer_frame: pd.DataFrame,
    variant: str,
    features: list[str],
    min_train_rows: int,
    outer_years: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    return estimate_component_ladder(
        answer_frame,
        component="answer",
        feature_sets={variant: features},
        learner=tuned_ridge_spec_with_alphas(RIDGE_ALPHAS),
        min_train_rows=min_train_rows,
        outer_years=outer_years,
    )


def _merge_llm_variant_results(results: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    call_innovations = results[0]["call_innovations"].copy()
    for result in results[1:]:
        call_innovations = call_innovations.merge(
            result["call_innovations"],
            on="transcriptid",
            how="inner",
            validate="one_to_one",
        )
    return {
        "turn_innovations": pd.concat(
            [result["turn_innovations"] for result in results], ignore_index=True
        ),
        "call_innovations": call_innovations,
        "first_stage": pd.concat(
            [result["first_stage"] for result in results], ignore_index=True
        ),
        "fold_metrics": pd.concat(
            [result["fold_metrics"] for result in results], ignore_index=True
        ),
        "selections": pd.concat(
            [result["selections"] for result in results], ignore_index=True
        ),
    }


def answer_filtration_features(frame: pd.DataFrame) -> dict[str, list[str]]:
    return revision_answer_filtration_features(frame, label_prefix="cached LLM answer filtration")


def observable_correlations(data: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for variant in feature_sets:
        innovation = f"llm_answer_{variant}_z"
        if innovation not in data:
            continue
        for observable in OBSERVABLES:
            if observable not in data:
                continue
            clean = data[[innovation, observable]].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "variant": variant,
                    "observable": observable,
                    "n_obs": int(len(clean)),
                    "correlation": safe_corr(clean[innovation], clean[observable]),
                }
            )
    return pd.DataFrame(rows)


def placebo_responses(data: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    controls = response_controls(data)
    rows = []
    for variant in feature_sets:
        treatment = f"llm_answer_{variant}_z"
        if treatment not in data:
            continue
        for outcome in ["placebo_CAR_m5_m1", "placebo_CAR_m10_m1"]:
            result = fit_response_regression(
                data,
                outcome=outcome,
                treatments=[treatment],
                controls=[col for col in controls if col != treatment],
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            out = result.coefficients.iloc[0].to_dict()
            rows.append({"variant": variant, "outcome": outcome, **out})
    return pd.DataFrame(rows)


def main_responses(data: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    controls = response_controls(data)
    rows = []
    for variant in feature_sets:
        treatment = f"llm_answer_{variant}_z"
        if treatment not in data:
            continue
        for horizon in HORIZONS:
            result = fit_response_regression(
                data,
                outcome=f"CAR_0_{horizon}",
                treatments=[treatment],
                controls=[col for col in controls if col != treatment],
                firm_fe="permno",
                quarter_fe="event_calendar_quarter",
                cluster_cols=("permno", "event_trading_date"),
            )
            out = result.coefficients.iloc[0].to_dict()
            rows.append(
                {
                    "variant": variant,
                    "horizon": horizon,
                    "outcome": f"CAR_0_{horizon}",
                    **out,
                }
            )
    return pd.DataFrame(rows)


def build_ladder_table(ladder: dict[str, pd.DataFrame]) -> pd.DataFrame:
    first_stage = ladder["first_stage"].copy()
    corr = grouped_abs_max(
        ladder["correlations"],
        "variant",
        "correlation",
        "max_abs_observable_corr",
    )
    placebo = grouped_abs_max(ladder["placebo"], "variant", "t_stat", "max_abs_placebo_t")
    response = ladder["response"]
    beta = (
        response.pivot(index="variant", columns="horizon", values="beta").add_prefix("car_beta_h")
        if not response.empty
        else pd.DataFrame()
    )
    tstat = (
        response.pivot(index="variant", columns="horizon", values="t_stat").add_prefix("car_t_h")
        if not response.empty
        else pd.DataFrame()
    )
    out = first_stage.merge(corr, on="variant", how="left").merge(placebo, on="variant", how="left")
    if not beta.empty:
        out = out.merge(beta.join(tstat, how="outer").reset_index(), on="variant", how="left")
    for horizon in [1, 20]:
        src = f"car_beta_h{horizon}"
        out[f"car_beta_bp_h{horizon}"] = 10000.0 * out[src] if src in out else np.nan
    cols = [
        "variant",
        "variant_label",
        "n_turns",
        "n_calls",
        "n_features",
        "oof_r2",
        "oof_mae",
        "turn_residual_sd",
        "max_abs_observable_corr",
        "max_abs_placebo_t",
        "car_beta_bp_h1",
        "car_t_h1",
        "car_beta_bp_h20",
        "car_t_h20",
    ]
    return round_numeric(out[[col for col in cols if col in out]])


def build_score_distribution(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    dist = scores.groupby(["status", "score"], dropna=False).size().reset_index(name="n_rows")
    dist["share"] = dist["n_rows"] / len(scores)
    return round_numeric(dist)


def build_diagnostics(
    scores: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    ladder: dict[str, pd.DataFrame],
    response_sample: pd.DataFrame,
    *,
    max_failure_rate: float,
    min_duplicate_exact_agreement: float,
) -> pd.DataFrame:
    n_scores = len(scores)
    failure_rate = (
        float(1.0 - scores["status"].eq("ok").mean()) if n_scores and "status" in scores else np.nan
    )
    exact = (
        float(duplicate_audit["exact_agreement"].mean()) if not duplicate_audit.empty else np.nan
    )
    within = (
        float(duplicate_audit["within_one_agreement"].mean())
        if not duplicate_audit.empty
        else np.nan
    )
    call_sample = ladder["call_sample"].copy()
    lm_corr = np.nan
    terminal_variant = "A3" if "llm_answer_A3_z" in call_sample else "A2"
    terminal_column = f"llm_answer_{terminal_variant}_z"
    if "answer_innovation_z" in call_sample and terminal_column in call_sample:
        clean = call_sample[["answer_innovation_z", terminal_column]].dropna()
        lm_corr = safe_corr(clean["answer_innovation_z"], clean[terminal_column])
    pass_failure = bool(pd.notna(failure_rate) and failure_rate <= max_failure_rate)
    pass_stability = bool(pd.notna(exact) and exact >= min_duplicate_exact_agreement)
    allow_market = pass_failure and pass_stability

    response = ladder["response"]
    terminal_response = (
        response.loc[response["variant"].eq(terminal_variant)].copy()
        if "variant" in response
        else pd.DataFrame()
    )
    car1 = response_at(terminal_response, 1) if allow_market else {}
    car20 = response_at(terminal_response, 20) if allow_market else {}
    rows = [
        {"metric": "n_score_rows", "value": n_scores},
        {"metric": "failure_rate", "value": failure_rate},
        {"metric": "duplicate_exact_agreement", "value": exact},
        {"metric": "duplicate_within_one_agreement", "value": within},
        {
            "metric": f"corr_with_lm_{terminal_variant.lower()}_innovation",
            "value": lm_corr,
        },
        {"metric": "market_response_interpretation_allowed", "value": allow_market},
        {
            "metric": f"{terminal_variant.lower()}_car1_bps_if_allowed",
            "value": float(car1.get("beta", np.nan)) * 10000,
        },
        {
            "metric": f"{terminal_variant.lower()}_car1_t_if_allowed",
            "value": car1.get("t_stat", np.nan),
        },
        {
            "metric": f"{terminal_variant.lower()}_car20_bps_if_allowed",
            "value": float(car20.get("beta", np.nan)) * 10000,
        },
        {
            "metric": f"{terminal_variant.lower()}_car20_t_if_allowed",
            "value": car20.get("t_stat", np.nan),
        },
        {"metric": "response_sample_rows", "value": len(response_sample)},
    ]
    return round_numeric(pd.DataFrame(rows))


def cache_key(data: pd.DataFrame) -> pd.Series:
    required = ["transcriptid", "turn_index", "text_sha256", "provider", "model", "prompt_version"]
    missing = [col for col in required if col not in data]
    if missing:
        raise ValueError(f"LLM score cache missing columns: {missing}")
    return data.apply(
        lambda row: cache_key_for_values(
            row["transcriptid"],
            row["turn_index"],
            row["text_sha256"],
            row["provider"],
            row["model"],
            row["prompt_version"],
        ),
        axis=1,
    )


def cache_key_for_values(
    transcriptid: object,
    turn_index: object,
    text_sha256: str,
    provider: str,
    model: str,
    prompt_version: str,
) -> str:
    return "||".join(
        [
            str(transcriptid),
            str(turn_index),
            str(text_sha256),
            str(provider),
            str(model),
            str(prompt_version),
        ]
    )


def config_suffix(config: LLMScoreConfig) -> str:
    return "||".join([config.provider, config.model, config.prompt_version])


def within_one(left: object, right: object) -> bool:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return False
    return abs(int(left) - int(right)) <= 1


def existing_columns(data: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([col for col in columns if col in data.columns]))


def merge_on_transcript(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = frames[0].copy()
    out["transcriptid"] = out["transcriptid"].astype(str)
    for frame in frames[1:]:
        right = frame.copy()
        right["transcriptid"] = right["transcriptid"].astype(str)
        out = out.merge(right, on="transcriptid", how="inner", validate="one_to_one")
    return out


def grouped_abs_max(
    data: pd.DataFrame, group_col: str, value_col: str, output_col: str
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=[group_col, output_col])
    return (
        data.groupby(group_col, as_index=False)[value_col]
        .apply(lambda values: float(np.nanmax(np.abs(values))))
        .rename(columns={value_col: output_col})
    )


def response_at(data: pd.DataFrame, horizon: int) -> dict[str, object]:
    if data.empty or "horizon" not in data:
        return {}
    subset = data.loc[data["horizon"].eq(horizon)]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.std() == 0 or right.std() == 0:
        return np.nan
    value = left.corr(right)
    return float(value) if pd.notna(value) else np.nan


def round_numeric(data: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = data.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(digits)
    return out


def write_latex(data: pd.DataFrame, path: Path) -> None:
    path.write_text(data.to_latex(index=False, escape=True, float_format="%.4f"), encoding="utf-8")


if __name__ == "__main__":
    legacy_main()
