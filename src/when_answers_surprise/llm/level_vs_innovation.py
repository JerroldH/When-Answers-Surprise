"""Provider-level versus innovation diagnostic for appendix answer-tone scores.

This diagnostic compares raw LLM score levels with A2 filtration residual
innovations. It is a measurement-sensitivity check, not a model benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.llm.answer_pipeline import (  # noqa: E402
    ANALYSIS_ROOT,
    BASELINE_DIR,
    FEATURE_CACHE_DIR,
    answer_filtration_features,
    merge_scores,
)
from when_answers_surprise.sp500_lm import (  # noqa: E402
    OOF_TEST_YEARS,
    aggregate_turn_innovations,
    base_categorical_features,
    oof_predict_by_year,
)


OUTPUT_DIR = ANALYSIS_ROOT / "llm_level_vs_innovation_diagnostic"
TURN_KEYS = ["transcriptid", "turn_index"]
SCORERS = ["Gemini", "DeepSeek", "LM"]
LLM_SCORERS = ["Gemini", "DeepSeek"]
PAIRWISE_SCORERS = list(combinations(SCORERS, 2))
RAW_SCORE_RANGE = [-2, -1, 0, 1, 2]
FILTRATION_VARIANTS = ["A0", "A1", "A2"]
TRANSFORM_LABELS = {
    "raw": "Raw z-score",
    "demean": "Fold demean",
    "A0": "A0 residual",
    "A1": "A1 residual",
    "A2": "A2 residual",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LLM raw answer-tone levels with A2 innovations."
    )
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=FEATURE_CACHE_DIR)
    parser.add_argument("--gemini-dir", type=Path, default=default_gemini_dir())
    parser.add_argument(
        "--deepseek-dir",
        type=Path,
        default=ANALYSIS_ROOT / "llm_answer_tone_deepseek_v4_flash",
    )
    parser.add_argument(
        "--lm-ladder-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sequential_validation" / "table_a_filtration_ladder.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-train-rows", type=int, default=120)
    args = parser.parse_args()

    run_diagnostic(
        baseline_dir=args.baseline_dir,
        feature_cache_dir=args.feature_cache_dir,
        gemini_dir=args.gemini_dir,
        deepseek_dir=args.deepseek_dir,
        lm_ladder_path=args.lm_ladder_path,
        output_dir=args.output_dir,
        min_train_rows=args.min_train_rows,
    )


def default_gemini_dir() -> Path:
    preferred = ANALYSIS_ROOT / "llm_answer_tone_gemini_3_5_flash"
    if preferred.exists():
        return preferred
    return ANALYSIS_ROOT / "llm_answer_tone"


def run_diagnostic(
    *,
    baseline_dir: Path,
    feature_cache_dir: Path,
    gemini_dir: Path,
    deepseek_dir: Path,
    lm_ladder_path: Path,
    output_dir: Path,
    min_train_rows: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_sample = pd.read_parquet(baseline_dir / "response_model_sample.parquet")
    answer_frame = pd.read_parquet(feature_cache_dir / "answer_first_stage_features.parquet")
    normalize_turn_keys(answer_frame)
    ensure_unique(answer_frame, TURN_KEYS, "LM answer first-stage features")

    gemini_scores = load_llm_score_cache(
        gemini_dir,
        "Gemini",
        expected_providers={"gemini", "gemini_adc"},
    )
    deepseek_scores = load_llm_score_cache(
        deepseek_dir,
        "DeepSeek",
        expected_providers={"deepseek"},
    )

    scorer_frames = {
        "Gemini": merge_scores(answer_frame, gemini_scores),
        "DeepSeek": merge_scores(answer_frame, deepseek_scores),
        "LM": answer_frame.copy(),
    }
    llm_variant_turns = {
        scorer: estimate_llm_variant_turns(frame, scorer, min_train_rows=min_train_rows)
        for scorer, frame in scorer_frames.items()
        if scorer in LLM_SCORERS
    }
    llm_demean_turns = {
        scorer: estimate_intercept_only_turns(frame, scorer, min_train_rows=min_train_rows)
        for scorer, frame in scorer_frames.items()
        if scorer in LLM_SCORERS
    }
    a2_turns = {
        "Gemini": llm_variant_turns["Gemini"]["A2"],
        "DeepSeek": llm_variant_turns["DeepSeek"]["A2"],
        "LM": estimate_a2_turns(scorer_frames["LM"], "LM", min_train_rows=min_train_rows),
    }
    common_turns = build_common_turn_sample(a2_turns)
    common_calls = build_common_call_sample(common_turns, response_sample)
    decorate_common_turns(common_turns)
    intercept_alignment = build_intercept_only_alignment(llm_variant_turns, llm_demean_turns)

    tables = {
        "table_llm_level_score_alignment": build_level_score_alignment(common_turns),
        "table_expected_vs_innovation_decomposition": build_expected_decomposition(common_turns),
        "table_innovation_alignment": build_innovation_alignment(common_turns),
        "table_measurement_filtration_ladders": build_measurement_ladders(
            lm_ladder_path=lm_ladder_path,
            gemini_dir=gemini_dir,
            deepseek_dir=deepseek_dir,
        ),
        "table_intercept_only_vs_filtration_alignment": intercept_alignment,
    }
    for name, table in tables.items():
        write_table(table, output_dir / f"{name}.csv", output_dir / f"{name}.tex")

    figure_path = output_dir / "figure_level_vs_innovation_diagnostic.pdf"
    write_diagnostic_figure(common_turns, figure_path)
    intercept_figure_path = output_dir / "figure_intercept_only_vs_filtration_alignment.pdf"
    write_intercept_alignment_figure(intercept_alignment, intercept_figure_path)

    summary = {
        "gemini_dir": str(gemini_dir),
        "deepseek_dir": str(deepseek_dir),
        "lm_ladder_path": str(lm_ladder_path),
        "output_dir": str(output_dir),
        "common_turn_rows": int(len(common_turns)),
        "common_call_rows": int(len(common_calls)),
        "common_response_call_rows": int(common_calls["in_response_sample"].sum()),
        "intercept_alignment_rows": int(len(intercept_alignment)),
        "interpretation": interpretation_from_alignment(tables["table_innovation_alignment"]),
        "outputs": sorted(
            [f"{name}.csv" for name in tables]
            + [figure_path.name, intercept_figure_path.name]
        ),
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def load_llm_score_cache(
    input_dir: Path,
    scorer: str,
    *,
    expected_providers: set[str],
) -> pd.DataFrame:
    path = input_dir / "answer_llm_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {scorer} score cache: {path}")
    scores = pd.read_parquet(path)
    required = [
        "transcriptid",
        "turn_index",
        "text_sha256",
        "provider",
        "model",
        "prompt_version",
        "score",
        "status",
    ]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise ValueError(f"{scorer} score cache missing columns: {missing}")
    normalize_turn_keys(scores)
    providers = set(scores["provider"].dropna().astype(str).unique())
    if not providers.intersection(expected_providers):
        raise ValueError(
            f"{scorer} score cache at {input_dir} has provider(s) {sorted(providers)}, "
            f"expected one of {sorted(expected_providers)}. Refusing to compare fake or wrong-provider scores."
        )
    scores = scores.loc[scores["provider"].isin(expected_providers)].copy()
    scores = scores.loc[scores["status"].eq("ok")].copy()
    ensure_unique(scores, TURN_KEYS, f"{scorer} ok score cache")
    if scores.empty:
        raise ValueError(f"{scorer} score cache has no ok rows after provider/status filtering.")
    return scores


def estimate_a2_turns(
    answer_frame: pd.DataFrame,
    scorer: str,
    *,
    min_train_rows: int,
) -> pd.DataFrame:
    return estimate_variant_turns(
        answer_frame,
        scorer,
        "A2",
        min_train_rows=min_train_rows,
    )


def estimate_llm_variant_turns(
    answer_frame: pd.DataFrame,
    scorer: str,
    *,
    min_train_rows: int,
) -> dict[str, pd.DataFrame]:
    return {
        variant: estimate_variant_turns(
            answer_frame,
            scorer,
            variant,
            min_train_rows=min_train_rows,
        )
        for variant in FILTRATION_VARIANTS
    }


def estimate_variant_turns(
    answer_frame: pd.DataFrame,
    scorer: str,
    variant: str,
    *,
    min_train_rows: int,
) -> pd.DataFrame:
    features = answer_filtration_features(answer_frame)[variant]
    print(
        f"[level-vs-innovation] estimating {variant} first stage for {scorer}: "
        f"{len(features)} features",
        flush=True,
    )
    pred, fold, _fold_metrics = oof_predict_by_year(
        answer_frame,
        target="raw_score",
        feature_columns=features,
        categorical_columns=[col for col in base_categorical_features() if col in features],
        min_train_rows=min_train_rows,
        learner="ridge",
        label=f"{slug(scorer)}_{variant.lower()}_level_vs_innovation",
    )
    columns = [
        "transcriptid",
        "ticker",
        "permno",
        "event_trading_date",
        "event_year",
        "turn_index",
        "word_count",
    ]
    turns = answer_frame[columns].copy()
    turns["component"] = "answer"
    turns["semantic_feature"] = f"{slug(scorer)}_answer_tone"
    turns["raw_score"] = pd.to_numeric(answer_frame["raw_score"], errors="coerce")
    turns["expected_score_oof"] = pred
    turns["narrative_innovation"] = turns["raw_score"] - turns["expected_score_oof"]
    turns["fold_year"] = fold
    turns = turns.loc[np.isfinite(turns["expected_score_oof"])].copy()
    ensure_unique(turns, TURN_KEYS, f"{scorer} {variant} turn innovations")
    columns = [
        "transcriptid",
        "turn_index",
        "ticker",
        "permno",
        "event_trading_date",
        "event_year",
        "word_count",
        "component",
        "semantic_feature",
        "raw_score",
        "expected_score_oof",
        "narrative_innovation",
        "fold_year",
    ]
    out = turns[[col for col in columns if col in turns.columns]].copy()
    rename = {
        "raw_score": f"raw_{slug(scorer)}",
        "expected_score_oof": f"expected_{slug(scorer)}",
        "narrative_innovation": f"innovation_{slug(scorer)}",
        "fold_year": f"fold_year_{slug(scorer)}",
    }
    out = out.rename(columns=rename)
    if scorer != "LM":
        keep = [
            *TURN_KEYS,
            f"raw_{slug(scorer)}",
            f"expected_{slug(scorer)}",
            f"innovation_{slug(scorer)}",
            f"fold_year_{slug(scorer)}",
        ]
        out = out[[col for col in keep if col in out.columns]]
    return out


def estimate_intercept_only_turns(
    answer_frame: pd.DataFrame,
    scorer: str,
    *,
    min_train_rows: int,
) -> pd.DataFrame:
    print(f"[level-vs-innovation] estimating fold-demean residual for {scorer}", flush=True)
    values = pd.to_numeric(answer_frame["raw_score"], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(answer_frame["event_year"], errors="coerce").to_numpy()
    expected = np.full(len(answer_frame), np.nan, dtype=float)
    fold_year = np.full(len(answer_frame), -1, dtype=int)
    for year in OOF_TEST_YEARS:
        train_mask = (years < year) & np.isfinite(values)
        test_mask = years == year
        if train_mask.sum() < min_train_rows or test_mask.sum() == 0:
            continue
        expected[test_mask] = float(np.nanmean(values[train_mask]))
        fold_year[test_mask] = year
    out = answer_frame[TURN_KEYS].copy()
    out[f"raw_{slug(scorer)}"] = values
    out[f"expected_mean_{slug(scorer)}"] = expected
    out[f"demean_{slug(scorer)}"] = values - expected
    out[f"fold_year_mean_{slug(scorer)}"] = fold_year
    out = out.loc[np.isfinite(out[f"expected_mean_{slug(scorer)}"])].copy()
    ensure_unique(out, TURN_KEYS, f"{scorer} fold-demean turn residuals")
    return out


def build_common_turn_sample(a2_turns: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = a2_turns["LM"].copy()
    for scorer in ["Gemini", "DeepSeek"]:
        common = common.merge(a2_turns[scorer], on=TURN_KEYS, how="inner", validate="one_to_one")
    ensure_unique(common, TURN_KEYS, "common turn-level sample")
    if common.empty:
        raise ValueError("Common Gemini/DeepSeek/LM turn-level sample is empty.")
    required = [
        f"{prefix}_{slug(scorer)}"
        for scorer in SCORERS
        for prefix in ["raw", "expected", "innovation"]
    ]
    clean = common.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    if len(clean) != len(common):
        common = clean.copy()
        ensure_unique(common, TURN_KEYS, "common non-missing turn-level sample")
    if common.empty:
        raise ValueError("Common turn-level sample is empty after dropping non-finite scores.")
    return common.reset_index(drop=True)


def build_common_call_sample(common_turns: pd.DataFrame, response_sample: pd.DataFrame) -> pd.DataFrame:
    response = response_sample[["transcriptid"]].copy()
    response["transcriptid"] = response["transcriptid"].astype(str)
    ensure_unique(response, ["transcriptid"], "response sample")
    pieces = []
    base_columns = [
        "transcriptid",
        "ticker",
        "permno",
        "event_trading_date",
        "event_year",
        "turn_index",
        "word_count",
        "component",
        "semantic_feature",
    ]
    for scorer in SCORERS:
        key = slug(scorer)
        turns = common_turns[base_columns].copy()
        turns["raw_score"] = common_turns[f"raw_{key}"]
        turns["expected_score_oof"] = common_turns[f"expected_{key}"]
        turns["narrative_innovation"] = common_turns[f"innovation_{key}"]
        turns["fold_year"] = common_turns.get(f"fold_year_{key}", np.nan)
        call = aggregate_turn_innovations(turns, "answer")
        call = call[["transcriptid", "narrative_innovation", "turn_count"]].rename(
            columns={
                "narrative_innovation": f"call_innovation_{key}",
                "turn_count": f"call_turn_count_{key}",
            }
        )
        ensure_unique(call, ["transcriptid"], f"{scorer} common call innovations")
        pieces.append(call)
    calls = pieces[0]
    for piece in pieces[1:]:
        calls = calls.merge(piece, on="transcriptid", how="inner", validate="one_to_one")
    calls = calls.merge(response.assign(in_response_sample=True), on="transcriptid", how="left")
    calls["in_response_sample"] = calls["in_response_sample"].eq(True)
    ensure_unique(calls, ["transcriptid"], "common call-level sample")
    if calls.empty:
        raise ValueError("Common call-level sample is empty.")
    return calls


def decorate_common_turns(data: pd.DataFrame) -> None:
    for scorer in SCORERS:
        key = slug(scorer)
        data[f"raw_z_{key}"] = zscore(data[f"raw_{key}"])
        data[f"innovation_z_{key}"] = zscore(data[f"innovation_{key}"])
    lm_raw = data["raw_lm"].astype(float)
    data["raw_lm_winsor_1_99"] = winsorize(lm_raw, 0.01, 0.99)
    data["raw_z_lm_winsor_1_99"] = zscore(data["raw_lm_winsor_1_99"])


def build_level_score_alignment(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_objects = {
        "Gemini": ("raw_score", "raw_gemini"),
        "DeepSeek": ("raw_score", "raw_deepseek"),
        "LM": ("raw_lm_net_tone", "raw_lm"),
        "LM winsorized": ("raw_lm_net_tone_winsor_1_99", "raw_lm_winsor_1_99"),
        "LM z-scored": ("raw_lm_net_tone_z", "raw_z_lm"),
    }
    for scorer, (score_object, column) in raw_objects.items():
        for metric, value in describe_series(data[column]).items():
            rows.append(
                {
                    "panel": "distribution",
                    "scorer": scorer,
                    "comparison": "",
                    "score_object": score_object,
                    "metric": metric,
                    "n_obs": int(data[column].notna().sum()),
                    "value": value,
                }
            )

    comparisons = {
        "Gemini-DeepSeek": ("raw_gemini", "raw_deepseek"),
        "Gemini-LM": ("raw_gemini", "raw_lm"),
        "DeepSeek-LM": ("raw_deepseek", "raw_lm"),
    }
    for label, (left, right) in comparisons.items():
        pair = data[[left, right]].dropna()
        metrics = {
            "pearson_corr": safe_corr(pair[left], pair[right], method="pearson"),
            "spearman_corr": safe_corr(pair[left], pair[right], method="spearman"),
        }
        if label == "Gemini-DeepSeek":
            metrics["exact_agreement"] = float((pair[left].astype(int) == pair[right].astype(int)).mean())
            metrics["within_one_agreement"] = float((pair[left] - pair[right]).abs().le(1).mean())
        for metric, value in metrics.items():
            rows.append(
                {
                    "panel": "pairwise_raw_alignment",
                    "scorer": "",
                    "comparison": label,
                    "score_object": "raw_score",
                    "metric": metric,
                    "n_obs": int(len(pair)),
                    "value": value,
                }
            )
    return round_numeric(pd.DataFrame(rows))


def build_expected_decomposition(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scorer in SCORERS:
        key = slug(scorer)
        raw = data[f"raw_{key}"]
        expected = data[f"expected_{key}"]
        innovation = data[f"innovation_{key}"]
        rows.append(
            {
                "scorer": scorer,
                "n_turns": int(len(data)),
                "mean_raw_score": float(raw.mean()),
                "mean_expected_score": float(expected.mean()),
                "mean_innovation": float(innovation.mean()),
                "sd_raw_score": float(raw.std(ddof=1)),
                "sd_expected_score": float(expected.std(ddof=1)),
                "sd_innovation": float(innovation.std(ddof=1)),
                "q05_innovation": float(innovation.quantile(0.05)),
                "q50_innovation": float(innovation.quantile(0.50)),
                "q95_innovation": float(innovation.quantile(0.95)),
            }
        )
    return round_numeric(pd.DataFrame(rows))


def build_innovation_alignment(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scorer in SCORERS:
        key = slug(scorer)
        for metric, value in describe_series(data[f"innovation_z_{key}"]).items():
            rows.append(
                {
                    "panel": "standardized_innovation_distribution",
                    "comparison": "",
                    "scorer": scorer,
                    "metric": metric,
                    "n_obs": int(len(data)),
                    "value": value,
                }
            )

    for left_scorer, right_scorer in PAIRWISE_SCORERS:
        left = slug(left_scorer)
        right = slug(right_scorer)
        pair = data[
            [
                f"raw_{left}",
                f"raw_{right}",
                f"raw_z_{left}",
                f"raw_z_{right}",
                f"innovation_z_{left}",
                f"innovation_z_{right}",
            ]
        ].dropna()
        label = f"{left_scorer}-{right_scorer}"
        metrics = {
            "pearson_corr_innovation_z": safe_corr(
                pair[f"innovation_z_{left}"],
                pair[f"innovation_z_{right}"],
                method="pearson",
            ),
            "spearman_corr_innovation_z": safe_corr(
                pair[f"innovation_z_{left}"],
                pair[f"innovation_z_{right}"],
                method="spearman",
            ),
            "sign_agreement_innovation_z": sign_agreement(
                pair[f"innovation_z_{left}"],
                pair[f"innovation_z_{right}"],
            ),
            "quintile_agreement_innovation_z": quintile_agreement(
                pair[f"innovation_z_{left}"],
                pair[f"innovation_z_{right}"],
            ),
            "ks_raw_score": ks_distance(pair[f"raw_{left}"], pair[f"raw_{right}"]),
            "ks_raw_z_score": ks_distance(pair[f"raw_z_{left}"], pair[f"raw_z_{right}"]),
            "ks_standardized_innovation": ks_distance(
                pair[f"innovation_z_{left}"],
                pair[f"innovation_z_{right}"],
            ),
            "wasserstein_raw_z_score": wasserstein_distance_1d(
                pair[f"raw_z_{left}"],
                pair[f"raw_z_{right}"],
            ),
            "wasserstein_standardized_innovation": wasserstein_distance_1d(
                pair[f"innovation_z_{left}"],
                pair[f"innovation_z_{right}"],
            ),
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "panel": "pairwise_innovation_alignment",
                    "comparison": label,
                    "scorer": "",
                    "metric": metric,
                    "n_obs": int(len(pair)),
                    "value": value,
                }
            )
    return round_numeric(pd.DataFrame(rows))


def build_intercept_only_alignment(
    variant_turns: dict[str, dict[str, pd.DataFrame]],
    demean_turns: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    common = build_intercept_alignment_sample(variant_turns, demean_turns)
    rows = []
    transforms = ["raw", "demean", *FILTRATION_VARIANTS]
    for transform in transforms:
        left = common[f"{transform}_z_gemini"]
        right = common[f"{transform}_z_deepseek"]
        metrics = {
            "ks_distance": ks_distance(left, right),
            "wasserstein_distance": wasserstein_distance_1d(left, right),
            "pearson_corr": safe_corr(left, right, method="pearson"),
            "spearman_corr": safe_corr(left, right, method="spearman"),
            "sign_agreement": sign_agreement(left, right),
            "exact_quintile_agreement": quintile_agreement(left, right),
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "transformation": transform,
                    "transformation_label": TRANSFORM_LABELS[transform],
                    "comparison": "Gemini-DeepSeek",
                    "metric": metric,
                    "n_obs": int(len(common)),
                    "value": value,
                }
            )
    return round_numeric(pd.DataFrame(rows))


def build_intercept_alignment_sample(
    variant_turns: dict[str, dict[str, pd.DataFrame]],
    demean_turns: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces = []
    for scorer in LLM_SCORERS:
        key = slug(scorer)
        base = variant_turns[scorer]["A2"][[*TURN_KEYS, f"raw_{key}"]].copy()
        mean_frame = demean_turns[scorer][[*TURN_KEYS, f"demean_{key}"]].copy()
        pieces.append(base)
        pieces.append(mean_frame)
        for variant in FILTRATION_VARIANTS:
            frame = variant_turns[scorer][variant][[*TURN_KEYS, f"innovation_{key}"]].copy()
            frame = frame.rename(columns={f"innovation_{key}": f"{variant}_{key}"})
            pieces.append(frame)

    common = pieces[0]
    for piece in pieces[1:]:
        common = common.merge(piece, on=TURN_KEYS, how="inner", validate="one_to_one")
    ensure_unique(common, TURN_KEYS, "Gemini/DeepSeek intercept-vs-filtration sample")

    value_cols = [
        f"{transform}_{slug(scorer)}"
        for scorer in LLM_SCORERS
        for transform in ["raw", "demean", *FILTRATION_VARIANTS]
    ]
    common = common.replace([np.inf, -np.inf], np.nan).dropna(subset=value_cols).copy()
    if common.empty:
        raise ValueError("Common Gemini/DeepSeek transformation sample is empty.")
    for scorer in LLM_SCORERS:
        key = slug(scorer)
        for transform in ["raw", "demean", *FILTRATION_VARIANTS]:
            common[f"{transform}_z_{key}"] = zscore(common[f"{transform}_{key}"])
    return common


def build_measurement_ladders(
    *,
    lm_ladder_path: Path,
    gemini_dir: Path,
    deepseek_dir: Path,
) -> pd.DataFrame:
    sources = [
        ("LM dictionary", lm_ladder_path),
        ("Gemini", gemini_dir / "table_llm_answer_filtration_ladder.csv"),
        ("DeepSeek", deepseek_dir / "table_llm_answer_filtration_ladder.csv"),
    ]
    frames = []
    for scorer, path in sources:
        if not path.exists():
            raise FileNotFoundError(f"Missing filtration ladder table for {scorer}: {path}")
        table = read_ladder_source(scorer, path)
        table.insert(0, "scorer", scorer)
        frames.append(table)
    columns = [
        "scorer",
        "variant",
        "variant_label",
        "n_turns",
        "n_calls",
        "n_features",
        "oof_r2",
        "turn_residual_sd",
        "max_abs_observable_corr",
        "max_abs_placebo_t",
        "car_beta_bp_h1",
        "car_t_h1",
        "car_beta_bp_h20",
        "car_t_h20",
    ]
    out = pd.concat(frames, ignore_index=True)
    return round_numeric(out[[col for col in columns if col in out.columns]])


def read_ladder_source(scorer: str, path: Path) -> pd.DataFrame:
    table = pd.read_csv(path).copy()
    if scorer == "Gemini" and "n_turns" in table and table["n_turns"].max() < 10000:
        # Keep the diagnostic reproducible when the local Gemini analysis CSV is a stale smoke run.
        # The paper table is a compact snapshot of the full Gemini ladder produced upstream.
        tex_path = PROJECT_ROOT / "papers" / "finnlp" / "tables" / "table_llm_answer_tone_ladder.tex"
        if tex_path.exists():
            return parse_compact_ladder_tex(tex_path)
    return table


def parse_compact_ladder_tex(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("A") or "&" not in stripped:
            continue
        parts = [part.strip().rstrip("\\") for part in stripped.split("&")]
        if len(parts) < 10:
            continue
        rows.append(
            {
                "variant": parts[0],
                "variant_label": parts[1],
                "n_turns": np.nan,
                "n_calls": np.nan,
                "n_features": np.nan,
                "oof_r2": parse_tex_number(parts[2]),
                "turn_residual_sd": parse_tex_number(parts[3]),
                "max_abs_observable_corr": parse_tex_number(parts[4]),
                "max_abs_placebo_t": parse_tex_number(parts[5]),
                "car_beta_bp_h1": parse_tex_number(parts[6]),
                "car_t_h1": parse_tex_number(parts[7]),
                "car_beta_bp_h20": parse_tex_number(parts[8]),
                "car_t_h20": parse_tex_number(parts[9]),
            }
        )
    if not rows:
        raise ValueError(f"Could not parse compact ladder rows from {path}")
    return pd.DataFrame(rows)


def parse_tex_number(value: str) -> float:
    clean = value.replace(",", "").replace("$", "").strip()
    return float(clean)


def write_diagnostic_figure(data: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    plot_raw_distributions(axes[0, 0], data)
    plot_raw_heatmap(axes[0, 1], data)
    plot_innovation_distributions(axes[1, 0], data)
    plot_innovation_corr_heatmap(axes[1, 1], data)
    fig.suptitle("Provider score levels versus filtration innovations", x=0.02, ha="left", fontsize=10)
    fig.text(
        0.02,
        0.01,
        "Measurement-sensitivity diagnostic only; main LM dictionary results remain primary.",
        fontsize=6.5,
        color="#555555",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95), pad=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_intercept_alignment_figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    order = ["raw", "demean", "A0", "A1", "A2"]
    labels = [TRANSFORM_LABELS[item] for item in order]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), sharex=True)
    plot_metric_panel(
        axes[0],
        table,
        order,
        labels,
        ["ks_distance", "wasserstein_distance"],
        "Distance",
        "Lower is closer",
    )
    plot_metric_panel(
        axes[1],
        table,
        order,
        labels,
        ["pearson_corr", "spearman_corr"],
        "Pointwise correlation",
        "Higher is closer",
    )
    plot_metric_panel(
        axes[2],
        table,
        order,
        labels,
        ["sign_agreement", "exact_quintile_agreement"],
        "Agreement",
        "Higher is closer",
    )
    fig.suptitle(
        "Intercept-only residuals versus filtration residuals",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    fig.text(
        0.02,
        0.01,
        "Gemini versus DeepSeek on the same common answer-turn sample.",
        fontsize=6.5,
        color="#555555",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), pad=0.7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_panel(
    ax,
    table: pd.DataFrame,
    order: list[str],
    labels: list[str],
    metrics: list[str],
    title: str,
    subtitle: str,
) -> None:
    colors = ["#4C78A8", "#E15759"]
    x = np.arange(len(order))
    for metric, color in zip(metrics, colors, strict=False):
        values = []
        for transform in order:
            subset = table.loc[
                table["transformation"].eq(transform) & table["metric"].eq(metric),
                "value",
            ]
            values.append(float(subset.iloc[0]) if not subset.empty else np.nan)
        ax.plot(
            x,
            values,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=color,
            label=metric.replace("_", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.2, frameon=False)
    soften_axes(ax)


def plot_raw_distributions(ax, data: pd.DataFrame) -> None:
    counts = []
    for scorer, column in [("Gemini", "raw_gemini"), ("DeepSeek", "raw_deepseek")]:
        dist = data[column].astype(int).value_counts(normalize=True).reindex(RAW_SCORE_RANGE, fill_value=0)
        for score, share in dist.items():
            counts.append({"scorer": scorer, "score": score, "share": share * 100})
    frame = pd.DataFrame(counts)
    width = 0.36
    x = np.arange(len(RAW_SCORE_RANGE))
    colors = {"Gemini": "#4C78A8", "DeepSeek": "#E15759"}
    for offset, scorer in [(-width / 2, "Gemini"), (width / 2, "DeepSeek")]:
        y = frame.loc[frame["scorer"].eq(scorer), "share"].to_numpy()
        ax.bar(x + offset, y, width=width, color=colors[scorer], label=scorer)
    ax.set_xticks(x)
    ax.set_xticklabels([str(score) for score in RAW_SCORE_RANGE])
    ax.set_ylabel("Share of common turns (%)")
    ax.set_xlabel("Raw integer score")
    ax.set_title("A. Raw score distributions", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False)
    soften_axes(ax)


def plot_raw_heatmap(ax, data: pd.DataFrame) -> None:
    cross = pd.crosstab(
        data["raw_gemini"].astype(int),
        data["raw_deepseek"].astype(int),
        normalize="all",
    ).reindex(index=RAW_SCORE_RANGE, columns=RAW_SCORE_RANGE, fill_value=0)
    image = ax.imshow(cross.to_numpy() * 100, cmap="Blues", vmin=0)
    for i, left in enumerate(cross.index):
        for j, right in enumerate(cross.columns):
            value = cross.loc[left, right] * 100
            if value >= 0.5:
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=5.8)
    ax.set_xticks(np.arange(len(RAW_SCORE_RANGE)))
    ax.set_yticks(np.arange(len(RAW_SCORE_RANGE)))
    ax.set_xticklabels(RAW_SCORE_RANGE)
    ax.set_yticklabels(RAW_SCORE_RANGE)
    ax.set_xlabel("DeepSeek raw score")
    ax.set_ylabel("Gemini raw score")
    ax.set_title("B. Gemini vs DeepSeek raw levels", loc="left", fontsize=8.5)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="% turns")


def plot_innovation_distributions(ax, data: pd.DataFrame) -> None:
    colors = {"Gemini": "#4C78A8", "DeepSeek": "#E15759", "LM": "#59A14F"}
    bins = np.linspace(-4, 4, 49)
    for scorer in SCORERS:
        key = slug(scorer)
        ax.hist(
            data[f"innovation_z_{key}"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.35,
            color=colors[scorer],
            label=scorer,
        )
    ax.set_xlabel("Standardized A2 innovation")
    ax.set_ylabel("Density")
    ax.set_title("C. Innovation distributions", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False)
    soften_axes(ax)


def plot_innovation_corr_heatmap(ax, data: pd.DataFrame) -> None:
    columns = [f"innovation_z_{slug(scorer)}" for scorer in SCORERS]
    corr = data[columns].corr().to_numpy()
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(SCORERS)):
        for j in range(len(SCORERS)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_xticks(np.arange(len(SCORERS)))
    ax.set_yticks(np.arange(len(SCORERS)))
    ax.set_xticklabels(SCORERS, rotation=20, ha="right")
    ax.set_yticklabels(SCORERS)
    ax.set_title("D. Pairwise innovation correlations", loc="left", fontsize=8.5)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def describe_series(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {metric: np.nan for metric in ["mean", "sd", "q05", "q25", "q50", "q75", "q95"]}
    return {
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "q05": float(clean.quantile(0.05)),
        "q25": float(clean.quantile(0.25)),
        "q50": float(clean.quantile(0.50)),
        "q75": float(clean.quantile(0.75)),
        "q95": float(clean.quantile(0.95)),
    }


def safe_corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    clean = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3 or clean.iloc[:, 0].std() == 0 or clean.iloc[:, 1].std() == 0:
        return np.nan
    value = clean.iloc[:, 0].corr(clean.iloc[:, 1], method=method)
    return float(value) if pd.notna(value) else np.nan


def sign_agreement(left: pd.Series, right: pd.Series) -> float:
    clean = pd.concat([left, right], axis=1).dropna()
    if clean.empty:
        return np.nan
    return float((np.sign(clean.iloc[:, 0]) == np.sign(clean.iloc[:, 1])).mean())


def quintile_agreement(left: pd.Series, right: pd.Series) -> float:
    clean = pd.concat([left, right], axis=1).dropna()
    if len(clean) < 5:
        return np.nan
    left_q = pd.qcut(clean.iloc[:, 0].rank(method="first"), 5, labels=False, duplicates="drop")
    right_q = pd.qcut(clean.iloc[:, 1].rank(method="first"), 5, labels=False, duplicates="drop")
    if left_q.isna().any() or right_q.isna().any():
        return np.nan
    return float((left_q.to_numpy() == right_q.to_numpy()).mean())


def ks_distance(left: pd.Series, right: pd.Series) -> float:
    x = clean_array(left)
    y = clean_array(right)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x = np.sort(x)
    y = np.sort(y)
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def wasserstein_distance_1d(left: pd.Series, right: pd.Series) -> float:
    x = np.sort(clean_array(left))
    y = np.sort(clean_array(right))
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n = max(len(x), len(y))
    grid = (np.arange(n) + 0.5) / n
    qx = np.quantile(x, grid)
    qy = np.quantile(y, grid)
    return float(np.mean(np.abs(qx - qy)))


def clean_array(values: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
        .to_numpy()
    )


def winsorize(values: pd.Series, lower: float, upper: float) -> pd.Series:
    lo = values.quantile(lower)
    hi = values.quantile(upper)
    return values.clip(lower=lo, upper=hi)


def zscore(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    sd = clean.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return (clean - clean.mean()) / sd


def interpretation_from_alignment(table: pd.DataFrame) -> str:
    distance = table.loc[
        table["metric"].isin(["ks_raw_z_score", "ks_standardized_innovation"]),
        ["comparison", "metric", "value"],
    ]
    if distance.empty:
        return "Provider scale differences should be treated as measurement sensitivity."
    wide = distance.pivot(index="comparison", columns="metric", values="value")
    if {"ks_raw_z_score", "ks_standardized_innovation"}.issubset(wide.columns):
        improvement = (wide["ks_standardized_innovation"] < wide["ks_raw_z_score"]).mean()
        if improvement >= 0.5:
            return (
                "Provider disagreement in score levels is partly absorbed by the filtration "
                "benchmark; the residual innovation is more comparable than the raw level score."
            )
    return (
        "Provider scale differences remain after residualization; the LLM check should be treated "
        "as measurement sensitivity, not supportive evidence."
    )


def normalize_turn_keys(data: pd.DataFrame) -> None:
    data["transcriptid"] = data["transcriptid"].astype(str)
    data["turn_index"] = pd.to_numeric(data["turn_index"], errors="raise").astype(int)


def ensure_unique(data: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicate_count = int(data.duplicated(keys).sum())
    if duplicate_count:
        examples = data.loc[data.duplicated(keys, keep=False), keys].head(5).to_dict("records")
        raise ValueError(f"{label} has {duplicate_count} duplicate key rows for {keys}: {examples}")


def write_table(data: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    tex_path.write_text(
        data.to_latex(index=False, escape=True, float_format="%.4f"),
        encoding="utf-8",
    )


def round_numeric(data: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = data.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(digits)
    return out


def soften_axes(ax) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", labelsize=7)


def slug(scorer: str) -> str:
    return scorer.lower().replace(" ", "_")


if __name__ == "__main__":
    main()
