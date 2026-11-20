"""Paper-facing validation experiments for sequential LM tone.

This stage reads frozen baseline artifacts and first-stage feature frames. It
does not rebuild raw WRDS data and does not use upstream estimated residuals as
downstream first-stage inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from when_answers_surprise.sp500_lm import (
    HORIZONS,
    LM_FEATURES,
    OOF_TEST_YEARS,
    PRIMARY_FEATURE,
    aggregate_turn_innovations,
    base_categorical_features,
    base_state_features,
    calibration_slope,
    fit_response_regression,
    oof_predict_by_year,
    response_controls,
    zscore,
)
from when_answers_surprise.sp500_lm_diagnostics import (
    add_nuisance_time_features,
    oof_outcome_residuals,
    placebo_nuisance_controls,
)
from when_answers_surprise.empirical.feature_spec import (
    all_feature_manifests,
    assert_feature_manifest_closure,
    answer_filtration_features as canonical_answer_filtration_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025" / "analysis"
BASELINE_DIR = ANALYSIS_ROOT / "lm_baseline"
FEATURE_CACHE_DIR = ANALYSIS_ROOT / "lm_baseline_learner_audit_cache"
OUTPUT_DIR = ANALYSIS_ROOT / "sequential_validation"

VARIANT_LABELS = {
    "A0": "A0: pre-call state",
    "A1": "A1: + prepared/prior dialogue",
    "A2": "A2: + current question",
}
OBSERVABLES = [
    "eps_surprise",
    "forecast_dispersion",
    "analyst_coverage",
    "pre_ret_20",
    "pre_vol_20",
]
STAGES = ("prepared", "question", "answer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential validation experiments.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=FEATURE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument(
        "--only-lm-intercept-diagnostic",
        action="store_true",
        help="Only build the LM raw/fold-demean/filtration appendix diagnostic.",
    )
    args = parser.parse_args()
    if args.only_lm_intercept_diagnostic:
        run_lm_intercept_only_diagnostic(
            args.baseline_dir,
            args.feature_cache_dir,
            args.output_dir,
            args.min_train_rows,
        )
    else:
        run_experiments(args.baseline_dir, args.feature_cache_dir, args.output_dir, args.min_train_rows)


def run_experiments(
    baseline_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    min_train_rows: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest_path = baseline_dir / "baseline_feature_manifest.json"
    cache_manifest_path = feature_cache_dir / "cache_manifest.json"
    if not baseline_manifest_path.exists() or not cache_manifest_path.exists():
        raise FileNotFoundError("Sequential validation requires baseline and cache feature manifests")
    assert_feature_manifest_closure(
        json.loads(baseline_manifest_path.read_text(encoding="utf-8")),
        json.loads(cache_manifest_path.read_text(encoding="utf-8")),
    )
    response_sample = pd.read_parquet(baseline_dir / "response_model_sample.parquet")
    answer_frame = pd.read_parquet(feature_cache_dir / "answer_first_stage_features.parquet")
    qa_turns = read_optional_parquet(baseline_dir / "qa_turns.parquet")

    ladder = estimate_answer_filtration_ladder(
        answer_frame,
        response_sample,
        min_train_rows=min_train_rows,
    )
    stress = estimate_nuisance_stress(
        feature_cache_dir,
        response_sample,
        min_train_rows=min_train_rows,
    )

    table_a = build_table_a(ladder)
    table_b = build_table_b(stress["detail"])
    sample_audit = same_sample_audit(stress["detail"])
    examples = build_a2_examples(ladder["turn_innovations"], response_sample, qa_turns)
    lm_intercept = estimate_lm_intercept_only_vs_filtration(
        answer_frame,
        response_sample,
        min_train_rows=min_train_rows,
        ladder_table=table_a,
    )

    outputs = {
        "table_a_filtration_ladder": table_a,
        "table_b_nuisance_stress_diagnostic": table_b,
        "table_lm_intercept_only_vs_filtration": lm_intercept,
        "filtration_ladder_first_stage_detail": ladder["first_stage"],
        "filtration_ladder_correlations": ladder["correlations"],
        "filtration_ladder_placebo_response": ladder["placebo"],
        "filtration_ladder_main_response": ladder["response"],
        "filtration_ladder_a2_examples": examples,
        "nuisance_stress_detail": stress["detail"],
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        if name.startswith("table_"):
            write_latex(table, output_dir / f"{name}.tex")

    summary = {
        "baseline_dir": str(baseline_dir),
        "feature_cache_dir": str(feature_cache_dir),
        "output_dir": str(output_dir),
        "answer_turn_rows": int(len(answer_frame)),
        "response_sample_rows": int(len(response_sample)),
        "table_a_rows": int(len(table_a)),
        "table_b_rows": int(len(table_b)),
        "lm_intercept_only_rows": int(len(lm_intercept)),
        "table_a_interpretation": "Answer-response coefficients attenuate but remain positive/significant as the filtration expands.",
        "table_b_description": "Nuisance-stress diagnostic; not proof that orthogonalization is uniformly more stable.",
        "table_b_note": "All naive and orthogonal estimates are computed on the same row intersection.",
        **sample_audit,
        "outputs": sorted(f"{name}.csv" for name in outputs)
        + ["figure_lm_intercept_only_vs_filtration.pdf"],
    }
    write_lm_intercept_only_figure(
        lm_intercept,
        output_dir / "figure_lm_intercept_only_vs_filtration.pdf",
    )
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def run_lm_intercept_only_diagnostic(
    baseline_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    min_train_rows: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_sample = pd.read_parquet(baseline_dir / "response_model_sample.parquet")
    answer_frame = pd.read_parquet(feature_cache_dir / "answer_first_stage_features.parquet")
    ladder_table = load_existing_lm_ladder_table(output_dir)
    if ladder_table is None:
        ladder = estimate_answer_filtration_ladder(
            answer_frame,
            response_sample,
            min_train_rows=min_train_rows,
        )
        ladder_table = build_table_a(ladder)
    table = estimate_lm_intercept_only_vs_filtration(
        answer_frame,
        response_sample,
        min_train_rows=min_train_rows,
        ladder_table=ladder_table,
    )
    table.to_csv(output_dir / "table_lm_intercept_only_vs_filtration.csv", index=False)
    write_latex(table, output_dir / "table_lm_intercept_only_vs_filtration.tex")
    write_lm_intercept_only_figure(
        table,
        output_dir / "figure_lm_intercept_only_vs_filtration.pdf",
    )
    summary = {
        "baseline_dir": str(baseline_dir),
        "feature_cache_dir": str(feature_cache_dir),
        "output_dir": str(output_dir),
        "answer_turn_rows": int(len(answer_frame)),
        "response_sample_rows": int(len(response_sample)),
        "lm_intercept_only_rows": int(len(table)),
        "outputs": [
            "table_lm_intercept_only_vs_filtration.csv",
            "table_lm_intercept_only_vs_filtration.tex",
            "figure_lm_intercept_only_vs_filtration.pdf",
        ],
    }
    (output_dir / "lm_intercept_only_vs_filtration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def estimate_answer_filtration_ladder(
    answer_frame: pd.DataFrame,
    response_sample: pd.DataFrame,
    *,
    min_train_rows: int,
) -> dict[str, pd.DataFrame]:
    feature_sets = answer_filtration_features(answer_frame)
    turn_rows = []
    first_stage_rows = []
    for variant, features in feature_sets.items():
        print(f"[validation] filtration_ladder answer {variant} n_features={len(features)}", flush=True)
        pred, fold, fold_metrics = oof_predict_by_year(
            answer_frame,
            target="raw_score",
            feature_columns=features,
            categorical_columns=[col for col in base_categorical_features() if col in features],
            min_train_rows=min_train_rows,
            learner="ridge",
            label=f"answer_{variant}",
        )
        turns = answer_frame[
            ["transcriptid", "ticker", "permno", "event_trading_date", "event_year", "turn_index", "word_count"]
        ].copy()
        turns["component"] = "answer"
        turns["semantic_feature"] = PRIMARY_FEATURE
        turns["raw_score"] = pd.to_numeric(answer_frame["raw_score"], errors="coerce")
        turns["expected_score_oof"] = pred
        turns["narrative_innovation"] = turns["raw_score"] - turns["expected_score_oof"]
        turns["fold_year"] = fold
        turns["first_stage_estimable"] = np.isfinite(pred)
        turns = turns.loc[turns["first_stage_estimable"]].copy()
        turns["variant"] = variant
        turn_rows.append(turns)
        first_stage_rows.append(first_stage_summary(variant, turns, fold_metrics, len(features)))

    turn_innovations = pd.concat(turn_rows, ignore_index=True)
    call_variants = []
    for variant, group in turn_innovations.groupby("variant", sort=True):
        call = aggregate_turn_innovations(group, "answer")
        call = call[["transcriptid", "narrative_innovation", "turn_count"]].rename(
            columns={"narrative_innovation": f"answer_{variant}", "turn_count": f"answer_{variant}_turn_count"}
        )
        call_variants.append(call)
    call_wide = merge_on_transcript(call_variants)
    data = response_sample.assign(transcriptid=response_sample["transcriptid"].astype(str)).merge(
        call_wide, on="transcriptid", how="inner", validate="one_to_one"
    )
    for variant in feature_sets:
        data[f"answer_{variant}_z"] = zscore(data[f"answer_{variant}"])

    return {
        "turn_innovations": turn_innovations,
        "call_sample": data,
        "first_stage": pd.DataFrame(first_stage_rows),
        "correlations": observable_correlations(data, feature_sets),
        "placebo": placebo_responses(data, feature_sets),
        "response": main_responses(data, feature_sets),
    }


def answer_filtration_features(frame: pd.DataFrame) -> dict[str, list[str]]:
    return canonical_answer_filtration_features(frame, label_prefix="sequential answer filtration")


def estimate_lm_intercept_only_vs_filtration(
    answer_frame: pd.DataFrame,
    response_sample: pd.DataFrame,
    *,
    min_train_rows: int,
    ladder_table: pd.DataFrame,
) -> pd.DataFrame:
    raw_demean = build_lm_raw_demean_rows(
        answer_frame,
        response_sample,
        min_train_rows=min_train_rows,
    )
    filtration = lm_intercept_rows_from_ladder_table(
        ladder_table,
        response_n_calls=int(response_sample["transcriptid"].nunique()),
    )
    return round_numeric(pd.concat([raw_demean, filtration], ignore_index=True))


def build_lm_raw_demean_rows(
    answer_frame: pd.DataFrame,
    response_sample: pd.DataFrame,
    *,
    min_train_rows: int,
) -> pd.DataFrame:
    common_turns = build_lm_raw_demean_turns(answer_frame, min_train_rows=min_train_rows)
    call_sample = build_lm_intercept_call_sample(common_turns, response_sample, ["raw", "demean"])
    rows = []
    for measure, label in list(lm_intercept_measure_labels().items())[:2]:
        treatment = f"{measure}_z"
        transformed = common_turns[f"{measure}_value"]
        rows.append(
            {
                "measure": measure,
                "measure_label": label,
                "n_turns": int(transformed.notna().sum()),
                "n_calls": int(call_sample["transcriptid"].nunique()),
                "turn_transformed_sd": float(transformed.std(ddof=1)),
                "max_abs_observable_corr": max_abs_observable_corr(call_sample, treatment),
                "max_abs_placebo_t": max_abs_placebo_t(call_sample, treatment),
                **response_at_horizons(call_sample, treatment),
            }
        )
    return pd.DataFrame(rows)


def build_lm_raw_demean_turns(
    answer_frame: pd.DataFrame,
    *,
    min_train_rows: int,
) -> pd.DataFrame:
    frame = answer_frame.copy()
    frame["transcriptid"] = frame["transcriptid"].astype(str)
    raw = frame[
        [
            "transcriptid",
            "ticker",
            "permno",
            "event_trading_date",
            "event_year",
            "turn_index",
            "word_count",
            "raw_score",
        ]
    ].copy()
    raw = raw.rename(columns={"raw_score": "raw_value"})
    raw["demean_value"] = fold_demeaned_values(frame, min_train_rows=min_train_rows)

    common = raw.copy()
    value_cols = ["raw_value", "demean_value"]
    common = common.replace([np.inf, -np.inf], np.nan).dropna(subset=value_cols).copy()
    ensure_unique_turns(common, "LM intercept common turns")
    if common.empty:
        raise ValueError("LM raw/demean diagnostic has no common answer turns.")
    return common


def fold_demeaned_values(answer_frame: pd.DataFrame, *, min_train_rows: int) -> np.ndarray:
    values = pd.to_numeric(answer_frame["raw_score"], errors="coerce").to_numpy(dtype=float)
    years = pd.to_numeric(answer_frame["event_year"], errors="coerce").to_numpy()
    expected = np.full(len(answer_frame), np.nan, dtype=float)
    for year in OOF_TEST_YEARS:
        train_mask = (years < year) & np.isfinite(values)
        test_mask = years == year
        if train_mask.sum() < min_train_rows or test_mask.sum() == 0:
            continue
        expected[test_mask] = float(np.nanmean(values[train_mask]))
    return values - expected


def build_lm_intercept_call_sample(
    common_turns: pd.DataFrame,
    response_sample: pd.DataFrame,
    measures: Sequence[str],
) -> pd.DataFrame:
    calls = []
    for measure in measures:
        turns = common_turns[
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
        turns["component"] = "answer"
        turns["semantic_feature"] = f"lm_answer_{measure}"
        turns["raw_score"] = common_turns[f"{measure}_value"]
        turns["expected_score_oof"] = 0.0
        turns["narrative_innovation"] = common_turns[f"{measure}_value"]
        turns["fold_year"] = common_turns["event_year"]
        call = aggregate_turn_innovations(turns, "answer")
        call = call[["transcriptid", "narrative_innovation"]].rename(
            columns={"narrative_innovation": measure}
        )
        calls.append(call)
    call_wide = merge_on_transcript(calls)
    data = response_sample.assign(transcriptid=response_sample["transcriptid"].astype(str)).merge(
        call_wide,
        on="transcriptid",
        how="inner",
        validate="one_to_one",
    )
    for measure in measures:
        data[f"{measure}_z"] = zscore(data[measure])
    return data


def lm_intercept_rows_from_ladder_table(
    ladder_table: pd.DataFrame,
    *,
    response_n_calls: int,
) -> pd.DataFrame:
    labels = lm_intercept_measure_labels()
    rows = []
    for variant in ["A0", "A1", "A2"]:
        row = ladder_table.loc[ladder_table["variant"].eq(variant)]
        if row.empty:
            raise ValueError(f"Missing {variant} row in LM filtration ladder table.")
        item = row.iloc[0]
        rows.append(
            {
                "measure": variant,
                "measure_label": labels[variant],
                "n_turns": int(item["n_turns"]) if pd.notna(item.get("n_turns")) else np.nan,
                "n_calls": response_n_calls,
                "turn_transformed_sd": float(item["turn_residual_sd"]),
                "max_abs_observable_corr": float(item["max_abs_observable_corr"]),
                "max_abs_placebo_t": float(item["max_abs_placebo_t"]),
                "car_beta_bp_h1": float(item["car_beta_bp_h1"]),
                "car_t_h1": float(item["car_t_h1"]),
                "car_beta_bp_h20": float(item["car_beta_bp_h20"]),
                "car_t_h20": float(item["car_t_h20"]),
            }
        )
    return pd.DataFrame(rows)


def load_existing_lm_ladder_table(output_dir: Path) -> pd.DataFrame | None:
    candidates = [
        output_dir / "table_a_filtration_ladder.csv",
        PROJECT_ROOT / "outputs" / "sequential_validation" / "table_a_filtration_ladder.csv",
        ANALYSIS_ROOT / "sequential_validation" / "table_a_filtration_ladder.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def lm_intercept_measure_labels() -> dict[str, str]:
    return {
        "raw": "Raw LM tone",
        "demean": "Fold-demeaned",
        "A0": "A0 residual",
        "A1": "A1 residual",
        "A2": "A2 residual",
    }


def max_abs_observable_corr(data: pd.DataFrame, treatment: str) -> float:
    values = []
    for observable in OBSERVABLES:
        if observable not in data:
            continue
        clean = data[[treatment, observable]].replace([np.inf, -np.inf], np.nan).dropna()
        values.append(abs(safe_corr(clean[treatment], clean[observable])))
    return float(np.nanmax(values)) if values else np.nan


def max_abs_placebo_t(data: pd.DataFrame, treatment: str) -> float:
    work = add_nuisance_time_features(data)
    controls = placebo_nuisance_controls(work, response_controls(work))
    values = []
    for outcome in ["placebo_CAR_m5_m1", "placebo_CAR_m10_m1"]:
        result = fit_response_regression(
            work,
            outcome=outcome,
            treatments=[treatment],
            controls=[col for col in controls if col != treatment],
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        values.append(abs(float(result.coefficients.iloc[0]["t_stat"])))
    return float(np.nanmax(values)) if values else np.nan


def response_at_horizons(data: pd.DataFrame, treatment: str) -> dict[str, float]:
    controls = response_controls(data)
    out = {}
    for horizon in [1, 20]:
        result = fit_response_regression(
            data,
            outcome=f"CAR_0_{horizon}",
            treatments=[treatment],
            controls=[col for col in controls if col != treatment],
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        coef = result.coefficients.iloc[0]
        out[f"car_beta_bp_h{horizon}"] = float(coef["beta"] * 10000)
        out[f"car_t_h{horizon}"] = float(coef["t_stat"])
    return out


def ensure_unique_turns(data: pd.DataFrame, label: str) -> None:
    duplicate_count = int(data.duplicated(["transcriptid", "turn_index"]).sum())
    if duplicate_count:
        examples = (
            data.loc[data.duplicated(["transcriptid", "turn_index"], keep=False), ["transcriptid", "turn_index"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"{label} has duplicate transcript-turn keys: {examples}")


def write_lm_intercept_only_figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(lm_intercept_measure_labels())
    labels = [lm_intercept_measure_labels()[measure] for measure in order]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    plot_lm_intercept_metric(
        axes[0],
        table,
        order,
        labels,
        ["max_abs_observable_corr", "max_abs_placebo_t"],
        "Confounding diagnostics",
    )
    plot_lm_intercept_metric(
        axes[1],
        table,
        order,
        labels,
        ["car_beta_bp_h1", "car_beta_bp_h20"],
        "Post-call response (bps)",
    )
    plot_lm_intercept_metric(
        axes[2],
        table,
        order,
        labels,
        ["turn_transformed_sd"],
        "Turn-level scale",
    )
    fig.suptitle("LM raw tone, fold mean subtraction, and filtration residuals", x=0.02, ha="left", fontsize=10)
    fig.text(
        0.02,
        0.01,
        "Appendix diagnostic only; main LM filtration estimates are unchanged.",
        fontsize=6.5,
        color="#555555",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), pad=0.7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_lm_intercept_metric(
    ax,
    table: pd.DataFrame,
    order: list[str],
    labels: list[str],
    columns: list[str],
    title: str,
) -> None:
    x = np.arange(len(order))
    colors = ["#4C78A8", "#E15759", "#59A14F"]
    for col, color in zip(columns, colors, strict=False):
        indexed = table.set_index("measure")
        values = indexed.loc[order, col].astype(float).to_numpy()
        ax.plot(
            x,
            values,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            label=clean_metric_label(col),
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False)


def clean_metric_label(column: str) -> str:
    labels = {
        "max_abs_observable_corr": "max observable corr.",
        "max_abs_placebo_t": "max placebo |t|",
        "car_beta_bp_h1": "CAR[0,1]",
        "car_beta_bp_h20": "CAR[0,20]",
        "turn_transformed_sd": "turn SD",
    }
    return labels.get(column, column)


def first_stage_summary(
    variant: str,
    turns: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    n_features: int,
) -> dict[str, object]:
    clean = turns.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["raw_score", "expected_score_oof", "narrative_innovation"]
    )
    actual = clean["raw_score"].astype(float).to_numpy()
    pred = clean["expected_score_oof"].astype(float).to_numpy()
    resid = clean["narrative_innovation"].astype(float).to_numpy()
    return {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "n_turns": int(len(clean)),
        "n_calls": int(clean["transcriptid"].nunique()),
        "n_features": int(n_features),
        "oof_r2": float(r2_score(actual, pred)) if len(clean) > 2 else np.nan,
        "oof_mae": float(mean_absolute_error(actual, pred)) if len(clean) else np.nan,
        "calibration_slope": calibration_slope(actual, pred),
        "turn_residual_sd": float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else np.nan,
        "mean_fold_r2": float(fold_metrics["r2"].mean(skipna=True)) if not fold_metrics.empty else np.nan,
    }


def observable_correlations(data: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for variant in feature_sets:
        innovation = f"answer_{variant}_z"
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
    work = add_nuisance_time_features(data)
    controls = placebo_nuisance_controls(work, response_controls(work))
    rows = []
    for variant in feature_sets:
        treatment = f"answer_{variant}_z"
        for outcome in ["placebo_CAR_m5_m1", "placebo_CAR_m10_m1"]:
            result = fit_response_regression(
                work,
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
        treatment = f"answer_{variant}_z"
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
            rows.append({"variant": variant, "horizon": horizon, "outcome": f"CAR_0_{horizon}", **out})
    return pd.DataFrame(rows)


def estimate_nuisance_stress(
    feature_cache_dir: Path,
    response_sample: pd.DataFrame,
    *,
    min_train_rows: int,
) -> dict[str, pd.DataFrame]:
    m_results = {}
    for stage in STAGES:
        frame = pd.read_parquet(feature_cache_dir / f"{stage}_first_stage_features.parquet")
        for m_spec, features in stage_m_features(stage, frame).items():
            print(f"[validation] first_stage {stage} {m_spec}", flush=True)
            pred, fold, _ = oof_predict_by_year(
                frame,
                target="raw_score",
                feature_columns=features,
                categorical_columns=[col for col in base_categorical_features() if col in features],
                min_train_rows=min_train_rows,
                learner="ridge",
                label=f"{stage}_{m_spec}",
            )
            innovations = feature_frame_to_stage_innovations(stage, frame, pred, fold)
            call = (
                aggregate_turn_innovations(innovations, stage)
                if stage in {"question", "answer"}
                else innovations
            )
            m_results[(stage, m_spec)] = call[["transcriptid", "narrative_innovation"]].rename(
                columns={"narrative_innovation": "u_raw"}
            )

    outcome_base = add_nuisance_time_features(response_sample.assign(transcriptid=response_sample["transcriptid"].astype(str)))
    g_residuals = {}
    g_controls = outcome_control_grid(outcome_base)
    for horizon in HORIZONS:
        outcome = f"CAR_0_{horizon}"
        for g_spec, controls in g_controls.items():
            print(f"[validation] outcome_nuisance {outcome} {g_spec}", flush=True)
            v_hat, g_diag = oof_outcome_residuals(
                outcome_base,
                outcome=outcome,
                controls=controls,
                min_train_rows=min_train_rows,
                profile=g_spec,
            )
            g_residuals[(outcome, g_spec)] = (
                outcome_base[["transcriptid"]].assign(v_hat=v_hat),
                g_diag,
            )

    rows = []
    for stage in STAGES:
        for m_spec in ["m_weak", "m_full"]:
            sample = response_sample.assign(transcriptid=response_sample["transcriptid"].astype(str)).merge(
                m_results[(stage, m_spec)], on="transcriptid", how="inner", validate="one_to_one"
            )
            sample["u_hat"] = zscore(sample["u_raw"])
            for horizon in HORIZONS:
                outcome = f"CAR_0_{horizon}"
                sample_with_g = sample.copy()
                for g_spec in ["g_weak", "g_full"]:
                    v_frame, _ = g_residuals[(outcome, g_spec)]
                    sample_with_g = sample_with_g.merge(
                        v_frame.rename(columns={"v_hat": f"v_hat_{g_spec}"}),
                        on="transcriptid",
                        how="left",
                        validate="one_to_one",
                    )
                common = sample_with_g[
                    ["u_hat", outcome, "v_hat_g_weak", "v_hat_g_full"]
                ].replace([np.inf, -np.inf], np.nan).dropna()
                theta_naive = theta_on_common(common, outcome)
                n_common = int(len(common))
                for g_spec in ["g_weak", "g_full"]:
                    _, g_diag = g_residuals[(outcome, g_spec)]
                    theta_orthogonal = theta_on_common(common, f"v_hat_{g_spec}")
                    rows.extend(
                        [
                            stress_row(stage, horizon, m_spec, g_spec, "naive", theta_naive, n_common, g_diag),
                            stress_row(
                                stage,
                                horizon,
                                m_spec,
                                g_spec,
                                "orthogonal",
                                theta_orthogonal,
                                n_common,
                                g_diag,
                            ),
                        ]
                    )
    detail = pd.DataFrame(rows)
    assert_same_sample(detail)
    detail = add_sensitivity(detail)
    return {"detail": detail}


def stage_m_features(stage: str, frame: pd.DataFrame) -> dict[str, list[str]]:
    del frame
    specs = all_feature_manifests()
    state = specs["A0"]["feature_columns"]
    if stage == "prepared":
        full = state
    elif stage == "question":
        full = specs["question"]["feature_columns"]
    elif stage == "answer":
        full = specs["A2"]["feature_columns"]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return {"m_weak": state, "m_full": full}


def feature_frame_to_stage_innovations(
    stage: str,
    frame: pd.DataFrame,
    pred: np.ndarray,
    fold: np.ndarray,
) -> pd.DataFrame:
    id_cols = ["transcriptid", "ticker", "permno", "event_trading_date", "event_year"]
    if stage in {"question", "answer"}:
        id_cols.extend(["turn_index", "word_count"])
    out = frame[id_cols].copy()
    out["component"] = stage
    out["semantic_feature"] = PRIMARY_FEATURE
    out["raw_score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
    out["expected_score_oof"] = pred
    out["narrative_innovation"] = out["raw_score"] - out["expected_score_oof"]
    out["fold_year"] = fold
    out["first_stage_estimable"] = np.isfinite(pred)
    return out.loc[out["first_stage_estimable"]].copy()


def outcome_control_grid(data: pd.DataFrame) -> dict[str, list[str]]:
    weak = existing_columns(data, ["scaled_eps_surprise", "pre_ret_20", "pre_vol_20", "log_market_cap"])
    full = response_controls(data)
    return {"g_weak": weak, "g_full": full}


def theta_on_common(common: pd.DataFrame, target: str) -> float:
    denom = float(np.square(common["u_hat"]).sum())
    if len(common) < 3 or denom <= 0:
        return np.nan
    return float((common["u_hat"] * common[target]).sum() / denom)


def stress_row(
    stage: str,
    horizon: int,
    m_spec: str,
    g_spec: str,
    estimator: str,
    theta: float,
    n_obs: int,
    g_diag: pd.DataFrame,
) -> dict[str, object]:
    return {
        "stage": stage,
        "horizon": horizon,
        "outcome": f"CAR_0_{horizon}",
        "m_spec": m_spec,
        "g_spec": g_spec,
        "estimator": estimator,
        "theta": theta,
        "n_obs": n_obs,
        "n_common": n_obs,
        "same_sample_flag": True,
        "common_sample_scope": "intersection_g_weak_g_full",
        "g_estimated_folds": int(g_diag["status"].eq("estimated").sum()) if "status" in g_diag else np.nan,
    }


def assert_same_sample(detail: pd.DataFrame) -> None:
    counts = (
        detail.groupby(["stage", "horizon", "m_spec", "g_spec"], dropna=False)["n_obs"]
        .nunique()
        .reset_index(name="n_unique_n_obs")
    )
    bad = counts.loc[counts["n_unique_n_obs"].ne(1)]
    if not bad.empty:
        raise ValueError(f"Naive and orthogonal estimators use different samples: {bad.to_dict('records')[:5]}")


def add_sensitivity(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    base = out.loc[out["m_spec"].eq("m_full") & out["g_spec"].eq("g_full")][
        ["stage", "horizon", "estimator", "theta"]
    ].rename(columns={"theta": "full_full_theta"})
    out = out.merge(base, on=["stage", "horizon", "estimator"], how="left")
    out["abs_deviation_from_full_full"] = (out["theta"] - out["full_full_theta"]).abs()
    max_dev = (
        out.groupby(["stage", "horizon", "estimator"], dropna=False)["abs_deviation_from_full_full"]
        .max()
        .reset_index(name="max_abs_deviation_from_full_full")
    )
    return out.merge(max_dev, on=["stage", "horizon", "estimator"], how="left")


def build_table_a(ladder: dict[str, pd.DataFrame]) -> pd.DataFrame:
    first_stage = ladder["first_stage"].copy()
    corr = (
        ladder["correlations"]
        .groupby("variant", as_index=False)["correlation"]
        .apply(lambda x: float(np.nanmax(np.abs(x))))
        .rename(columns={"correlation": "max_abs_observable_corr"})
    )
    placebo = (
        ladder["placebo"]
        .groupby("variant", as_index=False)["t_stat"]
        .apply(lambda x: float(np.nanmax(np.abs(x))))
        .rename(columns={"t_stat": "max_abs_placebo_t"})
    )
    response_beta = ladder["response"].pivot(index="variant", columns="horizon", values="beta").add_prefix("car_beta_h")
    response_t = ladder["response"].pivot(index="variant", columns="horizon", values="t_stat").add_prefix("car_t_h")
    response = response_beta.join(response_t, how="outer")
    out = first_stage.merge(corr, on="variant", how="left").merge(placebo, on="variant", how="left")
    out = out.merge(response.reset_index(), on="variant", how="left")
    for horizon in [1, 20]:
        out[f"car_beta_bp_h{horizon}"] = 10000.0 * out[f"car_beta_h{horizon}"]
    return round_numeric(
        out[
            [
                "variant",
                "variant_label",
                "n_turns",
                "n_calls",
                "n_features",
                "oof_r2",
                "oof_mae",
                "calibration_slope",
                "turn_residual_sd",
                "max_abs_observable_corr",
                "max_abs_placebo_t",
                "car_beta_bp_h1",
                "car_t_h1",
                "car_beta_bp_h20",
                "car_t_h20",
            ]
        ]
    )


def build_table_b(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, estimator), group in detail.groupby(["stage", "estimator"], sort=True):
        wide = group.pivot_table(
            index="horizon",
            columns=["m_spec", "g_spec"],
            values="theta",
            aggfunc="first",
        ).reindex(
            columns=pd.MultiIndex.from_product(
                [["m_weak", "m_full"], ["g_weak", "g_full"]],
                names=["m_spec", "g_spec"],
            )
        )
        full_full = wide[("m_full", "g_full")]
        g_sensitivity = (wide[("m_full", "g_weak")] - full_full).abs()
        m_sensitivity = (wide[("m_weak", "g_full")] - full_full).abs()
        all_grid = group.groupby("horizon")["abs_deviation_from_full_full"].max()
        rows.append(
            {
                "stage": stage,
                "estimator": estimator,
                "diagnostic": "nuisance_stress",
                "mean_full_full_theta": float(full_full.mean(skipna=True)),
                "max_g_sensitivity_at_m_full": float(g_sensitivity.max(skipna=True)),
                "mean_g_sensitivity_at_m_full": float(g_sensitivity.mean(skipna=True)),
                "max_m_sensitivity_at_g_full": float(m_sensitivity.max(skipna=True)),
                "mean_m_sensitivity_at_g_full": float(m_sensitivity.mean(skipna=True)),
                "all_grid_sensitivity": float(all_grid.max(skipna=True)),
                "n_horizons": int(all_grid.notna().sum()),
            }
        )
    return round_numeric(pd.DataFrame(rows))


def same_sample_audit(detail: pd.DataFrame) -> dict[str, object]:
    bad_groups = int(
        (
            detail.groupby(["stage", "horizon", "m_spec", "g_spec"], dropna=False)["n_obs"].nunique()
            != 1
        ).sum()
    )
    return {
        "same_sample_all": bool(detail["same_sample_flag"].all() and bad_groups == 0),
        "same_sample_bad_groups": bad_groups,
    }


def build_a2_examples(
    turn_innovations: pd.DataFrame,
    response_sample: pd.DataFrame,
    qa_turns: pd.DataFrame,
    n_per_tail: int = 5,
) -> pd.DataFrame:
    if qa_turns.empty:
        return pd.DataFrame()
    turns = turn_innovations.loc[turn_innovations["variant"].eq("A2")].copy()
    turns = turns.loc[turns["word_count"].ge(20)].copy()
    turns["transcriptid"] = turns["transcriptid"].astype(str)
    qa = qa_turns.copy()
    qa["transcriptid"] = qa["transcriptid"].astype(str)
    cols = [col for col in ["transcriptid", "turn_index", "question_text", "answer_text"] if col in qa]
    if len(cols) < 4:
        return pd.DataFrame()
    meta_cols = ["transcriptid", "ticker", "call_date", "headline", "answer_word_count"]
    meta = response_sample[[col for col in meta_cols if col in response_sample]].copy()
    meta["transcriptid"] = meta["transcriptid"].astype(str)
    examples = pd.concat(
        [turns.nlargest(n_per_tail, "narrative_innovation"), turns.nsmallest(n_per_tail, "narrative_innovation")],
        ignore_index=True,
    )
    examples = examples.merge(qa[cols], on=["transcriptid", "turn_index"], how="left")
    examples = examples.merge(meta, on="transcriptid", how="left")
    examples["question_excerpt"] = examples["question_text"].map(excerpt)
    examples["answer_excerpt"] = examples["answer_text"].map(excerpt)
    examples["tail"] = np.where(examples["narrative_innovation"].ge(0), "top", "bottom")
    keep = [
        "tail",
        "transcriptid",
        "ticker",
        "call_date",
        "headline",
        "turn_index",
        "word_count",
        "narrative_innovation",
        "question_excerpt",
        "answer_excerpt",
    ]
    return round_numeric(examples[[col for col in keep if col in examples]])


def merge_on_transcript(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0].copy()
    out["transcriptid"] = out["transcriptid"].astype(str)
    for frame in frames[1:]:
        right = frame.copy()
        right["transcriptid"] = right["transcriptid"].astype(str)
        out = out.merge(right, on="transcriptid", how="inner", validate="one_to_one")
    return out


def existing_columns(data: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([col for col in columns if col in data.columns]))


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.std() == 0 or right.std() == 0:
        return np.nan
    value = left.corr(right)
    return float(value) if pd.notna(value) else np.nan


def read_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def excerpt(text: object, limit: int = 240) -> str:
    clean = " ".join(str(text or "").split())
    return clean[: limit - 3] + "..." if len(clean) > limit else clean


def round_numeric(data: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = data.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(digits)
    return out


def write_latex(data: pd.DataFrame, path: Path) -> None:
    path.write_text(data.to_latex(index=False, escape=True, float_format="%.4f"), encoding="utf-8")


if __name__ == "__main__":
    main()
