"""Freeze existing S&P 500 LM tone outputs into the paper result package.

This paper-facing stage reads baseline, diagnostic, and learner-audit outputs,
checks closure, and writes CSV/TeX/figure artifacts. It is not an empirical
rerun and should preserve the frozen v1 facts used by the submission draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025"
ANALYSIS_ROOT = DATA_ROOT / "analysis"
BASELINE_DIR = ANALYSIS_ROOT / "lm_baseline"
DIAGNOSTICS_DIR = ANALYSIS_ROOT / "lm_baseline_diagnostics"
LEARNER_DIR = ANALYSIS_ROOT / "lm_baseline_learner_audit"
OUTPUT_DIR = ANALYSIS_ROOT / "lm_baseline_paper_results_v1"
HORIZONS = (1, 3, 5, 10, 20)
COMPONENT_LABELS = {
    "prepared": "Prepared",
    "question": "Question",
    "answer": "Answer",
}
TREATMENT_LABELS = {
    "prepared_innovation_z": "Prepared innovation",
    "question_innovation_z": "Question innovation",
    "answer_innovation_z": "Answer innovation",
}
LEARNER_LABELS = {
    "fixed_ridge": "Fixed Ridge",
    "tuned_ridge": "Tuned Ridge",
    "tuned_elasticnet": "Tuned ElasticNet",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
    "extra_trees": "Extra Trees",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze S&P 500 LM tone v1 empirical outputs into a paper-facing result package."
    )
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--learner-dir", type=Path, default=LEARNER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    build_package(args.baseline_dir, args.diagnostics_dir, args.learner_dir, args.output_dir)


def build_package(
    baseline_dir: Path,
    diagnostics_dir: Path,
    learner_dir: Path,
    output_dir: Path,
) -> None:
    csv_dir = output_dir / "csv"
    tex_dir = output_dir / "tex"
    fig_dir = output_dir / "figures"
    for path in [output_dir, csv_dir, tex_dir, fig_dir]:
        path.mkdir(parents=True, exist_ok=True)

    tables = load_inputs(baseline_dir, diagnostics_dir, learner_dir)
    closure = closure_checks(tables)
    if not bool(closure["passes"].all()):
        closure.to_csv(output_dir / "closure_checks.csv", index=False)
        raise RuntimeError("Closure checks failed; refusing to freeze paper package.")

    clean_tables = build_clean_tables(tables)
    clean_tables["closure_checks"] = closure
    for name, table in clean_tables.items():
        table.to_csv(csv_dir / f"{name}.csv", index=False)
        write_latex_table(table, tex_dir / f"{name}.tex", caption=latex_caption(name), label=f"tab:{name}")

    make_figures(tables, clean_tables, fig_dir)
    manifest = build_manifest(baseline_dir, diagnostics_dir, learner_dir, output_dir, tables, clean_tables)
    (output_dir / "paper_results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    closure.to_csv(output_dir / "closure_checks.csv", index=False)
    print(json.dumps({"output_dir": str(output_dir), "classification": manifest["audit_status"]["classification"]}, indent=2))


def load_inputs(baseline_dir: Path, diagnostics_dir: Path, learner_dir: Path) -> dict[str, pd.DataFrame | dict]:
    return {
        "baseline_summary": read_json(baseline_dir / "build_summary.json"),
        "diagnostics_summary": read_json(diagnostics_dir / "build_summary.json"),
        "learner_summary": read_json(learner_dir / "build_summary.json"),
        "attrition": pd.read_csv(baseline_dir / "attrition.csv"),
        "response": pd.read_csv(baseline_dir / "response_coefficients.csv"),
        "coefficient_tests": pd.read_csv(baseline_dir / "coefficient_tests.csv"),
        "placebos": pd.read_csv(baseline_dir / "validation_placebos.csv"),
        "first_stage": pd.read_csv(baseline_dir / "first_stage_diagnostics.csv"),
        "by_year": pd.read_csv(baseline_dir / "first_stage_diagnostics_by_year.csv"),
        "baseline_sample": pd.read_parquet(baseline_dir / "response_model_sample.parquet"),
        "call_innovations": pd.read_parquet(baseline_dir / "call_level_innovations.parquet"),
        "diagnostic_manifest": pd.read_csv(diagnostics_dir / "manifest.csv"),
        "diagnostic_sample_audit": pd.read_csv(diagnostics_dir / "diagnostic_sample_audit.csv"),
        "baseline_parity": pd.read_csv(diagnostics_dir / "baseline_parity.csv"),
        "control_stability": pd.read_csv(diagnostics_dir / "control_set_stability.csv"),
        "raw_decomposition_validation": pd.read_csv(diagnostics_dir / "raw_decomposition_validation.csv"),
        "raw_innovation_response": pd.read_csv(diagnostics_dir / "raw_innovation_response.csv"),
        "raw_innovation_correlations": pd.read_csv(diagnostics_dir / "raw_innovation_correlations.csv"),
        "raw_vs_innovation_differences": pd.read_csv(diagnostics_dir / "raw_vs_innovation_descriptive_differences.csv"),
        "outcome_residualized": pd.read_csv(diagnostics_dir / "outcome_residualized_responses.csv"),
        "outcome_residualization_diagnostics": pd.read_csv(diagnostics_dir / "outcome_residualization_diagnostics.csv"),
        "heterogeneity": pd.read_csv(diagnostics_dir / "heterogeneity_minimal_interactions.csv"),
        "learner_summary_table": pd.read_csv(learner_dir / "learner_invariance_summary.csv"),
        "learner_response": pd.read_csv(learner_dir / "learner_response_coefficients.csv"),
        "learner_car20_minus_1": pd.read_csv(learner_dir / "learner_car20_minus_1_coefficients.csv"),
        "learner_placebos": pd.read_csv(learner_dir / "learner_placebo_responses.csv"),
        "learner_correlations": pd.read_csv(learner_dir / "learner_innovation_correlations.csv"),
        "learner_tuning": pd.read_csv(learner_dir / "learner_hyperparameter_selection.csv"),
        "learner_first_stage_metrics": pd.read_csv(learner_dir / "learner_first_stage_metrics.csv"),
        "learner_first_stage_samples": pd.read_csv(learner_dir / "learner_first_stage_samples.csv"),
        "learner_manifest": pd.read_csv(learner_dir / "learner_common_response_manifest.csv"),
        "learner_parity": pd.read_csv(learner_dir / "learner_parity_checks.csv"),
        "learner_leakage": pd.read_csv(learner_dir / "learner_leakage_checks.csv"),
        "learner_timing": read_json(learner_dir / "timing.json"),
    }


def closure_checks(tables: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_sample = table(tables, "baseline_sample")
    learner_manifest = table(tables, "learner_manifest")
    learner_response = table(tables, "learner_response")
    learner_summary = dict(tables["learner_summary"])
    raw_validation = table(tables, "raw_decomposition_validation")
    placebos = table(tables, "placebos")
    residualized = table(tables, "outcome_residualized")

    add_check(rows, "fixed_ridge_parity", bool(table(tables, "learner_parity")["passes"].all()))
    add_check(rows, "diagnostic_baseline_parity", bool(table(tables, "baseline_parity")["passes"].all()))
    add_check(rows, "learner_leakage_checks", bool(table(tables, "learner_leakage")["passes"].all()))
    add_check(
        rows,
        "common_sample_equals_baseline_response_sample",
        set(learner_manifest["transcriptid"].astype(str)) == set(baseline_sample["transcriptid"].astype(str))
        and len(learner_manifest) == len(baseline_sample),
        n_rows=len(learner_manifest),
    )
    add_check(
        rows,
        "learner_common_sample_size_matches_summary",
        int(learner_summary["common_response_rows"]) == len(learner_manifest),
        n_rows=len(learner_manifest),
    )
    add_check(
        rows,
        "learner_response_uses_common_sample",
        learner_response["n_obs"].dropna().astype(int).eq(len(learner_manifest)).all(),
        n_rows=int(learner_response["n_obs"].min()),
    )
    add_check(
        rows,
        "raw_equals_expected_plus_innovation",
        raw_validation["max_abs_raw_minus_expected_minus_innovation"].abs().max() < 1e-10,
    )
    for treatment in ["prepared_innovation_z", "question_innovation_z", "answer_innovation_z"]:
        available = set(placebos.loc[placebos["treatment"].eq(treatment), "outcome"]) >= {
            "placebo_CAR_m5_m1",
            "placebo_CAR_m10_m1",
        }
        add_check(rows, f"placebo_available_{treatment}", available)
    residual_profiles = set(residualized["profile"].astype(str))
    add_check(
        rows,
        "residualized_response_comparator_present",
        {"main_car", "placebo_no_overlap_return_controls"}.issubset(residual_profiles),
        n_rows=len(residualized),
    )
    return pd.DataFrame(rows)


def build_clean_tables(tables: dict[str, pd.DataFrame | dict]) -> dict[str, pd.DataFrame]:
    clean = {
        "main_response_bps": main_response_table(table(tables, "response")),
        "diagnostic_summary": diagnostic_summary_table(tables),
        "learner_invariance_answer": learner_invariance_answer_table(tables),
        "appendix_attrition": table(tables, "attrition").copy(),
        "appendix_first_stage_diagnostics": first_stage_appendix_table(table(tables, "first_stage")),
        "appendix_innovation_correlations": innovation_correlation_appendix(tables),
        "appendix_learner_tuning": learner_tuning_appendix(table(tables, "learner_tuning")),
        "appendix_heterogeneity": heterogeneity_appendix(table(tables, "heterogeneity")),
        "appendix_timing": timing_appendix(tables["learner_timing"]),
    }
    return clean


def main_response_table(response: pd.DataFrame) -> pd.DataFrame:
    data = response.copy()
    data["component"] = data["treatment"].str.replace("_innovation_z", "", regex=False).map(COMPONENT_LABELS)
    data["beta_bps"] = data["beta"] * 10000
    data["se_bps"] = data["standard_error"] * 10000
    out = data[
        ["horizon", "component", "beta_bps", "se_bps", "t_stat", "n_obs", "n_firms", "n_event_dates"]
    ].copy()
    return round_numeric(out)


def diagnostic_summary_table(tables: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    response = table(tables, "response")
    control = table(tables, "control_stability")
    raw = table(tables, "raw_innovation_response")
    residual = table(tables, "outcome_residualized")
    placebos = table(tables, "placebos")
    learner = table(tables, "learner_summary_table")
    heterogeneity = table(tables, "heterogeneity")
    car = table(tables, "learner_car20_minus_1")

    rows = []
    rows.append(summary_row("Control stability", answer_at(control, "full_controls_current", 20), answer_at(control, "no_ibes_controls", 20)))
    rows.append(summary_row("Raw score response", answer_raw_at(raw, "raw_only", 20), None))
    rows.append(summary_row("Innovation response", answer_raw_at(raw, "innovation_only", 20), None))
    rows.append(summary_row("Outcome-residualized response", answer_residualized_at(residual, "CAR_0_20"), None))
    rows.append(summary_row("Placebo CAR[-5,-1]", answer_placebo_at(placebos, "placebo_CAR_m5_m1"), None))
    rows.append(summary_row("Placebo CAR[-10,-1]", answer_placebo_at(placebos, "placebo_CAR_m10_m1"), None))
    rows.append(
        {
            "diagnostic": "Learner invariance",
            "primary_estimate_bps": np.nan,
            "primary_t_stat": np.nan,
            "comparison_estimate_bps": np.nan,
            "comparison_t_stat": np.nan,
            "status": str(learner["overall_classification"].iloc[0]),
            "n_obs": int(answer_at(response.assign(model="baseline"), "baseline", 20)["n_obs"]),
        }
    )
    rows.append(summary_row("Minimal heterogeneity", answer_heterogeneity_at(heterogeneity, 20), None))
    rows.append(summary_row("CAR20 minus CAR1", answer_car20_minus_1_at(car, "fixed_ridge"), None))
    return round_numeric(pd.DataFrame(rows))


def learner_invariance_answer_table(tables: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    response = table(tables, "learner_response")
    placebos = table(tables, "learner_placebos")
    car = table(tables, "learner_car20_minus_1")
    corr = table(tables, "learner_correlations")
    rows = []
    for learner in sorted(response["learner"].unique()):
        main = response.loc[
            response["learner"].eq(learner)
            & response["treatment"].eq("answer_innovation_z")
            & response["horizon"].eq(20)
        ].iloc[0]
        placebo = placebos.loc[
            placebos["learner"].eq(learner)
            & placebos["treatment"].eq("answer_innovation_z")
            & placebos["outcome"].eq("placebo_CAR_m5_m1")
        ].iloc[0]
        car_row = car.loc[
            car["learner"].eq(learner)
            & car["treatment"].eq("answer_innovation_z")
        ].iloc[0]
        learner_corr = min_answer_corr(corr, learner)
        rows.append(
            {
                "learner": LEARNER_LABELS.get(learner, learner),
                "beta_bps_h20": main["beta"] * 10000,
                "se_bps_h20": main["standard_error"] * 10000,
                "t_stat_h20": main["t_stat"],
                "ci95_low_bps_h20": (main["beta"] - 1.96 * main["standard_error"]) * 10000,
                "ci95_high_bps_h20": (main["beta"] + 1.96 * main["standard_error"]) * 10000,
                "placebo_m5_m1_t_stat": placebo["t_stat"],
                "car20_minus_1_beta_bps": car_row["beta"] * 10000,
                "car20_minus_1_t_stat": car_row["t_stat"],
                "min_answer_innovation_corr": learner_corr,
                "n_obs": int(main["n_obs"]),
            }
        )
    return round_numeric(pd.DataFrame(rows))


def first_stage_appendix_table(first_stage: pd.DataFrame) -> pd.DataFrame:
    data = first_stage.loc[first_stage["semantic_feature"].eq("lm_net_tone")].copy()
    out = data[
        ["component", "grain", "n_obs", "first_stage_r2", "mae", "predicted_actual_corr", "calibration_slope"]
    ].copy()
    out["component"] = out["component"].map(COMPONENT_LABELS)
    return round_numeric(out)


def innovation_correlation_appendix(tables: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    corr = table(tables, "learner_correlations").copy()
    corr["component"] = corr["component"].map(COMPONENT_LABELS)
    corr["left_learner"] = corr["left_learner"].map(lambda x: LEARNER_LABELS.get(x, x))
    corr["right_learner"] = corr["right_learner"].map(lambda x: LEARNER_LABELS.get(x, x))
    return round_numeric(corr)


def learner_tuning_appendix(tuning: pd.DataFrame) -> pd.DataFrame:
    data = tuning.loc[tuning["selected"].astype(bool)].copy()
    data["learner"] = data["learner"].map(lambda x: LEARNER_LABELS.get(x, x))
    data["component"] = data["component"].map(COMPONENT_LABELS)
    return round_numeric(
        data[
            [
                "learner",
                "component",
                "outer_year",
                "candidate_json",
                "selection_source",
                "n_inner_folds",
                "inner_mae",
            ]
        ]
    )


def heterogeneity_appendix(heterogeneity: pd.DataFrame) -> pd.DataFrame:
    data = heterogeneity.loc[heterogeneity["treatment"].str.contains("answer", na=False)].copy()
    data["beta_bps"] = data["beta"] * 10000
    data["se_bps"] = data["standard_error"] * 10000
    return round_numeric(
        data[["moderator", "horizon", "treatment", "beta_bps", "se_bps", "t_stat", "n_obs"]]
    )


def timing_appendix(timing: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": item["stage"],
                "status": item["status"],
                "seconds": item["seconds"],
            }
            for item in timing
        ]
    )


def make_figures(tables: dict[str, pd.DataFrame | dict], clean_tables: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    figure_design_schematic(fig_dir, plt, Rectangle, FancyArrowPatch)
    figure_main_response(table(tables, "response"), fig_dir, plt)
    figure_answer_by_learner(table(tables, "learner_response"), fig_dir, plt)
    figure_raw_decomposition(table(tables, "raw_innovation_response"), fig_dir, plt)
    figure_placebo_residualized(tables, fig_dir, plt)


def figure_design_schematic(fig_dir: Path, plt, Rectangle, FancyArrowPatch) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.0), constrained_layout=True)
    ax.set_axis_off()
    boxes = [
        (0.02, 0.55, 0.18, 0.25, "Pre-call state\nX before event"),
        (0.28, 0.55, 0.18, 0.25, "Prepared\ninnovation"),
        (0.52, 0.55, 0.18, 0.25, "Question\ninnovation"),
        (0.76, 0.55, 0.18, 0.25, "Answer\ninnovation"),
        (0.52, 0.12, 0.22, 0.25, "CAR response\nh = 1, 3, 5, 10, 20"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#f2f4f7", edgecolor="#2f3b52", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
    arrows = [
        ((0.20, 0.675), (0.28, 0.675)),
        ((0.46, 0.675), (0.52, 0.675)),
        ((0.70, 0.675), (0.76, 0.675)),
        ((0.61, 0.55), (0.61, 0.37)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#2f3b52"))
    ax.text(0.5, 0.93, "Sequential earnings-call narrative innovation design", ha="center", weight="bold")
    save_figure(fig, fig_dir / "figure_1_sequential_tone_design")


def figure_main_response(response: pd.DataFrame, fig_dir: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    colors = {"prepared_innovation_z": "#1f77b4", "question_innovation_z": "#2ca02c", "answer_innovation_z": "#d62728"}
    for treatment, group in response.groupby("treatment", sort=False):
        group = group.sort_values("horizon")
        y = group["beta"] * 10000
        se = group["standard_error"] * 10000
        ax.errorbar(group["horizon"], y, yerr=1.96 * se, marker="o", linewidth=1.8, capsize=3, color=colors[treatment], label=TREATMENT_LABELS[treatment])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Trading-day CAR horizon")
    ax.set_ylabel("Basis points per 1 SD innovation")
    ax.set_title("Price response to sequential narrative innovations")
    ax.legend(loc="best")
    save_figure(fig, fig_dir / "figure_2_main_response_curves")


def figure_answer_by_learner(response: pd.DataFrame, fig_dir: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    data = response.loc[response["treatment"].eq("answer_innovation_z")].copy()
    for learner, group in data.groupby("learner", sort=False):
        group = group.sort_values("horizon")
        y = group["beta"] * 10000
        se = group["standard_error"] * 10000
        ax.errorbar(group["horizon"], y, yerr=1.96 * se, marker="o", linewidth=1.5, capsize=3, label=LEARNER_LABELS.get(learner, learner))
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Trading-day CAR horizon")
    ax.set_ylabel("Basis points per 1 SD answer innovation")
    ax.set_title("Answer response is learner-invariant")
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, fig_dir / "figure_3_answer_response_by_learner")


def figure_raw_decomposition(raw_response: pd.DataFrame, fig_dir: Path, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True, constrained_layout=True)
    series_specs = [
        ("raw_only", "_raw_", "Raw"),
        ("joint_expected_innovation", "_expected_", "Expected"),
        ("joint_expected_innovation", "_innovation_score_", "Innovation"),
    ]
    for ax, component in zip(axes, ["prepared", "question", "answer"], strict=True):
        for model, treatment_needle, label in series_specs:
            data = raw_response.loc[
                raw_response["model"].eq(model)
                & raw_response["treatment"].str.contains(component)
                & raw_response["treatment"].str.contains(treatment_needle, regex=False)
            ].copy()
            data = data.sort_values("horizon")
            ax.plot(data["horizon"], data["beta"] * 10000, marker="o", linewidth=1.6, label=label)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(COMPONENT_LABELS[component])
        ax.set_xlabel("Horizon")
    axes[0].set_ylabel("Basis points")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Raw score decomposition: expected and innovation components")
    save_figure(fig, fig_dir / "figure_4_raw_vs_innovation_decomposition")


def figure_placebo_residualized(tables: dict[str, pd.DataFrame | dict], fig_dir: Path, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    placebo = table(tables, "placebos").copy()
    placebo = placebo.loc[placebo["treatment"].eq("answer_innovation_z")]
    axes[0].bar(placebo["outcome"], placebo["beta"] * 10000, yerr=1.96 * placebo["standard_error"] * 10000, color="#8c8c8c", capsize=3)
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_title("Answer placebo checks")
    axes[0].set_ylabel("Basis points")
    axes[0].tick_params(axis="x", rotation=20)

    residual = table(tables, "outcome_residualized").copy()
    residual = residual.loc[
        residual["profile"].eq("main_car") & residual["treatment"].eq("answer_innovation_z")
    ].copy()
    residual["horizon"] = residual["source_outcome"].str.replace("CAR_0_", "", regex=False).astype(int)
    axes[1].errorbar(
        residual["horizon"],
        residual["beta"] * 10000,
        yerr=1.96 * residual["standard_error"] * 10000,
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#d62728",
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("Outcome-residualized answer response")
    axes[1].set_xlabel("Horizon")
    save_figure(fig, fig_dir / "figure_5_placebo_residualized_checks")


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))


def build_manifest(
    baseline_dir: Path,
    diagnostics_dir: Path,
    learner_dir: Path,
    output_dir: Path,
    tables: dict[str, pd.DataFrame | dict],
    clean_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    baseline_summary = dict(tables["baseline_summary"])
    learner_summary = dict(tables["learner_summary"])
    sample = table(tables, "baseline_sample")
    manifest = {
        "package": "lm_baseline_paper_results_v1",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "git": git_info(),
        "input_dirs": {
            "baseline": rel(baseline_dir),
            "diagnostics": rel(diagnostics_dir),
            "learner_audit": rel(learner_dir),
        },
        "output_dir": rel(output_dir),
        "sample_sizes": {
            "all_staged_transcript_events": int(baseline_summary["all_staged_transcript_events"]),
            "earnings_call_events_through_sample_end": int(baseline_summary["earnings_call_events_through_sample_end"]),
            "point_in_time_sp500_earnings_calls": int(baseline_summary["point_in_time_sp500_earnings_calls"]),
            "crsp_event_window_ready": int(baseline_summary["crsp_event_window_ready"]),
            "main_response_sample": int(len(sample)),
            "main_unique_firms": int(sample["permno"].nunique()),
            "learner_common_response_sample": int(learner_summary["common_response_rows"]),
        },
        "audit_status": {
            "fixed_ridge_parity_passed": bool(learner_summary["fixed_ridge_parity_passed"]),
            "classification": str(learner_summary["classification"]),
            "selected_nonlinear": str(learner_summary["selected_nonlinear"]),
            "closure_checks_passed": bool(clean_tables["closure_checks"]["passes"].all()),
        },
        "source_files": source_files(baseline_dir, diagnostics_dir, learner_dir),
        "outputs": sorted(rel(path) for path in output_dir.rglob("*") if path.is_file()),
    }
    return manifest


def source_files(*dirs: Path) -> list[dict[str, object]]:
    wanted = {".csv", ".json", ".parquet"}
    rows = []
    for root in dirs:
        for path in sorted(root.iterdir()):
            if path.suffix.lower() not in wanted:
                continue
            stat = path.stat()
            row: dict[str, object] = {
                "path": rel(path),
                "bytes": stat.st_size,
                "sha256": sha256(path) if stat.st_size <= 50_000_000 else None,
            }
            rows.append(row)
    return rows


def summary_row(label: str, primary: pd.Series | dict | None, comparison: pd.Series | dict | None) -> dict[str, object]:
    if primary is None:
        return {
            "diagnostic": label,
            "primary_estimate_bps": np.nan,
            "primary_t_stat": np.nan,
            "comparison_estimate_bps": np.nan,
            "comparison_t_stat": np.nan,
            "status": "",
            "n_obs": np.nan,
        }
    return {
        "diagnostic": label,
        "primary_estimate_bps": float(primary["beta"]) * 10000,
        "primary_t_stat": float(primary["t_stat"]),
        "comparison_estimate_bps": float(comparison["beta"]) * 10000 if comparison is not None else np.nan,
        "comparison_t_stat": float(comparison["t_stat"]) if comparison is not None else np.nan,
        "status": diagnostic_status(primary),
        "n_obs": int(primary["n_obs"]),
    }


def answer_at(data: pd.DataFrame, model: str, horizon: int) -> pd.Series:
    return data.loc[
        data["model"].eq(model)
        & data["horizon"].astype(str).eq(str(horizon))
        & data["treatment"].eq("answer_innovation_z")
    ].iloc[0]


def answer_raw_at(data: pd.DataFrame, model: str, horizon: int) -> pd.Series:
    treatment = "answer_raw_z_diag" if model == "raw_only" else "answer_innovation_score_z_diag"
    return data.loc[
        data["model"].eq(model)
        & data["horizon"].astype(str).eq(str(horizon))
        & data["treatment"].eq(treatment)
    ].iloc[0]


def answer_residualized_at(data: pd.DataFrame, source_outcome: str) -> pd.Series:
    return data.loc[
        data["profile"].eq("main_car")
        & data["source_outcome"].eq(source_outcome)
        & data["treatment"].eq("answer_innovation_z")
    ].iloc[0]


def answer_placebo_at(data: pd.DataFrame, outcome: str) -> pd.Series:
    return data.loc[data["outcome"].eq(outcome) & data["treatment"].eq("answer_innovation_z")].iloc[0]


def answer_heterogeneity_at(data: pd.DataFrame, horizon: int) -> pd.Series:
    subset = data.loc[
        data["horizon"].eq(horizon)
        & data["treatment"].str.startswith("answer_x_")
    ].copy()
    return subset.iloc[subset["t_stat"].abs().argmax()]


def answer_car20_minus_1_at(data: pd.DataFrame, learner: str) -> pd.Series:
    return data.loc[data["learner"].eq(learner) & data["treatment"].eq("answer_innovation_z")].iloc[0]


def min_answer_corr(corr: pd.DataFrame, learner: str) -> float:
    if learner == "fixed_ridge":
        data = corr.loc[corr["component"].eq("answer")]
    else:
        data = corr.loc[
            corr["component"].eq("answer")
            & (corr["left_learner"].eq(learner) | corr["right_learner"].eq(learner))
        ]
    return float(data["correlation"].min())


def diagnostic_status(row: pd.Series | dict) -> str:
    t = float(row["t_stat"])
    if abs(t) >= 1.96:
        return "statistically distinguishable from zero at 5%"
    return "not statistically distinguishable from zero at 5%"


def add_check(rows: list[dict[str, object]], name: str, passes: bool, n_rows: int | None = None) -> None:
    rows.append({"check": name, "passes": bool(passes), "n_rows": n_rows if n_rows is not None else np.nan})


def write_latex_table(table_df: pd.DataFrame, path: Path, *, caption: str, label: str) -> None:
    latex = table_df.to_latex(index=False, escape=True, caption=caption, label=label, float_format="%.3f")
    path.write_text(latex, encoding="utf-8")


def latex_caption(name: str) -> str:
    return name.replace("_", " ").title()


def round_numeric(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(digits)
    return out


def table(tables: dict[str, pd.DataFrame | dict], name: str) -> pd.DataFrame:
    value = tables[name]
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} is not a table")
    return value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, object]:
    commit = run_git(["rev-parse", "HEAD"]) or "uncommitted"
    status = run_git(["status", "--short"])
    branch = run_git(["branch", "--show-current"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
    }


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
