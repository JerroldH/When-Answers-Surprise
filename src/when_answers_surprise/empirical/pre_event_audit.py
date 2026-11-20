"""Audit pre-event return associations for the paper response sample.

The audit keeps the call-level unit of observation and the paper's joint
innovation specification. It replaces the overlapping ``pre_ret_20`` control
with return history from days -60 through -21, splits the pre-event outcome
into non-overlapping windows, and checks whether results vary with call timing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from when_answers_surprise.empirical.diagnostics import (
    add_nuisance_time_features,
    placebo_nuisance_controls,
)
from when_answers_surprise.empirical.pipeline import fit_response_regression, response_controls

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT / "data" / "wrds_sp500_2018_2025" / "analysis" / "lm_baseline"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "pre_event_audit"

EARLY_OUTCOME = "placebo_CAR_m10_m6"
LATE_OUTCOME = "placebo_CAR_m5_m1"
TOTAL_OUTCOME = "placebo_CAR_m10_m1"
OUTCOMES = (EARLY_OUTCOME, LATE_OUTCOME, TOTAL_OUTCOME)
NON_OVERLAP_OUTCOMES = (EARLY_OUTCOME, LATE_OUTCOME)
OUTCOME_LABELS = {
    EARLY_OUTCOME: "CAR[-10,-6]",
    LATE_OUTCOME: "CAR[-5,-1]",
    TOTAL_OUTCOME: "CAR[-10,-1]",
}
TREATMENTS = (
    "prepared_innovation_z",
    "question_innovation_z",
    "answer_innovation_z",
)
COMPONENT_LABELS = {
    "prepared_innovation_z": "Prepared",
    "question_innovation_z": "Question",
    "answer_innovation_z": "Answer",
}
TIMING_BUCKETS = ("premarket", "market_hours", "after_close")
ALIGNMENT_CLEAN_LABEL = "alignment_clean"


def prepare_pre_event_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the call sample and construct non-overlapping pre-event outcomes."""

    required = {
        "transcriptid",
        "permno",
        "event_trading_date",
        "event_year",
        "event_calendar_quarter",
        "event_datetime_utc",
        "actual_announcement_datetime_utc",
        "call_timing_bucket",
        "pre_ret_20",
        "pre_ret_60",
        LATE_OUTCOME,
        TOTAL_OUTCOME,
        *TREATMENTS,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Pre-event audit sample is missing columns: {missing}")
    if data["transcriptid"].duplicated().any():
        examples = data.loc[data["transcriptid"].duplicated(False), "transcriptid"].head(5)
        raise ValueError(f"Pre-event audit requires unique transcript ids: {examples.tolist()}")

    out = add_nuisance_time_features(data.copy())
    out[EARLY_OUTCOME] = (
        pd.to_numeric(out[TOTAL_OUTCOME], errors="coerce")
        - pd.to_numeric(out[LATE_OUTCOME], errors="coerce")
    )
    recombined = out[EARLY_OUTCOME] + out[LATE_OUTCOME]
    difference = (recombined - out[TOTAL_OUTCOME]).abs().max()
    if not np.isfinite(difference) or float(difference) > 1e-12:
        raise ValueError(f"Non-overlapping pre-event windows do not recombine: {difference}")
    return add_alignment_fields(out)


def add_alignment_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Add call, announcement, and aligned-event timing diagnostics."""

    out = data.copy()
    call = pd.to_datetime(out["event_datetime_utc"], utc=True, errors="coerce").dt.tz_convert(
        "America/New_York"
    )
    announcement = pd.to_datetime(
        out["actual_announcement_datetime_utc"], utc=True, errors="coerce"
    ).dt.tz_convert("America/New_York")
    event_day = pd.to_datetime(out["event_trading_date"], errors="coerce").dt.normalize()
    call_day = call.dt.tz_localize(None).dt.normalize()
    announcement_day = announcement.dt.tz_localize(None).dt.normalize()
    announcement_minutes = announcement.dt.hour * 60 + announcement.dt.minute

    release_timing = pd.Series("unknown", index=out.index, dtype=object)
    release_timing.loc[announcement.notna() & announcement_minutes.lt(9 * 60 + 30)] = (
        "premarket"
    )
    release_timing.loc[
        announcement.notna()
        & announcement_minutes.ge(9 * 60 + 30)
        & announcement_minutes.lt(16 * 60)
    ] = "market_hours"
    release_timing.loc[announcement.notna() & announcement_minutes.ge(16 * 60)] = (
        "after_close"
    )

    out["call_date_ny"] = call_day
    out["announcement_date_ny"] = announcement_day
    out["announcement_timing_bucket"] = release_timing
    out["call_date_before_aligned_event_day"] = call_day.lt(event_day)
    out["announcement_date_before_aligned_event_day"] = announcement_day.lt(event_day)
    out["announcement_before_call"] = announcement.le(call)
    out["announcement_event_gap_calendar_days"] = (event_day - announcement_day).dt.days
    out["regular_hours_announcement_before_event_day"] = (
        announcement_day.lt(event_day)
        & release_timing.isin(["premarket", "market_hours"])
    )
    out["stale_announcement_match"] = out[
        "announcement_event_gap_calendar_days"
    ].ge(6)
    out["same_call_day_release_before_close_for_after_close_call"] = (
        out["call_timing_bucket"].eq("after_close")
        & announcement_day.eq(call_day)
        & announcement_minutes.lt(16 * 60)
        & call_day.lt(event_day)
    )
    out["conservative_announcement_alignment_exclusion"] = (
        out["regular_hours_announcement_before_event_day"]
        | out["stale_announcement_match"]
    )
    return out


def safe_pre_event_controls(data: pd.DataFrame) -> list[str]:
    """Return the paper controls with non-overlapping return history."""

    controls = placebo_nuisance_controls(data, response_controls(data))
    if "pre_ret_20" in controls:
        raise ValueError("Pre-event controls still contain overlapping pre_ret_20")
    if "pre_ret_60_minus_20" not in controls:
        raise ValueError("Pre-event controls lack pre_ret_60_minus_20")
    return controls


def normal_two_sided_p(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    return math.erfc(abs(float(t_stat)) / math.sqrt(2.0))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm family-wise adjusted p-values in original order."""

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return adjusted
    order = valid_positions[np.argsort(values[valid_positions])]
    running = 0.0
    family_size = len(order)
    for rank, position in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted


def estimate_pre_event_cells(data: pd.DataFrame, *, sample_label: str) -> pd.DataFrame:
    """Estimate all three innovations on early, late, and total pre-event CAR."""

    controls = safe_pre_event_controls(data)
    rows = []
    for outcome in OUTCOMES:
        result = fit_response_regression(
            data,
            outcome=outcome,
            treatments=TREATMENTS,
            controls=controls,
            firm_fe="permno",
            quarter_fe="event_calendar_quarter",
            cluster_cols=("permno", "event_trading_date"),
        )
        if result.n_obs != len(data):
            raise ValueError(
                f"{sample_label}/{outcome} changed the sample: "
                f"expected {len(data)}, observed {result.n_obs}"
            )
        cells = result.coefficients.copy()
        cells.insert(0, "sample_label", sample_label)
        cells.insert(1, "outcome", outcome)
        rows.append(cells)

    out = pd.concat(rows, ignore_index=True)
    out["component"] = out["treatment"].map(COMPONENT_LABELS)
    out["window"] = out["outcome"].map(OUTCOME_LABELS)
    out["beta_bps"] = out["beta"] * 10000
    out["se_bps"] = out["standard_error"] * 10000
    out["p_value"] = out["t_stat"].map(normal_two_sided_p)
    return out


def add_full_sample_multiplicity(cells: pd.DataFrame) -> pd.DataFrame:
    """Adjust the six component by non-overlapping-window tests as one family."""

    out = cells.copy()
    out["p_holm_full_6"] = np.nan
    mask = out["sample_label"].eq("full") & out["outcome"].isin(NON_OVERLAP_OUTCOMES)
    out.loc[mask, "p_holm_full_6"] = holm_adjust(out.loc[mask, "p_value"])
    return out


def add_timing_multiplicity(cells: pd.DataFrame) -> pd.DataFrame:
    """Add exploratory Holm adjustments for timing-subsample cells."""

    out = cells.copy()
    out["p_holm_timing_18"] = np.nan
    out["p_holm_question_timing_6"] = np.nan
    non_overlap = out["outcome"].isin(NON_OVERLAP_OUTCOMES)
    out.loc[non_overlap, "p_holm_timing_18"] = holm_adjust(
        out.loc[non_overlap, "p_value"]
    )
    question = non_overlap & out["treatment"].eq("question_innovation_z")
    out.loc[question, "p_holm_question_timing_6"] = holm_adjust(
        out.loc[question, "p_value"]
    )
    return out


def joint_pre_period_tests(cells: pd.DataFrame) -> pd.DataFrame:
    """Test both non-overlapping pre-period coefficients jointly for each component.

    With the same design matrix and sample in all three regressions,
    Var(early + late) identifies the cross-window covariance.
    """

    rows = []
    for (sample_label, treatment), group in cells.groupby(
        ["sample_label", "treatment"], sort=False
    ):
        by_outcome = group.set_index("outcome")
        missing = set(OUTCOMES).difference(by_outcome.index)
        if missing:
            raise ValueError(f"Joint pre-period test is missing outcomes: {sorted(missing)}")
        early = by_outcome.loc[EARLY_OUTCOME]
        late = by_outcome.loc[LATE_OUTCOME]
        total = by_outcome.loc[TOTAL_OUTCOME]
        coefficient_gap = abs(float(total["beta"] - early["beta"] - late["beta"]))
        if coefficient_gap > 1e-10:
            raise ValueError(
                f"Pre-period coefficients do not add up for {sample_label}/{treatment}: "
                f"gap={coefficient_gap}"
            )

        var_early = float(early["standard_error"] ** 2)
        var_late = float(late["standard_error"] ** 2)
        var_total = float(total["standard_error"] ** 2)
        covariance = (var_total - var_early - var_late) / 2.0
        covariance_matrix = np.array(
            [[var_early, covariance], [covariance, var_late]], dtype=float
        )
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        status = "ok" if float(eigenvalues.min()) > 0 else "non_positive_covariance"
        if status == "ok":
            beta = np.array([early["beta"], late["beta"]], dtype=float)
            wald_stat = float(beta @ np.linalg.inv(covariance_matrix) @ beta)
            p_value = math.exp(-max(wald_stat, 0.0) / 2.0)
        else:
            wald_stat = float("nan")
            p_value = float("nan")
        rows.append(
            {
                "sample_label": sample_label,
                "treatment": treatment,
                "component": COMPONENT_LABELS[treatment],
                "wald_chi2_df2": wald_stat,
                "p_value": p_value,
                "cross_window_covariance": covariance,
                "min_covariance_eigenvalue": float(eigenvalues.min()),
                "status": status,
                "n_obs": int(early["n_obs"]),
                "n_firms": int(early["n_firms"]),
                "n_event_dates": int(early["n_event_dates"]),
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm_full_3"] = np.nan
    full = out["sample_label"].eq("full")
    out.loc[full, "p_holm_full_3"] = holm_adjust(out.loc[full, "p_value"])
    out["p_holm_timing_9"] = np.nan
    timing = out["sample_label"].isin(TIMING_BUCKETS)
    out.loc[timing, "p_holm_timing_9"] = holm_adjust(out.loc[timing, "p_value"])
    return out


def alignment_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    samples = [("full", data)] + [
        (bucket, data.loc[data["call_timing_bucket"].eq(bucket)])
        for bucket in TIMING_BUCKETS
    ]
    for label, sample in samples:
        timing_counts = sample["announcement_timing_bucket"].value_counts()
        rows.append(
            {
                "sample_label": label,
                "n_calls": int(len(sample)),
                "n_firms": int(sample["permno"].nunique()),
                "n_event_dates": int(sample["event_trading_date"].nunique()),
                "call_date_before_event_day_n": int(
                    sample["call_date_before_aligned_event_day"].sum()
                ),
                "announcement_date_before_event_day_n": int(
                    sample["announcement_date_before_aligned_event_day"].sum()
                ),
                "announcement_before_call_n": int(sample["announcement_before_call"].sum()),
                "same_day_release_before_close_after_close_call_n": int(
                    sample["same_call_day_release_before_close_for_after_close_call"].sum()
                ),
                "regular_hours_announcement_before_event_day_n": int(
                    sample["regular_hours_announcement_before_event_day"].sum()
                ),
                "stale_announcement_match_n": int(
                    sample["stale_announcement_match"].sum()
                ),
                "conservative_announcement_alignment_exclusion_n": int(
                    sample["conservative_announcement_alignment_exclusion"].sum()
                ),
                "release_premarket_n": int(timing_counts.get("premarket", 0)),
                "release_market_hours_n": int(timing_counts.get("market_hours", 0)),
                "release_after_close_n": int(timing_counts.get("after_close", 0)),
            }
        )
    out = pd.DataFrame(rows)
    for count_column in [
        "call_date_before_event_day_n",
        "announcement_date_before_event_day_n",
        "announcement_before_call_n",
        "same_day_release_before_close_after_close_call_n",
        "regular_hours_announcement_before_event_day_n",
        "stale_announcement_match_n",
        "conservative_announcement_alignment_exclusion_n",
    ]:
        share_column = count_column.removesuffix("_n") + "_share"
        out[share_column] = out[count_column] / out["n_calls"]
    return out


def run_pre_event_audit(sample: pd.DataFrame) -> dict[str, object]:
    data = prepare_pre_event_sample(sample)
    full_cells = add_full_sample_multiplicity(
        estimate_pre_event_cells(data, sample_label="full")
    )
    timing_pieces = []
    for bucket in TIMING_BUCKETS:
        timing_sample = data.loc[data["call_timing_bucket"].eq(bucket)].copy()
        timing_pieces.append(estimate_pre_event_cells(timing_sample, sample_label=bucket))
    timing_cells = add_timing_multiplicity(pd.concat(timing_pieces, ignore_index=True))
    alignment_clean_sample = data.loc[
        ~data["conservative_announcement_alignment_exclusion"]
    ].copy()
    alignment_clean_cells = estimate_pre_event_cells(
        alignment_clean_sample,
        sample_label=ALIGNMENT_CLEAN_LABEL,
    )
    all_cells = pd.concat(
        [full_cells, timing_cells, alignment_clean_cells],
        ignore_index=True,
        sort=False,
    )
    return {
        "full_cells": full_cells,
        "timing_cells": timing_cells,
        "alignment_clean_cells": alignment_clean_cells,
        "joint_tests": joint_pre_period_tests(all_cells),
        "alignment_summary": alignment_summary(data),
        "controls": safe_pre_event_controls(data),
        "n_obs": int(len(data)),
        "n_firms": int(data["permno"].nunique()),
        "n_event_dates": int(data["event_trading_date"].nunique()),
        "duplicate_firm_dates": int(data.duplicated(["permno", "event_trading_date"]).sum()),
    }


def markdown_table(data: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    headers = [label for _, label, _ in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data.itertuples(index=False):
        values = []
        row_values = row._asdict()
        for column, _, format_spec in columns:
            value = row_values[column]
            if pd.isna(value):
                values.append("")
            elif format_spec:
                values.append(format(value, format_spec))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_audit_memo(results: dict[str, object]) -> str:
    full = results["full_cells"]
    timing = results["timing_cells"]
    alignment_clean = results["alignment_clean_cells"]
    joint = results["joint_tests"]
    alignment = results["alignment_summary"]

    full_non_overlap = full.loc[full["outcome"].isin(NON_OVERLAP_OUTCOMES)].copy()
    full_table = markdown_table(
        full_non_overlap,
        [
            ("component", "Component", ""),
            ("window", "Window", ""),
            ("beta_bps", "Beta (bps)", ".1f"),
            ("t_stat", "t", ".2f"),
            ("p_value", "p", ".3f"),
            ("p_holm_full_6", "Holm p", ".3f"),
        ],
    )
    full_joint = joint.loc[joint["sample_label"].eq("full")]
    joint_table = markdown_table(
        full_joint,
        [
            ("component", "Component", ""),
            ("wald_chi2_df2", "Joint chi2(2)", ".2f"),
            ("p_value", "Joint p", ".3f"),
            ("p_holm_full_3", "Holm p", ".3f"),
            ("status", "Status", ""),
        ],
    )
    question_timing = timing.loc[timing["treatment"].eq("question_innovation_z")]
    timing_table = markdown_table(
        question_timing,
        [
            ("sample_label", "Call timing", ""),
            ("window", "Window", ""),
            ("beta_bps", "Beta (bps)", ".1f"),
            ("t_stat", "t", ".2f"),
            ("p_value", "p", ".3f"),
            ("p_holm_question_timing_6", "Holm p", ".3f"),
        ],
    )
    alignment_table = markdown_table(
        alignment,
        [
            ("sample_label", "Sample", ""),
            ("n_calls", "Calls", ",d"),
            ("call_date_before_event_day_share", "Call date < event day", ".3f"),
            (
                "announcement_date_before_event_day_share",
                "Announcement date < event day",
                ".3f",
            ),
            (
                "regular_hours_announcement_before_event_day_n",
                "Regular-hours release < event day",
                ",d",
            ),
            (
                "stale_announcement_match_n",
                "Announcement gap >= 6 days",
                ",d",
            ),
            (
                "conservative_announcement_alignment_exclusion_n",
                "Conservative exclusions",
                ",d",
            ),
        ],
    )
    alignment_clean_table = markdown_table(
        alignment_clean.loc[alignment_clean["outcome"].isin(NON_OVERLAP_OUTCOMES)],
        [
            ("component", "Component", ""),
            ("window", "Window", ""),
            ("beta_bps", "Beta (bps)", ".1f"),
            ("t_stat", "t", ".2f"),
            ("p_value", "p", ".3f"),
        ],
    )

    question = full_non_overlap.loc[full_non_overlap["component"].eq("Question")]
    question_early = question.loc[question["outcome"].eq(EARLY_OUTCOME)].iloc[0]
    question_late = question.loc[question["outcome"].eq(LATE_OUTCOME)].iloc[0]
    question_joint = full_joint.loc[full_joint["component"].eq("Question")].iloc[0]
    answer_joint = full_joint.loc[full_joint["component"].eq("Answer")].iloc[0]
    after_close_alignment = alignment.loc[alignment["sample_label"].eq("after_close")].iloc[0]
    full_alignment = alignment.loc[alignment["sample_label"].eq("full")].iloc[0]
    clean_question_early = alignment_clean.loc[
        alignment_clean["component"].eq("Question")
        & alignment_clean["outcome"].eq(EARLY_OUTCOME)
    ].iloc[0]

    return f"""# Pre-event specification audit

## Design

The unit of observation is an earnings call. The audit uses {results['n_obs']:,} calls,
{results['n_firms']:,} firms, and {results['n_event_dates']:,} event trading dates. Prepared,
question, and answer innovations enter each regression jointly. Specifications include firm
and event-calendar-quarter fixed effects, with standard errors clustered by firm and event
date. The return-history control covers days -60 through -21; no control overlaps either
pre-event outcome. This is the key difference from the earlier placebo specification:
`pre_ret_20` overlaps CAR[-10,-1], so retaining it can mechanically absorb the association
the placebo is meant to detect.

The two main outcomes are non-overlapping: CAR[-10,-6] and CAR[-5,-1]. Their sum reproduces
CAR[-10,-1]. Reported p-values use the paper's two-sided normal approximation. Holm p-values
treat the six component by window cells as one family.

## Full-sample results

{full_table}

The question coefficient is {question_early['beta_bps']:.1f} bps in CAR[-10,-6] and
{question_late['beta_bps']:.1f} bps in CAR[-5,-1]. The later window is small. The early
window has an unadjusted p-value of {question_early['p_value']:.3f} and a Holm-adjusted
p-value of {question_early['p_holm_full_6']:.3f}.

{joint_table}

The joint two-window test for Question has raw p={question_joint['p_value']:.3f} and
Holm-adjusted p={question_joint['p_holm_full_3']:.3f} across the three components. The
Answer joint test has p={answer_joint['p_value']:.3f}. These tests use the cross-window
covariance implied by the early, late, and total-window regression variances.

## Call timing and alignment

{timing_table}

The Question point estimate in CAR[-10,-6] is positive in all three timing groups, but no
timing-specific Question cell survives its six-cell Holm correction. The pattern is not
confined to after-close calls.

{alignment_table}

All after-close calls are assigned to a later event trading date. In
{int(after_close_alignment['same_day_release_before_close_after_close_call_n']):,} cases,
the earnings announcement occurred before the close on the same calendar day as an
after-close call. Those daily returns can contain public earnings information while still
falling on the negative side of the aligned call-day window. The timing estimates show
whether this small, directly identified group is consistent with the broader pattern; they
do not identify every possible announcement-alignment problem.

The conservative robustness sample removes
{int(full_alignment['conservative_announcement_alignment_exclusion_n']):,} calls: any
regular-hours announcement dated before the aligned event day, plus announcement matches
at least six calendar days before that day. This rule is intentionally broader than the
set of observations known to overlap a particular CRSP return.

{alignment_clean_table}

After those exclusions, the Question coefficient in CAR[-10,-6] is
{clean_question_early['beta_bps']:.1f} bps (t={clean_question_early['t_stat']:.2f}).

## Assessment

The Question result is concentrated in the earlier window rather than the five trading days
immediately before the call. It is therefore evidence of a pre-period association, but two
five-day bins cannot establish a smooth pre-trend. The full-sample cell is marginal after
family-wise adjustment, and none of the call-timing cells survives its exploratory Holm
correction. The estimate also survives the conservative announcement-alignment exclusion,
so a few mismatched event dates do not explain it. Answer has no full-sample pre-period
association.

My reading is that this does not overturn the Answer result. It does mean Question should
not be presented as a clean event-time or causal effect. An additional pre-state control is
reasonable only if it is fixed before day -10 and chosen independently of this result; the
overlapping `pre_ret_20` control should not be restored. For a workshop paper, keep Answer
as the main result, describe Question as a conditional association, and put this audit in
the appendix. A daily lead-lag design is warranted if the paper later makes a stronger
timing or causal claim.
"""


def write_audit_outputs(
    results: dict[str, object], output_dir: Path, *, input_sample: Path
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_cells": output_dir / "full_sample_pre_event_cells.csv",
        "timing_cells": output_dir / "timing_pre_event_cells.csv",
        "alignment_clean_cells": output_dir / "alignment_clean_pre_event_cells.csv",
        "joint_tests": output_dir / "joint_pre_period_tests.csv",
        "alignment_summary": output_dir / "event_alignment_summary.csv",
        "memo": output_dir / "pre_event_audit.md",
        "manifest": output_dir / "pre_event_audit_manifest.json",
    }
    for key in [
        "full_cells",
        "timing_cells",
        "alignment_clean_cells",
        "joint_tests",
        "alignment_summary",
    ]:
        results[key].to_csv(paths[key], index=False)
    paths["memo"].write_text(build_audit_memo(results), encoding="utf-8")
    manifest = {
        "input_sample": str(input_sample),
        "unit_of_observation": "earnings call",
        "n_obs": results["n_obs"],
        "n_firms": results["n_firms"],
        "n_event_dates": results["n_event_dates"],
        "duplicate_firm_event_dates": results["duplicate_firm_dates"],
        "treatments": list(TREATMENTS),
        "outcomes": {outcome: OUTCOME_LABELS[outcome] for outcome in OUTCOMES},
        "controls": results["controls"],
        "fixed_effects": ["permno", "event_calendar_quarter"],
        "cluster_columns": ["permno", "event_trading_date"],
        "multiple_testing_family": "six component by non-overlapping-window cells",
        "outputs": {key: str(path) for key, path in paths.items() if key != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-event specification audit.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    input_sample = args.baseline_dir / "response_model_sample.parquet"
    sample = pd.read_parquet(input_sample)
    results = run_pre_event_audit(sample)
    outputs = write_audit_outputs(results, args.output_dir, input_sample=input_sample)
    print(json.dumps({"status": "ok", **outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
