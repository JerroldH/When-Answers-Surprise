"""Complete pre-event timing audit for the frozen 2025 holdout.

This module derives secondary diagnostics from the fixed 2025 response sample.
It does not modify the compact holdout freeze.  The formal placebo inference
uses four non-overlapping five-day CAR windows over [-20, -1].  A day-level
AR[-20, -1] system and an announcement-alignment-clean sensitivity diagnose
whether aggregation hides a localized pre-event association.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from when_answers_surprise.empirical.diagnostics import add_nuisance_time_features
from when_answers_surprise.empirical.holdout_2025 import (
    STANDARDIZED_TREATMENTS,
    _apply_training_standardization,
    _complete_holdout_response_sample,
    _frozen_response_controls,
)
from when_answers_surprise.empirical.pre_event_audit import (
    add_alignment_fields,
    alignment_summary,
    holm_adjust,
)
from when_answers_surprise.empirical.revision_v2_event_audit import (
    BIN_DEFINITIONS,
    DAILY_OUTCOMES,
    build_daily_pre_event_sample,
    safe_controls,
)
from when_answers_surprise.empirical.revision_v2_inference import (
    JointSystemResult,
    curve_joint_tests,
    fit_joint_horizon_system,
    simultaneous_bands,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOLDOUT_FREEZE = (
    PROJECT_ROOT
    / "outputs"
    / "frozen"
    / "holdout_2025"
    / "20260801_sockeye_v2"
    / "freeze"
)
DEFAULT_RESPONSE_SAMPLE = (
    PROJECT_ROOT
    / "outputs"
    / "sockeye"
    / "results"
    / "holdout_2025"
    / "runs"
    / "20260801_holdout_v1"
    / "analysis"
    / "response"
    / "holdout_2025_response_sample.parquet"
)
DEFAULT_DAILY_DATA = (
    PROJECT_ROOT
    / "data"
    / "wrds_holdout_2025_ciz"
    / "archive"
    / "2026-08-01-missing-q2-a3"
    / "build_v1"
    / "ciz_legacy_daily.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "frozen"
    / "holdout_2025_timing"
    / "20260801_local_v1"
)
PRIMARY_PARITY_TOLERANCE = 1e-10
COMPACT_WINDOW_PARITY_TOLERANCE = 1e-10


def estimate_holdout_timing_robustness(
    sample: pd.DataFrame,
    *,
    controls: Sequence[str],
) -> pd.DataFrame:
    """Estimate the one-day A3 response by call timing and in interactions.

    These are conditional response regressions on the frozen standardized
    residuals.  They do not repeat the 200 first-stage refits within timing
    subsamples.
    """

    required = {
        "call_timing_bucket",
        "market_hours_call",
        "call_after_close",
        *STANDARDIZED_TREATMENTS,
        *controls,
    }
    if missing := sorted(required.difference(sample.columns)):
        raise ValueError(f"Holdout timing robustness is missing columns: {missing}")
    samples = {
        "full": sample,
        "exclude_market_hours": sample.loc[~sample["market_hours_call"].astype(bool)],
        "premarket": sample.loc[sample["call_timing_bucket"].eq("premarket")],
        "market_hours": sample.loc[
            sample["call_timing_bucket"].eq("market_hours")
        ],
        "after_close": sample.loc[
            sample["call_timing_bucket"].eq("after_close")
        ],
    }
    rows: list[dict[str, object]] = []
    for label, subset in samples.items():
        system = fit_joint_horizon_system(
            subset,
            outcomes=["CAR_0_1"],
            treatments=STANDARDIZED_TREATMENTS,
            controls=controls,
        )
        row = system.coefficients.loc[
            system.coefficients["treatment"].eq("answer_innovation_z")
        ].iloc[0]
        rows.append(
            {
                "result_type": "subsample_estimate",
                "specification": label,
                "term": "answer_innovation_z",
                "beta": float(row["beta"]),
                "standard_error": float(row["standard_error"]),
                "p_value": float(row["p_value"]),
                "beta_bps": 10_000 * float(row["beta"]),
                "standard_error_bps": 10_000 * float(row["standard_error"]),
                "ci_low_bps": 10_000
                * (float(row["beta"]) - 1.96 * float(row["standard_error"])),
                "ci_high_bps": 10_000
                * (float(row["beta"]) + 1.96 * float(row["standard_error"])),
                "n_calls": int(len(subset)),
                "n_firms": int(subset["permno"].nunique()),
                "n_event_dates": int(subset["event_trading_date"].nunique()),
                "joint_wald_chi2": np.nan,
                "joint_df": np.nan,
                "uncertainty": "conditional firm/date two-way clustered",
            }
        )

    interaction = sample.copy()
    interaction["timing_market_hours"] = interaction[
        "call_timing_bucket"
    ].eq("market_hours").astype(float)
    interaction["answer_x_market_hours"] = (
        interaction["answer_innovation_z"] * interaction["timing_market_hours"]
    )
    interaction["answer_x_after_close"] = (
        interaction["answer_innovation_z"]
        * interaction["call_after_close"].astype(float)
    )
    interaction_treatments = [
        *STANDARDIZED_TREATMENTS,
        "answer_x_market_hours",
        "answer_x_after_close",
    ]
    interaction_system = fit_joint_horizon_system(
        interaction,
        outcomes=["CAR_0_1"],
        treatments=interaction_treatments,
        controls=[*controls, "timing_market_hours"],
    )
    for term in (
        "answer_innovation_z",
        "answer_x_market_hours",
        "answer_x_after_close",
    ):
        row = interaction_system.coefficients.loc[
            interaction_system.coefficients["treatment"].eq(term)
        ].iloc[0]
        rows.append(
            {
                "result_type": "timing_interaction_estimate",
                "specification": "premarket_base_interactions",
                "term": term,
                "beta": float(row["beta"]),
                "standard_error": float(row["standard_error"]),
                "p_value": float(row["p_value"]),
                "beta_bps": 10_000 * float(row["beta"]),
                "standard_error_bps": 10_000 * float(row["standard_error"]),
                "ci_low_bps": 10_000
                * (float(row["beta"]) - 1.96 * float(row["standard_error"])),
                "ci_high_bps": 10_000
                * (float(row["beta"]) + 1.96 * float(row["standard_error"])),
                "n_calls": int(len(interaction)),
                "n_firms": int(interaction["permno"].nunique()),
                "n_event_dates": int(interaction["event_trading_date"].nunique()),
                "joint_wald_chi2": np.nan,
                "joint_df": np.nan,
                "uncertainty": "conditional firm/date two-way clustered",
            }
        )
    positions = interaction_system.coefficient_index.set_index("treatment")
    interaction_positions = positions.loc[
        ["answer_x_market_hours", "answer_x_after_close"], "position"
    ].astype(int).to_numpy()
    beta = interaction_system.coefficients.set_index("position").loc[
        interaction_positions, "beta"
    ].to_numpy(dtype=float)
    covariance = interaction_system.covariance[
        np.ix_(interaction_positions, interaction_positions)
    ]
    statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
    rows.append(
        {
            "result_type": "joint_timing_equality_test",
            "specification": "premarket_base_interactions",
            "term": "answer_effect_equal_across_timing_buckets",
            "beta": np.nan,
            "standard_error": np.nan,
            "p_value": float(chi2.sf(statistic, df=2)),
            "beta_bps": np.nan,
            "standard_error_bps": np.nan,
            "ci_low_bps": np.nan,
            "ci_high_bps": np.nan,
            "n_calls": int(len(interaction)),
            "n_firms": int(interaction["permno"].nunique()),
            "n_event_dates": int(interaction["event_trading_date"].nunique()),
            "joint_wald_chi2": statistic,
            "joint_df": 2,
            "uncertainty": "conditional firm/date two-way clustered Wald test",
        }
    )
    return pd.DataFrame(rows)


def reconstruct_frozen_response_sample(
    response_path: Path,
    protocol: dict[str, object],
    frozen_primary: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Recreate the protocol-complete 1,901-call response sample and prove parity."""

    sample = pd.read_parquet(response_path)
    sample = _apply_training_standardization(sample, protocol)
    sample["log_market_cap"] = np.log(
        pd.to_numeric(sample["market_cap"], errors="coerce").clip(lower=1.0)
    )
    response_controls = _frozen_response_controls(sample, protocol)
    complete, _ = _complete_holdout_response_sample(
        sample,
        controls=response_controls,
    )
    if len(complete) != 1_901:
        raise ValueError(
            "Timing audit must recover exactly 1,901 protocol-complete calls; "
            f"observed {len(complete)}"
        )
    if complete["permno"].nunique() != 499:
        raise ValueError("Timing audit did not recover the frozen 499-firm sample")
    if complete["event_trading_date"].nunique() != 173:
        raise ValueError("Timing audit did not recover the frozen 173 event dates")

    primary = fit_joint_horizon_system(
        complete,
        outcomes=["CAR_0_1"],
        treatments=STANDARDIZED_TREATMENTS,
        controls=response_controls,
    ).coefficients
    _assert_coefficient_parity(
        primary,
        frozen_primary,
        keys=("outcome", "treatment"),
        columns=("beta", "standard_error"),
        tolerance=PRIMARY_PARITY_TOLERANCE,
        label="frozen primary response",
    )
    return complete, response_controls


def validate_daily_source(
    daily_path: Path, frozen_build_manifest: dict[str, object]
) -> dict[str, object]:
    """Validate exact bytes or a checksummed prior-build serialization.

    The local archive can contain the same CIZ panel serialized by an earlier
    PyArrow build.  In that case, accept it only if its own sibling build
    manifest verifies the file.  Event-level return identities and the two
    frozen compact placebo windows are checked separately before any output is
    accepted.
    """

    records = [
        record
        for record in frozen_build_manifest.get("files", [])
        if record.get("path") == "ciz_legacy_daily.parquet"
    ]
    if len(records) != 1:
        raise ValueError("Frozen build manifest lacks one CIZ legacy daily record")
    record = records[0]
    expected_bytes = int(record["bytes"])
    expected_sha256 = str(record["sha256"])
    observed_sha256 = _sha256(daily_path)
    if daily_path.stat().st_size != expected_bytes:
        raise ValueError("Local CIZ daily byte size differs from the frozen build")
    if observed_sha256 == expected_sha256:
        return {
            "status": "exact_frozen_file_hash",
            "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256,
        }

    sibling_manifest_path = daily_path.parent / "build_manifest.json"
    if not sibling_manifest_path.is_file():
        raise ValueError(
            "Local CIZ daily SHA-256 differs from the frozen build and has no "
            "sibling build manifest"
        )
    sibling_manifest = json.loads(sibling_manifest_path.read_text(encoding="utf-8"))
    sibling_records = [
        item
        for item in sibling_manifest.get("files", [])
        if item.get("path") == "ciz_legacy_daily.parquet"
    ]
    if (
        len(sibling_records) != 1
        or int(sibling_records[0]["bytes"]) != expected_bytes
        or sibling_records[0]["sha256"] != observed_sha256
    ):
        raise ValueError(
            "Local CIZ daily differs from the frozen build and fails its prior-build manifest"
        )
    return {
        "status": "checksummed_prior_build_serialization_pending_content_gates",
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "sibling_build_manifest": str(sibling_manifest_path.resolve()),
        "sibling_build_manifest_sha256": _sha256(sibling_manifest_path),
    }


def prepare_timing_sample(
    response_sample: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    expected_pre_event_controls: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Attach complete daily outcomes without changing the fixed response sample."""

    sample = build_daily_pre_event_sample(response_sample, daily)
    sample = add_alignment_fields(sample)
    sample = add_nuisance_time_features(sample)
    if not sample[DAILY_OUTCOMES].notna().all(axis=1).all():
        incomplete = int((~sample[DAILY_OUTCOMES].notna().all(axis=1)).sum())
        raise ValueError(
            "Daily pre-event returns do not cover the fixed response sample: "
            f"{incomplete} incomplete calls"
        )
    controls = safe_controls(sample)
    if controls != list(expected_pre_event_controls):
        raise ValueError(
            "Timing-audit controls differ from the frozen pre-event contract: "
            f"expected={list(expected_pre_event_controls)}, observed={controls}"
        )
    return sample, controls


def validate_event_return_identities(sample: pd.DataFrame) -> dict[str, float]:
    """Verify daily outcomes against call-level values fixed before this audit."""

    required = {
        "pre_ret_20",
        "pre_market_ret_20",
        "placebo_CAR_m5_m1",
        "placebo_CAR_m10_m1",
        *DAILY_OUTCOMES,
    }
    if missing := sorted(required.difference(sample.columns)):
        raise ValueError(f"Return-identity check is missing columns: {missing}")
    full_twenty = sample[DAILY_OUTCOMES].sum(axis=1)
    last_ten = sample[BIN_DEFINITIONS["CAR_m10_m6"]].sum(axis=1) + sample[
        BIN_DEFINITIONS["CAR_m5_m1"]
    ].sum(axis=1)
    errors = {
        "ar_m20_m1_vs_pre_ret_minus_market": float(
            (
                full_twenty
                - (
                    pd.to_numeric(sample["pre_ret_20"], errors="coerce")
                    - pd.to_numeric(sample["pre_market_ret_20"], errors="coerce")
                )
            )
            .abs()
            .max()
        ),
        "ar_m5_m1_vs_frozen_call_value": float(
            (
                sample[BIN_DEFINITIONS["CAR_m5_m1"]].sum(axis=1)
                - pd.to_numeric(sample["placebo_CAR_m5_m1"], errors="coerce")
            )
            .abs()
            .max()
        ),
        "ar_m10_m1_vs_frozen_call_value": float(
            (
                last_ten
                - pd.to_numeric(sample["placebo_CAR_m10_m1"], errors="coerce")
            )
            .abs()
            .max()
        ),
    }
    if any(not np.isfinite(value) or value > 1e-12 for value in errors.values()):
        raise ValueError(f"Daily event-return identity failed: {errors}")
    return errors


def estimate_timing_systems(
    sample: pd.DataFrame,
    *,
    sample_label: str,
    controls: Sequence[str],
    seed: int,
) -> dict[str, object]:
    """Estimate daily and four-bin joint systems on one explicitly labeled sample."""

    daily = fit_joint_horizon_system(
        sample,
        outcomes=DAILY_OUTCOMES,
        treatments=STANDARDIZED_TREATMENTS,
        controls=controls,
    )
    five_day = fit_joint_horizon_system(
        sample,
        outcomes=list(BIN_DEFINITIONS),
        treatments=STANDARDIZED_TREATMENTS,
        controls=controls,
    )
    if daily.n_obs != len(sample) or five_day.n_obs != len(sample):
        raise ValueError(f"{sample_label} timing systems changed their input sample")

    daily_coefficients = _decorate_coefficients(
        daily.coefficients,
        sample_label=sample_label,
        multiplicity_column="p_holm_cell_60",
    )
    five_day_coefficients = _decorate_coefficients(
        five_day.coefficients,
        sample_label=sample_label,
        multiplicity_column="p_holm_cell_12",
    )
    daily_bands = _decorate_bands(
        simultaneous_bands(
            daily_coefficients,
            daily.covariance,
            daily.coefficient_index,
            seed=seed,
        )
    )
    five_day_bands = _decorate_bands(
        simultaneous_bands(
            five_day_coefficients,
            five_day.covariance,
            five_day.coefficient_index,
            seed=seed + 1,
        )
    )
    return {
        "daily_system": daily,
        "daily_coefficients": daily_coefficients,
        "daily_bands": daily_bands,
        "daily_joint_tests": _joint_tests(daily, sample_label),
        "five_day_system": five_day,
        "five_day_coefficients": five_day_coefficients,
        "five_day_bands": five_day_bands,
        "five_day_joint_tests": _joint_tests(five_day, sample_label),
    }


def run_holdout_2025_timing_audit(
    *,
    holdout_freeze: Path = DEFAULT_HOLDOUT_FREEZE,
    response_path: Path = DEFAULT_RESPONSE_SAMPLE,
    daily_path: Path = DEFAULT_DAILY_DATA,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Run and checksum the complete holdout timing audit in a new directory."""

    holdout_freeze = holdout_freeze.resolve()
    response_path = response_path.resolve()
    daily_path = daily_path.resolve()
    output_dir = output_dir.resolve()
    for required in (holdout_freeze, response_path, daily_path):
        if not required.exists():
            raise FileNotFoundError(required)
    if output_dir.exists():
        raise FileExistsError(f"Timing-audit output already exists: {output_dir}")

    protocol_path = holdout_freeze / "protocol" / "holdout_protocol.json"
    primary_path = (
        holdout_freeze
        / "analysis"
        / "response"
        / "holdout_2025_primary_coefficients.csv"
    )
    compact_placebo_path = (
        holdout_freeze
        / "analysis"
        / "response"
        / "holdout_2025_pre_event_placebos.csv"
    )
    build_manifest_path = holdout_freeze / "build" / "build_manifest.json"
    response_manifest_path = (
        holdout_freeze / "analysis" / "response" / "holdout_response_manifest.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    response_manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    frozen_primary = pd.read_csv(primary_path)
    compact_placebos = pd.read_csv(compact_placebo_path)

    daily_source_validation = validate_daily_source(daily_path, build_manifest)
    response_sample, response_controls = reconstruct_frozen_response_sample(
        response_path,
        protocol,
        frozen_primary,
    )
    post_event_timing = estimate_holdout_timing_robustness(
        response_sample,
        controls=response_controls,
    )
    daily = pd.read_parquet(daily_path)
    sample, controls = prepare_timing_sample(
        response_sample,
        daily,
        expected_pre_event_controls=response_manifest["pre_event_controls"],
    )
    return_identity_errors = validate_event_return_identities(sample)
    full = estimate_timing_systems(
        sample,
        sample_label="full",
        controls=controls,
        seed=20260801,
    )
    _validate_compact_window_parity(full["five_day_coefficients"], compact_placebos)
    if daily_source_validation["status"].endswith("pending_content_gates"):
        daily_source_validation["status"] = (
            "checksummed_prior_build_serialization_with_passed_content_gates"
        )

    clean_mask = ~sample["conservative_announcement_alignment_exclusion"].astype(bool)
    clean_sample = sample.loc[clean_mask].copy()
    clean = estimate_timing_systems(
        clean_sample,
        sample_label="alignment_clean",
        controls=controls,
        seed=20260803,
    )

    parent = output_dir.parent
    temporary = parent / f".{output_dir.name}.incomplete"
    if temporary.exists():
        raise FileExistsError(f"Incomplete timing-audit directory exists: {temporary}")
    parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        design = {
            "status": "fixed_secondary_diagnostic",
            "contract": "finnlp-2025-holdout-timing-audit-v1",
            "analysis_timing": "specified after the primary holdout response was observed",
            "reporting_rule": "report every day and every five-day bin regardless of result",
            "unit_of_observation": "earnings call",
            "full_sample_calls": int(len(sample)),
            "alignment_clean_calls": int(len(clean_sample)),
            "treatments": list(STANDARDIZED_TREATMENTS),
            "daily_outcomes": list(DAILY_OUTCOMES),
            "five_day_bins": BIN_DEFINITIONS,
            "controls": controls,
            "fixed_effects": ["permno", "event_calendar_quarter"],
            "clusters": ["permno", "event_trading_date"],
            "formal_placebo_inference": "four non-overlapping five-day bins",
            "day_level_role": "secondary timing and alignment diagnostic",
            "post_event_timing_role": (
                "secondary market-hours exclusion and timing-interaction diagnostic"
            ),
            "alignment_exclusion": "regular-hours announcement before aligned event day or announcement match at least six calendar days stale",
            "uncertainty": "conditional firm/date two-way clustered covariance",
            "simultaneous_intervals": "Gaussian max-absolute-t within stage",
        }
        (temporary / "timing_audit_design.json").write_text(
            json.dumps(design, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_system_outputs(temporary, "full", full)
        _write_system_outputs(temporary, "alignment_clean", clean)
        alignment_summary(sample).to_csv(
            temporary / "event_alignment_summary.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "sample_label": "full",
                    "calls": len(sample),
                    "firms": sample["permno"].nunique(),
                    "event_dates": sample["event_trading_date"].nunique(),
                    "complete_daily_share": sample[DAILY_OUTCOMES]
                    .notna()
                    .all(axis=1)
                    .mean(),
                },
                {
                    "sample_label": "alignment_clean",
                    "calls": len(clean_sample),
                    "firms": clean_sample["permno"].nunique(),
                    "event_dates": clean_sample["event_trading_date"].nunique(),
                    "complete_daily_share": clean_sample[DAILY_OUTCOMES]
                    .notna()
                    .all(axis=1)
                    .mean(),
                },
            ]
        ).to_csv(temporary / "sample_coverage.csv", index=False)
        post_event_timing.to_csv(
            temporary / "holdout_post_event_timing_robustness.csv", index=False
        )

        output_files = sorted(path for path in temporary.iterdir() if path.is_file())
        manifest = {
            "status": "complete",
            "contract": "finnlp-2025-holdout-timing-audit-v1",
            "input_files": {
                "response_sample": _source_record(response_path),
                "ciz_legacy_daily": _source_record(daily_path),
                "frozen_protocol": _source_record(protocol_path),
                "frozen_primary_coefficients": _source_record(primary_path),
                "frozen_compact_placebos": _source_record(compact_placebo_path),
            },
            "primary_parity_tolerance": PRIMARY_PARITY_TOLERANCE,
            "compact_window_parity_tolerance": COMPACT_WINDOW_PARITY_TOLERANCE,
            "daily_source_validation": daily_source_validation,
            "event_return_identity_max_abs_errors": return_identity_errors,
            "full_sample_calls": int(len(sample)),
            "alignment_clean_calls": int(len(clean_sample)),
            "excluded_for_alignment": int((~clean_mask).sum()),
            "post_event_timing_first_stage_adjustment": False,
            "files": [_output_record(path, temporary) for path in output_files],
        }
        (temporary / "holdout_2025_timing_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output_dir)
    except Exception:
        raise RuntimeError(
            "Timing audit did not complete. Inspect the preserved temporary directory "
            f"at {temporary}."
        )
    return manifest


def _decorate_coefficients(
    coefficients: pd.DataFrame,
    *,
    sample_label: str,
    multiplicity_column: str,
) -> pd.DataFrame:
    out = coefficients.copy()
    out.insert(0, "sample_label", sample_label)
    out[multiplicity_column] = holm_adjust(out["p_value"])
    out["beta_bps"] = out["beta"] * 10_000
    out["se_bps"] = out["standard_error"] * 10_000
    return out


def _decorate_bands(bands: pd.DataFrame) -> pd.DataFrame:
    out = bands.copy()
    out["simultaneous_ci_low_bps"] = out["simultaneous_ci_low"] * 10_000
    out["simultaneous_ci_high_bps"] = out["simultaneous_ci_high"] * 10_000
    return out


def _joint_tests(system: JointSystemResult, sample_label: str) -> pd.DataFrame:
    tests = curve_joint_tests(
        system.coefficients,
        system.covariance,
        system.coefficient_index,
    )
    tests.insert(0, "sample_label", sample_label)
    tests["p_holm_stage_3"] = holm_adjust(tests["p_value"])
    tests["n_obs"] = system.n_obs
    return tests


def _write_system_outputs(
    output_dir: Path,
    prefix: str,
    results: dict[str, object],
) -> None:
    for family in ("daily", "five_day"):
        results[f"{family}_coefficients"].to_csv(
            output_dir / f"{prefix}_{family}_pre_event_coefficients.csv", index=False
        )
        results[f"{family}_bands"].to_csv(
            output_dir / f"{prefix}_{family}_pre_event_simultaneous_bands.csv",
            index=False,
        )
        results[f"{family}_joint_tests"].to_csv(
            output_dir / f"{prefix}_{family}_pre_event_joint_tests.csv", index=False
        )
        system: JointSystemResult = results[f"{family}_system"]
        np.save(output_dir / f"{prefix}_{family}_pre_event_covariance.npy", system.covariance)
        system.coefficient_index.to_csv(
            output_dir / f"{prefix}_{family}_pre_event_coefficient_index.csv",
            index=False,
        )


def _validate_compact_window_parity(
    four_bin_coefficients: pd.DataFrame,
    compact_placebos: pd.DataFrame,
) -> None:
    mapping = {
        "CAR_m10_m6": "placebo_CAR_m10_m6",
        "CAR_m5_m1": "placebo_CAR_m5_m1",
    }
    pieces = []
    for new_outcome, frozen_outcome in mapping.items():
        piece = four_bin_coefficients.loc[
            four_bin_coefficients["outcome"].eq(new_outcome)
        ].copy()
        piece["outcome"] = frozen_outcome
        pieces.append(piece)
    observed = pd.concat(pieces, ignore_index=True)
    _assert_coefficient_parity(
        observed,
        compact_placebos,
        keys=("outcome", "treatment"),
        columns=("beta", "standard_error"),
        tolerance=COMPACT_WINDOW_PARITY_TOLERANCE,
        label="frozen compact pre-event windows",
    )


def _assert_coefficient_parity(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    columns: Sequence[str],
    tolerance: float,
    label: str,
) -> None:
    left = observed[[*keys, *columns]].copy()
    right = expected[[*keys, *columns]].copy()
    merged = left.merge(
        right,
        on=list(keys),
        how="outer",
        validate="one_to_one",
        suffixes=("_observed", "_expected"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{label} parity keys differ")
    differences = []
    for column in columns:
        differences.append(
            np.abs(
                merged[f"{column}_observed"].to_numpy(dtype=float)
                - merged[f"{column}_expected"].to_numpy(dtype=float)
            )
        )
    maximum = float(np.max(np.concatenate(differences)))
    if not np.isfinite(maximum) or maximum > tolerance:
        raise ValueError(
            f"{label} parity failed: max absolute difference {maximum:.3e} "
            f"exceeds {tolerance:.3e}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _output_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete timing audit for the frozen 2025 holdout."
    )
    parser.add_argument("--holdout-freeze", type=Path, default=DEFAULT_HOLDOUT_FREEZE)
    parser.add_argument("--response-sample", type=Path, default=DEFAULT_RESPONSE_SAMPLE)
    parser.add_argument("--daily-data", type=Path, default=DEFAULT_DAILY_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_holdout_2025_timing_audit(
        holdout_freeze=args.holdout_freeze,
        response_path=args.response_sample,
        daily_path=args.daily_data,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
