"""Camera-ready LM response ladder and context-revision test from saved predictions.

No first-stage fitting, outcome-driven selection, or API calls. Original paper
outputs are read-only; this diagnostic uses a separate output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from when_answers_surprise.empirical.holdout_2025 import (  # noqa: E402
    DEFAULT_LM_HOLDOUT_LADDER_DIR,
    DEFAULT_PROTOCOL_PATH,
    STANDARDIZED_TREATMENTS,
    _read_and_verify_protocol,
)
from when_answers_surprise.empirical.revision_v2_inference import fit_joint_horizon_system  # noqa: E402
from when_answers_surprise.empirical.revision_v2_supplement import aggregate_answer_components  # noqa: E402

VARIANTS = ("A0", "A1", "A2", "A3")
OUTCOME = "CAR_0_1"


def verify_response(response: pd.DataFrame, protocol: dict, primary: pd.DataFrame) -> dict:
    """Require the exact primary specification and numerical anchor before new fits."""
    if response["transcriptid"].isna().any() or response["transcriptid"].duplicated().any():
        raise ValueError("Response transcript IDs must be non-missing and unique")
    system = fit_joint_horizon_system(
        response,
        outcomes=[OUTCOME],
        treatments=STANDARDIZED_TREATMENTS,
        controls=protocol["controls"],
    )
    reference = primary.set_index("treatment").loc[STANDARDIZED_TREATMENTS]
    actual = system.coefficients.set_index("treatment").loc[STANDARDIZED_TREATMENTS]
    expected_n = reference["n_calls"].unique()
    if len(expected_n) != 1 or system.n_obs != len(response) or len(response) != expected_n[0]:
        raise ValueError("Response sample does not match the frozen primary call count")
    for field, column in (("permno", "n_firms"), ("event_trading_date", "n_event_dates")):
        if not reference[column].eq(response[field].nunique()).all():
            raise ValueError(f"Response sample does not match frozen {column}")
    beta_error = (actual["beta"] * 10_000 - reference["coefficient_bps"]).abs().max()
    se_error = (actual["standard_error"] * 10_000 - reference["conditional_se_bps"]).abs().max()
    # Allow floating-point variation, not a different local reconstruction.
    if not np.isfinite([beta_error, se_error]).all() or max(beta_error, se_error) > 1e-4:
        raise ValueError(
            "Frozen primary parity failed; do not mix a rebuilt sample with the paper. "
            f"Max coefficient difference={beta_error:.6f} bps, SE difference={se_error:.6f} bps"
        )
    return {
        "primary_parity_passed": True,
        "max_beta_error_bps": float(beta_error),
        "max_se_error_bps": float(se_error),
        "n_calls": len(response),
    }


def estimate_diagnostic(
    response: pd.DataFrame,
    turns: pd.DataFrame,
    protocol: dict,
    common_sd: float,
    *,
    use_frozen_a3: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Fit five separate measures and U_A3 + (m_A3 - m_A0) on identical calls."""
    keys = ["transcriptid", "turn_index"]
    values = ["raw_score", "expected_score_oof", "narrative_innovation", "word_count"]
    if (
        set(turns["variant"]) != set(VARIANTS)
        or not turns["semantic_feature"].eq("lm_net_tone").all()
    ):
        raise ValueError("Expected only LM answer turns for A0, A1, A2, and A3")
    if turns[[*keys, "variant"]].isna().any().any() or turns.duplicated([*keys, "variant"]).any():
        raise ValueError("Duplicate or missing transcript-turn-variant keys")
    if (
        not np.isfinite(turns[values].to_numpy(dtype=float)).all()
        or turns["word_count"].lt(0).any()
    ):
        raise ValueError("All turn components and non-negative word counts must be finite")
    if not np.isclose(
        turns["raw_score"],
        turns["expected_score_oof"] + turns["narrative_innovation"],
        rtol=0,
        atol=1e-10,
    ).all():
        raise ValueError("Turn components fail raw = expected + residual")
    reference = turns.loc[turns["variant"].eq("A0")].set_index(keys).sort_index()
    for variant in VARIANTS:
        group = turns.loc[turns["variant"].eq(variant)].set_index(keys).sort_index()
        if not group.index.equals(reference.index):
            raise ValueError(f"{variant} does not use the same answer-turn intersection")
        if not np.allclose(
            group[["raw_score", "word_count"]],
            reference[["raw_score", "word_count"]],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(f"{variant} changes the raw score or aggregation weights")
    if reference.groupby(level="transcriptid")["word_count"].sum().le(0).any():
        raise ValueError("Every call must have positive total answer word count")
    if not np.isfinite(common_sd) or common_sd <= 0:
        raise ValueError("The common development raw-tone SD must be positive")

    components = aggregate_answer_components(turns, variants=VARIANTS)
    wide = components.pivot(
        index="transcriptid",
        columns="variant",
        values=["raw_answer_tone", "expected_answer_tone", "answer_residual"],
    )
    wide.columns = [f"{field}_{variant}" for field, variant in wide.columns]
    work = response.merge(wide.reset_index(), on="transcriptid", how="left", validate="one_to_one")
    if work[list(wide.columns)].isna().any().any():
        raise ValueError("At least one frozen response call is missing answer components")
    a3_error = float((work["answer_residual_A3"] - work["answer_innovation"]).abs().max())
    a3_rmse = float(np.sqrt(np.mean((work["answer_residual_A3"] - work["answer_innovation"]) ** 2)))
    if use_frozen_a3:
        scale = protocol["training_standardization"]["answer_innovation_z"]["sd_2020_2024"]
        # A replay is not byte-identical across BLAS builds. Require negligible
        # measurement error, and retain the saved primary residual in all fits.
        if a3_error / scale > 0.01 or a3_rmse / scale > 0.001:
            raise ValueError("Replayed A3 differs materially from the saved call residual")
        work["answer_residual_A3"] = work["answer_innovation"]
        work["expected_answer_tone_A3"] = work["raw_answer_tone_A3"] - work["answer_innovation"]
    elif a3_error > 1e-7:
        raise ValueError(f"A3 predictions differ from frozen response residuals: {a3_error:.6g}")

    # A shared development SD preserves the additive decomposition. The intercept
    # absorbs means; no holdout moments define the scale or tune any benchmark.
    work["raw_common"] = work["raw_answer_tone_A3"] / common_sd
    for variant in VARIANTS:
        work[f"{variant}_common"] = work[f"answer_residual_{variant}"] / common_sd
    work["delta_common"] = (
        work["expected_answer_tone_A3"] - work["expected_answer_tone_A0"]
    ) / common_sd
    np.testing.assert_allclose(
        work["A0_common"] - work["A3_common"], work["delta_common"], rtol=0, atol=1e-10
    )
    base = ["prepared_innovation_z", "question_innovation_z"]
    specifications = {
        "Raw": ["raw_common"],
        **{v: [f"{v}_common"] for v in VARIANTS},
        "A3 + Delta": ["A3_common", "delta_common"],
    }
    rows = []
    for specification, focal in specifications.items():
        result = fit_joint_horizon_system(
            work,
            outcomes=[OUTCOME],
            treatments=[*base, *focal],
            controls=protocol["controls"],
        )
        if result.n_obs != len(response):
            raise ValueError(f"{specification} silently changed the frozen response sample")
        for term in focal:
            row = result.coefficients.set_index("treatment").loc[term]
            rows.append(
                {
                    "sample": "holdout_2025",
                    "specification": specification,
                    "term": term,
                    "outcome": OUTCOME,
                    "beta_bps": float(row["beta"] * 10_000),
                    "se_bps": float(row["standard_error"] * 10_000),
                    "t_stat": float(row["t_stat"]),
                    "p_value": float(row["p_value"]),
                    "n_calls": int(result.n_obs),
                    "common_development_raw_tone_sd": common_sd,
                    "unit": "bps_per_development_raw_tone_sd",
                    "uncertainty": "conditional_firm_date_clustered",
                    "analysis_status": "post_review_diagnostic",
                }
            )
    audit = {
        "same_sample_all": True,
        "a3_max_residual_difference": a3_error,
        "a3_replay_rmse": a3_rmse,
        "a3_source": "saved_call_residual" if use_frozen_a3 else "saved_turn_predictions",
        "n_calls": len(work),
        "n_first_stage_calls": len(wide),
        "n_turns_per_variant": len(reference),
        "common_development_raw_tone_sd": common_sd,
        "raw_A3_correlation_on_response_calls": float(
            work["raw_answer_tone_A3"].corr(work["answer_residual_A3"])
        ),
    }
    return pd.DataFrame(rows), audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-sample", type=Path, required=True)
    parser.add_argument("--lm-ladder-dir", type=Path, default=DEFAULT_LM_HOLDOUT_LADDER_DIR)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "camera_ready_lm_holdout"
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    protocol = _read_and_verify_protocol(args.protocol_path)
    primary_path = ROOT / "paper_results" / "holdout_primary.csv"
    response = pd.read_parquet(args.response_sample)
    parity = verify_response(response, protocol, pd.read_csv(primary_path))
    manifest_path = args.lm_ladder_dir / "lm_holdout_answer_ladder_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("outcomes_loaded") is not False
        or manifest.get("nested_time_forward_tuning") is not True
        or manifest.get("training_years") != list(range(2018, 2025))
    ):
        raise ValueError("The saved ladder must follow the original nested time-forward protocol")
    if (
        manifest["protocol"]["sha256"]
        != hashlib.sha256(args.protocol_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Prediction and response protocols differ")
    turns_path = args.lm_ladder_dir / "lm_holdout_answer_turn_innovations.parquet"
    sources = [args.response_sample, args.protocol_path, primary_path, manifest_path]
    for record in manifest["outputs"]:
        path = args.lm_ladder_dir / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Prediction cache checksum mismatch: {path}")
        sources.append(path)
    if turns_path not in sources:
        raise ValueError(
            "Ladder manifest lacks saved turn predictions; rerun lm-ladder into a new directory"
        )
    reference_ladder_path = ROOT / "paper_results" / "holdout_lm_ladders.csv"
    reference_ladder = pd.read_csv(reference_ladder_path).set_index("variant").loc[list(VARIANTS)]
    saved_ladder = pd.read_csv(args.lm_ladder_dir / "lm_holdout_answer_ladder.csv")
    if len(saved_ladder) != 4 or set(saved_ladder["variant"]) != set(VARIANTS):
        raise ValueError("Saved first-stage ladder must contain A0--A3 exactly once")
    columns = ["n_turns", "n_calls", "n_features", "oof_r2", "oof_mae"]
    replay = (
        manifest.get("reconstruction", {}).get("method")
        == "replay_archived_nested_forward_selections"
    )
    observed = saved_ladder.set_index("variant").loc[list(VARIANTS)]
    # Record numerical replay errors separately from the unchanged paper metrics.
    # Counts must be exact. R2/MAE allowances are for iterative-solver replay only.
    tolerances = np.array([0, 0, 0, 1e-5, 1e-3]) if replay else np.full(5, 1e-7)
    errors = (observed[columns] - reference_ladder[columns]).abs()
    if not np.isfinite(errors).all().all() or not (errors.to_numpy() <= tolerances).all():
        raise ValueError("Saved first-stage ladder differs from the paper's frozen A0--A3 fit")
    sources.append(reference_ladder_path)
    development_path = ROOT / "paper_results" / "development_answer_decomposition.csv"
    scales = pd.read_csv(development_path)["common_raw_answer_tone_sd"].unique()
    if len(scales) != 1:
        raise ValueError("Development decomposition has inconsistent common scales")
    sources.append(development_path)
    turns = pd.read_parquet(turns_path)
    table, audit = estimate_diagnostic(
        response, turns, protocol, float(scales[0]), use_frozen_a3=replay
    )
    audit["first_stage_replay_errors"] = errors.to_dict(orient="index")
    audit["reconstruction"] = manifest.get("reconstruction")
    if audit["n_turns_per_variant"] != 34_001 or audit["n_first_stage_calls"] != 1_954:
        raise ValueError("Saved predictions do not match the paper's 2025 first-stage sample")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "holdout_lm_response_diagnostic.csv", index=False)
    table[["specification", "term", "beta_bps", "se_bps", "p_value", "n_calls"]].to_latex(
        args.output_dir / "holdout_lm_response_diagnostic.tex",
        index=False,
        float_format="%.3f",
        escape=True,
    )
    audit.update(parity)
    audit.update(
        {
            "status": "complete",
            "analysis_status": "post_review_diagnostic",
            "first_stage_refit": False,
            "input_predictions_replayed": replay,
            "llm_api_calls": 0,
            "sources": [
                {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sources
            ],
        }
    )
    audit["outputs"] = [
        {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(args.output_dir.glob("holdout_lm_response_diagnostic.*"))
    ]
    (args.output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
