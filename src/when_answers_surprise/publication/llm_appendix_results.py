"""Build and validate aggregate-only LLM appendix results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from when_answers_surprise.common.manifests import manifest_digest

VARIANTS = ("A0", "A1", "A2", "A3")
HORIZONS = (1, 3, 5, 10, 20)
OUTPUTS = (
    "holdout_ladders.csv",
    "holdout_horizons.csv",
    "provider_alignment.csv",
    "score_provenance.csv",
    "table_llm_holdout_ladders.tex",
)


def build_llm_appendix_results(
    *,
    project_root: Path,
    provider_dirs: dict[str, Path],
    lm_holdout_sample: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Freeze small post-hoc aggregates from completed local LLM analyses."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ladders: list[pd.DataFrame] = []
    horizons: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    alignment_frames: dict[str, pd.DataFrame] = {}
    source_paths = [lm_holdout_sample]

    for scorer, provider_dir in provider_dirs.items():
        analysis_dir = provider_dir / "holdout_analysis_nested"
        score_cache = provider_dir / "answer_llm_scores.parquet"
        inputs = {
            "summary": analysis_dir / "build_summary.json",
            "first_stage": analysis_dir / "llm_holdout_first_stage.csv",
            "response": analysis_dir / "llm_holdout_response.csv",
            "placebo": analysis_dir / "llm_holdout_placebo.csv",
            "standardization": analysis_dir / "llm_development_standardization.csv",
            "selection": analysis_dir / "llm_tuned_ridge_hyperparameter_selection.csv",
            "response_sample": analysis_dir / "llm_holdout_response_sample.parquet",
            "score_cache": score_cache,
        }
        missing = [path for path in inputs.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing completed {scorer} inputs: {missing}")
        source_paths.extend(inputs.values())

        summary = json.loads(inputs["summary"].read_text(encoding="utf-8"))
        if not summary.get("nested_time_forward_tuning"):
            raise ValueError(f"{scorer} analysis did not record nested time-forward tuning")
        if not summary.get("outcomes_loaded_after_scoring"):
            raise ValueError(f"{scorer} analysis did not preserve outcome-blind scoring")

        scores = pd.read_parquet(score_cache)
        identity = scores[["provider", "model", "prompt_version"]].drop_duplicates()
        if len(identity) != 1:
            raise ValueError(f"{scorer} score cache mixes scorer identities")
        provider, model, prompt_version = identity.iloc[0].astype(str)
        if len(scores) != int(summary["score_rows"]) or not scores["status"].eq("ok").all():
            raise ValueError(f"{scorer} score cache is incomplete")
        if scores[["transcriptid", "turn_index"]].duplicated().any():
            raise ValueError(f"{scorer} score cache has duplicate transcript-turn keys")

        first_stage = pd.read_csv(inputs["first_stage"])
        first_stage = first_stage.loc[first_stage["scope"].eq("holdout_2025")].copy()
        _require_variants(first_stage, scorer, "first stage")
        response = pd.read_csv(inputs["response"])
        placebo = pd.read_csv(inputs["placebo"])
        standardization = pd.read_csv(inputs["standardization"])
        selections = pd.read_csv(inputs["selection"])
        selected = selections.loc[
            selections["scope"].eq("holdout_2025") & selections["selected"].astype(bool)
        ].copy()
        _require_variants(selected, scorer, "selected models")

        selected["selected_alpha"] = selected["candidate_json"].map(
            lambda value: json.loads(value)["params"]["alpha"]
        )
        selected = selected[
            ["variant", "selected_alpha", "n_inner_folds", "max_inner_train_year"]
        ]
        max_placebo = (
            placebo.assign(abs_t=pd.to_numeric(placebo["t_stat"], errors="raise").abs())
            .groupby("variant", as_index=False)["abs_t"]
            .max()
            .rename(columns={"abs_t": "max_placebo_abs_t"})
        )
        response_wide = _response_summary(response)
        ladder = (
            first_stage.merge(selected, on="variant", validate="one_to_one")
            .merge(max_placebo, on="variant", validate="one_to_one")
            .merge(response_wide, on="variant", validate="one_to_one")
            .merge(
                standardization[["variant", "development_n"]],
                on="variant",
                validate="one_to_one",
            )
        )
        ladder.insert(0, "scorer", scorer)
        ladder.insert(1, "provider", provider)
        ladder.insert(2, "model", model)
        ladder.insert(3, "prompt_version", prompt_version)
        ladder.insert(4, "sample", "holdout_2025")
        ladder["n_response_calls"] = int(summary["holdout_calls_analyzed"])
        ladders.append(ladder)

        response = response.copy()
        response.insert(0, "scorer", scorer)
        response.insert(1, "provider", provider)
        response.insert(2, "model", model)
        response.insert(3, "prompt_version", prompt_version)
        response.insert(4, "sample", "holdout_2025")
        horizons.append(response)

        score_counts = scores["score"].value_counts()
        override_count = 0
        if "score_source_override" in scores:
            overrides = scores["score_source_override"].fillna("").astype(str).str.strip()
            override_count = int(overrides.ne("").sum())
        provenance.append(
            {
                "scorer": scorer,
                "provider": provider,
                "model": model,
                "prompt_version": prompt_version,
                "prompt_sha256": _single_value(scores, "prompt_sha256"),
                "score_rows": len(scores),
                "ok_rows": int(scores["status"].eq("ok").sum()),
                "historical_fallback_rows": override_count,
                "score_min": float(pd.to_numeric(scores["score"], errors="raise").min()),
                "score_max": float(pd.to_numeric(scores["score"], errors="raise").max()),
                **{
                    f"score_{label}_share": float(score_counts.get(value, 0) / len(scores))
                    for value, label in ((-2, "m2"), (-1, "m1"), (0, "zero"), (1, "p1"), (2, "p2"))
                },
                "scored_at_min_utc": str(scores["scored_at_utc"].min()),
                "scored_at_max_utc": str(scores["scored_at_utc"].max()),
                "nested_time_forward_tuning": True,
                "outcomes_loaded_after_scoring": True,
            }
        )

        sample = pd.read_parquet(
            inputs["response_sample"], columns=["transcriptid", "llm_answer_A3_z"]
        )
        if sample["transcriptid"].duplicated().any():
            raise ValueError(f"{scorer} response sample has duplicate transcript IDs")
        alignment_frames[scorer] = sample.rename(columns={"llm_answer_A3_z": scorer})

    lm = pd.read_parquet(lm_holdout_sample, columns=["transcriptid", "answer_innovation_z"])
    if lm["transcriptid"].duplicated().any():
        raise ValueError("LM holdout response sample has duplicate transcript IDs")
    common = lm.rename(columns={"answer_innovation_z": "LM"})
    for frame in alignment_frames.values():
        common = common.merge(frame, on="transcriptid", how="inner", validate="one_to_one")
    expected_n = {int(len(frame)) for frame in alignment_frames.values()} | {len(lm)}
    if len(expected_n) != 1 or len(common) != expected_n.pop():
        raise ValueError("LM/Gemini/DeepSeek holdout samples are not the same call intersection")

    ladder_frame = pd.concat(ladders, ignore_index=True)
    horizon_frame = pd.concat(horizons, ignore_index=True)
    alignment = _pairwise_alignment(common, [*provider_dirs, "LM"])
    provenance_frame = pd.DataFrame(provenance)
    ladder_frame.to_csv(output_dir / OUTPUTS[0], index=False)
    horizon_frame.to_csv(output_dir / OUTPUTS[1], index=False)
    alignment.to_csv(output_dir / OUTPUTS[2], index=False)
    provenance_frame.to_csv(output_dir / OUTPUTS[3], index=False)
    (output_dir / OUTPUTS[4]).write_text(_latex_ladder(ladder_frame), encoding="utf-8")

    output_records = [
        {"path": name, **manifest_digest(output_dir / name)} for name in OUTPUTS
    ]
    manifest = {
        "contract": "when-answers-surprise-llm-appendix-results-v1",
        "status": "complete",
        "role": "post_hoc_cross_model_representation_robustness",
        "main_measurement": "LM dictionary tone remains primary",
        "sample": "holdout_2025",
        "variants": list(VARIANTS),
        "horizons": list(HORIZONS),
        "nested_time_forward_tuning": True,
        "outcomes_loaded_after_scoring": True,
        "inference_boundary": (
            "conditional two-way-clustered standard errors; "
            "no first-stage refit adjustment"
        ),
        "redistribution": {
            "licensed_or_observation_level_data_included": False,
            "aggregate_outputs_only": True,
        },
        "sample_boundary": {
            "llm_development_standardization_calls": int(ladder_frame["development_n"].iloc[0]),
            "core_frozen_development_calls": 9583,
            "llm_holdout_response_calls": int(ladder_frame["n_response_calls"].iloc[0]),
            "note": (
                "The LLM development intersection has one fewer call than the frozen LM "
                "contract and is not substituted for the core result."
            ),
        },
        "outputs": output_records,
        "sources": [_source_record(project_root, path) for path in sorted(set(source_paths))],
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_llm_appendix_results(output_dir)
    return manifest


def validate_llm_appendix_results(output_dir: Path) -> dict[str, Any]:
    """Validate the public aggregate LLM appendix contract without local row data."""

    manifest_path = output_dir / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing LLM appendix source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("LLM appendix source manifest is incomplete")
    if manifest.get("role") != "post_hoc_cross_model_representation_robustness":
        raise ValueError("LLM appendix role is not explicit")
    if not manifest.get("nested_time_forward_tuning"):
        raise ValueError("LLM appendix did not record nested time-forward tuning")
    if manifest.get("redistribution", {}).get("licensed_or_observation_level_data_included"):
        raise ValueError("LLM appendix contract cannot redistribute observation-level data")
    output_rows = manifest.get("outputs", [])
    if {row.get("path") for row in output_rows} != set(OUTPUTS):
        raise ValueError("LLM appendix output list is incomplete")
    for row in output_rows:
        path = output_dir / row["path"]
        if not path.is_file() or manifest_digest(path)["sha256"] != row.get("sha256"):
            raise ValueError(f"LLM appendix output digest mismatch: {path}")
        if path.suffix.lower() == ".parquet":
            raise ValueError("Observation-level parquet cannot enter the LLM appendix contract")
    ladder = pd.read_csv(output_dir / "holdout_ladders.csv")
    expected = {(scorer, variant) for scorer in ("Gemini", "DeepSeek") for variant in VARIANTS}
    if set(zip(ladder["scorer"], ladder["variant"], strict=True)) != expected:
        raise ValueError(
            "LLM appendix ladder does not contain the complete two-provider A0-A3 grid"
        )
    if not ladder["n_response_calls"].eq(1901).all():
        raise ValueError("LLM appendix response sample is not the fixed 1,901-call intersection")
    return {"status": "passed", "outputs": len(output_rows), "ladder_rows": len(ladder)}


def _response_summary(response: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        values: dict[str, Any] = {"variant": variant}
        for horizon in (1, 20):
            row = response.loc[
                response["variant"].eq(variant) & response["horizon"].eq(horizon)
            ]
            if len(row) != 1:
                raise ValueError(f"Expected one {variant} CAR[0,{horizon}] response row")
            item = row.iloc[0]
            values.update(
                {
                    f"car{horizon}_bps": float(item["beta_bps"]),
                    f"car{horizon}_se_bps": 10000.0 * float(item["standard_error"]),
                    f"car{horizon}_t_stat": float(item["t_stat"]),
                }
            )
        rows.append(values)
    return pd.DataFrame(rows)


def _pairwise_alignment(common: pd.DataFrame, scorers: list[str]) -> pd.DataFrame:
    rows = []
    for index, left in enumerate(scorers):
        for right in scorers[index + 1 :]:
            x = pd.to_numeric(common[left], errors="raise")
            y = pd.to_numeric(common[right], errors="raise")
            rows.append(
                {
                    "measure": "standardized_call_level_A3",
                    "left": left,
                    "right": right,
                    "n_common": len(common),
                    "pearson": float(x.corr(y, method="pearson")),
                    "spearman": float(x.corr(y, method="spearman")),
                    "sign_agreement": float((np.sign(x) == np.sign(y)).mean()),
                }
            )
    return pd.DataFrame(rows)


def _latex_ladder(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-model LLM answer tone representation check.}",
        r"\label{tab:llm-answer-tone-ladder}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scorer & Variant & OOF $R^2$ & Max placebo $|t|$ & CAR1 bps ($t$) & CAR20 bps ($t$) \\",
        r"\midrule",
    ]
    for scorer in ("Gemini", "DeepSeek"):
        subset = frame.loc[frame["scorer"].eq(scorer)].set_index("variant")
        for variant in VARIANTS:
            row = subset.loc[variant]
            lines.append(
                f"{scorer} & {variant} & {row['oof_r2']:.3f} & "
                f"{row['max_placebo_abs_t']:.2f} & "
                f"{row['car1_bps']:.1f} ({row['car1_t_stat']:.2f}) & "
                f"{row['car20_bps']:.1f} ({row['car20_t_stat']:.2f}) \\\\"
            )
        if scorer == "Gemini":
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize Notes: This outcome blind, post hoc check evaluates the "
            r"expanding filtration result with two LLM representations. "
            r"All variants use nested time forward Ridge tuning. First stage rows use 1,954 calls. "
            r"Response and placebo estimates use the same 1,901 call 2025 intersection. "
            r"Response standard errors use firm and event date clustering. "
            r"The prespecified LM analysis uses the 200 refit adjustment.",
            r"\end{flushleft}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _require_variants(frame: pd.DataFrame, scorer: str, label: str) -> None:
    if len(frame) != len(VARIANTS) or set(frame["variant"]) != set(VARIANTS):
        raise ValueError(f"{scorer} {label} does not contain exactly A0-A3")


def _single_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"Expected one {column}, found {values.tolist()}")
    return str(values[0])


def _source_record(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        **manifest_digest(path),
    }
