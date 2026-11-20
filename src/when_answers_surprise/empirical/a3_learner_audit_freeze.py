"""Freeze the completed Sockeye A3 learner audit as an auditable evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "outputs"
    / "sockeye"
    / "results"
    / "runs"
    / "20260716_a3_tree_full_v2"
    / "a3_learner_audit"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "frozen"
    / "a3_learner_audit"
    / "20260716_a3_tree_full_v2"
)
INPUTS = {
    "answer_features": PROJECT_ROOT
    / "data"
    / "wrds_sp500_2018_2025"
    / "analysis"
    / "revision_v2"
    / "answer_first_stage_features_revision_v2.parquet",
    "response_sample": PROJECT_ROOT
    / "data"
    / "wrds_sp500_2018_2025"
    / "analysis"
    / "revision_v2"
    / "response_model_sample_revision_v2.parquet",
    "frozen_first_stage": PROJECT_ROOT
    / "data"
    / "wrds_sp500_2018_2025"
    / "analysis"
    / "revision_v2"
    / "answer_ladder_first_stage.csv",
}
RELEVANT_CODE = (
    "src/when_answers_surprise/empirical/revision_v2_learner_audit.py",
    "src/when_answers_surprise/empirical/learner_audit.py",
    "src/when_answers_surprise/empirical/feature_spec.py",
    "src/when_answers_surprise/empirical/pipeline.py",
    "hpc/sockeye/jobs/a3_learner_audit.sbatch",
    "hpc/sockeye/submit/a3_learner_audit.sh",
)


def freeze_a3_learner_audit(source: Path, output: Path) -> dict[str, object]:
    """Validate and copy one completed audit without mutating its source."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Frozen A3 learner-audit directory is not empty: {output}")
    summary = _read_json(source / "build_summary.json")
    metadata = _read_json(source / "hpc_run_metadata.json")
    _validate_audit(source, summary)

    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = output / path.name
        shutil.copy2(path, destination)
        copied.append(_file_record(destination, output))

    input_records = []
    for role, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        record = _file_record(path, PROJECT_ROOT)
        record["role"] = role
        input_records.append(record)

    code_records = []
    for relative in RELEVANT_CODE:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code_records.append(_file_record(path, PROJECT_ROOT))

    provenance = git_provenance(PROJECT_ROOT)
    manifest = {
        "status": "frozen",
        "artifact_role": "appendix_learner_sensitivity",
        "main_procedure_decision": "time_forward_tuned_ridge",
        "main_procedure_requires_full_sequential_refit": True,
        "source": str(source),
        "output": str(output),
        "response_calls": int(summary["response_calls"]),
        "market_outcomes_used_for_tuning": False,
        "ridge_alpha_grid": summary["ridge_alpha_grid"],
        "ridge_boundary_selections": int(summary["ridge_boundary_selections"]),
        "source_summary": summary,
        "source_hpc_metadata": metadata,
        "git_provenance": provenance,
        "input_files": input_records,
        "relevant_code_files": code_records,
        "output_files": copied,
    }
    manifest_path = output / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _validate_audit(source: Path, summary: dict[str, object]) -> None:
    if not (source / "_SUCCESS").is_file():
        raise ValueError("A3 audit is missing its _SUCCESS marker")
    if summary.get("status") != "complete":
        raise ValueError("A3 audit did not report complete status")
    if int(summary.get("response_calls", 0)) != 9583:
        raise ValueError("A3 audit does not use the frozen 9,583-call response sample")
    if not bool(summary.get("fixed_ridge_structural_validation_passed")):
        raise ValueError("A3 audit failed its structural fixed-Ridge validation")
    if bool(summary.get("market_outcomes_used_for_tuning")):
        raise ValueError("A3 audit used market outcomes for first-stage tuning")
    if int(summary.get("ridge_boundary_selections", -1)) != 0:
        raise ValueError("A3 tuned Ridge selected a grid boundary")
    primary = pd.read_csv(source / "a3_learner_primary_comparison.csv")
    required = {
        "fixed_ridge",
        "tuned_ridge",
        "tuned_elasticnet",
        "tuned_hist_gradient_boosting",
        "tuned_random_forest",
    }
    observed = set(primary["learner"].astype(str))
    if observed != required:
        raise ValueError(f"Unexpected learner comparison rows: {sorted(observed)}")
    if primary.isna().any().any():
        raise ValueError("A3 primary comparison contains missing values")


def git_provenance(project_root: Path) -> dict[str, object]:
    head = _git(project_root, "rev-parse", "HEAD")
    status = _git(project_root, "status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
    ).stdout
    return {
        "head": head.strip() or "unavailable",
        "working_tree_clean": not bool(status.strip()),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a completed A3 learner audit.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = freeze_a3_learner_audit(args.source, args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": manifest["output"],
                "response_calls": manifest["response_calls"],
                "output_file_count": len(manifest["output_files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
