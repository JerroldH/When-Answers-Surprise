"""Batch-mode Gemini scoring helper for appendix LLM answer tone.

This script keeps the same prompt and cache schema as
95_score_llm_answer_online.py, but moves the slow scoring step to
Agent Platform batch prediction. It is deliberately split into explicit
prepare, submit, status, download, and import commands.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from when_answers_surprise.llm import answer_pipeline as pipeline  # noqa: E402
from when_answers_surprise.llm.tone_scoring import (  # noqa: E402
    LLMScoreConfig,
    build_payload,
    gcloud_command,
    gcloud_project,
    gemini_batch_model_path,
    gemini_batch_request,
    gemini_batch_response_content,
    gemini_batch_usage_fields,
    parse_integer_score,
    prompt_without_answer,
    provider_config,
    render_prompt,
    safe_text,
    sha256_text,
    utc_now,
)

ANALYSIS_ROOT = PROJECT_ROOT / "data" / "wrds_sp500_2018_2025" / "analysis"
BASELINE_DIR = ANALYSIS_ROOT / "lm_baseline"
FEATURE_CACHE_DIR = ANALYSIS_ROOT / "lm_baseline_learner_audit_cache"
OUTPUT_DIR = ANALYSIS_ROOT / "llm_answer_tone_gemini_3_5_flash"
BATCH_DIR = OUTPUT_DIR / "batch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Gemini batch answer-tone scoring.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Write local JSONL batch input.")
    add_common_paths(prepare)
    add_provider_args(prepare)
    prepare.add_argument("--input-jsonl", type=Path, default=BATCH_DIR / "requests.jsonl")
    prepare.add_argument("--manifest", type=Path, default=BATCH_DIR / "batch_manifest.json")
    prepare.add_argument("--max-prompts", type=int, default=None)
    prepare.add_argument("--upload-gcs-uri", default="")
    prepare.set_defaults(func=prepare_batch)

    submit = subparsers.add_parser("submit", help="Submit a prepared GCS JSONL file.")
    add_provider_args(submit)
    submit.add_argument("--gcs-input-uri", required=True)
    submit.add_argument("--gcs-output-uri", required=True)
    submit.add_argument("--display-name", default="llm-answer-tone-batch")
    submit.add_argument("--job-json", type=Path, default=BATCH_DIR / "batch_job.json")
    submit.set_defaults(func=submit_batch)

    status = subparsers.add_parser("status", help="Check a batch prediction job.")
    status.add_argument("--job-name", default="")
    status.add_argument("--job-json", type=Path, default=BATCH_DIR / "batch_job.json")
    status.set_defaults(func=status_batch)

    download = subparsers.add_parser("download", help="Download GCS batch output.")
    download.add_argument("--gcs-output-uri", required=True)
    download.add_argument("--local-output-dir", type=Path, default=BATCH_DIR / "output")
    download.set_defaults(func=download_batch_output)

    import_cmd = subparsers.add_parser("import", help="Import completed JSONL output.")
    add_common_paths(import_cmd)
    add_provider_args(import_cmd)
    import_cmd.add_argument("--batch-output-dir", type=Path, default=BATCH_DIR / "output")
    import_cmd.add_argument(
        "--score-path",
        type=Path,
        default=OUTPUT_DIR / "answer_llm_scores.parquet",
    )
    import_cmd.add_argument("--job-json", type=Path, default=BATCH_DIR / "batch_job.json")
    import_cmd.set_defaults(func=import_batch_output)

    args = parser.parse_args()
    args.func(args)


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=FEATURE_CACHE_DIR)


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="gemini_adc", choices=["gemini_adc"])
    parser.add_argument("--model", default="google/gemini-3.5-flash")
    parser.add_argument("--project", default="")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))


def prepare_batch(args: argparse.Namespace) -> None:
    pipeline = load_online_pipeline()
    config = provider_config(args.provider, model=args.model)
    scoring_input = load_scoring_input(args.baseline_dir, args.feature_cache_dir, pipeline)
    score_path = OUTPUT_DIR / "answer_llm_scores.parquet"
    pending = pending_rows(scoring_input, score_path, config, pipeline)
    unique_prompts = build_unique_prompt_rows(pending, config)
    if args.max_prompts is not None:
        unique_prompts = unique_prompts.head(args.max_prompts).copy()

    args.input_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in unique_prompts.itertuples(index=False):
            handle.write(json.dumps(gemini_batch_request(row.prompt), ensure_ascii=False) + "\n")

    manifest = {
        "created_at_utc": utc_now(),
        "provider": config.provider,
        "model": config.model,
        "batch_model_path": gemini_batch_model_path(config.model),
        "prompt_version": config.prompt_version,
        "prompt_sha256": manifest_prompt_sha(),
        "pending_rows": int(len(pending)),
        "unique_prompts": int(len(unique_prompts)),
        "input_jsonl": str(args.input_jsonl),
        "payload_policy": "model sees prompt text only; row ids stay local",
        "full_payload_saved_in_tracked_files": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.upload_gcs_uri:
        run_command(["gsutil", "cp", str(args.input_jsonl), args.upload_gcs_uri])
        manifest["uploaded_gcs_uri"] = args.upload_gcs_uri
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


def submit_batch(args: argparse.Namespace) -> None:
    project = resolve_project(args.project)
    location = args.location
    endpoint_prefix = "" if location == "global" else f"{location}-"
    url = (
        f"https://{endpoint_prefix}aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/batchPredictionJobs"
    )
    body = {
        "displayName": args.display_name,
        "model": gemini_batch_model_path(args.model),
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [args.gcs_input_uri]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": args.gcs_output_uri},
        },
    }
    response = requests.post(
        url,
        headers=auth_headers(),
        json=body,
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Batch submit failed: {response.status_code} {response.text}")
    args.job_json.parent.mkdir(parents=True, exist_ok=True)
    args.job_json.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    print(args.job_json.read_text(encoding="utf-8"))


def status_batch(args: argparse.Namespace) -> None:
    job_name = args.job_name or job_name_from_file(args.job_json)
    response = requests.get(batch_job_url(job_name), headers=auth_headers(), timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"Batch status failed: {response.status_code} {response.text}")
    print(json.dumps(response.json(), indent=2))


def download_batch_output(args: argparse.Namespace) -> None:
    args.local_output_dir.mkdir(parents=True, exist_ok=True)
    run_command(["gsutil", "-m", "cp", "-r", args.gcs_output_uri, str(args.local_output_dir)])
    print(json.dumps({"local_output_dir": str(args.local_output_dir)}, indent=2))


def import_batch_output(args: argparse.Namespace) -> None:
    pipeline = load_online_pipeline()
    config = provider_config(args.provider, model=args.model)
    scoring_input = load_scoring_input(args.baseline_dir, args.feature_cache_dir, pipeline)
    score_path = args.score_path
    existing = pd.read_parquet(score_path) if score_path.exists() else pd.DataFrame()
    terminal_by_key = terminal_cache_keys(existing, config, pipeline)
    rows_by_text = rows_pending_by_text(scoring_input, terminal_by_key, config, pipeline)
    job_name = job_name_from_file(args.job_json) if args.job_json.exists() else ""

    imported = []
    for line in iter_batch_output_lines(args.batch_output_dir):
        prompt = prompt_from_batch_line(line)
        text_sha = answer_text_sha_from_prompt(prompt)
        rows = rows_by_text.get(text_sha, [])
        if not rows:
            continue
        result = score_result_from_batch_line(line, config, text_sha, job_name)
        for row in rows:
            key = pipeline.cache_key_for_values(
                row.transcriptid,
                row.turn_index,
                row.text_sha256,
                config.provider,
                config.model,
                config.prompt_version,
            )
            if key in terminal_by_key:
                continue
            imported.append(
                {
                    "transcriptid": str(row.transcriptid),
                    "turn_index": int(row.turn_index),
                    "text_sha256": row.text_sha256,
                    **result,
                }
            )
            terminal_by_key.add(key)

    if imported:
        combined = pipeline.append_score_rows(existing, pd.DataFrame(imported), score_path)
    else:
        combined = existing
    summary = {
        "score_path": str(score_path),
        "imported_rows": int(len(imported)),
        "cache_rows": int(len(combined)),
        "rows_successful": int(combined["status"].eq("ok").sum()) if "status" in combined else 0,
    }
    print(json.dumps(summary, indent=2))


def load_online_pipeline():
    return pipeline


def load_scoring_input(baseline_dir: Path, feature_cache_dir: Path, pipeline) -> pd.DataFrame:
    answer_frame = pd.read_parquet(feature_cache_dir / "answer_first_stage_features.parquet")
    qa_turns = pd.read_parquet(baseline_dir / "qa_turns.parquet")
    return pipeline.build_scoring_input(answer_frame, qa_turns, max_rows=None)


def pending_rows(
    scoring_input: pd.DataFrame, score_path: Path, config: LLMScoreConfig, pipeline
) -> pd.DataFrame:
    existing = pd.read_parquet(score_path) if score_path.exists() else pd.DataFrame()
    terminal = terminal_cache_keys(existing, config, pipeline)
    keep = []
    for row in scoring_input.itertuples(index=False):
        key = pipeline.cache_key_for_values(
            row.transcriptid,
            row.turn_index,
            row.text_sha256,
            config.provider,
            config.model,
            config.prompt_version,
        )
        if key not in terminal:
            keep.append(row._asdict())
    return pd.DataFrame(keep)


def terminal_cache_keys(existing: pd.DataFrame, config: LLMScoreConfig, pipeline) -> set[str]:
    if existing.empty:
        return set()
    data = existing.copy()
    data["cache_key"] = pipeline.cache_key(data)
    current = data["cache_key"].map(lambda key: key.endswith(pipeline.config_suffix(config)))
    terminal = data["status"].isin(pipeline.TERMINAL_SCORE_STATUSES)
    return set(data.loc[current & terminal, "cache_key"])


def build_unique_prompt_rows(pending: pd.DataFrame, config: LLMScoreConfig) -> pd.DataFrame:
    if pending.empty:
        return pd.DataFrame(columns=["text_sha256", "prompt"])
    rows = []
    seen = set()
    for row in pending.itertuples(index=False):
        if row.text_sha256 in seen:
            continue
        payload = build_payload(row.answer_text, prompt_version=config.prompt_version)
        rows.append({"text_sha256": row.text_sha256, "prompt": render_prompt(payload)})
        seen.add(row.text_sha256)
    return pd.DataFrame(rows)


def rows_pending_by_text(
    scoring_input: pd.DataFrame,
    terminal_keys: set[str],
    config: LLMScoreConfig,
    pipeline,
) -> dict[str, list[object]]:
    rows = defaultdict(list)
    for row in scoring_input.itertuples(index=False):
        key = pipeline.cache_key_for_values(
            row.transcriptid,
            row.turn_index,
            row.text_sha256,
            config.provider,
            config.model,
            config.prompt_version,
        )
        if key not in terminal_keys:
            rows[row.text_sha256].append(row)
    return rows


def iter_batch_output_lines(output_dir: Path):
    files = sorted(output_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {output_dir}")
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield json.loads(text)


def prompt_from_batch_line(line: dict[str, Any]) -> str:
    request = line.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"Batch output line is missing request: {safe_text(line)[:500]}")
    contents = request.get("contents") or []
    parts = contents[0].get("parts") if contents and isinstance(contents[0], dict) else []
    return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))


def answer_text_sha_from_prompt(prompt: str) -> str:
    marker = "Text:\n"
    if marker not in prompt:
        raise ValueError("Batch prompt does not contain expected answer text marker.")
    return sha256_text(prompt.split(marker, maxsplit=1)[1])


def score_result_from_batch_line(
    line: dict[str, Any], config: LLMScoreConfig, text_sha: str, job_name: str
) -> dict[str, object]:
    processed_at = line.get("processed_time") or utc_now()
    base = {
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "prompt_sha256": manifest_prompt_sha(),
        "text_sha256": text_sha,
        "scored_at_utc": processed_at,
        "transport": "batch_prediction",
        "batch_job_name": job_name,
    }
    status_text = str(line.get("status") or "")
    response = line.get("response") if isinstance(line.get("response"), dict) else {}
    if status_text:
        return {
            **base,
            "score": None,
            "status": "api_error",
            "retry_count": 0,
            "raw_output": "",
            "error": safe_text(status_text),
        }
    try:
        raw_output = gemini_batch_response_content(response)
    except Exception as exc:
        return error_result(base, "api_error", "", exc)
    try:
        score = parse_integer_score(raw_output)
    except ValueError as exc:
        return error_result(base, "invalid_output", raw_output, exc)
    return {
        **base,
        "score": score,
        "status": "ok",
        "retry_count": 0,
        "raw_output": str(raw_output).strip(),
        "error": "",
        **gemini_batch_usage_fields(response),
    }


def error_result(
    base: dict[str, object], status: str, raw_output: str, exc: Exception
) -> dict[str, object]:
    return {
        **base,
        "score": None,
        "status": status,
        "retry_count": 0,
        "raw_output": str(raw_output).strip(),
        "error": safe_text(repr(exc)),
    }


def batch_job_url(job_name: str) -> str:
    return f"https://aiplatform.googleapis.com/v1/{job_name}"


def job_name_from_file(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = payload.get("name", "")
    if not name:
        raise ValueError(f"Batch job response has no name: {path}")
    return str(name)


def resolve_project(project: str) -> str:
    resolved = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or gcloud_project()
    if not resolved:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT or --project is required.")
    return resolved


def auth_headers() -> dict[str, str]:
    token = access_token_from_output(
        run_command([gcloud_command(), "auth", "print-access-token"])
    )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def access_token_from_output(output: str) -> str:
    for line in reversed([part.strip() for part in output.splitlines()]):
        if line.startswith(("ya29.", "eyJ")):
            return line
    raise RuntimeError("Could not parse gcloud access token output.")


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stderr}")
    return result.stdout


def manifest_prompt_sha() -> str:
    return sha256_text(prompt_without_answer(build_payload("")))


if __name__ == "__main__":
    main()
