"""Gemini API-key batch transport for extending an answer-score panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from when_answers_surprise.llm.answer_pipeline import (
    build_scoring_input,
    cache_key,
    cache_key_for_values,
    deduplicate_score_cache,
)
from when_answers_surprise.llm.tone_scoring import (
    PROMPT_VERSION,
    build_payload,
    gemini_batch_response_content,
    gemini_batch_usage_fields,
    parse_integer_score,
    prompt_without_answer,
    render_prompt,
    safe_text,
    sha256_text,
    utc_now,
)

MODEL = "gemini-3.5-flash"
FINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--baseline-dir", type=Path, required=True)
    submit.add_argument("--feature-dir", type=Path, required=True)
    submit.add_argument("--score-path", type=Path, required=True)
    submit.add_argument("--batch-dir", type=Path, required=True)
    submit.add_argument("--key-file", type=Path, default=default_key_file())
    submit.add_argument("--model", default=MODEL)
    submit.set_defaults(func=submit_batch)

    status = subparsers.add_parser("status")
    status.add_argument("--batch-dir", type=Path, required=True)
    status.add_argument("--key-file", type=Path, default=default_key_file())
    status.set_defaults(func=status_batch)

    import_cmd = subparsers.add_parser("import")
    import_cmd.add_argument("--batch-dir", type=Path, required=True)
    import_cmd.add_argument("--key-file", type=Path, default=default_key_file())
    import_cmd.set_defaults(func=import_batch)

    args = parser.parse_args()
    args.func(args)


def submit_batch(args: argparse.Namespace) -> None:
    from google.genai import types

    args.batch_dir.mkdir(parents=True, exist_ok=True)
    job_path = args.batch_dir / "batch_job.json"
    if job_path.exists():
        raise FileExistsError(job_path)
    features = pd.read_parquet(args.feature_dir / "answer_first_stage_features.parquet")
    qa_turns = pd.read_parquet(args.baseline_dir / "qa_turns.parquet")
    required = build_scoring_input(features, qa_turns, max_rows=None)
    required["cache_key"] = [
        cache_key_for_values(
            row.transcriptid,
            row.turn_index,
            row.text_sha256,
            "gemini",
            args.model,
            PROMPT_VERSION,
        )
        for row in required.itertuples(index=False)
    ]
    scores = pd.read_parquet(args.score_path)
    scores["cache_key"] = cache_key(scores)
    covered = set(scores.loc[scores["status"].eq("ok"), "cache_key"])
    pending = required.loc[~required["cache_key"].isin(covered)].copy()
    unique = pending.drop_duplicates("text_sha256").reset_index(drop=True)
    unique["batch_key"] = [f"r{index:08d}" for index in range(len(unique))]
    key_by_hash = dict(zip(unique["text_sha256"], unique["batch_key"], strict=True))
    pending["batch_key"] = pending["text_sha256"].map(key_by_hash)
    pending[["batch_key", "transcriptid", "turn_index", "text_sha256", "event_year"]].to_parquet(
        args.batch_dir / "pending_row_map.parquet", index=False
    )

    input_path = args.batch_dir / "requests.jsonl"
    with input_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in unique.itertuples(index=False):
            prompt = render_prompt(build_payload(row.answer_text))
            request = {
                "key": row.batch_key,
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": generation_config(),
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")

    client = api_client(args.key_file)
    uploaded = client.files.upload(
        file=input_path,
        config=types.UploadFileConfig(display_name="was-gemini-answer-batch", mime_type="jsonl"),
    )
    job = client.batches.create(
        model=args.model,
        src=uploaded.name,
        config={"display_name": f"was-answer-tone-{utc_now()[:10]}"},
    )
    manifest = {
        "created_at_utc": utc_now(),
        "required_rows": len(required),
        "covered_ok_rows": len(covered & set(required["panel_key"])),
        "pending_rows": len(pending),
        "unique_prompts": len(unique),
        "input_bytes": input_path.stat().st_size,
        "input_file_name": uploaded.name,
        "job_name": job.name,
        "job_state": job.state.name,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "score_path": str(args.score_path.resolve()),
    }
    write_json(manifest, job_path)
    print(json.dumps(manifest, indent=2))


def status_batch(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.batch_dir)
    client = api_client(args.key_file)
    job = client.batches.get(name=manifest["job_name"])
    manifest.update(
        {
            "job_state": job.state.name,
            "last_checked_at_utc": utc_now(),
            "result_file_name": job.dest.file_name if job.dest else None,
            "job_error": safe_text(job.error) if job.error else "",
        }
    )
    write_json(manifest, args.batch_dir / "batch_job.json")
    print(json.dumps(manifest, indent=2))


def import_batch(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.batch_dir)
    client = api_client(args.key_file)
    job = client.batches.get(name=manifest["job_name"])
    if job.state.name != "JOB_STATE_SUCCEEDED" or not job.dest or not job.dest.file_name:
        raise RuntimeError(f"Batch is not ready for import: {job.state.name}")
    content = client.files.download(file=job.dest.file_name)
    result_path = args.batch_dir / "responses.jsonl"
    result_path.write_bytes(content)

    mapping = pd.read_parquet(args.batch_dir / "pending_row_map.parquet")
    rows_by_key = {key: group for key, group in mapping.groupby("batch_key", sort=False)}
    imported = []
    for raw_line in content.decode("utf-8").splitlines():
        if not raw_line.strip():
            continue
        line = json.loads(raw_line)
        key = str(line.get("key", ""))
        if key not in rows_by_key:
            raise ValueError(f"Unknown Gemini batch response key: {key}")
        result = score_result(line, manifest)
        for row in rows_by_key[key].itertuples(index=False):
            imported.append(
                {
                    "transcriptid": str(row.transcriptid),
                    "turn_index": int(row.turn_index),
                    "text_sha256": row.text_sha256,
                    **result,
                }
            )
    if len(imported) != len(mapping):
        raise ValueError(f"Batch import row mismatch: {len(imported)} != {len(mapping)}")

    score_path = Path(manifest["score_path"])
    existing = pd.read_parquet(score_path)
    combined = deduplicate_score_cache(
        pd.concat([existing, pd.DataFrame(imported)], ignore_index=True, sort=False)
    ).drop(columns=["cache_key"])
    combined.to_parquet(score_path, index=False)
    summary = {
        "imported_rows": len(imported),
        "imported_ok_rows": sum(row["status"] == "ok" for row in imported),
        "score_rows": len(combined),
        "score_path": str(score_path),
        "result_file": str(result_path),
    }
    write_json(summary, args.batch_dir / "import_summary.json")
    print(json.dumps(summary, indent=2))


def score_result(line: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
    response = line.get("response") if isinstance(line.get("response"), dict) else {}
    error = line.get("status") or line.get("error")
    base = {
        "provider": "gemini",
        "model": manifest["model"],
        "prompt_version": manifest["prompt_version"],
        "prompt_sha256": sha256_text(prompt_without_answer(build_payload(""))),
        "scored_at_utc": utc_now(),
        "transport": "gemini_api_batch",
        "batch_job_name": manifest["job_name"],
        "retry_count": 0,
    }
    if error:
        return {
            **base,
            "score": None,
            "status": "api_error",
            "raw_output": "",
            "error": safe_text(error),
        }
    try:
        raw_output = gemini_batch_response_content(response)
        score = parse_integer_score(raw_output)
    except Exception as exc:
        return {
            **base,
            "score": None,
            "status": "invalid_output",
            "raw_output": str(locals().get("raw_output", "")).strip(),
            "error": safe_text(repr(exc)),
        }
    return {
        **base,
        "score": score,
        "status": "ok",
        "raw_output": str(raw_output).strip(),
        "error": "",
        **gemini_batch_usage_fields(response),
    }


def generation_config() -> dict[str, object]:
    return {
        "temperature": 0,
        "max_output_tokens": 256,
        "thinking_config": {"thinking_level": "LOW"},
        "response_mime_type": "text/x.enum",
        "response_schema": {"type": "STRING", "enum": ["-2", "-1", "0", "1", "2"]},
    }


def api_client(key_file: Path):
    from google import genai

    return genai.Client(api_key=read_api_key(key_file))


def read_api_key(path: Path) -> str:
    if path is None or not path.exists():
        raise FileNotFoundError("Set GEMINI_API_KEY_FILE or pass --key-file")
    return next(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def default_key_file() -> Path | None:
    value = os.environ.get("GEMINI_API_KEY_FILE", "").strip()
    return Path(value) if value else None


def read_manifest(batch_dir: Path) -> dict[str, object]:
    return json.loads((batch_dir / "batch_job.json").read_text(encoding="utf-8"))


def write_json(data: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
