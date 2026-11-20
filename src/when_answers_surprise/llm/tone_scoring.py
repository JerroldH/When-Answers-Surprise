from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import requests

PROMPT_VERSION = "answer_tone_integer_v1"
TEXT_TYPE = "management answer"
ALLOWED_SCORES = {-2, -1, 0, 1, 2}
PAYLOAD_KEYS = {"answer_text", "text_type", "prompt_version"}
DEFAULT_TIMEOUT_SECONDS = 45
MAX_OUTPUT_TOKENS = 128
ADC_TOKEN_TTL_SECONDS = 45 * 60
MAX_API_ATTEMPTS = 3
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
BASE_RETRY_SLEEP_SECONDS = 1.0
MAX_RETRY_SLEEP_SECONDS = 10.0
GEMINI_BATCH_PROVIDER = "gemini_adc"

PROVIDER_DEFAULTS = {
    "gemini_adc": {
        "api_key_env": "GOOGLE_OAUTH_ACCESS_TOKEN",
        "base_url": "",
        "model": "google/gemini-3.5-flash",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-3.5-flash",
        "max_output_tokens": 256,
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.4-nano",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "max_output_tokens": 8,
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.6-flash",
    },
}

_AUTH_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_AUTH_TOKEN_LOCK = Lock()
_API_KEY_POOL_CACHE: dict[tuple[str, str], tuple[str, ...]] = {}
_API_KEY_POOL_INDEX: dict[tuple[str, str], int] = {}


@dataclass(frozen=True)
class LLMScoreConfig:
    provider: str = "fake"
    model: str = "fake-answer-tone-v1"
    api_key_env: str = ""
    base_url: str = ""
    prompt_version: str = PROMPT_VERSION
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    retry_invalid_once: bool = True
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    extra_body: dict[str, Any] | None = None


def provider_config(provider: str, *, model: str | None = None) -> LLMScoreConfig:
    provider = provider.lower().strip()
    if provider == "fake":
        return LLMScoreConfig()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unknown LLM scoring provider: {provider}")
    defaults = PROVIDER_DEFAULTS[provider]
    base_url = str(defaults["base_url"])
    if provider == "gemini_adc":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or gcloud_project()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip()
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for provider=gemini_adc")
        base_url = (
            f"https://aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/endpoints/openapi"
        )
    return LLMScoreConfig(
        provider=provider,
        model=model or str(defaults["model"]),
        api_key_env=str(defaults["api_key_env"]),
        base_url=base_url,
        max_output_tokens=int(defaults.get("max_output_tokens", MAX_OUTPUT_TOKENS)),
        extra_body=defaults.get("extra_body"),
    )


def build_payload(answer_text: str, *, prompt_version: str = PROMPT_VERSION) -> dict[str, str]:
    payload = {
        "answer_text": str(answer_text or ""),
        "text_type": TEXT_TYPE,
        "prompt_version": prompt_version,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, object]) -> None:
    extra = set(payload).difference(PAYLOAD_KEYS)
    missing = PAYLOAD_KEYS.difference(payload)
    if extra or missing:
        raise ValueError(f"LLM payload keys must be exactly {sorted(PAYLOAD_KEYS)}")
    if payload.get("text_type") != TEXT_TYPE:
        raise ValueError("LLM payload text_type must be 'management answer'.")


def render_prompt(payload: dict[str, object]) -> str:
    validate_payload(payload)
    return "\n".join(
        [
            "Task: score the financial tone expressed in this management answer.",
            "Use only the text below. Do not use outside knowledge or prior dialogue.",
            "Reply with only one integer: -2, -1, 0, 1, or 2.",
            "-2 = very negative; -1 = negative; 0 = neutral, mixed, or factual;",
            "1 = positive; 2 = very positive.",
            f"Prompt version: {payload['prompt_version']}",
            f"Text type: {payload['text_type']}",
            "Text:",
            str(payload["answer_text"]),
        ]
    )


def parse_integer_score(text: object) -> int:
    value = str(text or "").strip()
    if not re.fullmatch(r"[+-]?\d+", value):
        raise ValueError(f"LLM score output is not a bare integer: {value!r}")
    score = int(value)
    if score not in ALLOWED_SCORES:
        raise ValueError(f"LLM score is outside allowed set: {score}")
    return score


def score_answer_text(answer_text: str, config: LLMScoreConfig) -> dict[str, object]:
    payload = build_payload(answer_text, prompt_version=config.prompt_version)
    prompt = render_prompt(payload)
    prompt_sha = sha256_text(prompt_without_answer(payload))
    text_sha = sha256_text(payload["answer_text"])
    base = {
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "prompt_sha256": prompt_sha,
        "text_sha256": text_sha,
        "scored_at_utc": utc_now(),
    }

    attempts = 2 if config.retry_invalid_once else 1
    last_error = ""
    usage: dict[str, object] = {}
    raw_output = ""
    for attempt in range(attempts):
        try:
            raw_output, usage = call_provider(prompt, config)
        except Exception as exc:
            return {
                **base,
                "score": None,
                "status": "api_error",
                "retry_count": attempt,
                "raw_output": str(raw_output).strip(),
                "error": safe_error(exc),
                **usage,
            }
        try:
            score = parse_integer_score(raw_output)
        except ValueError as exc:
            last_error = safe_error(exc)
            if attempt + 1 >= attempts:
                break
            continue
        return {
            **base,
            "score": score,
            "status": "ok",
            "retry_count": attempt,
            "raw_output": str(raw_output).strip(),
            "error": "",
            **usage,
        }
    return {
        **base,
        "score": None,
        "status": "invalid_output",
        "retry_count": attempts - 1,
        "raw_output": str(raw_output).strip(),
        "error": safe_text(last_error),
        **usage,
    }


def call_provider(prompt: str, config: LLMScoreConfig) -> tuple[str, dict[str, object]]:
    if config.provider == "fake":
        return fake_score(prompt), {}
    if config.provider == "gemini":
        return call_gemini_native(prompt, config)

    url = chat_completions_url(config.base_url)
    body = chat_completion_body(prompt, config)
    force_refresh = False
    response = None
    for attempt in range(MAX_API_ATTEMPTS):
        api_key = auth_token(config, force_refresh=force_refresh)
        force_refresh = False
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=config.timeout_seconds,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt + 1 >= MAX_API_ATTEMPTS:
                raise
            sleep_before_retry(attempt, None)
            continue
        if response.status_code == 401 and config.provider == "gemini_adc":
            clear_auth_token(config)
            force_refresh = True
            if attempt + 1 >= MAX_API_ATTEMPTS:
                raise_response_error(response)
            sleep_before_retry(attempt, response)
            continue
        if response.status_code in TRANSIENT_STATUS_CODES:
            if attempt + 1 >= MAX_API_ATTEMPTS:
                raise_response_error(response)
            sleep_before_retry(attempt, response)
            continue
        break
    if response is None:
        raise RuntimeError("Provider request did not return a response.")
    if response.status_code >= 400:
        raise_response_error(response)
    data = response.json()
    content = response_content(data)
    return str(content), usage_fields(data.get("usage"))


def call_gemini_native(prompt: str, config: LLMScoreConfig) -> tuple[str, dict[str, object]]:
    model = config.model.removeprefix("google/")
    url = f"{config.base_url.rstrip('/')}/models/{model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": config.max_output_tokens,
            "thinkingConfig": {"thinkingLevel": "LOW"},
            "responseMimeType": "text/x.enum",
            "responseSchema": {
                "type": "STRING",
                "enum": ["-2", "-1", "0", "1", "2"],
            },
        },
    }
    response = None
    for attempt in range(MAX_API_ATTEMPTS):
        api_key = auth_token(config)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body,
                timeout=config.timeout_seconds,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt + 1 >= MAX_API_ATTEMPTS:
                raise
            sleep_before_retry(attempt, None)
            continue
        if response.status_code in TRANSIENT_STATUS_CODES:
            if attempt + 1 >= MAX_API_ATTEMPTS:
                raise_response_error(response)
            sleep_before_retry(attempt, response)
            continue
        break
    if response is None:
        raise RuntimeError("Gemini request did not return a response.")
    if response.status_code >= 400:
        raise_response_error(response)
    data = response.json()
    return gemini_batch_response_content(data), gemini_batch_usage_fields(data)


def chat_completion_body(prompt: str, config: LLMScoreConfig) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model,
        "temperature": 0,
        "max_tokens": config.max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if config.extra_body:
        body.update(config.extra_body)
    return body


def sleep_before_retry(attempt: int, response: requests.Response | None) -> None:
    retry_after = retry_after_seconds(response.headers.get("Retry-After") if response else None)
    if retry_after is not None:
        time.sleep(min(retry_after, MAX_RETRY_SLEEP_SECONDS))
        return
    if response is not None and response.status_code == 429 and "spend-based" in response.text:
        time.sleep(30.0 * (attempt + 1) + random.random() * 10.0)
        return
    delay = min(MAX_RETRY_SLEEP_SECONDS, BASE_RETRY_SLEEP_SECONDS * (2**attempt))
    jitter = 0.75 + random.random() * 0.5
    time.sleep(delay * jitter)


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def raise_response_error(response: requests.Response) -> None:
    body = safe_text(response.text)[:500]
    message = f"{response.status_code} {response.reason} for url: {response.url}; body={body}"
    raise requests.HTTPError(message, response=response)


def response_content(data: dict[str, Any]) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        snippet = json.dumps(data, ensure_ascii=False)[:1000]
        raise RuntimeError(f"Missing chat completion message content: {snippet}") from exc


def gemini_batch_request(prompt: str) -> dict[str, object]:
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": MAX_OUTPUT_TOKENS},
        }
    }


def gemini_batch_model_path(model: str) -> str:
    clean = str(model or "").strip()
    if clean.startswith("publishers/google/models/"):
        return clean
    if clean.startswith("google/"):
        clean = clean.split("/", maxsplit=1)[1]
    return f"publishers/google/models/{clean}"


def gemini_batch_response_content(data: dict[str, Any]) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(str(part.get("text", "")) for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        snippet = json.dumps(data, ensure_ascii=False)[:1000]
        raise RuntimeError(f"Missing Gemini batch response text: {snippet}") from exc


def gemini_batch_usage_fields(response: dict[str, Any]) -> dict[str, object]:
    usage = response.get("usageMetadata") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
    }


def auth_token(config: LLMScoreConfig, *, force_refresh: bool = False) -> str:
    if config.provider != "gemini_adc":
        return static_auth_token(config)
    cache_key = (config.provider, config.base_url)
    now = time.time()
    with _AUTH_TOKEN_LOCK:
        cached = _AUTH_TOKEN_CACHE.get(cache_key)
        if cached and not force_refresh and cached[1] > now:
            return cached[0]
        token = adc_auth_token()
        _AUTH_TOKEN_CACHE[cache_key] = (token, now + ADC_TOKEN_TTL_SECONDS)
        return token


def static_auth_token(config: LLMScoreConfig) -> str:
    keys = static_auth_keys(config)
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    cache_key = (config.provider, config.api_key_env)
    with _AUTH_TOKEN_LOCK:
        index = _API_KEY_POOL_INDEX.get(cache_key, 0)
        _API_KEY_POOL_INDEX[cache_key] = index + 1
    return keys[index % len(keys)]


def static_auth_keys(config: LLMScoreConfig) -> tuple[str, ...]:
    cache_key = (config.provider, config.api_key_env)
    cached = _API_KEY_POOL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    keys = []
    direct = os.environ.get(config.api_key_env, "").strip()
    if direct:
        keys.extend(split_api_keys(direct))
    key_list = os.environ.get(f"{config.api_key_env}S", "").strip()
    if key_list:
        keys.extend(split_api_keys(key_list))
    key_file = (
        os.environ.get(f"{config.api_key_env}_FILE", "").strip()
        or os.environ.get(f"{config.provider.upper()}_API_KEY_FILE", "").strip()
    )
    if key_file:
        try:
            keys.extend(split_api_keys(Path(key_file).read_text(encoding="utf-8")))
        except OSError as exc:
            raise RuntimeError(f"Could not read API key file for {config.provider}") from exc
    unique = tuple(dict.fromkeys(key for key in keys if key))
    _API_KEY_POOL_CACHE[cache_key] = unique
    return unique


def split_api_keys(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\r\n,;]+", value) if part.strip()]


def clear_auth_token(config: LLMScoreConfig) -> None:
    cache_key = (config.provider, config.base_url)
    with _AUTH_TOKEN_LOCK:
        _AUTH_TOKEN_CACHE.pop(cache_key, None)


def adc_auth_token() -> str:
    # ponytail: no google-auth dependency; use the local ADC/gcloud token if present.
    gcloud = gcloud_command()
    for command in [
        [gcloud, "auth", "application-default", "print-access-token"],
        [gcloud, "auth", "print-access-token"],
    ]:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    raise RuntimeError("Could not get a Google Cloud access token from ADC or gcloud.")


def safe_error(exc: Exception) -> str:
    return safe_text(repr(exc))


def safe_text(text: object) -> str:
    value = str(text or "")
    value = re.sub(r"ya29\.[A-Za-z0-9._~+/=-]+", "[REDACTED_GOOGLE_ACCESS_TOKEN]", value)
    value = re.sub(r"Bearer\s+[^'\"\\r\\n]+", "Bearer [REDACTED]", value)
    return value.replace("\r", "\\r").replace("\n", "\\n")


def gcloud_command() -> str:
    candidates = [
        "gcloud",
        "gcloud.cmd",
        r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
    ]
    for command in candidates:
        try:
            result = subprocess.run(
                [command, "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            continue
        if result.returncode == 0:
            return command
    return "gcloud"


def gcloud_project() -> str:
    try:
        result = subprocess.run(
            [gcloud_command(), "config", "get-value", "project"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def fake_score(prompt: str) -> str:
    text = prompt.split("Text:", maxsplit=1)[-1].upper()
    positive_words = ["GROWTH", "STRONG", "IMPROVED", "RECORD", "SUCCESS"]
    negative_words = ["DECLINE", "WEAK", "LOSS", "RISK", "CHARGE", "PRESSURE"]
    positive = sum(text.count(word) for word in positive_words)
    negative = sum(text.count(word) for word in negative_words)
    net = positive - negative
    if net >= 2:
        return "2"
    if net == 1:
        return "1"
    if net == -1:
        return "-1"
    if net <= -2:
        return "-2"
    return "0"


def chat_completions_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return clean + "/chat/completions"


def usage_fields(usage: Any) -> dict[str, object]:
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def prompt_without_answer(payload: dict[str, object]) -> str:
    scrubbed = dict(payload)
    scrubbed["answer_text"] = "<answer_text>"
    return render_prompt(scrubbed)


def sha256_text(text: object) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def manifest_json(
    config: LLMScoreConfig, *, rows_requested: int, rows_scored: int, rows_successful: int = 0
) -> str:
    payload = {
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "prompt_sha256": sha256_text(prompt_without_answer(build_payload(""))),
        "rows_requested": int(rows_requested),
        "rows_scored": int(rows_scored),
        "rows_successful": int(rows_successful),
        "created_at_utc": utc_now(),
        "payload_policy": "answer_text + text_type + prompt_version only",
        "invalid_output_policy": "retry at most once for format only",
        "api_key_saved": False,
        "full_payload_saved": False,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
