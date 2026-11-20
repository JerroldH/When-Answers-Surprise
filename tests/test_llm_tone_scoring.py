import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import when_answers_surprise.llm.tone_scoring as llm_tone_scoring  # noqa: E402
from when_answers_surprise.llm.tone_scoring import (  # noqa: E402
    PROMPT_VERSION,
    LLMScoreConfig,
    build_payload,
    call_provider,
    chat_completion_body,
    parse_integer_score,
    provider_config,
    response_content,
    safe_text,
    score_answer_text,
    validate_payload,
)


def test_parse_integer_score_accepts_only_allowed_bare_integer() -> None:
    assert parse_integer_score("2") == 2
    assert parse_integer_score(" -1 ") == -1

    for value in ["score: 1", "1 because tone is positive", "3", "-4", ""]:
        with pytest.raises(ValueError):
            parse_integer_score(value)


def test_payload_allows_only_answer_text_type_and_prompt_version() -> None:
    payload = build_payload("We delivered strong growth.", prompt_version=PROMPT_VERSION)
    assert sorted(payload) == ["answer_text", "prompt_version", "text_type"]

    with pytest.raises(ValueError):
        validate_payload({**payload, "ticker": "AAPL"})

    with pytest.raises(ValueError):
        validate_payload({**payload, "return": "CAR_0_1"})


def test_fake_provider_scores_without_api_key_and_records_audit_fields() -> None:
    result = score_answer_text(
        "We delivered strong growth and improved margins.",
        provider_config("fake"),
    )

    assert result["status"] == "ok"
    assert result["score"] in {-2, -1, 0, 1, 2}
    assert result["provider"] == "fake"
    assert result["prompt_sha256"]
    assert result["text_sha256"]
    assert result["retry_count"] == 0


def test_gemini_adc_requires_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(llm_tone_scoring, "gcloud_project", lambda: "")

    with pytest.raises(RuntimeError):
        provider_config("gemini_adc")


def test_deepseek_v4_flash_disables_thinking_mode() -> None:
    config = provider_config("deepseek")
    body = chat_completion_body("prompt", config)

    assert config.model == "deepseek-v4-flash"
    assert config.base_url == "https://api.deepseek.com"
    assert config.max_output_tokens == 8
    assert body["max_tokens"] == 8
    assert body["thinking"] == {"type": "disabled"}


def test_deepseek_key_file_round_robins_without_logging_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "deepseek_keys.txt"
    key_file.write_text("key-a\nkey-b\nkey-a\n", encoding="utf-8")
    config = provider_config("deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEYS", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY_FILE", str(key_file))
    llm_tone_scoring._API_KEY_POOL_CACHE.clear()
    llm_tone_scoring._API_KEY_POOL_INDEX.clear()

    tokens = [llm_tone_scoring.auth_token(config) for _ in range(4)]

    assert tokens == ["key-a", "key-b", "key-a", "key-b"]
    assert llm_tone_scoring.static_auth_keys(config) == ("key-a", "key-b")


def test_gemini_body_does_not_include_deepseek_thinking_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "dummy-project")
    config = provider_config("gemini_adc", model="google/gemini-3.5-flash")
    body = chat_completion_body("prompt", config)

    assert "thinking" not in body
    assert body["model"] == "google/gemini-3.5-flash"


def test_gemini_api_uses_native_endpoint() -> None:
    config = provider_config("gemini")

    assert config.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert config.max_output_tokens == 256


def test_response_content_reports_missing_message_shape() -> None:
    response = {"choices": [{"finish_reason": "length", "index": 0}]}

    with pytest.raises(RuntimeError, match="Missing chat completion message content"):
        response_content(response)


def test_safe_text_redacts_google_access_tokens() -> None:
    text = "Bearer ya29.a0AT3oNZ_secret\nmore"

    redacted = safe_text(text)

    assert "ya29." not in redacted
    assert "\n" not in redacted
    assert "[REDACTED" in redacted


def test_gemini_adc_refreshes_token_after_401(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLMScoreConfig(
        provider="gemini_adc",
        model="google/gemini-test",
        api_key_env="GOOGLE_OAUTH_ACCESS_TOKEN",
        base_url="https://example.invalid/v1",
    )
    tokens = iter(["expired-token", "fresh-token"])
    calls = []

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.headers = {}
            self.reason = "OK"
            self.url = "https://example.invalid/v1/chat/completions"
            self.text = ""

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("request failed")

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "1"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
            }

    def post(*args: object, **kwargs: object) -> Response:
        calls.append(kwargs["headers"]["Authorization"])
        return Response(401 if len(calls) == 1 else 200)

    monkeypatch.delenv("GOOGLE_OAUTH_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(llm_tone_scoring, "adc_auth_token", lambda: next(tokens))
    monkeypatch.setattr(llm_tone_scoring.requests, "post", post)
    monkeypatch.setattr(llm_tone_scoring.time, "sleep", lambda _: None)
    llm_tone_scoring._AUTH_TOKEN_CACHE.clear()

    content, usage = call_provider("prompt", config)

    assert content == "1"
    assert usage["total_tokens"] == 8
    assert calls == ["Bearer expired-token", "Bearer fresh-token"]


def test_call_provider_retries_429(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLMScoreConfig(
        provider="openai",
        model="test-model",
        api_key_env="OPENAI_API_KEY",
        base_url="https://example.invalid/v1",
    )
    calls = []

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.headers = {}
            self.reason = "Too Many Requests" if status_code == 429 else "OK"
            self.url = "https://example.invalid/v1/chat/completions"
            self.text = "slow down" if status_code == 429 else ""

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "-1"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }

    def post(*args: object, **kwargs: object) -> Response:
        calls.append(kwargs["json"]["model"])
        return Response(429 if len(calls) == 1 else 200)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_tone_scoring.requests, "post", post)
    monkeypatch.setattr(llm_tone_scoring.time, "sleep", lambda _: None)

    content, usage = call_provider("prompt", config)

    assert content == "-1"
    assert usage["total_tokens"] == 4
    assert calls == ["test-model", "test-model"]
