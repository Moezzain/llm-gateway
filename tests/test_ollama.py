"""Tests for the Ollama provider (translation + error mapping, no live server)."""

import httpx
import pytest

from llm_gateway.models.llm import CompletionRequest, Message, Role
from llm_gateway.providers.exceptions import (
    ModelNotFoundError,
    ProviderError,
    ServiceUnavailableError,
)
from llm_gateway.providers.ollama import OllamaProvider


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(base_url="http://localhost:11434")


def _request(**kwargs) -> CompletionRequest:
    return CompletionRequest(
        model="llama3",
        messages=[Message(role=Role.USER, content="Hello")],
        **kwargs,
    )


@pytest.mark.unit
class TestBuildRequest:
    def test_maps_messages_and_temperature(self, provider: OllamaProvider) -> None:
        payload = provider._build_request(_request(temperature=0.5), stream=False)
        assert payload["model"] == "llama3"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["options"]["temperature"] == 0.5
        assert payload["stream"] is False

    def test_max_tokens_becomes_num_predict(self, provider: OllamaProvider) -> None:
        payload = provider._build_request(_request(max_tokens=128), stream=True)
        assert payload["options"]["num_predict"] == 128
        assert payload["stream"] is True

    def test_no_num_predict_when_max_tokens_unset(self, provider: OllamaProvider) -> None:
        payload = provider._build_request(_request(), stream=False)
        assert "num_predict" not in payload["options"]


@pytest.mark.unit
class TestParseResponse:
    def test_extracts_content_and_usage(self, provider: OllamaProvider) -> None:
        data = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "Hi there"},
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 7,
        }
        result = provider._parse_response(data)
        assert result.content == "Hi there"
        assert result.model == "llama3"
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 7
        assert result.finish_reason == "stop"

    def test_missing_counts_default_to_zero(self, provider: OllamaProvider) -> None:
        data = {"message": {"content": "x"}}
        result = provider._parse_response(data)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


@pytest.mark.unit
class TestParseNdjsonLine:
    def test_intermediate_chunk_has_content_no_usage(self, provider: OllamaProvider) -> None:
        chunk = provider._parse_ndjson_line('{"message":{"content":"Hel"},"done":false}')
        assert chunk is not None
        assert chunk.content == "Hel"
        assert chunk.usage is None
        assert chunk.finish_reason is None

    def test_final_chunk_carries_usage_and_finish(self, provider: OllamaProvider) -> None:
        line = (
            '{"message":{"content":""},"done":true,"done_reason":"stop",'
            '"prompt_eval_count":5,"eval_count":9}'
        )
        chunk = provider._parse_ndjson_line(line)
        assert chunk is not None
        assert chunk.finish_reason == "stop"
        assert chunk.usage.input_tokens == 5
        assert chunk.usage.output_tokens == 9

    def test_empty_line_returns_none(self, provider: OllamaProvider) -> None:
        assert provider._parse_ndjson_line("   ") is None

    def test_malformed_json_returns_none(self, provider: OllamaProvider) -> None:
        assert provider._parse_ndjson_line("{not json") is None

    def test_empty_non_final_chunk_skipped(self, provider: OllamaProvider) -> None:
        assert provider._parse_ndjson_line('{"message":{"content":""},"done":false}') is None


@pytest.mark.unit
class TestCheckResponse:
    def _resp(self, status: int, body: dict) -> httpx.Response:
        return httpx.Response(status_code=status, json=body)

    def test_200_does_not_raise(self, provider: OllamaProvider) -> None:
        provider._check_response(self._resp(200, {"message": {"content": "ok"}}))

    def test_404_is_model_not_found(self, provider: OllamaProvider) -> None:
        with pytest.raises(ModelNotFoundError):
            provider._check_response(self._resp(404, {"error": "model not found"}))

    def test_500_is_service_unavailable(self, provider: OllamaProvider) -> None:
        with pytest.raises(ServiceUnavailableError):
            provider._check_response(self._resp(500, {"error": "boom"}))

    def test_other_4xx_is_generic_provider_error(self, provider: OllamaProvider) -> None:
        with pytest.raises(ProviderError):
            provider._check_response(self._resp(400, {"error": "bad request"}))


@pytest.mark.unit
def test_name(provider: OllamaProvider) -> None:
    assert provider.name == "ollama"
