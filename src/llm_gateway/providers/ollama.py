"""Ollama provider implementation (local models).

Ollama runs open-weight models (Llama, Mistral, Qwen, etc.) on local hardware.
It speaks its own HTTP API on localhost:11434 — NOT OpenAI-compatible on the
native endpoints — so we translate the same way we do for OpenAI/Anthropic.

Why support it: cost ($0 per token, runs on your own box), privacy (data never
leaves the network — pairs naturally with the PII concerns), and as a fallback
target when paid providers are down. The tradeoff is you own the ops and the
hardware, and quality/throughput depend on the local model.

Two API shape differences worth knowing (Python/HTTP notes inline):
- Streaming is NDJSON (one JSON object per line), not SSE — no "data:" prefix,
  no [DONE] sentinel; a final object carries done=true + token counts.
- No API key. Auth is network-level (you firewall the port), so this provider
  is constructed without credentials — the registry special-cases that.
"""

import json
from typing import AsyncIterator, Optional

import httpx

from llm_gateway.models.llm import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    TokenUsage,
)
from llm_gateway.providers.exceptions import (
    ModelNotFoundError,
    ProviderError,
    ServiceUnavailableError,
    TimeoutError,
)


class OllamaProvider:
    """Ollama local-model provider.

    Translates unified requests to Ollama's /api/chat format and back.
    """

    BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        timeout: float = 60.0,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            timeout: Request timeout in seconds. Defaults higher than cloud
                providers — local inference on CPU can be slow.
            base_url: Override base URL (e.g. a remote Ollama host or docker
                service name).
        """
        self._base_url = base_url or self.BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send completion request to Ollama."""
        payload = self._build_request(request, stream=False)

        try:
            response = await self._client.post("/api/chat", json=payload)
            self._check_response(response)
            return self._parse_response(response.json())

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}", provider=self.name)
        except httpx.RequestError as e:
            raise ServiceUnavailableError(f"Request failed: {e}", provider=self.name)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream completion from Ollama.

        Ollama streams newline-delimited JSON (NDJSON). Each line is a full
        JSON object: {"message":{"content":"Hi"},"done":false}. The final
        object has done=true and the token counts.
        """
        payload = self._build_request(request, stream=True)

        try:
            async with self._client.stream(
                "POST", "/api/chat", json=payload
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._check_response(response)

                async for line in response.aiter_lines():
                    chunk = self._parse_ndjson_line(line)
                    if chunk is not None:
                        yield chunk

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Stream timed out: {e}", provider=self.name)
        except httpx.RequestError as e:
            raise ServiceUnavailableError(f"Stream failed: {e}", provider=self.name)

    def _parse_ndjson_line(self, line: str) -> Optional[StreamChunk]:
        """Parse one NDJSON line into a StreamChunk."""
        line = line.strip()
        if not line:
            return None

        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return None

        done = parsed.get("done", False)
        content = parsed.get("message", {}).get("content", "")

        # Final object carries usage + a done_reason; intermediate ones don't.
        usage = None
        finish_reason = None
        if done:
            finish_reason = self._map_finish_reason(parsed.get("done_reason"))
            usage = TokenUsage(
                input_tokens=parsed.get("prompt_eval_count", 0),
                output_tokens=parsed.get("eval_count", 0),
            )

        # Skip empty non-final chunks (nothing useful to forward).
        if not content and not done:
            return None

        return StreamChunk(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def health_check(self) -> bool:
        """Check Ollama health via the model-list endpoint."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _build_request(self, request: CompletionRequest, stream: bool) -> dict:
        """Translate unified request to Ollama format."""
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        # Ollama tuning params live under "options". num_predict is its
        # max_tokens equivalent.
        options: dict = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        return {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }

    def _parse_response(self, data: dict) -> CompletionResponse:
        """Translate Ollama response to unified format."""
        return CompletionResponse(
            content=data["message"]["content"],
            model=data.get("model", "unknown"),
            usage=TokenUsage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            ),
            finish_reason=self._map_finish_reason(data.get("done_reason")),
        )

    def _map_finish_reason(self, reason: Optional[str]) -> Optional[str]:
        """Map Ollama done reasons to unified format."""
        mapping = {
            "stop": "stop",
            "length": "length",
        }
        return mapping.get(reason)

    def _check_response(self, response: httpx.Response) -> None:
        """Check response for errors and raise appropriate exceptions."""
        if response.status_code == 200:
            return

        try:
            message = response.json().get("error", response.text)
        except Exception:
            message = response.text

        # Ollama returns 404 when the model isn't pulled locally.
        if response.status_code == 404:
            raise ModelNotFoundError(message, provider=self.name, model="unknown")

        if response.status_code >= 500:
            raise ServiceUnavailableError(message, provider=self.name)

        raise ProviderError(
            f"Ollama API error ({response.status_code}): {message}",
            provider=self.name,
            retryable=response.status_code >= 500,
        )
