"""OpenAI provider implementation.

Uses httpx directly instead of openai SDK for:
1. Consistent async patterns across all providers
2. Full control over request/response handling
3. Unified error handling
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
    AuthenticationError,
    ContentFilterError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
)


class OpenAIProvider:
    """OpenAI API provider.

    Translates unified requests to OpenAI format and back.
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
            timeout: Request timeout in seconds.
            base_url: Override base URL (useful for proxies/testing).
        """
        self._api_key = api_key
        self._base_url = base_url or self.BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        return "openai"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send completion request to OpenAI."""
        payload = self._build_request(request)

        try:
            response = await self._client.post("/chat/completions", json=payload)
            self._check_response(response)
            return self._parse_response(response.json())

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}", provider=self.name)
        except httpx.RequestError as e:
            raise ServiceUnavailableError(f"Request failed: {e}", provider=self.name)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream completion from OpenAI.

        Uses Server-Sent Events (SSE). Each line is:
        data: {"choices":[{"delta":{"content":"Hi"}}]}
        ...
        data: [DONE]
        """
        payload = self._build_request(request)
        payload["stream"] = True

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                # Check for errors before streaming
                if response.status_code != 200:
                    await response.aread()
                    self._check_response(response)

                async for line in response.aiter_lines():
                    chunk = self._parse_sse_line(line)
                    if chunk is not None:
                        yield chunk

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Stream timed out: {e}", provider=self.name)
        except httpx.RequestError as e:
            raise ServiceUnavailableError(f"Stream failed: {e}", provider=self.name)

    def _parse_sse_line(self, line: str) -> Optional[StreamChunk]:
        """Parse a single SSE line into a StreamChunk."""
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(":"):
            return None

        # Remove "data: " prefix
        if not line.startswith("data: "):
            return None

        data = line[6:]  # Remove "data: "

        # End of stream
        if data == "[DONE]":
            return None

        try:
            parsed = json.loads(data)
            choice = parsed["choices"][0]
            delta = choice.get("delta", {})

            return StreamChunk(
                content=delta.get("content", ""),
                finish_reason=self._map_finish_reason(choice.get("finish_reason")),
                usage=None,  # OpenAI doesn't include usage in stream chunks
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    async def health_check(self) -> bool:
        """Check OpenAI API health via models endpoint."""
        try:
            response = await self._client.get("/models", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _build_request(self, request: CompletionRequest) -> dict:
        """Translate unified request to OpenAI format."""
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        return payload

    def _parse_response(self, data: dict) -> CompletionResponse:
        """Translate OpenAI response to unified format."""
        choice = data["choices"][0]
        usage = data["usage"]

        return CompletionResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=TokenUsage(
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
            ),
            finish_reason=self._map_finish_reason(choice.get("finish_reason")),
        )

    def _map_finish_reason(self, reason: Optional[str]) -> Optional[str]:
        """Map OpenAI finish reasons to unified format."""
        mapping = {
            "stop": "stop",
            "length": "length",
            "content_filter": "content_filter",
        }
        return mapping.get(reason)

    def _check_response(self, response: httpx.Response) -> None:
        """Check response for errors and raise appropriate exceptions."""
        if response.status_code == 200:
            return

        try:
            error_data = response.json().get("error", {})
            message = error_data.get("message", response.text)
            error_type = error_data.get("type", "")
        except Exception:
            message = response.text
            error_type = ""

        if response.status_code == 401:
            raise AuthenticationError(message, provider=self.name)

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                message,
                provider=self.name,
                retry_after=float(retry_after) if retry_after else None,
            )

        if response.status_code == 404:
            raise ModelNotFoundError(
                message, provider=self.name, model="unknown"
            )

        if response.status_code == 400 and "content_filter" in error_type.lower():
            raise ContentFilterError(message, provider=self.name)

        if response.status_code >= 500:
            raise ServiceUnavailableError(message, provider=self.name)

        # Generic error for other cases
        raise ProviderError(
            f"OpenAI API error ({response.status_code}): {message}",
            provider=self.name,
            retryable=response.status_code >= 500,
        )
