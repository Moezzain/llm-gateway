"""Anthropic provider implementation.

Anthropic's API differs from OpenAI:
- System message is separate field, not in messages array
- Different response format (content is array of blocks)
- Different streaming event format
"""

import json
from typing import AsyncIterator, List, Optional

import httpx

from llm_gateway.models.llm import (
    CompletionRequest,
    CompletionResponse,
    Role,
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


class AnthropicProvider:
    """Anthropic API provider.

    Translates unified requests to Anthropic format and back.
    """

    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url or self.BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": api_key,  # Anthropic uses x-api-key, not Bearer
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send completion request to Anthropic."""
        payload = self._build_request(request)

        try:
            response = await self._client.post("/messages", json=payload)
            self._check_response(response)
            return self._parse_response(response.json())

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}", provider=self.name)
        except httpx.RequestError as e:
            raise ServiceUnavailableError(f"Request failed: {e}", provider=self.name)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream completion from Anthropic."""
        payload = self._build_request(request)
        payload["stream"] = True

        try:
            async with self._client.stream(
                "POST", "/messages", json=payload
            ) as response:
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

    async def health_check(self) -> bool:
        """Check Anthropic API health."""
        # Anthropic doesn't have a simple health endpoint
        # We'll just check if we can reach the API
        try:
            # Send minimal request that will fail auth but proves API is up
            response = await self._client.post(
                "/messages",
                json={"model": "test", "max_tokens": 1, "messages": []},
                timeout=5.0,
            )
            # Any response (even error) means API is reachable
            return response.status_code != 503
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _build_request(self, request: CompletionRequest) -> dict:
        """Translate unified request to Anthropic format."""
        # Extract system message (Anthropic wants it separate)
        system_content: Optional[str] = None
        messages: List[dict] = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,  # Anthropic requires max_tokens
        }

        if system_content:
            payload["system"] = system_content

        # Anthropic temperature range is 0-1, OpenAI is 0-2
        # Scale if needed
        if request.temperature <= 1.0:
            payload["temperature"] = request.temperature
        else:
            payload["temperature"] = 1.0  # Cap at 1

        return payload

    def _parse_response(self, data: dict) -> CompletionResponse:
        """Translate Anthropic response to unified format."""
        # Anthropic content is array of blocks
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})

        return CompletionResponse(
            content=content,
            model=data.get("model", ""),
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            finish_reason=self._map_stop_reason(data.get("stop_reason")),
        )

    def _map_stop_reason(self, reason: Optional[str]) -> Optional[str]:
        """Map Anthropic stop reasons to unified format."""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        return mapping.get(reason) if reason else None

    def _parse_sse_line(self, line: str) -> Optional[StreamChunk]:
        """Parse Anthropic SSE line into StreamChunk.

        Anthropic uses event types:
        - content_block_delta: contains text delta
        - message_stop: end of message
        - message_delta: contains stop_reason and usage
        """
        line = line.strip()

        if not line or line.startswith(":"):
            return None

        # Anthropic sends "event: type" then "data: json"
        if line.startswith("event:"):
            return None  # Skip event lines, we parse data

        if not line.startswith("data: "):
            return None

        data = line[6:]  # Remove "data: "

        try:
            parsed = json.loads(data)
            event_type = parsed.get("type", "")

            if event_type == "content_block_delta":
                delta = parsed.get("delta", {})
                if delta.get("type") == "text_delta":
                    return StreamChunk(
                        content=delta.get("text", ""),
                        finish_reason=None,
                    )

            elif event_type == "message_delta":
                # Final delta with stop_reason
                return StreamChunk(
                    content="",
                    finish_reason=self._map_stop_reason(parsed.get("delta", {}).get("stop_reason")),
                    usage=self._parse_usage(parsed.get("usage")),
                )

            elif event_type == "message_stop":
                return None  # End of stream

        except json.JSONDecodeError:
            pass

        return None

    def _parse_usage(self, usage: Optional[dict]) -> Optional[TokenUsage]:
        """Parse usage from message_delta."""
        if not usage:
            return None
        return TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

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
            raise RateLimitError(message, provider=self.name, retry_after=None)

        if response.status_code == 404:
            raise ModelNotFoundError(message, provider=self.name, model="unknown")

        if "content" in error_type.lower() and "block" in error_type.lower():
            raise ContentFilterError(message, provider=self.name)

        if response.status_code >= 500:
            raise ServiceUnavailableError(message, provider=self.name)

        raise ProviderError(
            f"Anthropic API error ({response.status_code}): {message}",
            provider=self.name,
            retryable=response.status_code >= 500,
        )
