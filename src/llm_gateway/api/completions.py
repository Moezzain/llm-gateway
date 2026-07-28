"""Completions API endpoint."""

import json
import logging
from typing import AsyncIterator, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from llm_gateway.core.auth import TeamContext, require_team
from llm_gateway.core.budget import budget_tracker
from llm_gateway.core.circuit_breaker import circuit_breaker
from llm_gateway.core.compression import prompt_compressor
from llm_gateway.core.guardrails import input_guardrail
from llm_gateway.core.health import health_checker
from llm_gateway.core.logging import RequestLogger
from llm_gateway.core.metrics import MetricsRecorder
from llm_gateway.core.rate_limit import rate_limiter
from llm_gateway.core.router import RouteResult, model_router
from llm_gateway.models.llm import CompletionRequest, CompletionResponse
from llm_gateway.providers.base import LLMProvider
from llm_gateway.providers.exceptions import (
    AuthenticationError,
    ContentFilterError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["completions"])


@router.post("/completions", response_model=None)
async def create_completion(
    request: CompletionRequest,
    team: TeamContext = Depends(require_team),
) -> Union[CompletionResponse, StreamingResponse]:
    """Create a completion.

    Requires authentication via X-API-Key or Authorization header.
    Routes to appropriate provider based on model name.
    """
    # Initialize request logger and metrics
    req_log = RequestLogger(
        team_id=team.team_id,
        model=request.model,
        is_streaming=request.stream,
    )
    metrics = MetricsRecorder(team_id=team.team_id, model=request.model)

    try:
        # Check model access
        _check_model_access(team, request.model)

        # Input guardrails — fail fast before spending route/budget/provider work
        _check_input_guardrails(request, req_log, metrics)

        # Route to provider
        route = _route_model(request.model)
        req_log.set_provider(route.provider.name, is_fallback=route.is_fallback)
        metrics.set_provider(route.provider.name)

        # Check budget
        await _check_budget(team, req_log, metrics)

        # Check rate limit (requests per minute)
        await _check_rate_limit(team, req_log, metrics)

        # Compress the prompt (no-op if disabled). Runs after the gate checks
        # so we don't spend work compressing a request we'd reject anyway.
        request = _compress_prompt(request, metrics)

        # Create provider-specific request with actual model ID
        provider_request = request.model_copy(update={"model": route.model_id})

        if request.stream:
            return StreamingResponse(
                stream_completion(route.provider, provider_request, team, request.model, req_log, metrics),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming: try primary, fallback on retryable error
        metrics.start_request()
        response = await _complete_with_fallback(
            route, provider_request, request.model, req_log, metrics
        )

        # Track spending after successful response
        cost = await _track_spending(team, request.model, response)

        # Log and record metrics
        req_log.set_tokens(response.usage.input_tokens, response.usage.output_tokens)
        if cost:
            req_log.set_cost(cost)
        req_log.success()

        metrics.record_success(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost or 0,
        )

        return response

    except HTTPException:
        raise  # Already logged by helper functions
    except Exception as e:
        req_log.error(e)
        raise


def _route_model(model: str) -> RouteResult:
    """Route model to provider. Raises 404 if not found."""
    route = model_router.route(model)

    if route is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Check config/gateway.yaml for available models.",
        )

    if route.is_fallback:
        logger.info(
            f"Using fallback provider '{route.provider.name}' for model '{model}' "
            f"(primary provider unhealthy)"
        )

    return route


def _check_model_access(team: TeamContext, model: str) -> None:
    """Check if team is allowed to use the requested model."""
    allowed = team.config.allowed_models

    # Empty list = all models allowed
    if not allowed:
        return

    if model not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{model}' not allowed for team '{team.team_id}'. "
            f"Allowed models: {', '.join(allowed)}",
        )


def _check_input_guardrails(
    request: CompletionRequest,
    req_log: RequestLogger,
    metrics: MetricsRecorder,
) -> None:
    """Scan input for prompt injection. Raises 400 in block mode.

    In flag mode, a match is logged and metered but the request proceeds —
    shadow mode for tuning rules before enforcing.
    """
    result = input_guardrail.check(request.messages)
    if not result.triggered:
        return

    blocked = input_guardrail.should_block(result)
    metrics.record_guardrail(result.rule_name or "unknown", blocked=blocked)

    if blocked:
        req_log.error(
            Exception(f"Blocked by guardrail '{result.rule_name}': {result.reason}"),
            error_type="GuardrailBlocked",
        )
        raise HTTPException(
            status_code=400,
            detail=f"Request blocked by input guardrail ({result.reason}).",
        )

    # Flag mode: allow but record.
    logger.warning(
        f"Guardrail '{result.rule_name}' matched (flag mode, allowed): "
        f"{result.reason!r} — matched: {result.matched_text!r}"
    )


async def _check_budget(
    team: TeamContext, req_log: RequestLogger, metrics: MetricsRecorder
) -> None:
    """Check if team has budget remaining. Raises 402 if exceeded."""
    status = await budget_tracker.get_status(team.team_id, team.config.budget)

    if status.is_exceeded:
        req_log.budget_exceeded()
        metrics.record_budget_exceeded()
        raise HTTPException(
            status_code=402,  # Payment Required
            detail=f"Budget exceeded for team '{team.team_id}'. "
            f"Spent: ${status.spent_usd:.2f}, Limit: ${status.limit_usd:.2f}",
        )

    if status.is_warning:
        logger.warning(
            f"Team '{team.team_id}' approaching budget limit: "
            f"${status.spent_usd:.2f} / ${status.limit_usd:.2f} "
            f"({status.percentage_used:.0%})"
        )


async def _check_rate_limit(
    team: TeamContext, req_log: RequestLogger, metrics: MetricsRecorder
) -> None:
    """Check rate limit for team. Raises 429 if exceeded."""
    rpm = team.config.rate_limit.requests_per_minute
    result = await rate_limiter.check_requests_per_minute(team.team_id, rpm)

    if not result.allowed:
        req_log.rate_limited()
        metrics.record_rate_limited()
        headers = {
            "X-RateLimit-Limit": str(rpm),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_at)),
        }
        if result.retry_after:
            headers["Retry-After"] = str(int(result.retry_after) + 1)

        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for team '{team.team_id}'. "
            f"Limit: {rpm} requests/minute.",
            headers=headers,
        )


def _compress_prompt(
    request: CompletionRequest, metrics: MetricsRecorder
) -> CompletionRequest:
    """Apply prompt compression. Returns a new request (or the same if no-op)."""
    result = prompt_compressor.compress(request.messages)

    if not result.was_compressed:
        return request

    metrics.record_compression(result.est_tokens_saved)
    logger.debug(
        f"Compressed prompt: {result.original_chars}→{result.compressed_chars} chars "
        f"(~{result.est_tokens_saved} tokens, {result.ratio:.0%} reduction)"
    )
    return request.model_copy(update={"messages": result.messages})


async def _track_spending(
    team: TeamContext, model: str, response: CompletionResponse
) -> Optional[float]:
    """Track spending after a successful request. Returns cost in USD."""
    cost = budget_tracker.calculate_cost(
        model,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    if cost and cost.total_cost_usd > 0:
        new_total = await budget_tracker.record_spending(
            team.team_id, cost.total_cost_usd
        )
        logger.debug(
            f"Team '{team.team_id}' spent ${cost.total_cost_usd:.4f} "
            f"(total: ${new_total:.2f})"
        )
        return cost.total_cost_usd

    return None


async def _complete_with_fallback(
    route: RouteResult,
    request: CompletionRequest,
    original_model: str,
    req_log: RequestLogger,
    metrics: MetricsRecorder,
) -> CompletionResponse:
    """Try completion with fallback on retryable errors.

    Flow:
    1. Check circuit breaker — skip if open
    2. Try primary provider
    3. On success → record success, return
    4. On retryable error → record failure, try fallback
    5. On non-retryable error → raise immediately
    """
    provider_name = route.provider.name
    req_log.set_circuit_state(circuit_breaker.get_state(provider_name).value)

    # Check circuit breaker first
    if not circuit_breaker.allow_request(provider_name):
        logger.info(f"Circuit open for '{provider_name}', skipping to fallback")
        metrics.record_fallback("circuit_open")
        return await _try_fallback(original_model, request, req_log, metrics, skip_primary=True)

    try:
        response = await route.provider.complete(request)
        circuit_breaker.record_success(provider_name)
        return response

    except ProviderError as e:
        # Non-retryable errors: don't try fallback, don't record as circuit failure
        if not e.retryable:
            req_log.error(e, error_type=type(e).__name__)
            metrics.record_error(type(e).__name__)
            _raise_provider_error(e)

        # Record failure for circuit breaker and health tracking
        circuit_breaker.record_failure(provider_name)
        await health_checker.check_provider(provider_name)
        metrics.record_fallback("request_failed")

        return await _try_fallback(original_model, request, req_log, metrics, primary_error=e)


async def _try_fallback(
    original_model: str,
    request: CompletionRequest,
    req_log: RequestLogger,
    metrics: MetricsRecorder,
    primary_error: Optional[ProviderError] = None,
    skip_primary: bool = False,
) -> CompletionResponse:
    """Try fallback provider.

    Args:
        original_model: Original requested model name
        request: The completion request (with primary model ID)
        req_log: Request logger
        metrics: Metrics recorder
        primary_error: Error from primary provider (if any)
        skip_primary: True if we skipped primary due to circuit breaker
    """
    fallback_route = model_router.get_fallback(original_model)

    if not fallback_route:
        if skip_primary:
            req_log.error(Exception("Circuit open, no fallback"), error_type="CircuitOpen")
            metrics.record_error("CircuitOpen")
            raise HTTPException(
                status_code=503,
                detail=f"Primary provider circuit open, no fallback configured for '{original_model}'"
            )
        logger.warning(
            f"No fallback configured for '{original_model}'"
        )
        req_log.error(primary_error, error_type=type(primary_error).__name__)  # type: ignore
        metrics.record_error(type(primary_error).__name__)  # type: ignore
        _raise_provider_error(primary_error)  # type: ignore

    fallback_name = fallback_route.provider.name
    logger.info(f"Falling back to '{fallback_name}' for '{original_model}'")
    req_log.set_provider(fallback_name, is_fallback=True)
    metrics.set_provider(fallback_name)

    # Check fallback circuit too
    if not circuit_breaker.allow_request(fallback_name):
        req_log.error(Exception("Both circuits open"), error_type="CircuitOpen")
        metrics.record_error("CircuitOpen")
        raise HTTPException(
            status_code=503,
            detail=f"Both primary and fallback circuits open for '{original_model}'"
        )

    # Build fallback request with correct model ID
    fallback_request = request.model_copy(
        update={"model": fallback_route.model_id}
    )

    try:
        response = await fallback_route.provider.complete(fallback_request)
        circuit_breaker.record_success(fallback_name)
        return response
    except ProviderError as fallback_error:
        if fallback_error.retryable:
            circuit_breaker.record_failure(fallback_name)
        logger.error(f"Fallback provider '{fallback_name}' also failed")
        req_log.error(fallback_error, error_type=type(fallback_error).__name__)
        metrics.record_error(type(fallback_error).__name__)
        _raise_provider_error(fallback_error)


def _raise_provider_error(e: ProviderError) -> None:
    """Convert ProviderError to appropriate HTTPException."""
    if isinstance(e, AuthenticationError):
        raise HTTPException(status_code=401, detail=str(e))

    if isinstance(e, RateLimitError):
        headers = {}
        if e.retry_after:
            headers["Retry-After"] = str(int(e.retry_after))
        raise HTTPException(status_code=429, detail=str(e), headers=headers)

    if isinstance(e, ModelNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))

    if isinstance(e, ContentFilterError):
        raise HTTPException(status_code=400, detail=str(e))

    status = 502 if e.retryable else 500
    raise HTTPException(status_code=status, detail=str(e))


async def stream_completion(
    provider: LLMProvider,
    request: CompletionRequest,
    _team: TeamContext,  # Reserved for streaming spending tracking
    original_model: str,
    req_log: RequestLogger,
    metrics: MetricsRecorder,
) -> AsyncIterator[str]:
    """Generate SSE events from provider stream.

    Supports fallback: if primary fails before first chunk,
    transparently retry with fallback provider.
    Also respects circuit breaker state.
    """
    provider_name = provider.name
    started_streaming = False
    metrics.start_request()

    # Check circuit breaker — if open, go straight to fallback
    if not circuit_breaker.allow_request(provider_name):
        logger.info(f"Circuit open for '{provider_name}', streaming via fallback")
        metrics.record_fallback("circuit_open")
        async for event in _stream_fallback(original_model, request, req_log, metrics):
            yield event
        return

    try:
        async for chunk in provider.stream(request):
            started_streaming = True

            if not chunk.content and chunk.finish_reason is None:
                continue

            data = {
                "content": chunk.content,
                "finish_reason": chunk.finish_reason,
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"
        circuit_breaker.record_success(provider_name)
        req_log.success()
        metrics.record_success()  # No token count for streaming

        # TODO: Track streaming spending

    except ProviderError as e:
        # Record failure if retryable
        if e.retryable:
            circuit_breaker.record_failure(provider_name)

        # If we already started streaming, can't fallback cleanly
        if started_streaming or not e.retryable:
            req_log.error(e, error_type=type(e).__name__)
            metrics.record_error(type(e).__name__)
            error_data = {"error": str(e), "retryable": e.retryable}
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        # Try fallback before first chunk was sent
        logger.info(f"Stream failed on '{provider_name}', trying fallback")
        metrics.record_fallback("request_failed")
        async for event in _stream_fallback(original_model, request, req_log, metrics):
            yield event


async def _stream_fallback(
    original_model: str,
    request: CompletionRequest,
    req_log: RequestLogger,
    metrics: MetricsRecorder,
) -> AsyncIterator[str]:
    """Stream from fallback provider."""
    fallback_route = model_router.get_fallback(original_model)

    if not fallback_route:
        req_log.error(Exception("No fallback configured"), error_type="NoFallback")
        metrics.record_error("NoFallback")
        error_data = {"error": "No fallback configured", "retryable": False}
        yield f"data: {json.dumps(error_data)}\n\n"
        return

    fallback_name = fallback_route.provider.name
    req_log.set_provider(fallback_name, is_fallback=True)
    metrics.set_provider(fallback_name)

    # Check fallback circuit
    if not circuit_breaker.allow_request(fallback_name):
        req_log.error(Exception("Fallback circuit open"), error_type="CircuitOpen")
        metrics.record_error("CircuitOpen")
        error_data = {"error": "Fallback circuit also open", "retryable": False}
        yield f"data: {json.dumps(error_data)}\n\n"
        return

    logger.info(f"Streaming via fallback '{fallback_name}'")

    fallback_request = request.model_copy(
        update={"model": fallback_route.model_id}
    )

    try:
        async for chunk in fallback_route.provider.stream(fallback_request):
            if not chunk.content and chunk.finish_reason is None:
                continue

            data = {
                "content": chunk.content,
                "finish_reason": chunk.finish_reason,
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"
        circuit_breaker.record_success(fallback_name)
        req_log.success()
        metrics.record_success()

    except ProviderError as fallback_error:
        if fallback_error.retryable:
            circuit_breaker.record_failure(fallback_name)
        req_log.error(fallback_error, error_type=type(fallback_error).__name__)
        metrics.record_error(type(fallback_error).__name__)
        error_data = {"error": str(fallback_error), "retryable": fallback_error.retryable}
        yield f"data: {json.dumps(error_data)}\n\n"
