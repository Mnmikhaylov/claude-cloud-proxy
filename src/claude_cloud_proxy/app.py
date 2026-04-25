from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from claude_cloud_proxy.config import Settings
from claude_cloud_proxy.errors import ProxyError
from claude_cloud_proxy.models import AnthropicCountTokensRequest, AnthropicMessageRequest
from claude_cloud_proxy.token_counting import TokenEstimator
from claude_cloud_proxy.translator import (
    AnthropicTranslator,
    OpenAIStreamToAnthropicAdapter,
    anthropic_sse,
)
from claude_cloud_proxy.upstream import CloudRUClient

ANTHROPIC_PATH_PREFIXES = ("", "/anthropic")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
CLOUD_RU_API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9+/=_-]{20,}\.[A-Fa-f0-9]{32,}$")


@dataclass(frozen=True, slots=True)
class AuthCandidate:
    source: str
    value: str


def create_app(
    settings: Settings | None = None,
    upstream_transport: Any | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    _validate_settings(settings)
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger = logging.getLogger("claude_cloud_proxy")

    estimator = TokenEstimator()
    translator = AnthropicTranslator(logger=logger, estimator=estimator)
    upstream = CloudRUClient(settings=settings, transport=upstream_transport)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        try:
            yield
        finally:
            await upstream.close()

    app = FastAPI(title="Claude Code Cloud.ru Proxy", lifespan=lifespan)
    app.add_middleware(MaxRequestBodySizeMiddleware, max_bytes=settings.max_request_bytes)
    app.state.settings = settings
    app.state.logger = logger
    app.state.translator = translator
    app.state.upstream = upstream

    @app.exception_handler(ProxyError)
    async def proxy_error_handler(_: Request, exc: ProxyError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.as_payload())

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    async def count_tokens(request: Request) -> JSONResponse:
        payload = AnthropicCountTokensRequest.model_validate(await request.json())
        translated = translator.translate_count_tokens_request(payload)
        tokens = translator.estimate_request_tokens(translated)
        return JSONResponse(content={"input_tokens": tokens})

    async def messages(request: Request) -> Any:
        body = AnthropicMessageRequest.model_validate(await request.json())
        session_id = request.headers.get("X-Claude-Code-Session-Id")
        headers = _build_upstream_headers(settings, request, logger)
        translated = translator.translate_message_request(body)
        input_tokens_estimate = translator.estimate_request_tokens(translated)

        if not body.stream:
            upstream_response = await upstream.create_chat_completion(
                translated,
                headers=headers,
            )
            return JSONResponse(
                content=translator.translate_non_stream_response(
                    upstream=upstream_response,
                    fallback_model=body.model,
                    input_tokens_estimate=input_tokens_estimate,
                )
            )

        if session_id:
            logger.info("Streaming request for Claude Code session %s", session_id)

        adapter = OpenAIStreamToAnthropicAdapter(
            translator=translator,
            fallback_model=body.model,
            input_tokens_estimate=input_tokens_estimate,
        )

        async def stream() -> Any:
            try:
                async with upstream.client.stream(
                    "POST",
                    "chat/completions",
                    json=translated,
                    headers=headers,
                ) as response:
                    await upstream.raise_for_status(response)
                    async for chunk in adapter.transform(response):
                        yield chunk
            except ProxyError as exc:
                logger.info("Upstream streaming request failed: %s", exc.message)
                yield anthropic_sse("error", exc.as_payload())

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
        )

    for prefix in ANTHROPIC_PATH_PREFIXES:
        app.add_api_route(
            f"{prefix}/v1/messages/count_tokens",
            count_tokens,
            methods=["POST"],
            response_model=None,
        )
        app.add_api_route(
            f"{prefix}/v1/messages",
            messages,
            methods=["POST"],
            response_model=None,
        )

    return app


def _build_upstream_headers(
    settings: Settings,
    request: Request,
    logger: logging.Logger,
) -> dict[str, str]:
    cloud_ru_key = settings.cloud_ru_api_key
    if cloud_ru_key:
        _require_proxy_authorization(settings, request)
    if not cloud_ru_key:
        selected = _select_incoming_cloud_ru_key(request)
        if selected:
            cloud_ru_key = selected.value
            _log_auth_selection(logger, selected)

    if not cloud_ru_key:
        raise ProxyError(
            status_code=401,
            error_type="authentication_error",
            message=(
                "Missing Cloud.ru API key. Set CLOUD_RU_API_KEY or pass "
                "ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN to the proxy."
            ),
        )

    headers = {"Authorization": f"Bearer {cloud_ru_key}"}
    session_id = request.headers.get("X-Claude-Code-Session-Id")
    if session_id:
        headers["X-Claude-Code-Session-Id"] = session_id
    return headers


def _select_incoming_cloud_ru_key(request: Request) -> AuthCandidate | None:
    candidates = _incoming_auth_candidates(request)
    for candidate in candidates:
        if CLOUD_RU_API_KEY_PATTERN.fullmatch(candidate.value):
            return candidate
    return candidates[0] if candidates else None


def _incoming_auth_candidates(request: Request) -> list[AuthCandidate]:
    candidates: list[AuthCandidate] = []

    for header_name in (
        "x-api-key",
        "authorization",
        "proxy-authorization",
        "anthropic-auth-token",
    ):
        header_value = request.headers.get(header_name)
        if header_value:
            normalized = _normalize_auth_value(header_value)
            if normalized:
                candidates.append(AuthCandidate(source=header_name, value=normalized))

    return candidates


def _log_auth_selection(logger: logging.Logger, selected: AuthCandidate) -> None:
    cloud_like = bool(CLOUD_RU_API_KEY_PATTERN.fullmatch(selected.value))
    fingerprint = sha256(selected.value.encode()).hexdigest()[:12]
    logger.info(
        "Using incoming Cloud.ru API key source=%s len=%d sha256=%s cloud_like=%s",
        selected.source,
        len(selected.value),
        fingerprint,
        str(cloud_like).lower(),
    )


def _normalize_auth_value(value: str) -> str:
    normalized = value.strip().strip("'\"")
    if normalized.lower().startswith("bearer "):
        normalized = normalized.split(" ", 1)[1].strip()
    return normalized.strip().strip("'\"")


def _validate_settings(settings: Settings) -> None:
    cloud_ru_base_url = settings.cloud_ru_base_url.strip()
    parsed_base_url = urlparse(cloud_ru_base_url)
    if parsed_base_url.scheme != "https" or not parsed_base_url.netloc:
        raise ValueError("CLOUD_RU_BASE_URL must be an absolute https:// URL.")
    settings.cloud_ru_base_url = cloud_ru_base_url

    if settings.max_request_bytes < 1:
        raise ValueError("PROXY_MAX_REQUEST_BYTES must be greater than zero.")
    if settings.cloud_ru_api_key and settings.host not in LOOPBACK_HOSTS and not settings.proxy_api_key:
        raise ValueError(
            "PROXY_API_KEY is required when CLOUD_RU_API_KEY is configured "
            "and PROXY_HOST is not loopback."
        )


def _require_proxy_authorization(settings: Settings, request: Request) -> None:
    if not settings.proxy_api_key:
        return

    provided_key = request.headers.get("X-Proxy-Api-Key", "")
    if not compare_digest(provided_key, settings.proxy_api_key):
        raise ProxyError(
            status_code=401,
            error_type="authentication_error",
            message="Invalid or missing proxy API key.",
        )


class MaxRequestBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            name.decode("latin1").lower(): value.decode("latin1")
            for name, value in scope.get("headers", [])
        }
        content_length = headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > self.max_bytes:
                await _request_too_large_response(self.max_bytes)(scope, receive, send)
                return

        received = 0
        buffered: list[Message] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    await _request_too_large_response(self.max_bytes)(scope, receive, send)
                    return
            buffered.append(message)
            if message["type"] != "http.request" or not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)


def _request_too_large_response(max_bytes: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "type": "error",
            "error": {
                "type": "request_too_large",
                "message": f"Request body exceeds {max_bytes} bytes.",
            },
        },
    )
