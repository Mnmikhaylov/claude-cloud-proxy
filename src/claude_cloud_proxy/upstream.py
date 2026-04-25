from __future__ import annotations

import json
from typing import Any

import httpx

from claude_cloud_proxy.config import Settings
from claude_cloud_proxy.errors import ProxyError


def map_status_to_error_type(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code == 529:
        return "overloaded_error"
    return "api_error"


class CloudRUClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.cloud_ru_base_url.rstrip("/") + "/",
            timeout=settings.timeout_seconds,
            transport=transport,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def close(self) -> None:
        await self._client.aclose()

    async def create_chat_completion(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        response = await self._client.post(
            "chat/completions",
            json=payload,
            headers=headers,
        )
        await self.raise_for_status(response)
        return response.json()

    async def raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        await response.aread()
        message = response.text
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
            else:
                message = str(data.get("message") or message)

        raise ProxyError(
            status_code=response.status_code,
            error_type=map_status_to_error_type(response.status_code),
            message=message or "Upstream request failed.",
        )
