from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    cloud_ru_base_url: str = "https://foundation-models.api.cloud.ru/v1"
    cloud_ru_api_key: str | None = None
    proxy_api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 2222
    max_request_bytes: int = 33_554_432
    timeout_seconds: float = 120.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            cloud_ru_base_url=os.getenv(
                "CLOUD_RU_BASE_URL",
                defaults.cloud_ru_base_url,
            ),
            cloud_ru_api_key=os.getenv("CLOUD_RU_API_KEY") or None,
            proxy_api_key=os.getenv("PROXY_API_KEY") or None,
            host=os.getenv("PROXY_HOST", defaults.host),
            port=int(os.getenv("PROXY_PORT", str(defaults.port))),
            max_request_bytes=int(
                os.getenv("PROXY_MAX_REQUEST_BYTES", str(defaults.max_request_bytes))
            ),
            timeout_seconds=float(
                os.getenv("PROXY_TIMEOUT_SECONDS", str(defaults.timeout_seconds))
            ),
            log_level=os.getenv("PROXY_LOG_LEVEL", defaults.log_level),
        )
