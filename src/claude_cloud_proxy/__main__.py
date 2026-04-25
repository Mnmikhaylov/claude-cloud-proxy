from __future__ import annotations

import uvicorn

from claude_cloud_proxy.app import create_app
from claude_cloud_proxy.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
