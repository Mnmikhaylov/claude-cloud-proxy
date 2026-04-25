from __future__ import annotations


class ProxyError(Exception):
    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message

    def as_payload(self) -> dict[str, object]:
        return {
            "type": "error",
            "error": {
                "type": self.error_type,
                "message": self.message,
            },
        }
