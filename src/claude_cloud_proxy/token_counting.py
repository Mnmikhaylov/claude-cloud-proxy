from __future__ import annotations

import json
import math
from typing import Any

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class TokenEstimator:
    def __init__(self) -> None:
        self._encoding = None
        if tiktoken is not None:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, math.ceil(len(text) / 4))

    def estimate_openai_payload_tokens(self, payload: dict[str, Any]) -> int:
        total = 0
        total += self.estimate_text_tokens(payload.get("model", ""))
        total += self.estimate_text_tokens(compact_json(payload.get("tools", [])))
        total += self.estimate_text_tokens(compact_json(payload.get("tool_choice", {})))
        total += 8

        for message in payload.get("messages", []):
            total += 4
            total += self.estimate_text_tokens(message.get("role", ""))
            total += self.estimate_text_tokens(compact_json(message.get("content", "")))
            total += self.estimate_text_tokens(compact_json(message.get("tool_calls", [])))
            total += self.estimate_text_tokens(message.get("tool_call_id", ""))

        return max(1, total)

    def estimate_anthropic_content_tokens(self, content: list[dict[str, Any]]) -> int:
        total = 0
        for block in content:
            total += self.estimate_text_tokens(compact_json(block))
        return max(1, total)
