from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from claude_cloud_proxy.errors import ProxyError
from claude_cloud_proxy.models import (
    AnthropicCountTokensRequest,
    AnthropicMessageRequest,
    AnthropicRequestBase,
    MessageParam,
    ToolParam,
)
from claude_cloud_proxy.token_counting import TokenEstimator

IGNORED_REQUEST_FIELDS = {
    "betas",
    "container",
    "context_management",
    "metadata",
    "mcp_servers",
    "output_config",
    "service_tier",
    "thinking",
    "top_k",
}


def anthropic_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def map_finish_reason(finish_reason: str | None, has_tool_calls: bool = False) -> str:
    if finish_reason == "tool_calls" or has_tool_calls:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter":
        return "refusal"
    if finish_reason == "stop" or finish_reason is None:
        return "end_turn"
    return "end_turn"


@dataclass(slots=True)
class ToolStreamState:
    openai_index: int
    anthropic_index: int
    tool_id: str
    name: str | None = None
    started: bool = False
    arguments: list[str] = field(default_factory=list)
    buffered_arguments: list[str] = field(default_factory=list)


class AnthropicTranslator:
    def __init__(
        self,
        logger: logging.Logger,
        estimator: TokenEstimator,
    ) -> None:
        self._logger = logger
        self._estimator = estimator

    def translate_message_request(
        self,
        request: AnthropicMessageRequest,
    ) -> dict[str, Any]:
        payload = self.translate_common_request(request)
        payload["max_tokens"] = request.max_tokens
        payload["stream"] = request.stream
        return payload

    def translate_count_tokens_request(
        self,
        request: AnthropicCountTokensRequest,
    ) -> dict[str, Any]:
        return self.translate_common_request(request)

    def translate_common_request(
        self,
        request: AnthropicRequestBase,
    ) -> dict[str, Any]:
        self._validate_request_extras(request.model_extra or {})

        messages: list[dict[str, Any]] = []
        system_text = self._normalize_system(request.system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

        for message in request.messages:
            messages.extend(self._translate_message(message))

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if request.tools:
            payload["tools"] = [self._translate_tool(tool) for tool in request.tools]
        if request.tool_choice:
            payload["tool_choice"] = self._translate_tool_choice(request.tool_choice)
        return payload

    def translate_non_stream_response(
        self,
        upstream: dict[str, Any],
        fallback_model: str,
        input_tokens_estimate: int,
    ) -> dict[str, Any]:
        choices = upstream.get("choices") or []
        if not choices:
            raise ProxyError(
                status_code=502,
                error_type="api_error",
                message="Upstream response did not contain choices.",
            )

        choice = choices[0]
        message = choice.get("message") or {}
        content = self._openai_message_to_anthropic_blocks(message)
        usage = upstream.get("usage") or {}

        return {
            "id": upstream.get("id") or self._message_id(),
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": upstream.get("model") or fallback_model,
            "stop_reason": map_finish_reason(
                choice.get("finish_reason"),
                has_tool_calls=bool(message.get("tool_calls")),
            ),
            "stop_sequence": None,
            "usage": {
                "input_tokens": int(usage.get("prompt_tokens") or input_tokens_estimate),
                "output_tokens": int(
                    usage.get("completion_tokens")
                    or self._estimator.estimate_anthropic_content_tokens(content)
                ),
            },
        }

    def estimate_request_tokens(self, payload: dict[str, Any]) -> int:
        return self._estimator.estimate_openai_payload_tokens(payload)

    def _validate_request_extras(self, extras: dict[str, Any]) -> None:
        for key, value in extras.items():
            if value is None:
                continue
            if key in IGNORED_REQUEST_FIELDS:
                self._logger.info("Ignoring unsupported Anthropic field '%s'.", key)
                continue
            raise ProxyError(
                status_code=400,
                error_type="invalid_request_error",
                message=f"Unsupported Anthropic request field: {key}",
            )

    def _normalize_system(self, system: str | list[dict[str, Any]] | None) -> str:
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        parts: list[str] = []
        for block in system:
            if block.get("type") != "text":
                raise ProxyError(
                    status_code=400,
                    error_type="invalid_request_error",
                    message="Only text system blocks are supported.",
                )
            parts.append(str(block.get("text", "")))
        return "".join(parts)

    def _translate_message(self, message: MessageParam) -> list[dict[str, Any]]:
        blocks = self._normalize_content_blocks(message.content)
        if message.role == "assistant":
            return [self._translate_assistant_message(blocks)]
        return self._translate_user_message(blocks)

    def _translate_assistant_message(
        self,
        blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
                continue
            if block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or self._tool_id(),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(
                                block.get("input", {}),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
                continue
            raise ProxyError(
                status_code=400,
                error_type="invalid_request_error",
                message=f"Unsupported assistant content block: {block_type}",
            )

        translated: dict[str, Any] = {"role": "assistant"}
        if text_parts or not tool_calls:
            translated["content"] = "".join(text_parts)
        if tool_calls:
            translated["tool_calls"] = tool_calls
        return translated

    def _translate_user_message(
        self,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
                continue
            if block_type == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": self._tool_result_to_text(block.get("content", "")),
                    }
                )
                continue
            raise ProxyError(
                status_code=400,
                error_type="invalid_request_error",
                message=f"Unsupported user content block: {block_type}",
            )

        if text_parts or not messages:
            messages.append({"role": "user", "content": "".join(text_parts)})

        return messages

    def _translate_tool(self, tool: ToolParam) -> dict[str, Any]:
        if tool.type and tool.type not in {"custom"}:
            raise ProxyError(
                status_code=400,
                error_type="invalid_request_error",
                message=f"Unsupported Anthropic tool type: {tool.type}",
            )
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }

    def _translate_tool_choice(self, tool_choice: dict[str, Any]) -> str | dict[str, Any]:
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "none":
            return "none"
        if choice_type == "tool":
            name = tool_choice.get("name")
            if not name:
                raise ProxyError(
                    status_code=400,
                    error_type="invalid_request_error",
                    message="tool_choice.type=tool requires a tool name.",
                )
            return {"type": "function", "function": {"name": name}}
        raise ProxyError(
            status_code=400,
            error_type="invalid_request_error",
            message=f"Unsupported tool_choice type: {choice_type}",
        )

    def _tool_result_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                else:
                    text_parts.append(json.dumps(block, ensure_ascii=False))
            return "".join(text_parts)
        return json.dumps(content, ensure_ascii=False)

    def _normalize_content_blocks(self, content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return content

    def _openai_message_to_anthropic_blocks(
        self,
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        content = message.get("content")
        text = self._openai_content_to_text(content)
        if text:
            blocks.append({"type": "text", "text": text})

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ProxyError(
                    status_code=502,
                    error_type="api_error",
                    message=f"Upstream returned invalid tool arguments: {exc}",
                ) from exc

            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or self._tool_id(),
                    "name": function.get("name") or "tool",
                    "input": parsed_arguments,
                }
            )
        return blocks

    def _openai_content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            return "".join(parts)
        return str(content)

    def _message_id(self) -> str:
        return f"msg_{uuid.uuid4().hex}"

    def _tool_id(self) -> str:
        return f"toolu_{uuid.uuid4().hex}"


class OpenAIStreamToAnthropicAdapter:
    def __init__(
        self,
        translator: AnthropicTranslator,
        fallback_model: str,
        input_tokens_estimate: int,
    ) -> None:
        self._translator = translator
        self._fallback_model = fallback_model
        self._input_tokens_estimate = input_tokens_estimate
        self._next_content_index = 0
        self._text_block_open = False
        self._text_block_index: int | None = None
        self._tool_states: dict[int, ToolStreamState] = {}
        self._active_tool_index: int | None = None
        self._output_fragments: list[str] = []
        self._usage: dict[str, int] = {
            "input_tokens": input_tokens_estimate,
            "output_tokens": 0,
        }
        self._finish_reason: str | None = None
        self._response_id = self._translator._message_id()

    async def transform(self, response: Any) -> Any:
        yield anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self._response_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self._fallback_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": self._usage,
                },
            },
        )

        async for event_name, data in self._iter_sse_events(response):
            if event_name == "error":
                yield anthropic_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": data,
                        },
                    },
                )
                continue

            if data == "[DONE]":
                continue

            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            if payload.get("model"):
                self._fallback_model = payload["model"]
            if payload.get("usage"):
                self._maybe_update_usage(payload["usage"])

            choices = payload.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                self._finish_reason = finish_reason

            content_delta = delta.get("content")
            if content_delta:
                async for chunk in self._handle_text_delta(content_delta):
                    yield chunk

            tool_calls = delta.get("tool_calls") or []
            if tool_calls:
                async for chunk in self._handle_tool_calls(tool_calls):
                    yield chunk

        async for chunk in self._close_open_blocks():
            yield chunk

        if not self._usage["output_tokens"]:
            self._usage["output_tokens"] = self._translator._estimator.estimate_text_tokens(
                "".join(self._output_fragments)
            )

        yield anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": map_finish_reason(
                        self._finish_reason,
                        has_tool_calls=bool(self._tool_states),
                    ),
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": self._usage["output_tokens"],
                },
            },
        )
        yield anthropic_sse("message_stop", {"type": "message_stop"})

    async def _handle_text_delta(self, text: str) -> Any:
        if self._active_tool_index is not None:
            async for chunk in self._close_tool_block(self._active_tool_index):
                yield chunk
            self._active_tool_index = None

        if not self._text_block_open:
            self._text_block_index = self._next_content_index
            self._next_content_index += 1
            self._text_block_open = True
            yield anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._text_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )

        self._output_fragments.append(text)
        yield anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._text_block_index,
                "delta": {"type": "text_delta", "text": text},
            },
        )

    async def _handle_tool_calls(self, tool_calls: list[dict[str, Any]]) -> Any:
        if self._text_block_open:
            yield anthropic_sse(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": self._text_block_index,
                },
            )
            self._text_block_open = False
            self._text_block_index = None

        for tool_delta in tool_calls:
            openai_index = int(tool_delta.get("index", 0))
            state = self._tool_states.get(openai_index)
            if state is None:
                state = ToolStreamState(
                    openai_index=openai_index,
                    anthropic_index=self._next_content_index,
                    tool_id=tool_delta.get("id") or self._translator._tool_id(),
                )
                self._tool_states[openai_index] = state
                self._next_content_index += 1

            if self._active_tool_index is not None and self._active_tool_index != openai_index:
                async for chunk in self._close_tool_block(self._active_tool_index):
                    yield chunk
                self._active_tool_index = None

            function = tool_delta.get("function") or {}
            if tool_delta.get("id"):
                state.tool_id = tool_delta["id"]
            if function.get("name"):
                state.name = function["name"]

            arguments = function.get("arguments")
            if arguments is not None:
                if state.started and state.name:
                    state.arguments.append(arguments)
                    yield anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": state.anthropic_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        },
                    )
                else:
                    state.buffered_arguments.append(arguments)

            if state.name and not state.started:
                state.started = True
                self._active_tool_index = openai_index
                yield anthropic_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": state.anthropic_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": state.tool_id,
                            "name": state.name,
                            "input": {},
                        },
                    },
                )
                for buffered in state.buffered_arguments:
                    state.arguments.append(buffered)
                    yield anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": state.anthropic_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": buffered,
                            },
                        },
                    )
                state.buffered_arguments.clear()

    async def _close_open_blocks(self) -> Any:
        if self._text_block_open:
            yield anthropic_sse(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": self._text_block_index,
                },
            )
            self._text_block_open = False
            self._text_block_index = None

        if self._active_tool_index is not None:
            async for chunk in self._close_tool_block(self._active_tool_index):
                yield chunk
            self._active_tool_index = None

        for openai_index in sorted(self._tool_states):
            state = self._tool_states[openai_index]
            if state.started:
                continue
            state.started = True
            state.name = state.name or "tool"
            yield anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": state.tool_id,
                        "name": state.name,
                        "input": {},
                    },
                },
            )
            for buffered in state.buffered_arguments:
                yield anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": state.anthropic_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": buffered,
                        },
                    },
                )
            yield anthropic_sse(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": state.anthropic_index,
                },
            )

    async def _close_tool_block(self, openai_index: int) -> Any:
        state = self._tool_states[openai_index]
        if not state.started:
            return
        yield anthropic_sse(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": state.anthropic_index,
            },
        )

    def _maybe_update_usage(self, usage: dict[str, Any]) -> None:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            self._usage["input_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            self._usage["output_tokens"] = int(completion_tokens)

    async def _iter_sse_events(self, response: Any) -> Any:
        event_name = "message"
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if not line:
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            yield event_name, "\n".join(data_lines)
