from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageParam(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]
    model_config = ConfigDict(extra="allow")


class ToolParam(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    type: str | None = None
    model_config = ConfigDict(extra="allow")


class AnthropicRequestBase(BaseModel):
    model: str
    messages: list[MessageParam]
    system: str | list[dict[str, Any]] | None = None
    tools: list[ToolParam] | None = None
    tool_choice: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    model_config = ConfigDict(extra="allow")


class AnthropicMessageRequest(AnthropicRequestBase):
    max_tokens: int


class AnthropicCountTokensRequest(AnthropicRequestBase):
    pass
