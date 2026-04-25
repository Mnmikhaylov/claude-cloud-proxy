from __future__ import annotations

import json
import logging

import httpx
from fastapi.testclient import TestClient

from claude_cloud_proxy.app import create_app
from claude_cloud_proxy.config import Settings


def make_client(handler) -> TestClient:
    settings = Settings(
        cloud_ru_api_key="cloud-key",
        log_level="DEBUG",
    )
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_healthz() -> None:
    with make_client(lambda request: httpx.Response(200, json={})) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_server_side_key_requires_proxy_key_when_configured() -> None:
    settings = Settings(
        cloud_ru_api_key="cloud-key",
        proxy_api_key="proxy-key",
        log_level="DEBUG",
    )
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid or missing proxy API key."


def test_server_side_key_accepts_valid_proxy_key() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-proxy-key",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    settings = Settings(
        cloud_ru_api_key="cloud-key",
        proxy_api_key="proxy-key",
        log_level="DEBUG",
    )
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-proxy-api-key": "proxy-key"},
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen_headers["authorization"] == "Bearer cloud-key"


def test_non_loopback_server_side_key_requires_proxy_key() -> None:
    try:
        create_app(
            settings=Settings(
                cloud_ru_api_key="cloud-key",
                host="0.0.0.0",
            ),
            upstream_transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        )
    except ValueError as exc:
        assert "PROXY_API_KEY is required" in str(exc)
    else:
        raise AssertionError("Expected non-loopback server-side key to require PROXY_API_KEY")


def test_cloud_ru_base_url_requires_https() -> None:
    try:
        create_app(
            settings=Settings(
                cloud_ru_api_key="cloud-key",
                cloud_ru_base_url="http://127.0.0.1:9000/v1",
            ),
            upstream_transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        )
    except ValueError as exc:
        assert "CLOUD_RU_BASE_URL must be an absolute https:// URL" in str(exc)
    else:
        raise AssertionError("Expected non-HTTPS CLOUD_RU_BASE_URL to be rejected")


def test_request_body_size_limit() -> None:
    settings = Settings(
        cloud_ru_api_key="cloud-key",
        max_request_bytes=64,
        log_level="DEBUG",
    )
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            content=b'{"model":"Qwen/Qwen3-Coder-Next","max_tokens":1,"messages":[{"role":"user","content":"'
            + b"x" * 128
            + b'"}]}',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "request_too_large"


def test_anthropic_prefixed_route_and_x_api_key_work() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-prefixed",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    settings = Settings(cloud_ru_api_key=None, log_level="DEBUG")
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.post(
            "/anthropic/v1/messages",
            headers={"x-api-key": "cloud-ru-key"},
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen_headers["authorization"] == "Bearer cloud-ru-key"


def test_x_api_key_takes_precedence_over_authorization_header() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-auth-precedence",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    settings = Settings(cloud_ru_api_key=None, log_level="DEBUG")
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.post(
            "/anthropic/v1/messages",
            headers={
                "authorization": "Bearer stale-anthropic-token",
                "x-api-key": "cloud-ru-key",
            },
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen_headers["authorization"] == "Bearer cloud-ru-key"


def test_cloud_ru_like_authorization_wins_over_placeholder_x_api_key() -> None:
    seen_headers: dict[str, str] = {}
    cloud_ru_like_key = (
        "dGVzdC1jbG91ZC1ydS1rZXktaWQ.0123456789abcdef0123456789abcdef"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-cloud-key-detection",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    settings = Settings(cloud_ru_api_key=None, log_level="DEBUG")
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.post(
            "/anthropic/v1/messages",
            headers={
                "authorization": f"Bearer {cloud_ru_like_key}",
                "x-api-key": "local-proxy-key",
            },
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen_headers["authorization"] == f"Bearer {cloud_ru_like_key}"


def test_proxy_authorization_cloud_ru_key_is_supported() -> None:
    seen_headers: dict[str, str] = {}
    cloud_ru_like_key = (
        "dGVzdC1wcm94eS1hdXRob3JpemF0aW9u.abcdef0123456789abcdef0123456789"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-proxy-authorization",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    settings = Settings(cloud_ru_api_key=None, log_level="DEBUG")
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.post(
            "/anthropic/v1/messages",
            headers={
                "authorization": "Bearer stale-anthropic-token",
                "x-api-key": "local-proxy-key",
                "proxy-authorization": f"Bearer {cloud_ru_like_key}",
            },
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen_headers["authorization"] == f"Bearer {cloud_ru_like_key}"


def test_bearer_prefix_in_x_api_key_is_normalized() -> None:
    seen_headers: dict[str, str] = {}
    cloud_ru_like_key = (
        "dGVzdC14LWFwaS1rZXktYmVhcmVy.abcdef0123456789abcdef0123456789"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-normalized-x-api-key",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    settings = Settings(cloud_ru_api_key=None, log_level="DEBUG")
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.post(
            "/anthropic/v1/messages",
            headers={"x-api-key": f"Bearer {cloud_ru_like_key}"},
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen_headers["authorization"] == f"Bearer {cloud_ru_like_key}"


def test_incoming_auth_selection_is_logged_without_secret(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-auth-log",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    cloud_ru_like_key = (
        "dGVzdC1hdXRoLWxvZy1rZXktaWQ.abcdef0123456789abcdef0123456789"
    )
    settings = Settings(cloud_ru_api_key=None, log_level="INFO")
    app = create_app(
        settings=settings,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client, caplog.at_level(logging.INFO, logger="claude_cloud_proxy"):
        response = client.post(
            "/anthropic/v1/messages",
            headers={"x-api-key": cloud_ru_like_key},
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert "Using incoming Cloud.ru API key source=x-api-key" in caplog.text
    assert "sha256=" in caplog.text
    assert cloud_ru_like_key not in caplog.text


def test_count_tokens_returns_best_effort_estimate() -> None:
    with make_client(lambda request: httpx.Response(200, json={})) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "system": "You are a coding model.",
                "messages": [{"role": "user", "content": "Write a function."}],
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


def test_non_stream_request_translates_tool_result_and_system_messages() -> None:
    seen_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_request.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Tool result processed.",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 21, "completion_tokens": 4},
            },
        )

    body = {
        "model": "Qwen/Qwen3-Coder-Next",
        "max_tokens": 128,
        "system": "You are a careful coding assistant.",
        "messages": [
            {"role": "user", "content": "Inspect the repo."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll inspect the repo."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": [{"type": "text", "text": "file contents"}],
                    },
                    {"type": "text", "text": "Summarize it."},
                ],
            },
        ],
    }

    with make_client(handler) as client:
        response = client.post("/v1/messages", json=body)

    assert response.status_code == 200
    translated_messages = seen_request["messages"]
    assert translated_messages[0] == {
        "role": "system",
        "content": "You are a careful coding assistant.",
    }
    assert translated_messages[1] == {"role": "user", "content": "Inspect the repo."}
    assert translated_messages[2]["role"] == "assistant"
    assert translated_messages[2]["tool_calls"][0]["id"] == "toolu_123"
    assert translated_messages[3] == {
        "role": "tool",
        "tool_call_id": "toolu_123",
        "content": "file contents",
    }
    assert translated_messages[4] == {"role": "user", "content": "Summarize it."}
    assert response.json()["content"] == [{"type": "text", "text": "Tool result processed."}]


def test_non_stream_response_translates_tool_calls_back_to_anthropic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["tool_choice"] == "required"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Need to read the file.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\":\"README.md\"}",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 12},
            },
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 128,
                "tool_choice": {"type": "any"},
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "Read the README."}],
            },
        )

    assert response.status_code == 200
    content = response.json()["content"]
    assert content[0] == {"type": "text", "text": "Need to read the file."}
    assert content[1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_file",
        "input": {"path": "README.md"},
    }
    assert response.json()["stop_reason"] == "tool_use"


def test_streaming_text_response_is_converted_to_anthropic_sse() -> None:
    stream = (
        'data: {"id":"chatcmpl-3","model":"Qwen/Qwen3-Coder-Next","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-3","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream.encode(),
        )

    with make_client(handler) as client:
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 128,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: message_start" in body
    assert '"type":"text_delta","text":"Hello"' in body
    assert '"type":"text_delta","text":" world"' in body
    assert '"stop_reason":"end_turn"' in body
    assert "event: message_stop" in body


def test_streaming_tool_call_response_is_converted_to_tool_use_sse() -> None:
    stream = (
        'data: {"id":"chatcmpl-4","model":"Qwen/Qwen3-Coder-Next","choices":[{"index":0,"delta":{"content":"Checking."},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":\\"README.md\\"}"}}]},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":8}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream.encode(),
        )

    with make_client(handler) as client:
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 128,
                "stream": True,
                "messages": [{"role": "user", "content": "Read the README"}],
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"content_block":{"type":"tool_use","id":"call_2","name":"read_file","input":{}}' in body
    assert '"type":"input_json_delta","partial_json":"{\\"path\\":\\"README.md\\"}"' in body
    assert '"stop_reason":"tool_use"' in body


def test_upstream_auth_error_is_mapped_to_anthropic_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 401
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "bad key",
        },
    }


def test_streaming_upstream_error_is_returned_as_anthropic_sse_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "bad stream request"}},
        )

    with make_client(handler) as client:
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: error" in body
    assert '"type":"invalid_request_error","message":"bad stream request"' in body


def test_unknown_anthropic_field_returns_400() -> None:
    with make_client(lambda request: httpx.Response(200, json={})) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
                "foo": "bar",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_output_config_is_ignored() -> None:
    with make_client(
        lambda request: httpx.Response(
            200,
            json={
                "id": "chatcmpl-output-config",
                "model": "Qwen/Qwen3-Coder-Next",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )
    ) as client:
        response = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "Qwen/Qwen3-Coder-Next",
                "max_tokens": 64,
                "output_config": {"format": "text"},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "ok"}]
