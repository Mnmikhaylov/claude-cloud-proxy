# Claude Code -> Cloud.ru proxy

Languages: [English](README.md) | [Русский](README.ru.md)

Thin local Python proxy that accepts Anthropic Messages API requests from `Claude Code`
and forwards them to `Cloud.ru Foundation Models` through the OpenAI-compatible
`/chat/completions` endpoint.

## What it does

- Exposes `POST /v1/messages`
- Exposes `POST /v1/messages/count_tokens`
- Exposes compatible aliases under `/anthropic/v1/...`
- Exposes `GET /healthz`
- Translates Anthropic text and tool-use payloads into OpenAI chat-completions payloads
- Converts OpenAI streaming responses into Anthropic SSE events for `Claude Code`

## Known limitations

`Claude Code` sends some Anthropic-specific request fields that the Cloud.ru
OpenAI-compatible endpoint does not understand directly. The proxy intentionally
logs and drops these fields instead of forwarding them upstream.

Ignored fields:

- `metadata`: request-level service metadata. It usually does not affect model output.
- `thinking`: Claude extended-thinking settings. Cloud.ru/Qwen models do not support
  this Anthropic wire format through the OpenAI-compatible endpoint.
- `context_management`: Anthropic context-management controls. Cloud.ru's
  OpenAI-compatible API does not expose a direct equivalent.
- `output_config`: Claude Code output-format settings. The proxy treats this field as
  safe to ignore because forwarding or rejecting it can make Claude Code fail with a
  `400` response.

Expected to work:

- Basic chat requests
- Streaming responses
- Tool use and tool results

May not work:

- Anthropic extended thinking
- Anthropic beta flags that depend on native Anthropic API behavior
- Anthropic-specific context management
- Structured output through `output_config` when Claude Code expects Anthropic-native
  behavior

## Quick start (macOS)

https://github.com/user-attachments/assets/27ca57a1-d43f-44e0-bca5-7c80263aed9e

Open Terminal, create a test folder, remove any previous checkout, download the
latest repository version, and enter the project root:

```bash
mkdir -p test-proxy
cd test-proxy
rm -rf claude-cloud-proxy
git clone https://github.com/Mnmikhaylov/claude-cloud-proxy.git
cd claude-cloud-proxy
```

Run from the project root:

This will install or upgrade Homebrew `python@3.11` and create the virtual
environment with Python 3.11.

```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the proxy:

```bash
python -m claude_cloud_proxy
```

The proxy listens on `127.0.0.1:2222` by default.

Run Claude Code in a new terminal:

Set the Foundation Models key only once, as `ANTHROPIC_AUTH_TOKEN` below.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:2222/anthropic
export ANTHROPIC_AUTH_TOKEN="<foundation-models-cloud-ru-key>"
export ANTHROPIC_CUSTOM_MODEL_OPTION="Qwen/Qwen3-Coder-Next"
export ANTHROPIC_DEFAULT_SONNET_MODEL="Qwen/Qwen3-Coder-Next"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Qwen/Qwen3-Coder-Next"
export ANTHROPIC_DEFAULT_OPUS_MODEL="Qwen/Qwen3-Coder-Next"

claude --model Qwen/Qwen3-Coder-Next
```

To return to regular Claude models, close both terminals: the proxy terminal and the
Claude Code terminal.

## Notes

- This is a local single-user proxy, not a production gateway.
- For team usage, fallback routing, observability, and managed keys, `LiteLLM` is the
  better fit.
- If you use `LiteLLM`, avoid versions `1.82.7` and `1.82.8`.

## Run As A Local Service (Ubuntu systemd)

On Ubuntu, use the `systemd --user` scripts. The Ubuntu service binds to `127.0.0.1`
by default and should stay loopback-only for local Claude Code usage.

Install Python and the proxy:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

cd /path/to/claude-cloud-ru-proxy
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Install and start the user service:

```bash
chmod +x scripts/install-systemd-user-service.sh scripts/uninstall-systemd-user-service.sh
./scripts/install-systemd-user-service.sh
```

The installer creates:

```bash
~/.config/systemd/user/claude-cloud-proxy.service
~/.config/claude-cloud-proxy/env
```

Put the Cloud.ru key and proxy settings in `~/.config/claude-cloud-proxy/env`:

```bash
CLOUD_RU_BASE_URL=https://foundation-models.api.cloud.ru/v1
CLOUD_RU_API_KEY="<foundation-models-cloud-ru-key>"
PROXY_HOST=127.0.0.1
PROXY_PORT=2222
PROXY_MAX_REQUEST_BYTES=33554432
PROXY_TIMEOUT_SECONDS=120
PROXY_LOG_LEVEL=INFO
```

Then restart and check the service:

```bash
systemctl --user restart claude-cloud-proxy.service
systemctl --user status claude-cloud-proxy.service
curl http://127.0.0.1:2222/healthz
```

Logs:

```bash
journalctl --user -u claude-cloud-proxy.service -f
```

Remove the Ubuntu service:

```bash
./scripts/uninstall-systemd-user-service.sh
```

If this is a headless Ubuntu server and the service must start before interactive
login, enable linger once:

```bash
sudo loginctl enable-linger "$USER"
```
