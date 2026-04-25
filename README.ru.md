# Claude Code -> Cloud.ru proxy

Языки: [English](README.md) | [Русский](README.ru.md)

Тонкий локальный Python-прокси, который принимает запросы Anthropic Messages API от
`Claude Code` и отправляет их в `Cloud.ru Foundation Models` через OpenAI-compatible
endpoint `/chat/completions`.

## Что делает прокси

- Экспортирует `POST /v1/messages`
- Экспортирует `POST /v1/messages/count_tokens`
- Экспортирует совместимые алиасы `/anthropic/v1/...`
- Экспортирует `GET /healthz`
- Переводит Anthropic text и tool-use payloads в OpenAI chat-completions payloads
- Конвертирует OpenAI streaming responses в Anthropic SSE events для `Claude Code`

## Известные ограничения

`Claude Code` отправляет некоторые Anthropic-specific поля запроса, которые Cloud.ru
OpenAI-compatible endpoint напрямую не понимает. Прокси сознательно логирует и
отбрасывает эти поля вместо того, чтобы отправлять их upstream.

Игнорируемые поля:

- `metadata`: служебные метаданные запроса. Обычно не влияют на ответ модели.
- `thinking`: настройки Claude extended thinking. Модели Cloud.ru/Qwen не поддерживают
  этот Anthropic wire format через OpenAI-compatible endpoint.
- `context_management`: Anthropic-механика управления контекстом. В Cloud.ru
  OpenAI-compatible API нет прямого аналога.
- `output_config`: настройки формата/вывода от Claude Code. Прокси считает это поле
  безопасным для игнорирования, потому что его forwarding или rejection может приводить
  к ошибке `400` в Claude Code.

Должно работать:

- Базовый чат
- Streaming responses
- Tool use и tool results

Может не работать:

- Anthropic extended thinking
- Anthropic beta-флаги, завязанные на native Anthropic API behavior
- Anthropic-specific context management
- Structured output через `output_config`, если Claude Code ожидает Anthropic-native
  behavior

## Быстрый старт (macOS)

https://github.com/user-attachments/assets/27ca57a1-d43f-44e0-bca5-7c80263aed9e

Откройте терминал, создайте тестовую папку, удалите предыдущую копию проекта,
скачайте последнюю версию репозитория и перейдите в корень проекта:

```bash
mkdir -p test-proxy
cd test-proxy
rm -rf claude-cloud-proxy
git clone https://github.com/Mnmikhaylov/claude-cloud-proxy.git
cd claude-cloud-proxy
```

Запускайте из корня проекта:

Эта команда установит или обновит Homebrew `python@3.11` и создаст virtualenv
на Python 3.11.

```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Запустите прокси:

```bash
python -m claude_cloud_proxy
```

По умолчанию прокси слушает `127.0.0.1:2222`.

Запустите Claude Code в новом терминале:

Укажите Foundation Models key только один раз, в `ANTHROPIC_AUTH_TOKEN` ниже.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:2222/anthropic
export ANTHROPIC_AUTH_TOKEN="<foundation-models-cloud-ru-key>"
export ANTHROPIC_CUSTOM_MODEL_OPTION="Qwen/Qwen3-Coder-Next"
export ANTHROPIC_DEFAULT_SONNET_MODEL="Qwen/Qwen3-Coder-Next"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Qwen/Qwen3-Coder-Next"
export ANTHROPIC_DEFAULT_OPUS_MODEL="Qwen/Qwen3-Coder-Next"

claude --model Qwen/Qwen3-Coder-Next
```

Чтобы вернуться к обычным моделям Claude, закройте оба терминала: терминал с прокси
и терминал с Claude Code.

## Заметки

- Это локальный single-user proxy, а не production gateway.
- Для командного использования, fallback routing, observability и managed keys лучше
  подходит `LiteLLM`.
- Если используете `LiteLLM`, избегайте версий `1.82.7` и `1.82.8`.

## Локальный сервис на Ubuntu systemd

На Ubuntu используйте `systemd --user` scripts. Ubuntu service по умолчанию bind-ится
на `127.0.0.1`; для локального Claude Code usage его стоит оставлять loopback-only.

Установите Python и прокси:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

cd /path/to/claude-cloud-ru-proxy
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Установите и запустите user service:

```bash
chmod +x scripts/install-systemd-user-service.sh scripts/uninstall-systemd-user-service.sh
./scripts/install-systemd-user-service.sh
```

Installer создает:

```bash
~/.config/systemd/user/claude-cloud-proxy.service
~/.config/claude-cloud-proxy/env
```

Укажите Cloud.ru key и настройки прокси в `~/.config/claude-cloud-proxy/env`:

```bash
CLOUD_RU_BASE_URL=https://foundation-models.api.cloud.ru/v1
CLOUD_RU_API_KEY="<foundation-models-cloud-ru-key>"
PROXY_HOST=127.0.0.1
PROXY_PORT=2222
PROXY_MAX_REQUEST_BYTES=33554432
PROXY_TIMEOUT_SECONDS=120
PROXY_LOG_LEVEL=INFO
```

Перезапустите и проверьте сервис:

```bash
systemctl --user restart claude-cloud-proxy.service
systemctl --user status claude-cloud-proxy.service
curl http://127.0.0.1:2222/healthz
```

Логи:

```bash
journalctl --user -u claude-cloud-proxy.service -f
```

Удалить Ubuntu service:

```bash
./scripts/uninstall-systemd-user-service.sh
```

Если это headless Ubuntu server и сервис должен стартовать до interactive login,
один раз включите linger:

```bash
sudo loginctl enable-linger "$USER"
```
