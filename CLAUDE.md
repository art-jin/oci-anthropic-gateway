# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCI-Anthropic Gateway translates Anthropic Messages API requests into OCI GenAI requests and converts responses back into Anthropic-compatible JSON or SSE. The main complexity is not FastAPI itself, but the translation layer around model routing, content conversion, tool calling, streaming, guardrails, and debug observability.

## Common Commands

### Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.json.template config.json
```

### Run locally
```bash
python main.py
LOG_LEVEL=INFO python main.py
LOG_LEVEL=DEBUG python main.py
```

`main.py` reads bind host/port and uvicorn log level from `config.json` `server.*`.

### Run tests
```bash
# Unit tests
pytest
pytest test/test_tool_call_detection.py
pytest test/test_guardrails.py

# Integration-style scripts against a running gateway
PYTHONPATH=. python -m test.03_messages_non_stream
PYTHONPATH=. python -m test.04_messages_stream_sse
PYTHONPATH=. python -m test.05_tools_non_stream_and_tool_result_loop

# Override target gateway/model for script tests
GATEWAY_BASE_URL=http://localhost:8000 GATEWAY_MODEL=openai.gpt-oss-20b PYTHONPATH=. python -m test.03_messages_non_stream
```

### Container flow
```bash
docker build -t oci-anthropic-gateway:latest .
docker-compose up -d
```

## Architecture

### Request path

The app is intentionally thin at the HTTP layer:

1. `main.py` creates the FastAPI app, initializes config on startup, optionally starts the debug dump indexer, and applies in-memory rate limiting.
2. The catch-all POST route forwards `count_tokens` requests to `src/routes/handlers.py:handle_count_tokens` and `messages` requests to `src/routes/handlers.py:handle_messages_request`.
3. `handle_messages_request` validates Anthropic-format payloads, assigns `session_id` / `request_id` metadata, checks requested modalities against the selected model, runs input guardrails, and converts Anthropic content into OCI SDK message objects.
4. `src/services/generation.py` performs the OCI chat call and is responsible for the hard part: adapting parameters per model, injecting tool instructions for generic models, parsing OCI responses, converting tool calls, applying output guardrails, and emitting Anthropic-compatible non-stream or streaming responses.

### Model routing and config normalization

`src/config/__init__.py` does more than load JSON:

- Resolves requested Anthropic model names through `model_aliases`.
- Normalizes each `model_definitions` entry so runtime code can rely on defaults like `compartment_id`, `max_tokens_key`, and `model_types`.
- Initializes the OCI client with auth fallback order: OKE workload identity, OCI resource principal, then local `~/.oci/config` API key.

Important config fields future agents usually need:

- `model_definitions.<name>.api_format`: `cohere` uses OCI native tool calling; everything else follows the generic translation path.
- `model_definitions.<name>.model_types`: gates whether text/images/video/audio inputs are allowed.
- `debug`: enables JSON debug dumps in `debug_dumps/`.
- `debug_ui.*`: enables the debug API/UI backed by an indexed dump database.
- `guardrails.*`: controls OCI ApplyGuardrails on input and non-stream output.
- `rate_limit.*`: enables the in-memory sliding-window limiter in `main.py`.

### Two response paths

#### 1. Generic models

For non-Cohere models, the gateway simulates tool calling by injecting a system instruction that requires the model to emit:

```text
<TOOL_CALL>
{"name": "tool_name", "input": {...}}
</TOOL_CALL>
```

`src/services/generation.py` then:

- buffers or extracts text from OCI responses,
- looks for structured tool calls first,
- falls back to `<TOOL_CALL>` block detection via `src/utils/json_helper.py`,
- optionally uses natural-language fallback if `enable_nl_tool_fallback` is enabled,
- strips tool-call markup from assistant text,
- returns Anthropic `tool_use` blocks plus any remaining text.

Streaming generic responses are especially important: the service buffers text when tools are present, then emits Anthropic SSE events only after tool detection completes.

#### 2. Cohere models

For `api_format: "cohere"`, the gateway converts Anthropic messages/tools into Cohere-specific request structures and relies on OCI native tool calling. This path skips the prompt-engineered `<TOOL_CALL>` simulation.

### Content conversion boundaries

The conversion layer is split across utilities:

- `src/utils/content_converter.py` converts Anthropic content blocks to OCI SDK content and handles response-side extraction.
- `src/utils/tools.py` converts Anthropic tool schemas into Cohere tools and builds generic tool-use instructions.
- `src/utils/request_validation.py` validates payload shape and modality compatibility before conversion.
- `src/utils/token.py` powers `/v1/messages/count_tokens` and rough usage estimates on responses.

When debugging malformed model output or tool parsing, `src/utils/json_helper.py` is a primary file to inspect.

### Guardrails flow

Guardrails are integrated in two different phases:

- Input guardrails run in `src/routes/handlers.py` before the OCI chat request, using text extracted from user messages, and optionally system/tool-result content.
- Output guardrails run only in non-stream mode inside `src/services/generation.py` after text is produced.

If output guardrails are enabled and the request asks for streaming, the handler either rejects the request or downgrades it to non-stream based on `guardrails.streaming_behavior`.

The gateway depends on `oci>=2.164.0` because prompt-injection guardrails require newer SDK types.

### Debug UI and observability

There are two separate debug mechanisms:

- `debug=true` writes per-request JSON dumps into `debug_dumps/`.
- `debug_ui.enabled=true` starts the indexer and exposes `/debug/api/*` plus static files under `web/debug/`.

The debug UI is backed by indexed dump files, not live request state. `src/debug/routes.py` serves session/timeline APIs and SSE feeds; startup in `main.py` launches `DebugDumpIndexer` to ingest dumps and broadcast timeline events.

Use this when investigating tool-call parsing, stream ordering, or OCI request/response mismatches.

## Testing Notes

The repo has both pytest tests and numbered runnable scripts under `test/`.

- `pytest` covers focused units such as tool detection, normalization, config validation, rate limiting, request validation, debug dumps, and guardrails.
- `test/03` through `test/10` are scenario scripts that exercise the running gateway end to end.

If you change streaming, tool detection, multimodal validation, or guardrails, update the relevant script tests as well as unit tests.

## Known Important Behaviors

- The POST API is implemented as a catch-all route in `main.py`, so endpoint matching is substring-based (`messages`, `count_tokens`, `event_logging`) rather than explicit FastAPI path declarations.
- Streaming tool detection for generic models depends on buffering the full text when tools are present.
- Tool-call parsing is hardened against literal `</TOOL_CALL>` appearing inside JSON string fields by extracting the first balanced JSON object after `<TOOL_CALL>`.
- `metadata.session_id` and `metadata.request_id` are auto-filled if missing and are used by the debug timeline and rate-limit keying.
- Model modality validation happens before content conversion, so image/video/audio failures are usually config issues (`model_types`) rather than OCI SDK issues.
