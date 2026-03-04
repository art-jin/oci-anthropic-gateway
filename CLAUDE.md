# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCI-Anthropic Gateway is a translation layer that enables OCI GenAI models (Grok, GPT, Cohere Command-R, etc.) to work with Anthropic's API format. The key feature is enhanced tool calling support for models that don't natively support function calling.

**Key Features:**
- Full Anthropic Messages API compatibility
- Enhanced tool calling support (native + simulated)
- Streaming and non-streaming responses
- Prompt caching support
- Vision/image analysis
- Extended thinking mode
- Modular, maintainable codebase

## Development Commands

### Environment Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Server
```bash
python main.py                        # Binds to server.host:server.port from config.json (default 127.0.0.1:8000)
LOG_LEVEL=DEBUG python main.py        # Verbose gateway logging
LOG_LEVEL=WARNING python main.py      # Minimal logging (production)
```

### Run Tests
```bash
# Integration tests (require running server on localhost:8000)
PYTHONPATH=. python -m test.01_event_logging
PYTHONPATH=. python -m test.02_count_tokens
PYTHONPATH=. python -m test.03_messages_non_stream
PYTHONPATH=. python -m test.04_messages_stream_sse
PYTHONPATH=. python -m test.05_tools_non_stream_and_tool_result_loop

# With environment variables
GATEWAY_BASE_URL=http://localhost:8000 GATEWAY_MODEL=your-model PYTHONPATH=. python -m test.03_messages_non_stream

# Unit tests (no server required)
pytest                                    # Run all tests in tests/
pytest tests/test_tool_call_detection.py  # Run specific test file
```

## Architecture

### Two API Formats

1. **Cohere Format** (`api_format: "cohere"`):
   - Uses OCI's native function calling for Cohere models
   - Tools converted to `CohereTool` format

2. **Generic Format** (`api_format: "generic"`):
   - Uses prompt engineering to simulate tool calling
   - Models output `<TOOL_CALL>JSON</TOOL_CALL>` format
   - Gateway detects, parses, and converts to Anthropic format

### Key Modules

| Path | Purpose |
|------|---------|
| `src/config/__init__.py` | Config class for loading config.json; initializes OCI GenAI client |
| `src/services/generation.py` | Core generation logic (streaming & non-streaming OCI responses) |
| `src/utils/tools.py` | Converts Anthropic tools to OCI/Cohere format; builds tool use instructions |
| `src/utils/json_helper.py` | Parses and fixes malformed JSON from models; detects tool call blocks |
| `src/utils/content_converter.py` | Converts between Anthropic, OCI, and Cohere content formats |
| `src/routes/handlers.py` | FastAPI route handlers for `/v1/messages` and `/v1/messages/count_tokens` |
| `src/utils/cache.py` | Cache control utilities |
| `src/utils/token.py` | Token counting and estimation |
| `src/utils/constants.py` | Constants, stop reasons, pre-compiled regex patterns |
| `src/utils/logging_config.py` | Logging configuration with flexible levels |

### Tool Call Detection Flow

1. Check for structured tool calls (Cohere or Generic format)
2. If none, check for `<TOOL_CALL>` blocks in text
3. If still none, use natural language fallback mechanism
4. Remove tool call markers from response text
5. Format as Anthropic-compatible response

## Configuration

Copy `config.json.template` to `config.json`. Key fields:

```json
{
  "compartment_id": "ocid1.compartment.oc1...",
  "endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
  "model_aliases": {
    "claude-3-5-sonnet-20241022": "gpt5",
    "claude-3-opus-20240229": "cohere.command-r-plus"
  },
  "model_definitions": {
    "gpt5": {
      "ocid": "ocid1.generativeaimodel.oc1...",
      "api_format": "generic",
      "max_tokens_key": "max_completion_tokens",
      "temperature": 1.0
    },
    "cohere.command-r-plus": {
      "ocid": "ocid1.generativeaimodel.oc1...",
      "api_format": "cohere",
      "max_tokens_key": "max_tokens",
      "temperature": 0.7
    }
  },
  "default_model": "gpt5",
  "debug": false,
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "log_level": "warning"
  }
}
```

| Field | Description |
|-------|-------------|
| `ocid` | OCI model OCID |
| `api_format` | `"generic"` (simulated tools) or `"cohere"` (native tools) |
| `max_tokens_key` | Parameter name: `"max_tokens"` or `"max_completion_tokens"` |
| `temperature` | Fixed temperature (overrides request) |
| `debug` | Debug-only mode. When `true`, writes diagnostic dumps to `./debug_dumps/`. |
| `server.host` | Uvicorn bind host (e.g. `127.0.0.1`, `0.0.0.0`). |
| `server.port` | Uvicorn bind port (1-65535). |
| `server.log_level` | Uvicorn log level: `critical|error|warning|info|debug|trace`. |

## API Features

### Supported Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Create messages (streaming & non-streaming) |
| `/v1/messages/count_tokens` | POST | Count tokens in request |

### Messages API Features

#### System Prompts
```json
{
  "system": "You are a helpful programming assistant.",
  "messages": [...]
}
```

Array format with cache control:
```json
{
  "system": [
    {"type": "text", "text": "You are a helpful assistant."},
    {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}
  ]
}
```

#### Streaming
Emits Anthropic-compatible SSE events:
- `message_start`
- `content_block_start`
- `content_block_delta`
- `content_block_stop`
- `message_delta`
- `message_stop`

#### Vision / Images
```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {
        "type": "image",
        "source": {
          "type": "base64",
          "media_type": "image/png",
          "data": "iVBORw0KGgo..."
        }
      }
    ]
  }]
}
```

#### Extended Thinking
```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 16000
  },
  "messages": [...]
}
```

#### Sampling Parameters
```json
{
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "max_tokens": 4096,
  "stop_sequences": ["\n\n", "END"]
}
```

## Important Patterns

### JSON Repair (`src/utils/json_helper.py`)

The `fix_json_issues()` function handles malformed JSON from models:
- Auto-fixes missing quotes
- Replaces single quotes with double quotes
- Removes trailing commas
- Completes incomplete JSON objects
- Extracts JSON embedded in text

### Content Conversion (`src/utils/content_converter.py`)

Handles three formats:
- **Anthropic format**: `{"type": "text", "text": "..."}` or `{"type": "image", "source": {...}}`
- **OCI format**: `{"type": "TEXT", "text": "..."}`
- **Cohere format**: Different message structure for Cohere models

### Streaming Response Format

Responses are streamed as Server-Sent Events (SSE) in Anthropic format:
```
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "..."}}
```

## Logging and Debugging

Control log output using the `LOG_LEVEL` environment variable:

```bash
LOG_LEVEL=WARNING python main.py   # Minimal (production)
LOG_LEVEL=INFO python main.py      # Balanced (default, development)
LOG_LEVEL=DEBUG python main.py     # Verbose (debugging)
```

**What Gets Logged:**
- **WARNING**: Configuration loaded, server status, critical errors only
- **INFO**: Request summaries, tool call detection results, key operational events
- **DEBUG**: Detailed parameters, content conversion, JSON parsing, OCI SDK requests

## Troubleshooting

### Tool Calls Not Detected
1. Review logs for tool detection messages
2. Verify tool definition format
3. Try explicit tool names in user message
4. Test with `tool_choice: "required"`

**Known failure mode (fixed):**
- For long/complex tool calls (especially `Edit`), the literal string `</TOOL_CALL>` can appear inside JSON string fields such as `Edit.new_string`.
- Older parsing could be misled by that literal and truncate the JSON early, resulting in `detected=[]`.
- The gateway now extracts the *first balanced JSON object* after `<TOOL_CALL>` before searching for the real closing tag, making detection robust to `</TOOL_CALL>` appearing inside JSON strings.
- Tool names are normalized (e.g. `edit` -> `Edit`) to reduce failures from model casing/style drift.

**End-to-end verification:**
1. Restart the gateway.
2. Re-run the same Anthropic conversation that previously produced a long `Edit` tool call.
3. Inspect `debug_dumps/*_stream_tool_detection_primary.json` and confirm `detected` is non-empty, and verify the client actually executes the `Edit` tool call.

### JSON Parsing Errors
1. Review raw model output in logs
2. Check for special characters
3. Review `fix_json_issues` logs

### Model Not Following Tool Format
1. Switch to a Cohere model for native support
2. Use more explicit prompts
3. Set `tool_choice: "required"`

## Adding New Models

1. Get the OCI model OCID from OCI Console
2. Add to `model_definitions` in `config.json`
3. Optionally add alias in `model_aliases`

## Prerequisites

- Python 3.12+
- OCI CLI configured (`~/.oci/config`)
- OCI account with GenAI service access

## Dependencies

```
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
python-dotenv>=1.0.0
oci==2.131.1
```

## Security Notes

- `config.json` is in `.gitignore` - never commit it
- Use OCI IAM policies to restrict model access
- Consider implementing authentication for production

## Changelog

### 2026-03-04: Debug dumps for tool call instability + configurable server bind

**Problem:** Tool call detection could become unstable under long/complex outputs (e.g. missing `<TOOL_CALL>` wrapper, missing closing marker, or streaming timing issues), making root-cause hard to reproduce.

**Fix:**
- Added `debug` flag in `config.json` to enable debug-only observability dumps written to `./debug_dumps/`.
- Dumps include request/response summaries and tool detection evidence for both non-streaming and streaming flows.
- Fixed a core parsing edge case where the literal string `</TOOL_CALL>` can appear inside JSON string fields (e.g. `Edit.new_string`) and mislead older extraction logic. The gateway now extracts the first balanced JSON object after `<TOOL_CALL>` before searching for the real closing tag (see `src/utils/json_helper.py`).
- Added common tool-name normalization (e.g. `edit` -> `Edit`) to reduce failures from model casing/style drift (see `src/utils/json_helper.py`).
- Made Uvicorn bind configurable via `config.json`:
  - `server.host`
  - `server.port`
  - `server.log_level`

**Notes:**
- When `debug` is disabled, dump writing is a no-op.
- Dump write failures must not impact request flow.

### 2025-02-15: Fix Streaming Tool Call Detection

**Problem:** In streaming mode, tool calls were not being detected and converted. The `<TOOL_CALL>` blocks were passed through as plain text instead of being converted to Anthropic `tool_use` format.

**Root Cause:** `src/services/generation.py` line 620 had `buffer_text_for_tool_detection = False` hardcoded, disabling tool detection in streaming mode.

**Fix:**
```python
# Before (broken)
buffer_text_for_tool_detection = False

# After (fixed)
buffer_text_for_tool_detection = has_tools and not is_cohere
```

**Test:** Run `python test_tool_call.py` to verify tool calling works in both streaming and non-streaming modes.

**Verified Capabilities:**
- Maximum parallel tool calls tested: 6
- Supported tools: 17 (Task, Bash, Read, Write, Edit, Glob, Grep, etc.)
- Both streaming and non-streaming modes working
