# README.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an OCI (Oracle Cloud Infrastructure) GenAI to Anthropic API Gateway. It acts as a translation layer that:
1. Accepts Anthropic-compatible API requests (Messages API format)
2. Translates them to OCI GenAI format
3. Streams responses back in Anthropic's SSE format

This allows using OCI-hosted models (like Grok, GPT variants) with Anthropic's API clients and tools (e.g., Claude Code).

## Running the Gateway

```bash
# Setup virtual environment (first time)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure OCI credentials in ~/.oci/config (standard OCI SDK location)

# Edit config.json with your OCI compartment_id and model OCIDs
# Then run:
python oci-anthropic-gateway.py
```

The gateway runs on `0.0.0.0:8000` by default.

## Configuration

The gateway loads configuration from `config.json`:

### Setup (First Time)

1. **Copy the template**:
   ```bash
   cp config.json.template config.json
   ```

2. **Edit `config.json` with your OCI credentials**:
   - `compartment_id`: Your OCI compartment OCID
   - `endpoint`: OCI GenAI service endpoint (region-specific)
   - `model_definitions`: Replace `your-model-ocid` with actual OCI model OCIDs

3. **Configure OCI SDK** in `~/.oci/config`:
   ```ini
   [DEFAULT]
   user=ocid1.user.oc1...
   fingerprint=...
   tenancy=ocid1.tenancy.oc1...
   region=us-chicago-1
   key_file=~/.oci/oci_api_key.pem
   ```

### Configuration Options

- `compartment_id`: OCI compartment OCID
- `endpoint`: OCI GenAI service endpoint (region-specific)
- `model_aliases`: Maps Anthropic model names to OCI model keys
- `model_definitions`: OCI model configurations with:
  - `ocid`: The OCI model OCID to use
  - `max_tokens_key`: Parameter name for token limit (`max_tokens` or `max_completion_tokens`)
  - `temperature`: Fixed temperature (overrides request)
- `default_model`: Fallback model when requested model not found

> **⚠️ SECURITY**: `config.json` is in `.gitignore` and will not be committed. Never commit files containing real OCIDs or API keys.

## Supported Features

### Core Messages API

The gateway supports the full Anthropic Messages API with the following features:

#### 1. System Prompts

Define system-level instructions for the model:

```json
{
    "model": "claude-3-5-sonnet-20241022",
    "system": "You are a helpful assistant specialized in Python programming.",
    "messages": [
        {"role": "user", "content": "How do I read a file in Python?"}
    ]
}
```

**Array format** (supports cache_control):
```json
{
    "system": [
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": "Be concise and accurate."}
    ]
}
```

#### 2. Tool Calling (Function Calling)

Enable the model to call external functions. The gateway supports tool calling through two mechanisms:

**1. Native Function Calling (Cohere Models)**

Models configured with `api_format: "cohere"` use OCI's native Function Calling support:

```json
{
    "model": "cohere.command-r-plus",
    "tools": [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    ],
    "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ]
}
```

**2. Simulated Tool Calling (Generic Models)**

For models configured with `api_format: "generic"` (xAI Grok, OpenAI GPT, etc.), the gateway uses prompt engineering to simulate tool calling:

- The gateway injects a system prompt with tool definitions and usage instructions
- Models are instructed to output tool calls in `<TOOL_CALL>JSON</TOOL_CALL>` format
- The gateway detects and parses these tool call blocks
- Supports multiple tool calls in a single response

**Configuration Example**:

```json
{
    "model_definitions": {
        "cohere.command-r-plus": {
            "ocid": "ocid1.generativeaimodel.oc1...",
            "api_format": "cohere",
            "max_tokens_key": "max_tokens",
            "temperature": 0.7
        },
        "openai.gpt-5.2-2025-12-11": {
            "ocid": "ocid1.generativeaimodel.oc1...",
            "api_format": "generic",
            "max_tokens_key": "max_completion_tokens",
            "temperature": 1.0
        }
    }
}
```

**Tool Choice Options**:
- `"auto"` (default): Model decides when to use tools
- `"any"` or `"required"`: Must use at least one tool
- `"none"`: Don't use tools
- `{"type": "tool", "name": "tool_name"}`: Force specific tool

**Tool Response Format**:

The gateway converts tool results back to Anthropic format:

```json
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_abc123",
            "content": "The weather in Tokyo is 22°C and sunny."
        }
    ]
}
```

**Implementation Details**:

- **Native (Cohere)**: Uses `CohereChatRequest` with `tools` and `tool_results` parameters
- **Simulated (Generic)**: Injects system prompt with tool definitions, detects `<TOOL_CALL>` blocks in model output
- **Streaming**: Buffers text during streaming to detect tool calls, sends clean text + tool call events
- **Multi-tool**: Supports detecting and parsing multiple `<TOOL_CALL>` blocks in one response

#### 3. Thinking Mode (Extended Thinking)

Enable the model to show its reasoning process:

```json
{
    "model": "claude-3-5-sonnet-20241022",
    "thinking": {
        "type": "enabled",
        "budget_tokens": 16000
    },
    "messages": [
        {"role": "user", "content": "Solve this complex problem step by step."}
    ]
}
```

#### 4. Sampling Parameters

Control the model's output generation:

```json
{
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "max_tokens": 4096,
    "messages": [...]
}
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| `temperature` | 0.0 - 1.0 | Controls randomness (lower = more focused) |
| `top_k` | > 0 | Limit to top K most probable tokens |
| `top_p` | 0.0 - 1.0 | Nucleus sampling threshold |
| `max_tokens` | > 0 | Maximum tokens to generate |

#### 5. Stop Sequences

Define custom stop strings to end generation:

```json
{
    "model": "claude-3-5-sonnet-20241022",
    "stop_sequences": ["\n\n", "END", "###"],
    "messages": [...]
}
```

#### 6. Images / Vision

Send images for analysis (base64 encoded):

```json
{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
                    }
                }
            ]
        }
    ]
}
```

#### 7. Prompt Caching

Cache frequently used prompts to reduce costs:

```json
{
    "system": [
        {
            "type": "text",
            "text": "You are a helpful assistant with extensive knowledge...",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is a large document to analyze...",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
}
```

**Response includes cache metrics**:
```json
{
    "usage": {
        "input_tokens": 1500,
        "output_tokens": 200,
        "cache_creation_input_tokens": 1200,
        "cache_read_input_tokens": 300
    }
}
```

#### 8. Metadata

Attach custom metadata to requests for tracking:

```json
{
    "model": "claude-3-5-sonnet-20241022",
    "metadata": {
        "user_id": "usr_12345",
        "conversation_id": "conv_abc789",
        "request_id": "req_xyz456"
    },
    "messages": [...]
}
```

**Response echoes metadata**:
```json
{
    "type": "message",
    "content": [...],
    "metadata": {
        "user_id": "usr_12345",
        "conversation_id": "conv_abc789"
    }
}
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Create a message (streaming & non-streaming) |
| `/v1/messages/count_tokens` | POST | Count tokens in request |

### Token Counting

Estimate tokens before making a request:

```bash
curl -X POST http://localhost:8001/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

Response:
```json
{
  "type": "usage",
  "input_tokens": 12
}
```

## Architecture

**Single-file design**: All logic is in `oci-anthropic-gateway.py`

**Main components**:

1. **Configuration loading** (lines 16-39): Loads `config.json` and OCI SDK config

2. **`generate_oci_stream()` function** (lines 58-167): Core streaming logic
   - Builds OCI ChatDetails request
   - Handles dynamic parameter adaptation (max_tokens key, temperature)
   - Calls OCI GenAI API
   - Converts OCI streaming events to Anthropic SSE format
   - Emits proper Anthropic event sequence: `message_start` -> `content_block_start` -> `content_block_delta` (text chunks) -> `content_block_stop` -> `message_delta` -> `message_stop` -> `[DONE]`

3. **`@app.post("/{path:path}")` catch-all route** (lines 172-230):
   - Handles telemetry/token-count stubs
   - Processes `/v1/messages` requests
   - Model alias resolution (maps requested model to OCI config)
   - Message format conversion (Anthropic -> OCI)
   - Returns streaming response with proper SSE headers

**Key design patterns**:
- Model lookup tries aliases first, then direct match in definitions, falls back to default
- Content handling supports both string and structured content formats
- Error handling yields SSE error events for client compatibility

## Adding New Models

Edit `config.json`:
1. Add entry to `model_definitions` with the OCI model OCID
2. Optionally add alias in `model_aliases` to map Anthropic names
3. Set `max_tokens_key` based on model (check OCI docs)
4. Set `temperature` if model requires specific value

## Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `oci`: Oracle Cloud Infrastructure SDK (GenAI inference client)
- `httpx`: HTTP client (not currently used but listed)
- `python-dotenv`: Environment variable loading (not currently used)
