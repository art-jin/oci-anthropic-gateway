# OCI-Anthropic Gateway

[中文文档](README_CN.md) | English

A production-ready translation layer that enables OCI GenAI models to work seamlessly with Anthropic's API format, including comprehensive tool calling support.

## Overview

This gateway acts as a bridge between Oracle Cloud Infrastructure (OCI) GenAI services and Anthropic's API format, allowing you to:

1. **Use OCI-hosted models** (Grok, GPT variants, Cohere Command-R, etc.) with Anthropic API clients
2. **Enable tool calling** for models that don't natively support it through advanced prompt engineering
3. **Stream responses** in Anthropic's Server-Sent Events (SSE) format
4. **Access advanced features** like prompt caching, vision, extended thinking, and more

**Key Features:**
- ✅ Full Anthropic Messages API compatibility
- ✅ Enhanced tool calling support (native + simulated)
- ✅ Streaming and non-streaming responses
- ✅ Prompt caching support
- ✅ Vision/image analysis
- ✅ Modular, maintainable codebase
- ✅ Production-ready with comprehensive error handling

## Quick Start

### Prerequisites

- Python 3.12+
- OCI account with GenAI service access
- OCI CLI configured (`~/.oci/config`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd oci-anthropic-gateway

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Copy the configuration template:**
   ```bash
   cp config.json.template config.json
   ```

2. **Edit `config.json`:**
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
     "default_model": "gpt5"
   }
   ```

3. **Configure OCI SDK** (`~/.oci/config`):
   ```ini
   [DEFAULT]
   user=ocid1.user.oc1...
   fingerprint=aa:bb:cc:dd...
   tenancy=ocid1.tenancy.oc1...
   region=us-chicago-1
   key_file=~/.oci/oci_api_key.pem
   ```

### Running the Gateway

```bash
# Using the modular entry point (recommended)
python main.py

# Or using the legacy single-file version
python oci-anthropic-gateway.py
```

The gateway runs on `0.0.0.0:8001` by default.

## Architecture

The codebase has been refactored from a single 2000+ line file into a modular structure for better maintainability:

```
oci-anthropic-gateway/
├── main.py                      # Modern entry point
├── oci-anthropic-gateway.py     # Legacy single-file (backward compatible)
├── config.json                  # Configuration
├── src/
│   ├── config/                  # Configuration management
│   │   └── __init__.py         # Config class, OCI client initialization
│   ├── utils/                   # Utility modules
│   │   ├── constants.py        # Constants and stop reasons
│   │   ├── token.py            # Token counting utilities
│   │   ├── tools.py            # Tool calling conversion
│   │   ├── cache.py            # Cache control utilities
│   │   ├── json_helper.py      # JSON parsing and fixing
│   │   └── content_converter.py # Content format conversion
│   ├── services/                # Business logic
│   │   └── generation.py       # OCI generation service
│   └── routes/                  # API routes
│       └── handlers.py         # Request handlers
└── requirements.txt
```

### Key Components

| Module | Purpose |
|--------|---------|
| **config/** | Load configuration, initialize OCI clients |
| **utils/constants.py** | Define constants, stop reasons, pre-compiled regex patterns |
| **utils/token.py** | Token counting and estimation |
| **utils/tools.py** | Convert Anthropic tools to OCI/Cohere format, build tool instructions |
| **utils/json_helper.py** | Parse and fix malformed JSON from models |
| **utils/content_converter.py** | Convert between Anthropic, OCI, and Cohere formats |
| **services/generation.py** | Core generation logic (streaming & non-streaming) |
| **routes/handlers.py** | FastAPI route handlers |

## Tool Calling Support

A major focus of this gateway is providing robust tool calling support for OCI GenAI models that lack native function calling capabilities.

### Two Approaches

#### 1. Native Function Calling (Cohere Models)

Models configured with `api_format: "cohere"` use OCI's native function calling:

```json
{
  "model": "cohere.command-r-plus",
  "tools": [{
    "name": "get_weather",
    "description": "Get current weather",
    "input_schema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }],
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"}
  ]
}
```

#### 2. Simulated Tool Calling (Generic Models)

For models with `api_format: "generic"` (xAI Grok, OpenAI GPT, etc.), the gateway uses advanced prompt engineering:

**How it works:**
1. Injects detailed system prompt with tool definitions and usage instructions
2. Models output tool calls in `<TOOL_CALL>JSON</TOOL_CALL>` format
3. Gateway detects, parses, and converts to Anthropic format
4. Supports multiple tool calls in one response

**Example model output:**
```
<TOOL_CALL>
{"name": "get_weather", "input": {"location": "Tokyo"}}
</TOOL_CALL>
```

**Gateway converts to:**
```json
{
  "content": [{
    "type": "tool_use",
    "id": "toolu_abc123",
    "name": "get_weather",
    "input": {"location": "Tokyo"}
  }],
  "stop_reason": "tool_use"
}
```

### Enhanced JSON Parsing

The gateway includes sophisticated JSON parsing to handle malformed model outputs:

**Fixes applied:**
- Missing quotes around keys/values
- Single quotes → double quotes
- Trailing commas
- Unquoted property names
- Incomplete JSON objects
- Embedded JSON in text

**Example:**
```javascript
// Model outputs (malformed):
{name: "search", input: {query: 'test',}}

// Gateway fixes to:
{"name": "search", "input": {"query": "test"}}
```

### Natural Language Fallback

When models don't follow the exact `<TOOL_CALL>` format, the gateway has fallback detection:

**Patterns detected:**
- "I'll use the [tool_name] tool..."
- "Calling [tool_name] with..."
- "Let me [tool_name]..."
- Standalone JSON objects with "name" and "input" fields

### Tool Choice Strategies

```json
{
  "tool_choice": "auto"  // Options: auto, required, any, none, {"type": "tool", "name": "..."}
}
```

| Strategy | Behavior |
|----------|----------|
| `auto` (default) | Model decides when to use tools |
| `required`/`any` | Must use at least one tool |
| `none` | Don't use tools |
| `{"type": "tool", "name": "..."}` | Force specific tool |

### Tool Result Format

The gateway converts tool results to the Anthropic format with clear XML-like markers:

```xml
<TOOL_RESULT id='toolu_xxx' status='success'>
{"temperature": "22°C", "condition": "sunny"}
</TOOL_RESULT>
```

Or for errors:
```xml
<TOOL_RESULT id='toolu_xxx' status='error'>
Error message here
</TOOL_RESULT>
```

## API Features

### Supported Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Create messages (streaming & non-streaming) |
| `/v1/messages/count_tokens` | POST | Count tokens in request |

### Messages API Features

#### 1. System Prompts

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

#### 2. Streaming

```bash
curl -X POST http://localhost:8001/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Emits Anthropic-compatible SSE events:
- `message_start`
- `content_block_start`
- `content_block_delta`
- `content_block_stop`
- `message_delta`
- `message_stop`

#### 3. Vision / Images

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

#### 4. Extended Thinking

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 16000
  },
  "messages": [...]
}
```

#### 5. Prompt Caching

```json
{
  "system": [{
    "type": "text",
    "text": "Long system prompt...",
    "cache_control": {"type": "ephemeral"}
  }]
}
```

Response includes cache metrics:
```json
{
  "usage": {
    "input_tokens": 1500,
    "cache_creation_input_tokens": 1200,
    "cache_read_input_tokens": 300,
    "output_tokens": 200
  }
}
```

#### 6. Sampling Parameters

```json
{
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "max_tokens": 4096,
  "stop_sequences": ["\n\n", "END"]
}
```

#### 7. Metadata

```json
{
  "metadata": {
    "user_id": "usr_12345",
    "conversation_id": "conv_abc789"
  }
}
```

Echoed in response.

#### 8. Token Counting

```bash
curl -X POST http://localhost:8001/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Response:
```json
{
  "type": "usage",
  "input_tokens": 12
}
```

## Configuration Options

### Model Configuration

```json
{
  "model_definitions": {
    "model-key": {
      "ocid": "ocid1.generativeaimodel.oc1...",
      "api_format": "generic",  // or "cohere"
      "max_tokens_key": "max_completion_tokens",  // or "max_tokens"
      "temperature": 1.0
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `ocid` | OCI model OCID |
| `api_format` | `"generic"` (simulated tools) or `"cohere"` (native tools) |
| `max_tokens_key` | Parameter name: `"max_tokens"` or `"max_completion_tokens"` |
| `temperature` | Fixed temperature (overrides request) |

### Model Aliases

Map Anthropic model names to your OCI models:

```json
{
  "model_aliases": {
    "claude-3-5-sonnet-20241022": "gpt5",
    "claude-3-opus-20240229": "cohere.command-r-plus"
  }
}
```

## Logging and Debugging

The gateway provides flexible logging with multiple levels to control verbosity.

### Log Levels

Control log output using the `LOG_LEVEL` environment variable:

```bash
# Minimal logging (recommended for production)
LOG_LEVEL=WARNING python main.py

# Balanced logging (default, recommended for development)
LOG_LEVEL=INFO python main.py

# Verbose logging (for debugging issues)
LOG_LEVEL=DEBUG python main.py
```

### What Gets Logged

**WARNING (Minimal):**
- Configuration loaded
- Server status
- Critical errors only

**INFO (Default - Recommended):**
- Request summaries (stream, tool count)
- Tool call detection results
- Key operational events
- Empty tool result handling

**DEBUG (Verbose):**
- Detailed parameter settings
- Content conversion details
- JSON parsing attempts
- OCI SDK internal requests

### Example Logs

**INFO level (clean output):**
```
INFO:oci-gateway:Config loaded. Default model: xai.grok-code-fast-1
INFO:oci-gateway:OCI SDK initialized successfully
INFO:oci-gateway:Detected tool call: Bash with 1 parameters
INFO:oci-gateway:Tool detection result: found 1 tool calls
```

**DEBUG level (verbose output):**
```
INFO:oci-gateway:Config loaded. Default model: xai.grok-code-fast-1
DEBUG:oci-gateway:Added system prompt (150 chars)
DEBUG:oci-gateway:Set top_p: 0.9
INFO:oci-gateway:Detected tool call: Bash with 1 parameters
DEBUG:oci-gateway:Raw JSON string: {"name": "Bash",...
```

### Key Features

- **Auto-suppression of OCI SDK logs**: At INFO and WARNING levels, verbose OCI SDK internal logs are hidden
- **Environment variable control**: No code changes needed, just set `LOG_LEVEL`
- **Clean format**: `LEVEL:logger_name:message`

For complete logging documentation, see [LOGGING.md](LOGGING.md).

## Troubleshooting

### Tool Calls Not Detected

**Check:**
1. Review logs for tool detection messages
2. Verify tool definition format
3. Try explicit tool names in user message
4. Simplify tool descriptions
5. Test with `tool_choice: "required"`

### JSON Parsing Errors

**Check:**
1. Review raw model output in logs
2. Check for special characters
3. Verify JSON structure
4. Review `fix_json_issues` logs

### WebSearch Returns Empty

This is an issue with the Anthropic client environment, not the gateway:
- Check API keys/credentials
- Verify network access
- Try different search queries
- Test with other tools (Glob, Bash, etc.)

### Model Not Following Tool Format

**Solutions:**
1. Switch to a Cohere model for native support
2. Use more explicit prompts: "Use the ListFiles tool to..."
3. Set `tool_choice: "required"`
4. Check model compatibility

## Performance Considerations

- **Pre-compiled regex**: Patterns compiled at module level
- **Streaming efficiency**: Tool detection after complete text received
- **Token optimization**: Simplified instructions for 10+ tools
- **Caching**: Consider caching tool definition conversions

## Adding New Models

1. Get the OCI model OCID from OCI Console
2. Add to `model_definitions` in `config.json`:
   ```json
   {
     "my-model": {
       "ocid": "ocid1.generativeaimodel.oc1...",
       "api_format": "generic",  // or "cohere"
       "max_tokens_key": "max_tokens",
       "temperature": 0.7
     }
   }
   ```
3. Optionally add alias:
   ```json
   {
     "model_aliases": {
       "claude-3-5-sonnet-20241022": "my-model"
     }
   }
   ```

## Dependencies

```
fastapi>=0.104.0
uvicorn>=0.24.0
oci>=2.119.0
httpx>=0.25.0
python-dotenv>=1.0.0
```

## Security Notes

- `config.json` is in `.gitignore` - never commit it
- Use OCI IAM policies to restrict model access
- Consider implementing authentication for production
- Monitor token usage and costs

## Migration from Single-File Version

The original `oci-anthropic-gateway.py` remains functional for backward compatibility. To migrate:

1. **Test the new version:**
   ```bash
   python main.py
   ```

2. **Update any scripts** that reference the old file

3. **Keep using old version** if needed - both work identically

## Future Enhancements

Planned improvements:
- [ ] ML-driven tool call detection
- [ ] Adaptive prompt optimization
- [ ] Tool usage statistics and analytics
- [ ] Enhanced error recovery
- [ ] Performance monitoring and tracing

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the Apache License 2.0.

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check logs for detailed error information
- Review TOOLS_SUPPORT.md for tool calling details
- Review REFACTORING.md for architecture details

---

**Last Updated**: 2026-01-24  
**Version**: 2.0 (Modular Architecture with Enhanced Tool Support)
