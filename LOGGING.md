# Logging Configuration

## Log Levels

The gateway supports the following log levels:

- **ERROR**: Only errors
- **WARNING**: Warnings and errors
- **INFO**: Key events, warnings, and errors (default)
- **DEBUG**: Detailed debugging information

## Controlling Log Output

### Method 1: Environment Variable

Set the `LOG_LEVEL` environment variable before starting the gateway:

```bash
# Minimal logging (recommended for production)
export LOG_LEVEL=WARNING
python main.py

# Default logging (balanced)
export LOG_LEVEL=INFO
python main.py

# Verbose logging (for debugging)
export LOG_LEVEL=DEBUG
python main.py
```

### Method 2: One-line Command

```bash
# Minimal logging
LOG_LEVEL=WARNING python main.py

# Default logging
LOG_LEVEL=INFO python main.py

# Verbose logging
LOG_LEVEL=DEBUG python main.py
```

## What Gets Logged at Each Level

### WARNING (Minimal)
- Critical errors
- Configuration issues
- Tool call parsing failures

**Example output:**
```
INFO:oci-gateway:Config loaded. Default model: xai.grok-code-fast-1
INFO:oci-gateway:OCI SDK initialized successfully
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### INFO (Default - Recommended)
- All WARNING level logs
- Request summaries (stream, tools count)
- Tool call detection results
- Key operational events

**Example output:**
```
INFO:oci-gateway:Config loaded. Default model: xai.grok-code-fast-1
INFO:oci-gateway:OCI SDK initialized successfully
INFO:oci-gateway:REQ stream=True tools=17 tool_choice=None
INFO:oci-gateway:Detected tool call: Bash with 1 parameters
INFO:oci-gateway:Tool detection result: found 1 tool calls
```

### DEBUG (Verbose)
- All INFO level logs
- Detailed parameter settings
- Content conversion details
- JSON parsing attempts
- OCI SDK requests

**Example output:**
```
INFO:oci-gateway:Config loaded. Default model: xai.grok-code-fast-1
DEBUG:oci-gateway:Added system prompt (150 chars)
DEBUG:oci-gateway:Added message 0 (42 chars)
DEBUG:oci-gateway:Set top_p: 0.9
DEBUG:oci-gateway:Raw JSON string after strip: '{"name": "Bash",...'
INFO:oci-gateway:Detected tool call: Bash with 1 parameters
```

## Suppressing OCI SDK Logs

The gateway automatically suppresses verbose OCI SDK logging at WARNING and INFO levels. Only at DEBUG level will you see OCI SDK internal logs like:

```
INFO:oci.base_client: Request: POST https://inference.generativeai...
WARNING:oci.base_client: Received SSE response, returning an SSE client
```

## Recommended Settings

**Development:** `LOG_LEVEL=INFO` (balanced, shows key events)  
**Production:** `LOG_LEVEL=WARNING` (minimal noise)  
**Debugging:** `LOG_LEVEL=DEBUG` (detailed diagnostics)

## Log Format

All logs follow this format:
```
LEVEL:logger_name:message
```

Example:
```
INFO:oci-gateway:Detected tool call: Bash with 1 parameters
```

## Programmatic Control

You can also control logging programmatically in your code:

```python
from src.utils.logging_config import set_log_level

# Set to WARNING for minimal logging
set_log_level("WARNING")

# Set to DEBUG for verbose logging
set_log_level("DEBUG")
```
