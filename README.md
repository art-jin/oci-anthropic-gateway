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

- `compartment_id`: OCI compartment OCID
- `endpoint`: OCI GenAI service endpoint (region-specific)
- `model_aliases`: Maps Anthropic model names to OCI model keys
- `model_definitions`: OCI model configurations with:
  - `ocid`: The OCI model OCID to use
  - `max_tokens_key`: Parameter name for token limit (`max_tokens` or `max_completion_tokens`)
  - `temperature`: Fixed temperature (overrides request)
- `default_model`: Fallback model when requested model not found

OCI SDK authentication is loaded from `~/.oci/config` (DEFAULT profile).

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
