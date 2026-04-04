# AGENTS.md

## Architecture Overview
- Thin HTTP Layer: Catch-all POST route in `main.py` routes `messages` to `src/routes/handlers.py:handle_messages_request`.
- Modular Structure: `src/config/` loads JSON; `src/utils/` conversions; `src/services/generation.py` OCI calls; `src/routes/` validation.
- Translation Core: Anthropic to OCI GenAI, model routing, content conversion, tool calling, streaming, guardrails.
- Two Response Paths: Generic simulates tools via `<TOOL_CALL>{"name": "tool_name", "input": {...}}</TOOL_CALL>`; Cohere uses native OCI tools.
- Debug System: `debug=true` writes dumps to `debug_dumps/`; `debug_ui.enabled=true` serves swimlane UI at `/debug/`.

## Critical Workflows
- Setup: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cp config.json.template config.json`
- Run Locally: `python main.py`; `LOG_LEVEL=INFO python main.py`; `LOG_LEVEL=DEBUG python main.py`
- Run Tests: `pytest`; `PYTHONPATH=. python -m test.03_messages_non_stream`
- Container: `docker build -t oci-anthropic-gateway:latest . && docker run -d -p 8000:8000 -v $(pwd)/config.json:/app/config.json:ro -v $(pwd)/docker-oci:/root/.oci:ro oci-anthropic-gateway:latest`
- Kubernetes: Use `k8s/deployment-workload-identity.yaml` for OKE or `deployment-with-secrets.yaml` for secrets.

## Project Conventions
- Model Routing: `model_aliases` map Anthropic names (e.g., `"claude-3-5-sonnet-20241022": "gpt5"`); `model_definitions` specify `ocid`, `api_format` ("generic" or "cohere"), `max_tokens_key`, `model_types`.
- Tool Calling: Generic injects system prompt from `src/utils/tools.py:_build_tool_use_instruction()`; detects `<TOOL_CALL>` via `src/utils/json_helper.py:detect_all_tool_call_blocks()`.
- Streaming: Buffer text for tool detection when tools present; emit SSE after parsing.
- Config Normalization: `src/config/__init__.py` applies fallbacks and validates at startup.
- OCI Auth: Auto-detects OKE workload identity > resource principal > local API key.
- Guardrails: Input pre-OCI; output non-stream only; streaming downgrades/rejects based on config.
- Error Handling: Tool parsing robust to `</TOOL_CALL>` in JSON by extracting first balanced object.

## Key Files
- `main.py`: Entry point, FastAPI app, catch-all route.
- `src/routes/handlers.py`: Validation, conversion, guardrails.
- `src/services/generation.py`: OCI calls, tool detection.
- `src/config/__init__.py`: Config loading, OCI init.
- `src/utils/tools.py`: Tool instructions, conversions.
- `src/utils/json_helper.py`: JSON fixing, detection.
- `config.json`: Model defs, aliases, settings.
