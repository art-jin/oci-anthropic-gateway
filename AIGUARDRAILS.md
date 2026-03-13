# AI Guardrails Guide

This document describes the Guardrails support added to the OCI-Anthropic Gateway, including configuration, runtime behavior, testing, scenarios, and troubleshooting.

## 1. What Was Added

> Concept boundary (SDK-verified): OCI Guardrails (via `ApplyGuardrails`) does **not** expose a customer-configurable blocklist/regex interface in the current OCI Python SDK (SDK API Version: 20231130). `GuardrailConfigs` only includes content moderation / prompt injection / PII configs, and `GuardrailsResults` only returns those result types. The `local_blocklist` described below is a **gateway-local** regex/substring matcher, not an OCI Guardrails feature.

The gateway now supports OCI `ApplyGuardrails` in two places:

1. Input Guardrails
   - Runs before the request is sent to OCI chat inference
   - Supports content moderation
   - Supports prompt injection detection
   - Supports PII detection
   - Supports an optional local blocklist

2. Output Guardrails
   - Runs after OCI returns a non-streaming response
   - Supports content moderation
   - Supports PII detection
   - Supports text redaction or masking

Current scope:

1. Only text content is checked
2. Images and video are not sent to Guardrails in this version
3. Output Guardrails apply only to non-streaming responses
4. When output guardrails are enabled, streaming requests can either:
   - be rejected
   - or be downgraded to non-streaming JSON

## 2. SDK Requirement

Prompt injection support requires a recent OCI Python SDK.

Project requirement:

```text
oci>=2.164.0
```

Why this matters:

1. Older SDK versions do not expose `PromptInjectionConfiguration`
2. If the runtime uses an older SDK, input prompt injection checks will fail

Recommended check:

```bash
.venv/bin/python -c "import oci; print(oci.__version__)"
```

Recommended capability check:

```bash
.venv/bin/python -c "import oci.generative_ai_inference.models as m; print(hasattr(m, 'PromptInjectionConfiguration'))"
```

Expected result:

```text
True
```

## 3. Files Changed

Core code:

1. `src/utils/guardrails.py`
2. `src/config/__init__.py`
3. `src/routes/handlers.py`
4. `src/services/generation.py`

Config and templates:

1. `config.json.template`
2. `guardrails/blocklist.txt.template`
3. `.gitignore`

Tests:

1. `test/test_guardrails.py`
2. `test/08_guardrails_input_block.py`
3. `test/09_guardrails_output_redact.py`
4. `test/10_guardrails_stream_downgrade.py`

## 4. Request Flow

### 4.1 Input Flow

1. Request arrives at `src/routes/handlers.py`
2. Request payload is validated
3. Model routing is resolved
4. Input text is collected for guardrails
5. OCI `ApplyGuardrails` is called
6. If blocked:
   - return HTTP error in `block` mode
   - only log in `inform` mode
7. If allowed:
   - continue with OCI chat inference

Input coverage:

1. User text
2. Optional system text
3. Tool result text if `include_tool_results=true`

### 4.2 Output Flow

1. Non-streaming OCI response is received in `src/services/generation.py`
2. Text blocks are extracted
3. OCI `ApplyGuardrails` is called for output
4. If output moderation is triggered in `block` mode:
   - return an Anthropic-style error response
5. If output PII is detected and `action=redact` or `mask`:
   - rewrite only text blocks
6. Return Anthropic-compatible JSON

### 4.3 Streaming Flow

If `guardrails.output.enabled=true` and a request uses `stream=true`, the gateway applies `streaming_behavior`:

1. `reject`
   - return 400

2. `downgrade_to_non_stream`
   - run the request through the non-stream path
   - return `application/json`
   - add `metadata.guardrails_stream_downgraded=true`

## 5. Configuration Reference

Example:

```json
{
  "guardrails": {
    "enabled": true,
    "mode": "block",
    "default_language": "en",
    "config_dir": "guardrails",
    "block_http_status": 400,
    "block_message": "Request blocked by guardrails policy.",
    "streaming_behavior": "downgrade_to_non_stream",
    "log_details": true,
    "redact_logs": true,
    "input": {
      "enabled": true,
      "fail_mode": "closed",
      "include_system": false,
      "include_tool_results": true,
      "content_moderation": {
        "enabled": false,
        "categories": ["OVERALL"]
      },
      "prompt_injection": {
        "enabled": true,
        "threshold": 0.95
      },
      "pii": {
        "enabled": false,
        "types": ["EMAIL"]
      },
      "local_blocklist": {
        "enabled": false,
        "file": "blocklist.txt"
      }
    },
    "output": {
      "enabled": true,
      "fail_mode": "open",
      "content_moderation": {
        "enabled": false,
        "categories": ["OVERALL"]
      },
      "pii": {
        "enabled": true,
        "types": ["EMAIL"],
        "action": "redact",
        "placeholder": "[REDACTED]"
      }
    }
  }
}
```

### 5.1 Top-Level Fields

1. `enabled`
   - Master switch

2. `mode`
   - `block`
   - `inform`

3. `default_language`
   - Passed into OCI Guardrails

4. `config_dir`
   - Directory used for runtime files such as blocklist
   - Resolved relative to `config.json`

5. `block_http_status`
   - HTTP status returned when blocked

6. `block_message`
   - Safe message returned to the client

7. `streaming_behavior`
   - `reject`
   - `downgrade_to_non_stream`

8. `log_details`
   - Whether to log guardrails summaries

9. `redact_logs`
   - Whether to remove sensitive text from log summaries

### 5.2 Input Fields

1. `enabled`
2. `fail_mode`
   - `open`
   - `closed`
3. `include_system`
4. `include_tool_results`
5. `content_moderation.enabled`
6. `content_moderation.categories`
7. `content_moderation.threshold`
8. `prompt_injection.enabled`
9. `prompt_injection.threshold`
10. `pii.enabled`
11. `pii.types`
12. `local_blocklist.enabled`
13. `local_blocklist.file`

### 5.3 Output Fields

1. `enabled`
2. `fail_mode`
3. `content_moderation.enabled`
4. `content_moderation.categories`
5. `content_moderation.threshold`
6. `pii.enabled`
7. `pii.types`
8. `pii.action`
   - `none`
   - `redact`
   - `mask`
9. `pii.placeholder`

## 6. Prompt Injection Threshold

`prompt_injection.threshold` defines the score threshold used to decide whether OCI's prompt injection score counts as a hit.

Current rule:

```text
score >= threshold  => triggered
```

Interpretation:

1. Lower threshold
   - more sensitive
   - more blocking
   - more false positives

2. Higher threshold
   - less sensitive
   - fewer false positives
   - more false negatives

Examples:

1. `threshold = 0.5`
   - aggressive

2. `threshold = 0.95`
   - conservative

Recommended rollout:

1. Start with `mode="inform"`
2. Observe scores in logs
3. Choose a threshold
4. Switch to `block`

## 7. Local Blocklist

To enable a local blocklist:

1. Copy the template

```bash
cp guardrails/blocklist.txt.template guardrails/blocklist.txt
```

2. Add one rule per line

```text
secret project name
regex:\b\d{4}-\d{4}-\d{4}-\d{4}\b
```

3. Enable it in `config.json`

```json
{
  "guardrails": {
    "input": {
      "local_blocklist": {
        "enabled": true,
        "file": "blocklist.txt"
      }
    }
  }
}
```

Behavior:

1. Exact substring match is case-insensitive
2. `regex:` entries are interpreted as regular expressions
3. If the file is missing, the gateway logs a warning

## 8. Output Text Normalization

During this work, one response formatting bug was fixed:

Problem:

1. OCI Generic non-stream responses can contain SDK `TextContent` objects
2. If these objects are converted with `str(...)`, they become JSON-looking text such as:

```json
{"text":"Hi","type":"TEXT"}
```

Fix:

1. The gateway now extracts `.text` from OCI SDK objects
2. Anthropic clients now receive plain text again

This matters because:

1. Anthropic clients expect text content blocks, not OCI object dumps
2. Guardrails output rewriting should operate on plain text

## 9. Test Scripts

### 9.1 Unit Tests

Run:

```bash
pytest -q test/test_guardrails.py
```

Covers:

1. Input text extraction
2. Blocklist matching
3. PII redaction
4. Path resolution
5. Stream downgrade routing
6. Output redaction integration
7. OCI `TextContent` plain-text extraction

### 9.2 End-to-End Scripts

Run from repo root:

```bash
python3 test/08_guardrails_input_block.py
python3 test/09_guardrails_output_redact.py
python3 test/10_guardrails_stream_downgrade.py
```

These use:

1. `GATEWAY_BASE_URL`, default `http://localhost:8000`
2. `GATEWAY_MODEL`, default `default_model` from `config.json`

### 9.3 What Each Script Verifies

1. `08_guardrails_input_block.py`
   - sends a prompt injection-style request
   - expects a 400 block

2. `09_guardrails_output_redact.py`
   - exercises output PII redaction
   - expects the response body to contain `[REDACTED]`

3. `10_guardrails_stream_downgrade.py`
   - sends `stream=true`
   - expects a downgraded `application/json` response
   - expects `metadata.guardrails_stream_downgraded=true`

## 10. Recommended Scenarios

### 10.1 Safe Rollout

Use:

1. `mode = "inform"`
2. `input.prompt_injection.enabled = true`
3. high threshold such as `0.9` or `0.95`
4. `output.enabled = false`

Purpose:

1. learn the score distribution
2. avoid blocking production traffic too early

### 10.2 Input-First Protection

Use:

1. `mode = "block"`
2. `input.prompt_injection.enabled = true`
3. `input.fail_mode = "closed"`
4. `output.enabled = false`

Purpose:

1. protect the model from dangerous inputs
2. keep output flow simple

### 10.3 Output PII Protection

Use:

1. `output.enabled = true`
2. `output.pii.enabled = true`
3. `output.pii.action = "redact"`
4. `streaming_behavior = "downgrade_to_non_stream"`

Purpose:

1. prevent email or contact details from leaving the gateway

### 10.4 Tool-Rich Workflows

Keep:

1. `input.include_tool_results = true`

Reason:

1. external tool output is a prompt-injection risk
2. tool results are passed back into later model turns

## 11. Common Problems

### 11.1 `PromptInjectionConfiguration` AttributeError

Symptom:

```text
AttributeError: module 'oci.generative_ai_inference.models' has no attribute 'PromptInjectionConfiguration'
```

Cause:

1. runtime OCI SDK is too old

Fix:

```bash
.venv/bin/pip install -U "oci>=2.164.0"
```

Then restart the gateway.

### 11.2 Output Tests Keep Returning 400

Cause:

1. input prompt injection blocks the test request before output guardrails run

Fix options:

1. temporarily set `mode="inform"`
2. temporarily disable `input.prompt_injection.enabled`
3. raise `input.prompt_injection.threshold`

### 11.3 Anthropic Client Receives JSON-Looking Text

Symptom:

```json
{
  "text": "4",
  "type": "TEXT"
}
```

Cause:

1. older gateway process still running code before the `TextContent` normalization fix

Fix:

1. restart the gateway after updating code

### 11.4 Stream Request No Longer Returns SSE

Cause:

1. `guardrails.output.enabled=true`
2. `streaming_behavior="downgrade_to_non_stream"`

This is expected.

## 12. Operational Notes

1. Keep `redact_logs=true` in production
2. Prefer `fail_mode=closed` for input, `fail_mode=open` for output
3. Validate behavior with `08/09/10` after any config change
4. If you change SDK versions, verify the runtime interpreter and virtualenv match

## 13. Quick Checklist

1. OCI SDK upgraded
2. `config.json` updated
3. `guardrails/blocklist.txt` created if needed
4. gateway restarted
5. `08/09/10` executed
6. Anthropic client returns plain text content again
