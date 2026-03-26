# AI Guardrails Guide

This document describes the Guardrails support in OCI-Anthropic Gateway, with a strict separation between:

1. **OCI-native Guardrails** — features provided by OCI GenAI `ApplyGuardrails`
2. **Gateway-local Guardrails extensions** — enforcement, transport, logging, and response-rewrite behavior implemented by this gateway

## 1. Capability Boundary

### 1.1 OCI-native Guardrails

The OCI Python SDK `ApplyGuardrails` API currently exposes these detector families:

1. Content moderation
2. Prompt injection detection
3. PII detection

In this project, those map to:

- `guardrails.oci_native.input.content_moderation`
- `guardrails.oci_native.input.prompt_injection`
- `guardrails.oci_native.input.pii_detection`
- `guardrails.oci_native.output.content_moderation`
- `guardrails.oci_native.output.pii_detection`

### 1.2 Gateway-local Guardrails extensions

The following are **not** OCI-native Guardrails features. They are implemented by this gateway:

| Layer | Config path | Role | Current capabilities |
|-------|-------------|------|----------------------|
| Gateway-local policy | `guardrails.gateway_policy.*` | Decide how the gateway reacts to findings or Guardrails failures | - `mode`<br>- `block_http_status`<br>- `block_message`<br>- `log_details`<br>- `redact_logs`<br>- `input_failure_mode`<br>- `output_failure_mode` |
| Gateway-local extensions | `guardrails.gateway_extensions.*` | Add behavior OCI Guardrails does not provide directly. The current gateway already implements request/response collection rules, local matching, output rewrite, and stream handling; the same extension layer can also be used for future gateway-only controls that OCI does not natively expose, such as per-tenant/per-route/per-model policies, finer-grained output rewrite rules, audit/alert integrations, local custom rule engines, or multi-step approval / human-review workflows. | - Include/exclude `system`<br>- Include/exclude `tool_result`<br>- Local blocklist<br>- Output PII rewrite<br>- Stream reject/downgrade behavior |

> Important: OCI Guardrails does **not** expose a customer-configurable blocklist/regex interface in the current OCI Python SDK. `local_blocklist` is a gateway-local matcher, not an OCI Guardrails feature.

## 2. SDK Requirement

Prompt injection support requires a recent OCI Python SDK.

Project requirement:

```text
oci>=2.164.0
```

Why this matters:

1. Older SDK versions do not expose `PromptInjectionConfiguration`
2. If the runtime uses an older SDK, OCI-native prompt injection checks will fail

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

## 3. Configuration Schema

Example:

```json
{
  "guardrails": {
    "enabled": true,
    "config_dir": "guardrails",
    "oci_native": {
      "default_language": "en",
      "input": {
        "enabled": true,
        "content_moderation": {
          "enabled": true,
          "categories": ["OVERALL"],
          "threshold": 0.5
        },
        "prompt_injection": {
          "enabled": true,
          "threshold": 0.95
        },
        "pii_detection": {
          "enabled": false,
          "types": ["EMAIL"]
        }
      },
      "output": {
        "enabled": true,
        "content_moderation": {
          "enabled": false,
          "categories": ["OVERALL"],
          "threshold": 0.5
        },
        "pii_detection": {
          "enabled": true,
          "types": ["EMAIL"]
        }
      }
    },
    "gateway_policy": {
      "mode": "block",
      "block_http_status": 400,
      "block_message": "Request blocked by gateway guardrails policy.",
      "log_details": true,
      "redact_logs": true,
      "input_failure_mode": "closed",
      "output_failure_mode": "open"
    },
    "gateway_extensions": {
      "input": {
        "include_system": false,
        "include_tool_results": true,
        "local_blocklist": {
          "enabled": false,
          "file": "blocklist.txt"
        }
      },
      "output": {
        "pii_rewrite": {
          "enabled": true,
          "action": "redact",
          "placeholder": "[REDACTED]"
        }
      },
      "streaming": {
        "when_output_guardrails_enabled": "downgrade_to_non_stream"
      }
    }
  }
}
```

## 4. Configuration Reference

### 4.1 Top-level

1. `guardrails.enabled`
   - master switch for all guardrails behavior

2. `guardrails.config_dir`
   - directory used for runtime files such as local blocklist
   - resolved relative to `config.json`

### 4.2 OCI-native Guardrails

#### `guardrails.oci_native.default_language`
- passed into OCI `ApplyGuardrails`

#### `guardrails.oci_native.input`
1. `enabled`
2. `content_moderation.enabled`
3. `content_moderation.categories`
4. `content_moderation.threshold`
5. `prompt_injection.enabled`
6. `prompt_injection.threshold`
7. `pii_detection.enabled`
8. `pii_detection.types`

#### `guardrails.oci_native.output`
1. `enabled`
2. `content_moderation.enabled`
3. `content_moderation.categories`
4. `content_moderation.threshold`
5. `pii_detection.enabled`
6. `pii_detection.types`

### 4.3 Gateway policy

#### `guardrails.gateway_policy`
1. `mode`
   - `block`
   - `inform`

2. `block_http_status`
   - HTTP status returned when the gateway blocks a request/response

3. `block_message`
   - safe message returned by the gateway

4. `log_details`
   - whether to log detailed guardrails summaries

5. `redact_logs`
   - whether to remove sensitive text from summaries

6. `input_failure_mode`
   - `open`
   - `closed`

7. `output_failure_mode`
   - `open`
   - `closed`

### 4.4 Gateway extensions

#### `guardrails.gateway_extensions.input`
1. `include_system`
2. `include_tool_results`
3. `local_blocklist.enabled`
4. `local_blocklist.file`

#### `guardrails.gateway_extensions.output.pii_rewrite`
1. `enabled`
2. `action`
   - `redact`
   - `mask`
3. `placeholder`

#### `guardrails.gateway_extensions.streaming.when_output_guardrails_enabled`
1. `reject`
2. `downgrade_to_non_stream`

## 5. Request Flow

### 5.1 Input flow

1. Request arrives at `src/routes/handlers.py`
2. Request payload is validated
3. Model routing is resolved
4. Gateway-local input collection runs
   - user text is always included
   - system text is included only if `gateway_extensions.input.include_system=true`
   - tool result text is included only if `gateway_extensions.input.include_tool_results=true`
5. Gateway-local blocklist runs if enabled
6. OCI-native `ApplyGuardrails` runs using `guardrails.oci_native.input.*`
7. Gateway policy decides whether to:
   - block (`gateway_policy.mode=block`)
   - or only log (`gateway_policy.mode=inform`)
8. If allowed, request continues to OCI chat inference

### 5.2 Output flow

1. Non-streaming OCI response is received in `src/services/generation.py`
2. Text blocks are extracted
3. OCI-native `ApplyGuardrails` runs using `guardrails.oci_native.output.*`
4. Gateway policy decides whether moderation findings should block the response
5. If OCI-native output PII detection returns entities and `gateway_extensions.output.pii_rewrite.enabled=true`, the gateway rewrites only text blocks
6. Anthropic-compatible JSON is returned

### 5.3 Streaming flow

If `guardrails.oci_native.output.enabled=true` and a request uses `stream=true`, the gateway uses:

- `guardrails.gateway_extensions.streaming.when_output_guardrails_enabled`

Options:

1. `reject`
   - return 400

2. `downgrade_to_non_stream`
   - route through the non-stream path
   - return `application/json`
   - add `metadata.guardrails_stream_downgraded=true`

## 6. Prompt Injection Threshold

`guardrails.oci_native.input.prompt_injection.threshold` defines the score threshold used by the gateway to interpret OCI's prompt injection score.

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

1. Start with `gateway_policy.mode="inform"`
2. Observe scores in logs
3. Choose a threshold
4. Switch to `block`

## 7. Local Blocklist

To enable the gateway-local blocklist:

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
    "gateway_extensions": {
      "input": {
        "local_blocklist": {
          "enabled": true,
          "file": "blocklist.txt"
        }
      }
    }
  }
}
```

Behavior:

1. Exact substring match is case-insensitive
2. `regex:` entries are interpreted as regular expressions
3. If the file is missing, the gateway logs a warning

## 8. Output Text Rewrite

Gateway output rewrite is separate from OCI-native PII detection.

Flow:

1. OCI-native output PII detection identifies entities
2. The gateway receives offsets/lengths from OCI results
3. If `gateway_extensions.output.pii_rewrite.enabled=true`, the gateway rewrites Anthropic text blocks

This means:

- PII detection is OCI-native
- redaction/masking of Anthropic response text is gateway-local

## 9. Output Text Normalization

OCI Generic non-stream responses can contain SDK `TextContent` objects.
The gateway extracts `.text` so Anthropic clients receive plain text instead of OCI object dumps.

This matters because:

1. Anthropic clients expect text content blocks, not OCI object dumps
2. Gateway-local output rewrite operates on plain text blocks

## 10. Test Scripts

### 10.1 Unit tests

Run:

```bash
pytest -q test/test_guardrails.py
```

Covers:

1. Input text extraction
2. Gateway-local blocklist matching
3. Gateway-local PII rewrite
4. Path resolution
5. Stream downgrade routing
6. Output redaction integration
7. OCI `TextContent` plain-text extraction

### 10.2 End-to-end scripts

Run from repo root:

```bash
python3 test/08_guardrails_input_block.py
python3 test/09_guardrails_output_redact.py
python3 test/10_guardrails_stream_downgrade.py
```

These use:

1. `GATEWAY_BASE_URL`, default `http://localhost:8000`
2. `GATEWAY_MODEL`, default `default_model` from `config.json`

### 10.3 What each script verifies

1. `08_guardrails_input_block.py`
   - sends a prompt injection-style request
   - expects a 400 block

2. `09_guardrails_output_redact.py`
   - exercises OCI-native output PII detection plus gateway-local output rewrite
   - expects the response body to contain `[REDACTED]`

3. `10_guardrails_stream_downgrade.py`
   - sends `stream=true`
   - expects a downgraded `application/json` response
   - expects `metadata.guardrails_stream_downgraded=true`

## 11. Recommended Scenarios

### 11.1 Safe rollout

Use:

1. `gateway_policy.mode = "inform"`
2. `oci_native.input.prompt_injection.enabled = true`
3. high threshold such as `0.9` or `0.95`
4. `oci_native.output.enabled = false`

Purpose:

1. learn the score distribution
2. avoid blocking production traffic too early

### 11.2 Input-first protection

Use:

1. `gateway_policy.mode = "block"`
2. `oci_native.input.prompt_injection.enabled = true`
3. `gateway_policy.input_failure_mode = "closed"`
4. `oci_native.output.enabled = false`

Purpose:

1. protect the model from dangerous inputs
2. keep output flow simple

### 11.3 Output PII protection

Use:

1. `oci_native.output.enabled = true`
2. `oci_native.output.pii_detection.enabled = true`
3. `gateway_extensions.output.pii_rewrite.enabled = true`
4. `gateway_extensions.output.pii_rewrite.action = "redact"`
5. `gateway_extensions.streaming.when_output_guardrails_enabled = "downgrade_to_non_stream"`

Purpose:

1. prevent email or contact details from leaving the gateway

### 11.4 Tool-rich workflows

Keep:

1. `gateway_extensions.input.include_tool_results = true`

Reason:

1. external tool output is a prompt-injection risk
2. tool results are passed back into later model turns

## 12. Common Problems

### 12.1 `PromptInjectionConfiguration` AttributeError

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

### 12.2 Output tests keep returning 400

Cause:

1. OCI-native input prompt injection blocks the test request before output guardrails run

Fix options:

1. temporarily set `gateway_policy.mode="inform"`
2. temporarily disable `oci_native.input.prompt_injection.enabled`
3. raise `oci_native.input.prompt_injection.threshold`

### 12.3 Anthropic client receives JSON-looking text

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

### 12.4 Stream request no longer returns SSE

Cause:

1. `guardrails.oci_native.output.enabled=true`
2. `guardrails.gateway_extensions.streaming.when_output_guardrails_enabled="downgrade_to_non_stream"`

This is expected.

## 13. Operational Notes

1. Keep `gateway_policy.redact_logs=true` in production
2. Prefer `gateway_policy.input_failure_mode=closed` for input and `gateway_policy.output_failure_mode=open` for output
3. Validate behavior with `08/09/10` after any config change
4. If you change SDK versions, verify the runtime interpreter and virtualenv match

## 14. Quick Checklist

1. OCI SDK upgraded
2. `config.json` updated to the new schema
3. `guardrails/blocklist.txt` created if needed
4. gateway restarted
5. `08/09/10` executed
6. Anthropic client returns plain text content again
