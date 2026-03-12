import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import oci

from src.routes.handlers import handle_messages_request
from src.services.generation import generate_oci_non_stream
from src.utils.guardrails import (
    GuardrailsCheckResult,
    GuardrailsConfig,
    InputGuardrailsConfig,
    OutputGuardrailsConfig,
    PIIConfig,
    build_guardrails_config,
    check_local_blocklist,
    collect_input_text_for_guardrails,
    redact_pii_text,
)


def test_collect_input_text_for_guardrails_includes_tool_results():
    cfg = GuardrailsConfig(
        enabled=True,
        input=InputGuardrailsConfig(include_system=True, include_tool_results=True),
    )
    body = {
        "system": [{"type": "text", "text": "system rule"}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "user asks"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "tool says"}]},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "Zm9v"}},
                ],
            }
        ],
    }

    text = collect_input_text_for_guardrails(body, cfg)

    assert "system rule" in text
    assert "user asks" in text
    assert "tool says" in text
    assert "Zm9v" not in text


def test_check_local_blocklist_supports_plain_and_regex():
    matches = check_local_blocklist(
        "The project Falcon leaked card 1234-5678-9012-3456.",
        {"falcon", "regex:\\b\\d{4}-\\d{4}-\\d{4}-\\d{4}\\b"},
    )
    assert "falcon" in matches
    assert "regex:\\b\\d{4}-\\d{4}-\\d{4}-\\d{4}\\b" in matches


def test_redact_pii_text_replaces_entities_from_back_to_front():
    entities = [
        {"text": "alice@example.com", "offset": 8, "length": 17},
        {"text": "13900000000", "offset": 26, "length": 11},
    ]

    redacted = redact_pii_text(
        "Contact alice@example.com 13900000000",
        entities,
        action="redact",
        placeholder="[REDACTED]",
    )
    masked = redact_pii_text(
        "Contact alice@example.com 13900000000",
        entities,
        action="mask",
        placeholder="[REDACTED]",
    )

    assert redacted == "Contact [REDACTED] [REDACTED]"
    assert masked == "Contact ***************** ***********"


def test_build_guardrails_config_resolves_blocklist_relative_to_config(tmp_path: Path):
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    guardrails_dir = tmp_path / "guardrails"
    guardrails_dir.mkdir()
    (guardrails_dir / "blocklist.txt").write_text("secret\n", encoding="utf-8")

    cfg = build_guardrails_config(
        {
            "enabled": True,
            "config_dir": "guardrails",
            "input": {
                "local_blocklist": {
                    "enabled": True,
                    "file": "blocklist.txt",
                }
            },
        },
        config_file_path=str(config_file),
    )

    assert cfg.enabled is True
    assert cfg.config_dir == guardrails_dir.resolve()
    assert cfg.input.local_blocklist.resolved_path == (guardrails_dir / "blocklist.txt").resolve()
    assert "secret" in cfg.input.local_blocklist.entries


def test_handle_messages_request_rejects_stream_with_output_guardrails():
    guardrails = GuardrailsConfig(
        enabled=True,
        streaming_behavior="reject",
        input=InputGuardrailsConfig(enabled=False),
        output=OutputGuardrailsConfig(enabled=True),
    )

    class AppConfig:
        messages_max_items = 200
        genai_client = object()
        debug = False
        debug_redact_media = True

        def __init__(self, guardrails_config):
            self.guardrails = guardrails_config

        @staticmethod
        def get_model_config(_req_model: str):
            return {
                "api_format": "generic",
                "model_types": ["text"],
                "compartment_id": "ocid1.compartment.oc1..example",
                "ocid": "ocid1.generativeaimodel.oc1..example",
            }

    body = {
        "model": "demo",
        "stream": True,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    }

    response = asyncio.run(handle_messages_request(body, "demo", AppConfig(guardrails)))

    assert response.status_code == 400
    assert b"Streaming responses are not supported" in response.body


def test_handle_messages_request_downgrades_stream_to_non_stream():
    guardrails = GuardrailsConfig(
        enabled=True,
        streaming_behavior="downgrade_to_non_stream",
        input=InputGuardrailsConfig(enabled=False),
        output=OutputGuardrailsConfig(enabled=True),
    )

    class AppConfig:
        messages_max_items = 200
        genai_client = object()
        debug = False
        debug_redact_media = True

        def __init__(self, guardrails_config):
            self.guardrails = guardrails_config

        @staticmethod
        def get_model_config(_req_model: str):
            return {
                "api_format": "generic",
                "model_types": ["text"],
                "compartment_id": "ocid1.compartment.oc1..example",
                "ocid": "ocid1.generativeaimodel.oc1..example",
            }

    body = {
        "model": "demo",
        "stream": True,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    }

    async def fake_generate_oci_non_stream(
        oci_messages,
        params,
        message_id,
        model_conf,
        requested_model,
        genai_client,
        cohere_messages=None,
        **kwargs,
    ):
        payload = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "downgraded"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "metadata": params.get("metadata"),
        }
        from fastapi.responses import JSONResponse

        return JSONResponse(content=payload)

    with patch("src.routes.handlers.generate_oci_non_stream", fake_generate_oci_non_stream):
        response = asyncio.run(handle_messages_request(body, "demo", AppConfig(guardrails)))

    data = json.loads(response.body)
    assert response.status_code == 200
    assert body["stream"] is False
    assert data["metadata"]["guardrails_stream_downgraded"] is True
    assert data["content"][0]["text"] == "downgraded"


def test_generate_oci_non_stream_redacts_output_text_when_block_mode():
    chat_response = SimpleNamespace(
        message=SimpleNamespace(content="Email alice@example.com"),
    )
    fake_response = SimpleNamespace(data=SimpleNamespace(chat_response=chat_response))

    class FakeClient:
        def chat(self, _chat_detail):
            return fake_response

    guardrails = GuardrailsConfig(
        enabled=True,
        mode="block",
        output=OutputGuardrailsConfig(
            enabled=True,
            pii=PIIConfig(enabled=True, types=["EMAIL"], action="redact", placeholder="[REDACTED]"),
        ),
    )

    async def fake_apply_output_guardrails(**_kwargs):
        return GuardrailsCheckResult(
            passed=True,
            issue_detected=True,
            redacted_content="Email [REDACTED]",
        )

    with patch("src.services.generation.apply_output_guardrails", fake_apply_output_guardrails):
        response = asyncio.run(
            generate_oci_non_stream(
                oci_messages=[],
                params={"max_tokens": 32},
                message_id="msg_test",
                model_conf={
                    "api_format": "generic",
                    "max_tokens_key": "max_tokens",
                    "temperature": 0.7,
                    "compartment_id": "ocid1.compartment.oc1..example",
                    "ocid": "ocid1.generativeaimodel.oc1..example",
                },
                requested_model="demo",
                genai_client=FakeClient(),
                guardrails_config=guardrails,
            )
        )

    assert response.status_code == 200
    assert b"Email [REDACTED]" in response.body


def test_generate_oci_non_stream_extracts_plain_text_from_oci_textcontent():
    chat_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[oci.generative_ai_inference.models.TextContent(text="Hi there")]
                )
            )
        ]
    )
    fake_response = SimpleNamespace(data=SimpleNamespace(chat_response=chat_response))

    class FakeClient:
        def chat(self, _chat_detail):
            return fake_response

    response = asyncio.run(
        generate_oci_non_stream(
            oci_messages=[],
            params={"max_tokens": 32},
            message_id="msg_textcontent",
            model_conf={
                "api_format": "generic",
                "max_tokens_key": "max_tokens",
                "temperature": 0.7,
                "compartment_id": "ocid1.compartment.oc1..example",
                "ocid": "ocid1.generativeaimodel.oc1..example",
            },
            requested_model="demo",
            genai_client=FakeClient(),
            guardrails_config=None,
        )
    )

    data = json.loads(response.body)
    assert response.status_code == 200
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hi there"
