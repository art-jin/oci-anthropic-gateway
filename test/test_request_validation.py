from src.utils.request_validation import (
    validate_messages_payload,
    validate_system_payload,
    collect_requested_modalities,
    validate_model_modalities,
)


def test_validate_messages_payload_max_tokens_range():
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}
    err = validate_messages_payload(body)
    assert err is not None
    assert "max_tokens" in err


def test_validate_messages_payload_temperature_range():
    body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 2.1}
    err = validate_messages_payload(body)
    assert err == "'temperature' must be between 0.0 and 2.0"


def test_validate_messages_payload_messages_limit():
    body = {"messages": [{"role": "user", "content": "x"}] * 3}
    err = validate_messages_payload(body, max_messages=2)
    assert err == "'messages' exceeds limit (2)"


def test_validate_messages_payload_image_block_valid():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "Zm9v",
                        },
                    },
                ],
            }
        ]
    }
    err = validate_messages_payload(body)
    assert err is None


def test_collect_requested_modalities_text_and_image():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "http://x"}}]},
    ]
    assert collect_requested_modalities(messages) == {"text", "images"}


def test_collect_requested_modalities_includes_system():
    messages = [{"role": "user", "content": "hello"}]
    system = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "Zm8="}}]
    assert collect_requested_modalities(messages, system) == {"text", "images"}


def test_validate_model_modalities_reject_missing_images():
    err = validate_model_modalities({"text", "images"}, {"text"})
    assert err is not None
    assert "not supported" in err


def test_validate_model_modalities_keep_text_path():
    err = validate_model_modalities({"text"}, {"text"})
    assert err is None


def test_validate_model_modalities_cohere_reject_non_text():
    err = validate_model_modalities({"text", "images"}, {"text", "images"}, is_cohere=True)
    assert err is not None
    assert "Cohere" in err


def test_validate_messages_payload_reject_image_url():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "http://example.com/a.png"},
                    }
                ],
            }
        ]
    }
    err = validate_messages_payload(body)
    assert err is not None
    assert "must be 'base64'" in err


def test_validate_system_payload_reject_non_text_block():
    err = validate_system_payload([{"type": "image"}])
    assert err is not None
    assert "only supports 'text'" in err
