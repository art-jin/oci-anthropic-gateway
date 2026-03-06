from src.utils.request_validation import validate_messages_payload


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
