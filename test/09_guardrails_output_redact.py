import asyncio

import httpx

try:
    from test._common import load_test_config, assert_is_anthropic_message, print_ok
except ModuleNotFoundError:
    from _common import load_test_config, assert_is_anthropic_message, print_ok


async def main() -> None:
    cfg = load_test_config()
    url = f"{cfg.base_url}/v1/messages"

    payload = {
        "model": cfg.model,
        "max_tokens": 64,
        "system": "Reply with exactly this email address and nothing else: support@example.com",
        "messages": [
            {
                "role": "user",
                "content": "Follow the system instruction exactly.",
            }
        ],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as client:
        r = await client.post(url, json=payload)

    assert r.status_code == 200, (
        "expected 200, got "
        f"{r.status_code}: {r.text[:500]}. "
        "If you got 400 here, the live service is blocking this request during input guardrails, "
        "so output redaction could not be reached."
    )
    data = r.json()
    assert_is_anthropic_message(data)

    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    assert "[REDACTED]" in text, f"expected redacted output, got: {text[:500]}"
    assert "support@example.com" not in text, f"email should have been redacted, got: {text[:500]}"

    print_ok("guardrails_output_redact")


if __name__ == "__main__":
    asyncio.run(main())
