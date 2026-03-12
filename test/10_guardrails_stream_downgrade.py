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
        "stream": True,
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

    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:500]}"
    ct = r.headers.get("content-type", "")
    assert "application/json" in ct, f"expected downgrade to JSON response, got content-type={ct}"
    assert "text/event-stream" not in ct, f"expected non-stream downgrade, got content-type={ct}"

    data = r.json()
    assert_is_anthropic_message(data)

    metadata = data.get("metadata") or {}
    assert metadata.get("guardrails_stream_downgraded") is True, f"expected downgrade marker, got: {metadata}"
    assert isinstance(data.get("content"), list), f"expected content list, got: {data}"

    print_ok("guardrails_stream_downgrade")


if __name__ == "__main__":
    asyncio.run(main())
