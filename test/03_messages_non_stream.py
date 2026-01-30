import asyncio
import httpx

from test._common import load_test_config, assert_is_anthropic_message, print_ok


async def main() -> None:
    cfg = load_test_config()
    url = f"{cfg.base_url}/v1/messages"

    payload = {
        "model": cfg.model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hi in one sentence."}],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as client:
        r = await client.post(url, json=payload)

    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:500]}"
    data = r.json()

    assert_is_anthropic_message(data)

    # Minimal sanity: expect at least one text block
    blocks = data.get("content", [])
    assert any(b.get("type") == "text" for b in blocks if isinstance(b, dict)), f"no text block in content: {blocks}"

    print_ok("messages_non_stream")


if __name__ == "__main__":
    asyncio.run(main())
