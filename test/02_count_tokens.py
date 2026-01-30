import asyncio
import httpx

from test._common import load_test_config, print_ok


async def main() -> None:
    cfg = load_test_config()
    url = f"{cfg.base_url}/v1/count_tokens"

    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "Hello world"}],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload)

    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
    data = r.json()

    assert data.get("type") == "usage", f"expected type=usage, got {data.get('type')}"
    assert isinstance(data.get("input_tokens"), int), f"expected input_tokens int, got {data.get('input_tokens')}"
    assert data["input_tokens"] > 0, f"expected input_tokens > 0, got {data['input_tokens']}"

    print_ok("count_tokens")


if __name__ == "__main__":
    asyncio.run(main())
