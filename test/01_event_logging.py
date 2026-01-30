import asyncio
import httpx

from test._common import load_test_config, print_ok


async def main() -> None:
    cfg = load_test_config()
    url = f"{cfg.base_url}/v1/event_logging"

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, json={})

    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert data == {"status": "ok"}, f"unexpected response: {data}"
    print_ok("event_logging")


if __name__ == "__main__":
    asyncio.run(main())
