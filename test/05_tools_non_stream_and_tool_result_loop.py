import asyncio

import httpx

from test._common import load_test_config, assert_is_anthropic_message, print_ok


def _extract_first_tool_use_id(resp: dict) -> str:
    for b in resp.get("content", []):
        if isinstance(b, dict) and b.get("type") == "tool_use":
            tool_id = b.get("id")
            if isinstance(tool_id, str) and tool_id:
                return tool_id
    raise AssertionError(f"No tool_use block found in response content: {resp.get('content')}")


async def main() -> None:
    cfg = load_test_config()
    url = f"{cfg.base_url}/v1/messages"

    tools_payload = {
        "model": cfg.model,
        "max_tokens": 128,
        "tool_choice": "required",
        "tools": [
            {
                "name": "calc",
                "description": "Do simple arithmetic",
                "input_schema": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
            }
        ],
        "messages": [{"role": "user", "content": "Compute 12*13. Use the tool."}],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as client:
        r1 = await client.post(url, json=tools_payload)
        assert r1.status_code == 200, f"expected 200, got {r1.status_code}: {r1.text[:500]}"
        resp1 = r1.json()
        assert_is_anthropic_message(resp1)

        tool_id = _extract_first_tool_use_id(resp1)

        r2 = await client.post(
            url,
            json={
                "model": cfg.model,
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": "156",
                                "is_error": False,
                            }
                        ],
                    }
                ],
            },
        )

    assert r2.status_code == 200, f"expected 200, got {r2.status_code}: {r2.text[:500]}"
    resp2 = r2.json()
    assert_is_anthropic_message(resp2)

    text = "".join(
        b.get("text", "") for b in resp2.get("content", []) if isinstance(b, dict) and b.get("type") == "text"
    )
    assert "156" in text, f"expected assistant to reference 156, got: {text[:500]}"

    print_ok("tools_non_stream_and_tool_result_loop")


if __name__ == "__main__":
    asyncio.run(main())
