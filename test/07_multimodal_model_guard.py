import asyncio
import json
import os

import httpx

from test._common import load_test_config, print_ok


def _pick_text_only_model(config_path: str = "config.json") -> str:
    override = os.environ.get("TEXT_ONLY_MODEL", "").strip()
    if override:
        return override

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_defs = cfg.get("model_definitions", {})
    if not isinstance(model_defs, dict):
        raise RuntimeError("config.json missing valid model_definitions")

    for model_name, model_conf in model_defs.items():
        if not isinstance(model_conf, dict):
            continue
        model_types = model_conf.get("model_types", ["text"])
        if not isinstance(model_types, list):
            continue
        normalized = {str(x).strip().lower() for x in model_types if str(x).strip()}
        if normalized == {"text"}:
            return str(model_name)

    raise RuntimeError("No text-only model found. Set TEXT_ONLY_MODEL=<model_name> to run this test.")


async def main() -> None:
    cfg = load_test_config()
    url = f"{cfg.base_url}/v1/messages"
    text_model = _pick_text_only_model()

    payload = {
        "model": text_model,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgQXkM0QAAAAASUVORK5CYII=",
                        },
                    },
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as client:
        r = await client.post(url, json=payload)

    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:500]}"
    data = r.json()
    assert data.get("type") == "error", f"expected error response, got: {data}"
    err = data.get("error", {})
    assert err.get("type") == "invalid_request_error", f"unexpected error.type: {err}"
    msg = str(err.get("message", ""))
    assert "not supported" in msg and "model_types" in msg, f"unexpected error.message: {msg}"

    print_ok("multimodal_model_guard")


if __name__ == "__main__":
    asyncio.run(main())
