import json
import httpx

from test._common import load_test_config, print_ok


def main() -> None:
    cfg = load_test_config()
    url = f"{cfg.base_url}/v1/messages"

    payload = {
        "model": cfg.model,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    }

    saw_message_start = False
    saw_content_delta = False
    saw_done = False

    with httpx.stream(
        "POST",
        url,
        json=payload,
        timeout=httpx.Timeout(10.0, read=180.0),
        headers={"content-type": "application/json"},
    ) as r:
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
        ct = r.headers.get("content-type", "")
        assert "text/event-stream" in ct, f"expected text/event-stream, got {ct}"

        for raw_line in r.iter_lines():
            if raw_line is None or raw_line == b"" or raw_line == "":
                continue
            line = raw_line.decode() if isinstance(raw_line, (bytes, bytearray)) else raw_line

            if line.startswith("event: message_start"):
                saw_message_start = True

            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    saw_done = True
                    break

                # Best effort: ensure JSON lines are parseable
                try:
                    json.loads(data_str)
                except json.JSONDecodeError as e:
                    raise AssertionError(f"Invalid JSON in data line: {data_str[:200]} ({e})")

            if "content_block_delta" in line and "text_delta" in line:
                saw_content_delta = True

    assert saw_message_start, "did not see message_start event"
    assert saw_content_delta, "did not see any text deltas"
    assert saw_done, "did not see [DONE] terminator"

    print_ok("messages_stream_sse")


if __name__ == "__main__":
    main()
