from src.utils.debug_dump import _truncate_payload


def test_truncate_payload_redacts_media_fields():
    payload = {
        "full_request": {
            "messages": [
                {
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc123",
                                "url": "http://example.com/x.png",
                            },
                        }
                    ]
                }
            ]
        }
    }
    out = _truncate_payload(payload, 1024, True)
    source = out["full_request"]["messages"][0]["content"][0]["source"]
    assert source["data"] == {"redacted": True}
    assert source["url"] == {"redacted": True}


def test_truncate_payload_can_disable_redaction():
    payload = {"source": {"data": "abc123"}}
    out = _truncate_payload(payload, 1024, False)
    assert out["source"]["data"] == "abc123"
