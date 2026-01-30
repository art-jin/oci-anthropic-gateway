import json
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TestConfig:
    base_url: str
    model: str


def load_default_model(config_path: str = "config.json") -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = cfg.get("default_model")
    if not model or not isinstance(model, str):
        raise RuntimeError("config.json missing valid 'default_model'")
    return model


def load_test_config() -> TestConfig:
    base_url = os.environ.get("GATEWAY_BASE_URL", "http://localhost:8000").rstrip("/")
    model = os.environ.get("GATEWAY_MODEL")
    if not model:
        model = load_default_model()
    return TestConfig(base_url=base_url, model=model)


def assert_is_anthropic_message(obj: Dict[str, Any]) -> None:
    assert obj.get("type") == "message", f"expected type=message, got {obj.get('type')}"
    assert obj.get("role") == "assistant", f"expected role=assistant, got {obj.get('role')}"
    assert isinstance(obj.get("content"), list), "expected content to be a list"

    usage = obj.get("usage")
    assert isinstance(usage, dict), "expected usage dict"
    assert isinstance(usage.get("input_tokens"), int), "expected usage.input_tokens int"
    assert isinstance(usage.get("output_tokens"), int), "expected usage.output_tokens int"


def print_ok(name: str) -> None:
    print(f"OK: {name}")
