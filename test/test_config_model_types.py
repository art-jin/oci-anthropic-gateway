import pytest

from src.config import Config


def test_normalize_model_types_default_text():
    assert Config._normalize_model_types(None, "m1") == ["text"]


def test_normalize_model_types_invalid_value_raises():
    with pytest.raises(ValueError):
        Config._normalize_model_types(["text", "pdf"], "m2")
