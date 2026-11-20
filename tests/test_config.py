import pathlib

import pytest

from app.config import PLACEHOLDER_KEY, Settings


@pytest.mark.parametrize("value", ["", "   ", PLACEHOLDER_KEY, f"  {PLACEHOLDER_KEY}\n"])
def test_missing_or_placeholder_key_means_not_configured(value: str) -> None:
    assert Settings(_env_file=None, openrouter_api_key=value).openrouter_api_key is None


def test_real_key_is_kept_secret() -> None:
    s = Settings(_env_file=None, openrouter_api_key="sk-or-v1-abc123")
    assert s.openrouter_api_key is not None
    assert s.openrouter_api_key.get_secret_value() == "sk-or-v1-abc123"
    assert "abc123" not in repr(s)  # SecretStr never prints the value


def test_placeholder_matches_env_example() -> None:
    example = pathlib.Path(__file__).resolve().parent.parent / ".env.example"
    assert f"OPENROUTER_API_KEY={PLACEHOLDER_KEY}" in example.read_text()
