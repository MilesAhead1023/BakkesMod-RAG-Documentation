"""Tests for bakkesmod_rag.setup_keys module."""

import os
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from bakkesmod_rag.setup_keys import (
    mask_key,
    validate_key_format,
    check_missing_keys,
    save_keys,
    ensure_api_keys,
    KEY_DEFINITIONS,
)


class TestMaskKey:
    """Tests for mask_key()."""

    def test_normal_key(self):
        result = mask_key("sk-proj-abcdefgh1234")
        assert result == "sk-...1234"

    def test_short_key(self):
        result = mask_key("abc")
        assert result == "****"

    def test_exactly_four_chars(self):
        result = mask_key("abcd")
        assert result == "****"

    def test_five_chars(self):
        result = mask_key("abcde")
        assert result == "abc...bcde"

    def test_anthropic_key(self):
        result = mask_key("sk-ant-api03-xxxxxxxxxxxxxxxxxxWxYz")
        assert result.endswith("WxYz")
        assert result.startswith("sk-")


class TestValidateKeyFormat:
    """Tests for validate_key_format()."""

    def test_openai_valid(self):
        assert validate_key_format("OPENAI_API_KEY", "sk-proj-abc123") is True

    def test_openai_invalid_prefix(self):
        assert validate_key_format("OPENAI_API_KEY", "not-a-key") is False

    def test_anthropic_valid(self):
        assert validate_key_format("ANTHROPIC_API_KEY", "sk-ant-api03-xxx") is True

    def test_google_valid(self):
        assert validate_key_format("GOOGLE_API_KEY", "AIzaSyD_something") is True

    def test_openrouter_valid(self):
        assert validate_key_format("OPENROUTER_API_KEY", "sk-or-v1-abc") is True

    def test_cohere_any_value(self):
        # Cohere has no prefix requirement
        assert validate_key_format("COHERE_API_KEY", "anything-goes") is True

    def test_unknown_key(self):
        assert validate_key_format("UNKNOWN_KEY", "whatever") is True


class TestCheckMissingKeys:
    """Tests for check_missing_keys()."""

    def test_all_present(self):
        env = {name: "test-value" for name in KEY_DEFINITIONS}
        with patch.dict(os.environ, env, clear=False):
            missing = check_missing_keys()
        # Filter out any that might not be in our patched env
        assert len(missing) == 0 or all(
            name not in env for name in missing
        )

    def test_all_missing(self):
        env_clear = {name: "" for name in KEY_DEFINITIONS}
        with patch.dict(os.environ, env_clear, clear=False):
            # os.environ.get returns "" which is falsy
            missing = check_missing_keys()
        assert len(missing) == len(KEY_DEFINITIONS)

    def test_partial_missing(self):
        env = {"OPENAI_API_KEY": "sk-test123"}
        with patch.dict(os.environ, env, clear=False):
            missing = check_missing_keys()
        assert "OPENAI_API_KEY" not in missing


class TestSaveKeys:
    """Tests for save_keys()."""

    def test_creates_env_file(self, tmp_path):
        dotenv_path = str(tmp_path / ".env")
        save_keys({"OPENAI_API_KEY": "sk-test123"}, dotenv_path)

        assert Path(dotenv_path).exists()
        content = Path(dotenv_path).read_text()
        assert "OPENAI_API_KEY" in content
        assert "sk-test123" in content

    def test_appends_to_existing(self, tmp_path):
        dotenv_path = str(tmp_path / ".env")
        Path(dotenv_path).write_text("EXISTING_VAR='hello'\n")

        save_keys({"OPENAI_API_KEY": "sk-test123"}, dotenv_path)

        content = Path(dotenv_path).read_text()
        assert "EXISTING_VAR" in content
        assert "OPENAI_API_KEY" in content

    def test_sets_environ(self, tmp_path):
        dotenv_path = str(tmp_path / ".env")
        save_keys({"TEST_SETUP_KEY": "test-value"}, dotenv_path)

        assert os.environ.get("TEST_SETUP_KEY") == "test-value"
        # Cleanup
        del os.environ["TEST_SETUP_KEY"]


class TestEnsureApiKeys:
    """Tests for ensure_api_keys()."""

    def test_all_keys_present_skips_prompts(self, tmp_path):
        dotenv_path = str(tmp_path / ".env")
        Path(dotenv_path).touch()
        env = {name: "test-value" for name in KEY_DEFINITIONS}
        with patch.dict(os.environ, env, clear=False):
            result = ensure_api_keys(dotenv_path)
        assert result is True

    def test_required_key_missing_returns_false_on_skip(self, tmp_path):
        dotenv_path = str(tmp_path / ".env")
        Path(dotenv_path).touch()
        # Remove OPENAI_API_KEY, set all others
        env = {name: "test-value" for name in KEY_DEFINITIONS}
        env["OPENAI_API_KEY"] = ""
        with patch.dict(os.environ, env, clear=False):
            # Mock getpass to return empty (user skipping) then KeyboardInterrupt
            with patch("bakkesmod_rag.setup_keys.getpass.getpass", side_effect=KeyboardInterrupt):
                result = ensure_api_keys(dotenv_path)
        assert result is False


class TestKeyDefinitions:
    """Tests for KEY_DEFINITIONS structure."""

    def test_openai_is_required(self):
        required, _, _ = KEY_DEFINITIONS["OPENAI_API_KEY"]
        assert required is True

    def test_optional_keys_not_required(self):
        for name in ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                      "GOOGLE_API_KEY", "COHERE_API_KEY"]:
            required, _, _ = KEY_DEFINITIONS[name]
            assert required is False, f"{name} should be optional"

    def test_all_have_descriptions(self):
        for name, (_, _, desc) in KEY_DEFINITIONS.items():
            assert len(desc) > 10, f"{name} needs a description"
