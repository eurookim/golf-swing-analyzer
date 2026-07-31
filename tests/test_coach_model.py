"""Tests for model selection and credential loading in golfswing.coach."""

import importlib

import pytest

from golfswing import coach


class TestModelSelection:
    def test_defaults_to_haiku(self, monkeypatch):
        monkeypatch.delenv("GOLFSWING_MODEL", raising=False)
        importlib.reload(coach)
        assert coach.MODEL == "claude-haiku-4-5"

    def test_an_environment_variable_overrides_it(self, monkeypatch):
        monkeypatch.setenv("GOLFSWING_MODEL", "claude-opus-5")
        importlib.reload(coach)
        try:
            assert coach.MODEL == "claude-opus-5"
        finally:
            monkeypatch.delenv("GOLFSWING_MODEL")
            importlib.reload(coach)


class TestRequestParams:
    def test_haiku_gets_no_effort_or_adaptive_thinking(self):
        """output_config.effort errors on Haiku 4.5, and adaptive thinking is
        an Opus-family feature. Sending either turns every request into a 400."""
        params = coach.request_params("claude-haiku-4-5")
        assert "output_config" not in params
        assert "thinking" not in params

    def test_opus_gets_adaptive_thinking_and_effort(self):
        params = coach.request_params("claude-opus-5")
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"]["effort"] == "low"

    def test_every_model_carries_a_token_ceiling(self):
        for model in ("claude-haiku-4-5", "claude-opus-5"):
            assert params_of(model)["max_tokens"] > 0

    def test_haiku_stays_under_its_own_output_cap(self):
        """Haiku 4.5 caps at 64K output, unlike the 128K of the Opus family."""
        assert params_of("claude-haiku-4-5")["max_tokens"] <= 64000


def params_of(model):
    return coach.request_params(model)


class TestCredentials:
    def test_a_dotenv_file_supplies_the_key(self, tmp_path, monkeypatch):
        """A Finder-launched .app inherits no shell environment, so a key
        exported in .zshrc never reaches it. The project .env has to work."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")

        coach.load_env_file(env_file)

        import os
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"
        del os.environ["ANTHROPIC_API_KEY"]

    def test_an_already_set_variable_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")

        coach.load_env_file(env_file)

        import os
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"

    def test_comments_and_blank_lines_are_ignored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("# a comment\n\nSOME_KEY = value \n")

        coach.load_env_file(env_file)

        import os
        assert os.environ["SOME_KEY"] == "value"
        del os.environ["SOME_KEY"]

    def test_quotes_are_stripped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUOTED", raising=False)
        (tmp_path / ".env").write_text('QUOTED="sk-ant-quoted"\n')

        coach.load_env_file(tmp_path / ".env")

        import os
        assert os.environ["QUOTED"] == "sk-ant-quoted"
        del os.environ["QUOTED"]

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        coach.load_env_file(tmp_path / "nope.env")   # must not raise
