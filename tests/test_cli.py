import json

import pytest

from evidencelab import cli
from evidencelab.cli import app
from evidencelab.config import load_config


def setup_profile(runner, *extra):
    return runner.invoke(
        app,
        ["configure", "--non-interactive", "--provider", "openai", "--model", "test-model", *extra],
    )


def test_noninteractive_setup_and_profile_selection(runner, isolated_settings):
    assert setup_profile(runner).exit_code == 0
    assert setup_profile(runner, "--profile", "review").exit_code == 0
    assert runner.invoke(app, ["profiles", "use", "review"]).exit_code == 0
    result = runner.invoke(app, ["profiles", "list", "--json"])
    assert result.exit_code == 0
    profiles = json.loads(result.stdout)["profiles"]
    assert [p["name"] for p in profiles if p["default"]] == ["review"]
    result = runner.invoke(app, ["profiles", "show", "--model", "override", "--json"])
    assert json.loads(result.stdout)["model"] == "override"
    assert load_config(isolated_settings).profiles["review"].model == "test-model"


def test_switching_provider_resets_key_reference(runner, isolated_settings):
    assert setup_profile(runner, "--api-key-env", "TEAM_OPENAI_KEY").exit_code == 0
    result = runner.invoke(
        app,
        ["configure", "--non-interactive", "--provider", "anthropic", "--model", "chosen-model"],
    )
    assert result.exit_code == 0
    assert load_config(isolated_settings).profiles["default"].api_key_env == "ANTHROPIC_API_KEY"


def test_wizard_saves_profile_with_defaults(runner, monkeypatch, isolated_settings):
    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    # Provider, model, key env, depth, reranker, credential source, services, save.
    answers = ["gemini", "my-gemini", "", "", "", "environment", "n", "n", "y"]
    result = runner.invoke(app, ["configure"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    profile = load_config(isolated_settings).profiles["default"]
    assert profile.provider == "gemini"
    assert profile.model == "my-gemini"
    assert profile.depth == "thorough"
    assert profile.reranker == "none"


@pytest.mark.parametrize("save", ["y", "n"])
def test_hidden_key_and_cancelled_wizard(runner, monkeypatch, isolated_settings, vault, save):
    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    answers = ["openai", "m", "", "", "", "keyring", "secret-value", "n", "n", save]
    result = runner.invoke(app, ["configure"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    assert "secret-value" not in result.output
    assert bool(vault) == (save == "y")
    assert isolated_settings.exists() == (save == "y")
    if save == "y":
        assert "secret-value" not in isolated_settings.read_text()


def test_non_tty_does_not_prompt(runner, isolated_settings):
    result = runner.invoke(app, ["configure"])
    assert result.exit_code == 2
    assert "--non-interactive" in result.output
    assert not isolated_settings.exists()


def test_bad_config_is_not_overwritten(runner, isolated_settings):
    original = "not valid = [toml"
    isolated_settings.write_text(original)
    result = setup_profile(runner)
    assert result.exit_code == 2
    assert isolated_settings.read_text() == original


def test_json_errors_and_missing_credentials(runner):
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 2
    assert "error" in json.loads(result.stdout)
    assert setup_profile(runner).exit_code == 0
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["model_ready"] is False


def test_offline_doctor_never_invokes_model(runner, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-print")
    monkeypatch.setattr(
        cli, "check_model", lambda *args: pytest.fail("Unexpected inference request")
    )
    assert setup_profile(runner).exit_code == 0
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert "do-not-print" not in result.output
    assert json.loads(result.stdout)["model_check"] == "not-run"


def test_explicit_probe_and_json_stdout(runner, monkeypatch):
    called = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "check_model", lambda *args: called.append(args))
    assert setup_profile(runner).exit_code == 0
    result = runner.invoke(app, ["doctor", "--model-check", "--json"])
    assert result.exit_code == 0
    assert len(called) == 1
    assert json.loads(result.stdout)["model_check"].startswith("passed")
    assert "charges" in result.stderr


def test_config_option_and_compatible_no_auth(runner, tmp_path):
    path = tmp_path / "separate" / "config.toml"
    result = runner.invoke(
        app,
        [
            "--config",
            str(path),
            "configure",
            "--non-interactive",
            "--provider",
            "compatible",
            "--model",
            "custom",
            "--base-url",
            "http://localhost:1234/v1",
            "--no-api-key",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["--config", str(path), "doctor", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["model_credential"] == "not-required"


def test_invalid_provider_and_missing_model(runner, isolated_settings):
    result = runner.invoke(app, ["configure", "--non-interactive", "--provider", "invalid"])
    assert result.exit_code == 2
    assert not isolated_settings.exists()
    result = runner.invoke(app, ["configure", "--non-interactive", "--provider", "openai"])
    assert result.exit_code == 2
    assert "model" in result.output


def test_provider_catalog_needs_no_configuration(runner):
    result = runner.invoke(app, ["models", "providers", "--json"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["providers"]) == 5


def test_noninteractive_edit_preserves_other_settings(runner, isolated_settings):
    assert setup_profile(runner, "--depth", "quick", "--reranker", "jev").exit_code == 0
    result = runner.invoke(app, ["configure", "--non-interactive", "--model", "new-model"])
    assert result.exit_code == 0
    saved = load_config(isolated_settings).profiles["default"]
    assert saved.model == "new-model"
    assert saved.depth == "quick"
    assert saved.reranker == "jev"


def test_wizard_with_elastic_search_and_jev(runner, monkeypatch, isolated_settings, vault):
    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    answers = [
        "openrouter",
        "vendor/model",
        "",
        "balanced",
        "jev",
        "environment",
        "y",
        "https://elastic.example",
        "embedding-id",
        "keyring",
        "elastic-secret",
        "y",
        "keyring",
        "search-secret",
        "keyring",
        "jev-secret",
        "y",
    ]
    result = runner.invoke(app, ["configure"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    config = load_config(isolated_settings)
    assert config.services.elasticsearch_url == "https://elastic.example"
    assert config.services.inference_id == "embedding-id"
    assert config.profiles["default"].reranker == "jev"
    assert len(vault) == 3
    for secret in ("elastic-secret", "search-secret", "jev-secret"):
        assert secret not in result.output
        assert secret not in isolated_settings.read_text()


def test_custom_auth_can_be_reenabled(runner, isolated_settings):
    result = runner.invoke(
        app,
        [
            "configure",
            "--non-interactive",
            "--provider",
            "compatible",
            "--model",
            "m",
            "--base-url",
            "http://localhost:1234/v1",
            "--no-api-key",
        ],
    )
    assert result.exit_code == 0
    assert not load_config(isolated_settings).profiles["default"].api_key_required
    result = runner.invoke(app, ["configure", "--non-interactive", "--api-key-required"])
    assert result.exit_code == 0
    assert load_config(isolated_settings).profiles["default"].api_key_required
