import os

import keyring
import pytest
from keyring.errors import NoKeyringError
from pydantic import ValidationError

from evidencelab.config import (
    AppConfig,
    ConfigError,
    ModelProfile,
    Provider,
    Services,
    load_config,
    resolve_profile,
    resolve_services,
    save_config,
)
from evidencelab.credentials import (
    SERVICE,
    profile_slot,
    read_credential,
    store_credential,
)


def test_toml_round_trip_and_profile_override(isolated_settings):
    profile = ModelProfile(provider=Provider.OPENAI, model="a-model")
    config = AppConfig(default_profile="work", profiles={"work": profile})
    save_config(isolated_settings, config)
    restored = load_config(isolated_settings)
    assert restored == config
    _, selected = resolve_profile(restored, model="different-model")
    assert selected.model == "different-model"
    assert load_config(isolated_settings).profiles["work"].model == "a-model"
    _, selected = resolve_profile(restored, provider=Provider.GEMINI, model="custom-gemini")
    assert selected.api_key_env == "GOOGLE_API_KEY"
    assert selected.base_url is None
    with pytest.raises(ConfigError, match="explicit --model"):
        resolve_profile(restored, provider=Provider.GEMINI)


def test_atomic_write_preserves_existing_file_on_failure(isolated_settings, monkeypatch):
    save_config(isolated_settings, AppConfig())
    original = isolated_settings.read_bytes()

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(ConfigError, match="Cannot save"):
        save_config(isolated_settings, AppConfig(services=Services(inference_id="new")))
    assert isolated_settings.read_bytes() == original
    assert not list(isolated_settings.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "url",
    [
        "http://public.example/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "https://example.com/#secret",
        "file:///tmp/api",
        "https://example.com:bad/v1",
    ],
)
def test_invalid_custom_endpoint(url):
    with pytest.raises(ValidationError):
        ModelProfile(provider=Provider.COMPATIBLE, model="m", base_url=url)


def test_custom_and_hosted_endpoint_rules():
    custom = ModelProfile(
        provider=Provider.COMPATIBLE,
        model="local-model",
        base_url="http://localhost:11434/v1/",
        api_key_required=False,
    )
    assert custom.base_url == "http://localhost:11434/v1"
    with pytest.raises(ValidationError):
        ModelProfile(provider=Provider.COMPATIBLE, model="m")
    with pytest.raises(ValidationError):
        ModelProfile(provider=Provider.OPENAI, model="m", base_url="https://example.com")
    with pytest.raises(ValidationError):
        ModelProfile(provider=Provider.OPENAI, model="m", api_key_required=False)


def test_credentials_environment_precedence_and_endpoint_isolation(monkeypatch, vault):
    first = ModelProfile(provider=Provider.COMPATIBLE, model="m", base_url="https://one.example/v1")
    second = ModelProfile(
        provider=Provider.COMPATIBLE, model="m", base_url="https://two.example/v1"
    )
    slot = profile_slot("work", first)
    store_credential(slot, "vault-secret")
    assert vault[(SERVICE, slot)] == "vault-secret"
    assert read_credential(first.api_key_env, slot).source == "keyring"
    assert read_credential(second.api_key_env, profile_slot("work", second)).value is None
    monkeypatch.setenv(first.api_key_env, "environment-secret")
    result = read_credential(first.api_key_env, slot)
    assert result.value == "environment-secret"
    assert "environment-secret" not in repr(result)


def test_keyring_failure_has_no_plaintext_fallback(monkeypatch):
    def fail(*args):
        raise NoKeyringError("internal-secret")

    monkeypatch.setattr(keyring, "get_password", fail)
    monkeypatch.setattr(keyring, "set_password", fail)
    assert read_credential("NO_KEY", "slot").source == "keyring-unavailable"
    with pytest.raises(ConfigError, match="environment variable") as error:
        store_credential("slot", "private-secret")
    assert "secret" not in str(error.value)


def test_services_environment_overrides(monkeypatch):
    config = AppConfig(services=Services(elasticsearch_url="https://old.example"))
    monkeypatch.setenv("ELASTICSEARCH_URL", "https://new.example/")
    monkeypatch.setenv("EVIDENCE_INFERENCE_ID", "chosen-embedding")
    effective = resolve_services(config)
    assert effective.elasticsearch_url == "https://new.example"
    assert effective.inference_id == "chosen-embedding"
    assert config.services.elasticsearch_url == "https://old.example"


@pytest.mark.parametrize("data", [{"schema_version": 2}, {"default_profile": "missing"}])
def test_invalid_config_contract(data):
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_unreadable_config_encoding_is_actionable(isolated_settings):
    isolated_settings.write_bytes(b"\xff\xfeinvalid")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(isolated_settings)
