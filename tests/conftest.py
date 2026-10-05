"""Isolate configuration and credentials from the developer's machine."""

import keyring
import pytest
from typer.testing import CliRunner

from evidencelab.config import KEY_ENV


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("EVIDENCE_CONFIG_FILE", str(path))
    monkeypatch.setenv("EVIDENCE_DATA_DIR", str(tmp_path / "data"))
    for variable in [
        *KEY_ENV.values(),
        "ELASTICSEARCH_URL",
        "ELASTICSEARCH_API_KEY",
        "EVIDENCE_INFERENCE_ID",
        "EVIDENCE_CUSTOM_BASE_URL",
        "TAVILY_API_KEY",
        "TYPESAFE_API_KEY",
    ]:
        monkeypatch.delenv(variable, raising=False)
    return path


@pytest.fixture(autouse=True)
def vault(monkeypatch):
    passwords = {}
    monkeypatch.setattr(
        keyring, "get_password", lambda service, slot: passwords.get((service, slot))
    )
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda service, slot, value: passwords.update({(service, slot): value}),
    )
    return passwords


@pytest.fixture
def runner():
    return CliRunner()
