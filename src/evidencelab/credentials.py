"""Credential references; secrets never enter the TOML configuration."""

import hashlib
import os
from dataclasses import dataclass, field

import keyring
from keyring.errors import KeyringError

from evidencelab.config import ConfigError, ModelProfile

SERVICE = "evidencelab"


@dataclass(frozen=True)
class Credential:
    value: str | None = field(repr=False)
    source: str


def profile_slot(name: str, profile: ModelProfile) -> str:
    endpoint = hashlib.sha256((profile.base_url or "").encode()).hexdigest()[:16]
    return f"profile:{name}:{profile.provider}:{endpoint}"


def service_slot(name: str, endpoint: str = "") -> str:
    suffix = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    return f"service:{name}:{suffix}"


def read_credential(env_name: str, slot: str) -> Credential:
    value = os.getenv(env_name, "").strip()
    if value:
        return Credential(value, "environment")
    try:
        stored = keyring.get_password(SERVICE, slot)
    except (KeyringError, RuntimeError):
        return Credential(None, "keyring-unavailable")
    return Credential(stored or None, "keyring" if stored else "missing")


def store_credential(slot: str, value: str) -> None:
    if not value.strip():
        raise ConfigError("API key cannot be empty.")
    try:
        keyring.set_password(SERVICE, slot, value.strip())
    except (KeyringError, RuntimeError):
        raise ConfigError(
            "OS credential storage is unavailable. Use an environment variable instead."
        ) from None
