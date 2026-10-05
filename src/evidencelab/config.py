"""Validated, nonsecret configuration and atomic TOML persistence."""

import os
import re
import tempfile
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

import tomli_w
from platformdirs import user_config_path
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class ConfigError(Exception):
    """An actionable configuration error, safe to display."""


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    COMPATIBLE = "compatible"


class Depth(StrEnum):
    QUICK = "quick"
    BALANCED = "balanced"
    THOROUGH = "thorough"


class Reranker(StrEnum):
    NONE = "none"
    JEV = "jev"


KEY_ENV = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
    Provider.COMPATIBLE: "EVIDENCE_CUSTOM_API_KEY",
}

ModelID = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^\S+$")]
EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
ProfileName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]


def validate_endpoint(value: str, *, allow_loopback_http: bool = False) -> str:
    """Reject embedded credentials and insecure remote endpoints."""
    value = value.strip().rstrip("/")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ValueError("Endpoint must be a valid HTTP(S) URL") from None
    if (
        not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or re.search(r"\s|[\x00-\x1f\x7f]", value)
        or (port is not None and port == 0)
    ):
        raise ValueError("Endpoint must have a host and no credentials, query, or fragment")
    local_http = (
        allow_loopback_http
        and parts.scheme == "http"
        and parts.hostname in {"localhost", "127.0.0.1", "::1"}
    )
    if parts.scheme != "https" and not local_http:
        raise ValueError("Use HTTPS; HTTP is allowed only for a loopback custom endpoint")
    return value


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ModelProfile(SettingsModel):
    provider: Provider
    model: ModelID
    base_url: str | None = None
    api_key_env: EnvName | None = None
    api_key_required: bool = True
    depth: Depth = Depth.THOROUGH
    reranker: Reranker = Reranker.NONE
    jev_model: ModelID = "jev-latest"

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        if self.provider == Provider.COMPATIBLE:
            if not self.base_url:
                raise ValueError("A custom provider requires base_url")
            self.base_url = validate_endpoint(self.base_url, allow_loopback_http=True)
        elif self.base_url is not None or not self.api_key_required:
            raise ValueError("Custom base URLs and no-key authentication require compatible")
        if self.api_key_env is None:
            self.api_key_env = KEY_ENV[self.provider]
        return self


class Services(SettingsModel):
    elasticsearch_url: str | None = None
    elasticsearch_api_key_env: EnvName = "ELASTICSEARCH_API_KEY"
    inference_id: str | None = None
    tavily_api_key_env: EnvName = "TAVILY_API_KEY"
    typesafe_api_key_env: EnvName = "TYPESAFE_API_KEY"

    @model_validator(mode="after")
    def validate_elasticsearch(self) -> Self:
        if self.elasticsearch_url:
            self.elasticsearch_url = validate_endpoint(self.elasticsearch_url)
        return self


class AppConfig(SettingsModel):
    schema_version: Literal[1] = 1
    default_profile: ProfileName | None = None
    profiles: dict[ProfileName, ModelProfile] = Field(default_factory=dict)
    services: Services = Field(default_factory=Services)

    @model_validator(mode="after")
    def validate_default(self) -> Self:
        if self.default_profile is not None and self.default_profile not in self.profiles:
            raise ValueError("Default profile must refer to a saved profile")
        return self


def config_path() -> Path:
    override = os.getenv("EVIDENCE_CONFIG_FILE")
    return (
        Path(override).expanduser() if override else user_config_path("evidencelab") / "config.toml"
    )


def resolve_services(config: AppConfig) -> Services:
    data = config.services.model_dump()
    for variable, setting in [
        ("ELASTICSEARCH_URL", "elasticsearch_url"),
        ("EVIDENCE_INFERENCE_ID", "inference_id"),
    ]:
        value = os.getenv(variable, "").strip()
        if value:
            data[setting] = value
    return Services.model_validate(data)


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (tomllib.TOMLDecodeError, UnicodeError):
        raise ConfigError(
            "Configuration is not valid TOML; fix it before saving changes."
        ) from None
    except OSError:
        raise ConfigError("Cannot read configuration; check the path and permissions.") from None
    return AppConfig.model_validate(data)


def save_config(path: Path, config: AppConfig) -> None:
    validated = AppConfig.model_validate(config.model_dump())
    payload = tomli_w.dumps(validated.model_dump(mode="json", exclude_none=True))
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        raise ConfigError("Cannot save configuration; check the path and permissions.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_profile(
    config: AppConfig,
    name: str | None = None,
    *,
    provider: Provider | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> tuple[str, ModelProfile]:
    selected = name or config.default_profile
    if selected is None or selected not in config.profiles:
        raise ConfigError("Select a saved profile or run 'evidence configure' first.")
    data = config.profiles[selected].model_dump()
    if provider is not None and provider != data["provider"]:
        if not model:
            raise ConfigError("Changing provider requires an explicit --model.")
        data.update(provider=provider, base_url=None, api_key_env=None, api_key_required=True)
    if model is not None:
        data["model"] = model
    if base_url is not None:
        data["base_url"] = base_url
    return selected, ModelProfile.model_validate(data)
