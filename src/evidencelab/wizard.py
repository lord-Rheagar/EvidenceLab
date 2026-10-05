"""Interactive setup; collect changes before committing credentials or configuration."""

import os
from dataclasses import dataclass, field

import typer

from evidencelab.config import (
    KEY_ENV,
    AppConfig,
    Depth,
    ModelProfile,
    Provider,
    Reranker,
    Services,
    validate_endpoint,
)
from evidencelab.credentials import profile_slot, read_credential, service_slot
from evidencelab.providers import MODEL_CATALOGS


@dataclass(frozen=True)
class PendingSecret:
    slot: str
    value: str = field(repr=False)


def prompt_choice(label: str, choices: list[str], default: object = None) -> str:
    while True:
        value = str(typer.prompt(f"{label} ({', '.join(choices)})", default=default))
        if value in choices:
            return value
        typer.echo(f"Choose one of: {', '.join(choices)}", err=True)


def ask_credential(label: str, env_name: str, slot: str) -> list[PendingSecret]:
    credential = read_credential(env_name, slot)
    typer.echo(f"{label}: {credential.source}; environment variable: {env_name}")
    method = prompt_choice(
        "Credential source",
        ["environment", "keyring", "keep"],
        default="keep" if credential.value else "environment",
    )
    if method != "keyring":
        return []
    if os.getenv(env_name, "").strip():
        typer.echo(f"{env_name} takes precedence over the stored key while it is set.")
    value = typer.prompt("API key (stored in OS credential manager)", hide_input=True)
    while not value.strip():
        value = typer.prompt("API key cannot be empty", hide_input=True)
    return [PendingSecret(slot, value.strip())]


def ask_endpoint(label: str, default: str = "", *, custom: bool = False) -> str:
    while True:
        value = typer.prompt(label, default=default)
        try:
            return validate_endpoint(value, allow_loopback_http=custom)
        except ValueError as exc:
            typer.echo(str(exc), err=True)


def collect_profile(name: str, seed: dict[str, object]) -> tuple[ModelProfile, list[PendingSecret]]:
    typer.echo("Providers: openai, anthropic, gemini, openrouter, compatible")
    provider = Provider(
        prompt_choice(
            "Provider",
            [p.value for p in Provider],
            default=seed.get("provider"),
        )
    )
    typer.echo(f"Model catalog: {MODEL_CATALOGS[provider]}")
    same_provider = seed.get("provider") == provider
    model = typer.prompt("Model ID", default=seed.get("model") if same_provider else None)
    base_url = None
    api_key_required = True
    if provider == Provider.COMPATIBLE:
        base_url = ask_endpoint(
            "API base URL (include /v1 when required)",
            str(seed.get("base_url") or os.getenv("EVIDENCE_CUSTOM_BASE_URL", "")),
            custom=True,
        )
        api_key_required = typer.confirm(
            "Does this endpoint require an API key?",
            default=bool(seed.get("api_key_required", True)),
        )
    key_env = typer.prompt(
        "API key environment variable",
        default=(seed.get("api_key_env") if same_provider else None) or KEY_ENV[provider],
    )
    depth = Depth(
        prompt_choice(
            "Research depth",
            [d.value for d in Depth],
            default=seed.get("depth", Depth.THOROUGH),
        )
    )
    reranker = Reranker(
        prompt_choice(
            "Evidence reranker",
            [r.value for r in Reranker],
            default=seed.get("reranker", Reranker.NONE),
        )
    )
    profile = ModelProfile(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=key_env,
        api_key_required=api_key_required,
        depth=depth,
        reranker=reranker,
        jev_model=str(seed.get("jev_model", "jev-latest")),
    )
    secrets = (
        ask_credential("Model credential", key_env, profile_slot(name, profile))
        if api_key_required
        else []
    )
    return profile, secrets


def collect_services(
    services: Services, reranker: Reranker
) -> tuple[Services, list[PendingSecret]]:
    services = services.model_copy(deep=True)
    secrets: list[PendingSecret] = []
    typer.echo(
        "Live research sends indexed passages to Elastic Cloud and selected passages to your "
        "model provider, including supplied local documents. Enabling Jev also sends passages "
        "to TypeSafe."
    )
    if typer.confirm("Configure Elastic Cloud now?", default=False):
        services.elasticsearch_url = ask_endpoint(
            "Elasticsearch URL",
            str(services.elasticsearch_url or os.getenv("ELASTICSEARCH_URL", "")),
        )
        services.inference_id = (
            typer.prompt(
                "Embedding inference endpoint ID (optional)", default=services.inference_id or ""
            )
            or None
        )
        secrets.extend(
            ask_credential(
                "Elasticsearch credential",
                services.elasticsearch_api_key_env,
                service_slot("elasticsearch", services.elasticsearch_url),
            )
        )
    if typer.confirm("Configure a Tavily search key now?", default=False):
        secrets.extend(
            ask_credential("Search credential", services.tavily_api_key_env, service_slot("tavily"))
        )
    if reranker == Reranker.JEV:
        secrets.extend(
            ask_credential(
                "Jev credential", services.typesafe_api_key_env, service_slot("typesafe")
            )
        )
    return Services.model_validate(services.model_dump()), secrets


def with_profile(
    config: AppConfig, name: str, profile: ModelProfile, services: Services, *, make_default: bool
) -> AppConfig:
    return AppConfig(
        profiles={**config.profiles, name: profile},
        services=services,
        default_profile=name
        if make_default or not config.default_profile
        else config.default_profile,
    )
