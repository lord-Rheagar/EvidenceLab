"""Staged graphical configuration; no writes until the final Save action."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evidencelab.config import (
    KEY_ENV,
    AppConfig,
    ConfigError,
    Depth,
    ModelProfile,
    Provider,
    Reranker,
    Services,
    load_config,
    save_config,
    validate_endpoint,
)
from evidencelab.credentials import profile_slot, read_credential, service_slot, store_credential
from evidencelab.providers import MODEL_CATALOGS
from evidencelab.terminal import Back, Terminal
from evidencelab.wizard import PendingSecret, with_profile


def identifier(value: str) -> bool | str:
    return (bool(value.strip()) and not re.search(r"\s|[\x00-\x1f\x7f]", value.strip())) or (
        "Enter a model ID without whitespace."
    )


def environment_name(value: str) -> bool | str:
    return (
        not value
        or bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))
        or ("Enter a variable NAME, not an API key. Back returns to secure key entry.")
    )


def endpoint(value: str, *, custom: bool = False) -> bool | str:
    try:
        validate_endpoint(value, allow_loopback_http=custom)
        return True
    except ValueError:
        return "Use a valid HTTPS endpoint without keys, credentials, query or fragment."


@dataclass
class SetupDraft:
    config: AppConfig
    name: str
    profile: dict[str, Any]
    services: Services
    secrets: dict[str, PendingSecret] = field(default_factory=dict, repr=False)
    make_default: bool = True

    def selected(self) -> ModelProfile:
        return ModelProfile.model_validate(self.profile)

    def clear_model_secrets(self) -> None:
        for slot in list(self.secrets):
            if slot.startswith(f"profile:{self.name}:"):
                del self.secrets[slot]


def credential(ui: Terminal, draft: SetupDraft, label: str, env: str, slot: str) -> str:
    current = read_credential(env, slot)
    staged = slot in draft.secrets
    ui.tell(f"{label}: {'new key staged' if staged else current.source}")
    options = [
        ("paste", "Paste a key securely (OS credential manager)"),
        ("environment", "Use an environment variable"),
        ("later", "Configure later"),
        ("back", "Back"),
    ]
    if staged or current.value:
        options.insert(0, ("keep", "Keep current credential"))
    action = ui.select(label, options)
    if action == "back":
        raise Back()
    if action == "paste":
        value = ui.text(
            "API key",
            secret=True,
            validate=lambda text: bool(text.strip()) or "API key cannot be empty.",
        )
        draft.secrets[slot] = PendingSecret(slot, value.strip())
        if os.getenv(env, "").strip():
            ui.tell(f"{env} is set and takes precedence over the new stored key.")
    elif action == "environment":
        ui.tell(f"Default variable: {env}. Stored credentials remain available if it is unset.")
        variable = (
            ui.text(
                "Variable name (blank keeps default; input hidden)",
                secret=True,
                validate=environment_name,
            )
            or env
        )
        draft.secrets.pop(slot, None)
        ui.tell(f"{variable}: {'set' if os.getenv(variable, '').strip() else 'not set'}")
        return variable
    elif action == "later":
        draft.secrets.pop(slot, None)
        ui.tell("No new key staged. Existing credentials are not deleted.")
    return env


def model_step(ui: Terminal, draft: SetupDraft) -> None:
    provider = Provider(
        ui.select(
            "Model provider",
            [(p.value, p.value) for p in Provider],
            str(draft.profile.get("provider", "openrouter")),
        )
    )
    if provider != draft.profile.get("provider"):
        draft.clear_model_secrets()
        draft.profile = {
            "provider": provider,
            "model": "",
            "depth": draft.profile.get("depth", Depth.BALANCED),
            "reranker": draft.profile.get("reranker", Reranker.NONE),
            "jev_model": draft.profile.get("jev_model", "jev-latest"),
        }
    draft.profile["provider"] = provider
    ui.tell(f"Model catalog: {MODEL_CATALOGS[provider]}")
    draft.profile["model"] = ui.text(
        "Model ID", str(draft.profile.get("model", "")), validate=identifier
    ).strip()
    if provider == Provider.COMPATIBLE:
        value = ui.text(
            "API base URL",
            str(draft.profile.get("base_url") or ""),
            validate=lambda text: endpoint(text, custom=True),
        )
        normalized = validate_endpoint(value, allow_loopback_http=True)
        if normalized != draft.profile.get("base_url"):
            draft.clear_model_secrets()
        draft.profile["base_url"] = normalized
        draft.profile["api_key_required"] = (
            ui.select(
                "Authentication",
                [("key", "API key required"), ("none", "No authentication")],
                "key" if draft.profile.get("api_key_required", True) else "none",
            )
            == "key"
        )
        if not draft.profile["api_key_required"]:
            draft.clear_model_secrets()


def credential_step(ui: Terminal, draft: SetupDraft) -> None:
    profile = draft.selected()
    if profile.api_key_required:
        draft.profile["api_key_env"] = credential(
            ui,
            draft,
            "Model credential",
            profile.api_key_env or KEY_ENV[profile.provider],
            profile_slot(draft.name, profile),
        )


def preferences_step(ui: Terminal, draft: SetupDraft) -> None:
    draft.profile["depth"] = Depth(
        ui.select(
            "Default research depth",
            [
                ("quick", "Quick - focused answer"),
                ("balanced", "Balanced - compare evidence"),
                ("thorough", "Thorough - broader investigation"),
            ],
            str(draft.profile.get("depth", Depth.BALANCED)),
        )
    )
    draft.profile["reranker"] = Reranker(
        ui.select(
            "Evidence reranking",
            [("none", "Hybrid retrieval only"), ("jev", "Jev - additional TypeSafe requests")],
            str(draft.profile.get("reranker", Reranker.NONE)),
        )
    )


def elastic_step(ui: Terminal, draft: SetupDraft) -> None:
    action = ui.select(
        "Elasticsearch",
        [
            ("configure", "Configure connection"),
            ("keep", "Keep current / configure later"),
            ("back", "Back"),
        ],
        "keep" if draft.services.elasticsearch_url else "configure",
    )
    if action == "back":
        raise Back()
    if action == "keep":
        return
    old = draft.services.elasticsearch_url or ""
    url = ui.text("Elasticsearch URL", old, validate=endpoint)
    draft.services.elasticsearch_url = validate_endpoint(url)
    if old != draft.services.elasticsearch_url:
        draft.secrets.pop(service_slot("elasticsearch", old), None)
    draft.services.inference_id = ui.text(
        "Embedding inference endpoint ID (required for research)",
        draft.services.inference_id or "",
        validate=lambda text: (
            bool(text.strip()) or "Enter an endpoint ID, or go Back to configure later."
        ),
    ).strip()
    draft.services.elasticsearch_api_key_env = credential(
        ui,
        draft,
        "Elasticsearch credential",
        draft.services.elasticsearch_api_key_env,
        service_slot("elasticsearch", draft.services.elasticsearch_url),
    )


def optional_step(ui: Terminal, draft: SetupDraft) -> None:
    if (
        ui.select(
            "Web discovery",
            [("keep", "Keep current / configure later"), ("configure", "Configure Tavily key")],
        )
        == "configure"
    ):
        draft.services.tavily_api_key_env = credential(
            ui,
            draft,
            "Tavily credential",
            draft.services.tavily_api_key_env,
            service_slot("tavily"),
        )
    if draft.selected().reranker == Reranker.JEV:
        draft.services.typesafe_api_key_env = credential(
            ui,
            draft,
            "Jev credential",
            draft.services.typesafe_api_key_env,
            service_slot("typesafe"),
        )
    else:
        draft.secrets.pop(service_slot("typesafe"), None)


def save_draft(path: Path, draft: SetupDraft) -> None:
    selected = draft.selected()
    updated = with_profile(
        draft.config, draft.name, selected, draft.services, make_default=draft.make_default
    )
    written = 0
    try:
        for secret in draft.secrets.values():
            store_credential(secret.slot, secret.value)
            written += 1
        save_config(path, updated)
    except ConfigError:
        raise ConfigError(
            f"Save did not complete. {written} credential(s) were stored; "
            "configuration was not saved. Check credential-manager and file access."
        ) from None


def configure_profile(ui: Terminal, path: Path, name: str | None = None) -> None:
    config = load_config(path)
    if name is None:
        options = [(key, f"Edit {key}") for key in config.profiles]
        options.extend([("__new__", "New profile"), ("__back__", "Back")])
        name = ui.select("Profiles", options)
        if name == "__back__":
            return
        if name == "__new__":
            name = ui.text(
                "Profile name",
                "default" if not config.profiles else "",
                validate=lambda v: (
                    (
                        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", v))
                        and v not in config.profiles
                    )
                    or "Choose an unused name: letters, digits, underscores or hyphens."
                ),
            )
    existing = config.profiles.get(name)
    draft = SetupDraft(
        config,
        name,
        existing.model_dump() if existing else {},
        config.services.model_copy(deep=True),
        make_default=not config.default_profile or config.default_profile == name,
    )
    steps = [model_step, credential_step, preferences_step, elastic_step, optional_step]
    labels = ["Model", "Credentials", "Preferences", "Elasticsearch", "Optional services"]
    step = 0
    try:
        while True:
            try:
                if step < len(steps):
                    ui.heading(f"Configure {name}", f"{step + 1}/6 - {labels[step]}")
                    steps[step](ui, draft)
                    step += 1
                    continue
                profile = draft.selected()
                ui.heading(
                    "Review configuration",
                    "No paid requests. Keys stay in the OS credential manager.",
                )
                for label, value in [
                    ("Profile", name),
                    ("Provider", profile.provider.value),
                    ("Model", profile.model),
                    ("Depth", profile.depth.value),
                    ("Reranker", profile.reranker.value),
                    ("Elasticsearch", draft.services.elasticsearch_url or "Not configured"),
                    ("Inference", draft.services.inference_id or "Not configured"),
                    ("Default profile", draft.make_default),
                ]:
                    ui.tell(f"{label}: {value}")
                ui.tell(
                    "Research shares selected evidence with the model and Elasticsearch; "
                    "web queries go to Tavily and Jev excerpts to TypeSafe when enabled."
                )
                ui.tell(f"New credentials staged: {len(draft.secrets)}")
                if os.getenv("ELASTICSEARCH_URL") or os.getenv("EVIDENCE_INFERENCE_ID"):
                    ui.tell(
                        "Environment overrides are set for Elasticsearch; they take "
                        "precedence over saved connection settings."
                    )
                action = ui.select(
                    "Configuration",
                    [
                        ("save", "Save configuration"),
                        *[(str(i), f"Edit {label.lower()}") for i, label in enumerate(labels)],
                        ("advanced", "Advanced settings"),
                        ("cancel", "Cancel"),
                    ],
                )
                if action == "save":
                    save_draft(path, draft)
                    ui.tell("Configuration saved. Connections have not been tested.")
                    return
                if action == "cancel":
                    return
                if action == "advanced":
                    draft.make_default = (
                        ui.select(
                            "Make default profile?",
                            [("yes", "Yes"), ("no", "No")],
                            "yes" if draft.make_default else "no",
                        )
                        == "yes"
                    )
                    if profile.reranker == Reranker.JEV:
                        draft.profile["jev_model"] = ui.text(
                            "Jev model ID", profile.jev_model, validate=identifier
                        )
                else:
                    step = int(action)
            except Back:
                if step == 0:
                    return
                step -= 1
            except ValidationError:
                ui.tell("Invalid configuration. Return to the field and correct it.", error=True)
    finally:
        draft.secrets.clear()
