"""Explicit provider adapters and an opt-in capability probe."""

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr

from evidencelab.config import ConfigError, ModelProfile, Provider
from evidencelab.credentials import profile_slot, read_credential

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

MODEL_CATALOGS = {
    Provider.OPENAI: "https://platform.openai.com/docs/models",
    Provider.ANTHROPIC: "https://platform.claude.com/docs/en/about-claude/models/overview",
    Provider.GEMINI: "https://ai.google.dev/gemini-api/docs/models",
    Provider.OPENROUTER: "https://openrouter.ai/models",
    Provider.COMPATIBLE: "Use the model ID published by your endpoint operator.",
}


PROBE_ERRORS = {
    "authentication": "Model authentication or access failed; check your key and model access.",
    "rate_limit": "Model request was rate limited or quota was exhausted.",
    "request_rejected": "Model or tool request was rejected; check the model ID and tool support.",
    "timeout": "Model request timed out; the provider did not respond within the request timeout.",
    "provider_unavailable": "Model provider is temporarily unavailable; retry later.",
    "connection": "Could not connect to the model provider; check connectivity.",
    "unknown": "Model check failed; check connectivity, endpoint, and provider availability.",
    "missing_tool_call": (
        "Model did not return the required tool call; compatibility is unverified."
    ),
    "unexpected_tool": "Model returned an unexpected tool; compatibility is unverified.",
    "invalid_arguments": "Model returned invalid structured tool arguments.",
}


class ModelProbeError(ConfigError):
    """Allowlisted diagnostics safe to expose without raw provider responses."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(PROBE_ERRORS[code])


def build_model(name: str, profile: ModelProfile) -> "BaseChatModel":
    """Instantiate the selected adapter without making an inference request."""
    credential = read_credential(profile.api_key_env or "", profile_slot(name, profile))
    if profile.api_key_required and not credential.value:
        raise ConfigError(f"Set {profile.api_key_env} or store a key with 'evidence configure'.")
    # A no-auth custom endpoint must not inherit another provider's environment key.
    secret = SecretStr((credential.value or "") if profile.api_key_required else "not-required")
    if profile.provider in {Provider.OPENAI, Provider.COMPATIBLE}:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=profile.model,
            api_key=secret,
            base_url=profile.base_url or "https://api.openai.com/v1",
            timeout=30,
            max_retries=0,
            use_responses_api=False,
        )
    if profile.provider == Provider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=profile.model,
            api_key=secret,
            base_url="https://api.anthropic.com",
            timeout=30,
            max_retries=0,
            stop=None,
        )
    if profile.provider == Provider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=profile.model,
            api_key=secret,
            vertexai=False,
            client_options="https://generativelanguage.googleapis.com",
            request_timeout=30,
            retries=0,
        )
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        model=profile.model,
        api_key=secret,
        base_url="https://openrouter.ai/api/v1",
        timeout=30_000,  # This adapter uses milliseconds, unlike the other providers.
        max_retries=0,
        # The adapter leaves SDK retries unset for max_retries=0; None disables them.
        model_kwargs={"retries": None},
    )


class CapabilityProbe(BaseModel):
    """Return the requested fixed values to demonstrate structured tool calling."""

    model_config = ConfigDict(extra="forbid", strict=True)
    status: str
    value: int


def check_model(name: str, profile: ModelProfile) -> None:
    """One paid request checks tool calling and schema-valid tool arguments."""
    from langchain_core.messages import AIMessage
    from langsmith import tracing_context

    try:
        model = build_model(name, profile)
        with tracing_context(enabled=False):
            response = model.bind_tools([CapabilityProbe]).invoke(
                "Call CapabilityProbe exactly once with status='ok' and value=7. "
                "Do not answer in prose."
            )
    except ConfigError:
        raise
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            code = "authentication"
        elif status == 429:
            code = "rate_limit"
        elif status in {400, 404, 422}:
            code = "request_rejected"
        elif status in {408, 504} or isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            code = "timeout"
        elif isinstance(status, int) and 500 <= status < 600:
            code = "provider_unavailable"
        elif isinstance(exc, httpx.NetworkError):
            code = "connection"
        else:
            code = "unknown"
        raise ModelProbeError(code) from None
    if not isinstance(response, AIMessage) or len(response.tool_calls) != 1:
        raise ModelProbeError("missing_tool_call")
    call = response.tool_calls[0]
    if call["name"] != "CapabilityProbe":
        raise ModelProbeError("unexpected_tool")
    try:
        result = CapabilityProbe.model_validate(call["args"])
        if result.status != "ok" or result.value != 7:
            raise ValueError("Unexpected probe values")
    except ValueError:
        raise ModelProbeError("invalid_arguments") from None
