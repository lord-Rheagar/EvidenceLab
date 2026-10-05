import asyncio
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from evidencelab import providers
from evidencelab.config import KEY_ENV, ConfigError, ModelProfile, Provider


@pytest.mark.parametrize(
    ("provider", "module", "class_name", "model_field"),
    [
        (Provider.OPENAI, "langchain_openai", "ChatOpenAI", "model"),
        (Provider.ANTHROPIC, "langchain_anthropic", "ChatAnthropic", "model_name"),
        (Provider.GEMINI, "langchain_google_genai", "ChatGoogleGenerativeAI", "model"),
        (Provider.OPENROUTER, "langchain_openrouter", "ChatOpenRouter", "model"),
        (Provider.COMPATIBLE, "langchain_openai", "ChatOpenAI", "model"),
    ],
)
def test_adapter_selection_and_explicit_credentials(
    monkeypatch, provider, module, class_name, model_field
):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{class_name: constructor}))
    monkeypatch.setenv(KEY_ENV[provider], "provider-key")
    # Vendor credentials must not be routed to ambient OpenAI-compatible URLs.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example/v1")
    kwargs = {"base_url": "https://custom.example/v1"} if provider == Provider.COMPATIBLE else {}
    profile = ModelProfile(provider=provider, model="chosen-model", **kwargs)
    providers.build_model("work", profile)
    assert captured[model_field] == "chosen-model"
    assert captured["api_key"].get_secret_value() == "provider-key"
    if provider == Provider.OPENAI:
        assert captured["base_url"] == "https://api.openai.com/v1"
    if provider == Provider.OPENROUTER:
        assert captured["base_url"] == "https://openrouter.ai/api/v1"
    if provider == Provider.GEMINI:
        assert captured["vertexai"] is False
        assert captured["client_options"] == "https://generativelanguage.googleapis.com"


def test_missing_key_fails_before_adapter_import():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        providers.build_model("work", ModelProfile(provider=Provider.OPENAI, model="m"))


def test_custom_no_auth_does_not_forward_ambient_key(monkeypatch):
    captured = {}
    monkeypatch.setenv("EVIDENCE_CUSTOM_API_KEY", "do-not-forward")
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=lambda **kwargs: captured.update(kwargs)),
    )
    profile = ModelProfile(
        provider=Provider.COMPATIBLE,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key_required=False,
    )
    providers.build_model("work", profile)
    assert captured["api_key"].get_secret_value() == "not-required"


class StubModel:
    def __init__(self, response):
        self.response = response

    def bind_tools(self, tools):
        assert tools == [providers.CapabilityProbe]
        return self

    def invoke(self, prompt):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def tool_response(name="CapabilityProbe", args=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {"status": "ok", "value": 7}, "id": "probe"}],
    )


def test_valid_probe(monkeypatch):
    monkeypatch.setattr(providers, "build_model", lambda *args: StubModel(tool_response()))
    providers.check_model("work", ModelProfile(provider=Provider.OPENAI, model="m"))


@pytest.mark.parametrize(
    "response",
    [
        AIMessage(content="ok"),
        tool_response(name="wrong"),
        tool_response(args={"status": "ok", "value": 9}),
    ],
)
def test_probe_rejects_invalid_tool_responses(monkeypatch, response):
    monkeypatch.setattr(providers, "build_model", lambda *args: StubModel(response))
    with pytest.raises(ConfigError):
        providers.check_model("work", ModelProfile(provider=Provider.OPENAI, model="m"))


def test_provider_exception_is_redacted(monkeypatch):
    monkeypatch.setattr(
        providers,
        "build_model",
        lambda *args: StubModel(RuntimeError("Bearer secret-from-provider")),
    )
    with pytest.raises(ConfigError) as error:
        providers.check_model("work", ModelProfile(provider=Provider.OPENAI, model="m"))
    assert "secret-from-provider" not in str(error.value)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "rate_limit"),
        (400, "request_rejected"),
        (404, "request_rejected"),
        (422, "request_rejected"),
        (408, "timeout"),
        (504, "timeout"),
        (503, "provider_unavailable"),
    ],
)
def test_probe_exposes_only_safe_http_diagnostic(monkeypatch, status, code):
    error = RuntimeError("PRIVATE_PROVIDER_RESPONSE")
    error.status_code = status
    monkeypatch.setattr(providers, "build_model", lambda *args: StubModel(error))
    with pytest.raises(providers.ModelProbeError) as caught:
        providers.check_model("work", ModelProfile(provider=Provider.OPENAI, model="m"))
    assert caught.value.code == code
    assert "PRIVATE_PROVIDER_RESPONSE" not in str(caught.value)


@pytest.mark.parametrize("timeout", [True, False])
def test_probe_transport_diagnostic(monkeypatch, timeout):
    import httpx

    error_type = httpx.ReadTimeout if timeout else httpx.ConnectError
    monkeypatch.setattr(
        providers, "build_model", lambda *args: StubModel(error_type("PRIVATE_RESPONSE"))
    )
    with pytest.raises(providers.ModelProbeError) as caught:
        providers.check_model("work", ModelProfile(provider=Provider.OPENROUTER, model="m"))
    assert caught.value.code == ("timeout" if timeout else "connection")
    assert "PRIVATE_RESPONSE" not in str(caught.value)


@pytest.mark.parametrize("provider", list(Provider))
def test_installed_adapters_can_bind_tools_without_network(monkeypatch, provider):
    import httpx
    from langchain_core.language_models.chat_models import BaseChatModel

    def reject_network(*args, **kwargs):
        pytest.fail("Adapter construction must not send HTTP requests")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    monkeypatch.setenv(KEY_ENV[provider], "dummy-key-for-offline-test")
    extra = {"base_url": "http://localhost:8080/v1"} if provider == Provider.COMPATIBLE else {}
    profile = ModelProfile(provider=provider, model="user-supplied-model", **extra)
    model = providers.build_model("offline", profile)
    assert isinstance(model, BaseChatModel)
    assert model.bind_tools([providers.CapabilityProbe]) is not None


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.parametrize("failure", ["server_error", "connection_error"])
def test_openrouter_http_timeout_and_no_automatic_retries(monkeypatch, mode, failure):
    import httpx
    from langsmith import tracing_context
    from openrouter.errors import ServiceUnavailableResponseError

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    requests = []

    def response(request):
        requests.append(request)
        if len(requests) == 1 and failure == "connection_error":
            raise httpx.ConnectError("Synthetic connection failure", request=request)
        # A second attempt terminates even on regression, avoiding lengthy SDK backoff.
        status = 503 if len(requests) == 1 else 400
        return httpx.Response(
            status, request=request, json={"error": {"message": "Synthetic error", "code": status}}
        )

    def send(self, request, *args, **kwargs):
        return response(request)

    async def send_async(self, request, *args, **kwargs):
        return response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", send_async)
    model = providers.build_model(
        "offline", ModelProfile(provider=Provider.OPENROUTER, model="test-model")
    ).bind_tools([providers.CapabilityProbe])

    async def run_async():
        if mode == "ainvoke":
            await model.ainvoke("Offline transport check")
        else:
            async for _ in model.astream("Offline transport check"):
                pass

    expected = (
        httpx.ConnectError if failure == "connection_error" else ServiceUnavailableResponseError
    )
    with tracing_context(enabled=False), pytest.raises(expected):
        if mode.startswith("a"):
            asyncio.run(run_async())
        elif mode == "stream":
            list(model.stream("Offline transport check"))
        else:
            model.invoke("Offline transport check")
    assert len(requests) == 1
    assert requests[0].url == "https://openrouter.ai/api/v1/chat/completions"
    assert requests[0].extensions["timeout"] == dict.fromkeys(
        ["connect", "read", "write", "pool"], 30.0
    )
