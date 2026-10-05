import json

import httpx
import httpx2
import pytest
from pydantic import ValidationError

from evidencelab.diagnostics import Diagnostic, classify_exception


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (httpx.ReadTimeout("PRIVATE_KEY"), "timeout"),
        (httpx2.ReadTimeout("PRIVATE_KEY"), "timeout"),
        (TimeoutError("PRIVATE_KEY"), "timeout"),
        (httpx.ConnectError("PRIVATE_KEY"), "connection"),
        (httpx2.ConnectError("PRIVATE_KEY"), "connection"),
        (ValueError("PRIVATE_KEY"), "invalid_response"),
        (RuntimeError("PRIVATE_KEY"), "unknown"),
    ],
)
def test_exception_classification_never_uses_messages(error, category):
    assert classify_exception(error) == (category, None)
    wrapper = RuntimeError("PRIVATE_DOCUMENT")
    wrapper.__cause__ = error
    assert classify_exception(wrapper) == (category, None)


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (401, "authentication"),
        (402, "quota"),
        (403, "access_denied"),
        (429, "rate_limit"),
        (422, "request_rejected"),
        (503, "provider_unavailable"),
        (504, "timeout"),
    ],
)
def test_numeric_http_status_only(status, category):
    error = RuntimeError("PRIVATE_RESPONSE")
    error.status_code = status
    assert classify_exception(error) == (category, status)
    event = Diagnostic(service="model", code=category, http_status=status)
    assert "PRIVATE" not in event.model_dump_json()


def test_cycles_and_untrusted_status_fields_are_safe():
    error = RuntimeError("PRIVATE_RESPONSE")
    error.status_code = "PRIVATE_STATUS"
    error.__cause__ = error
    assert classify_exception(error) == ("unknown", None)
    with pytest.raises(ValidationError):
        Diagnostic(service="model", code="PRIVATE_CATEGORY")
    with pytest.raises(ValidationError):
        Diagnostic(service="jev", code="score_mismatch", probabilities=[float("nan")] * 4)
    assert (
        json.loads(Diagnostic(service="model", code="unknown").model_dump_json())["code"]
        == "unknown"
    )


def test_openrouter_sdk_validation_failure_preserves_safe_http_status():
    from openrouter.errors import ResponseValidationError

    error = ResponseValidationError(
        "PRIVATE_KEY", httpx.Response(200, text="PRIVATE_DOCUMENT"), ValueError("PRIVATE_BODY")
    )
    assert classify_exception(error) == ("invalid_response", 200)
