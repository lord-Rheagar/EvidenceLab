import socket

import httpx
import pytest

from evidencelab import fetching
from evidencelab.evidence import EvidenceError


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1"])
def test_private_destinations_blocked(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (address, 80))])
    with pytest.raises(EvidenceError, match="non-public"):
        fetching.public_address("example.com", 80)


def test_mixed_dns_answers_blocked(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("1.1.1.1", 443)), (2, 1, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(EvidenceError):
        fetching.public_address("example.com", 443)


def test_pinned_destination_preserves_host_and_tls_name(monkeypatch, respx_mock):
    monkeypatch.setattr(fetching, "public_address", lambda *a: "1.1.1.1")
    route = respx_mock.get("https://1.1.1.1/article").mock(
        return_value=httpx.Response(200, content=b"article", headers={"content-type": "text/plain"})
    )
    result = fetching.fetch_url("https://example.com/article#heading")
    request = route.calls[0].request
    assert request.headers["host"] == "example.com"
    assert request.extensions["sni_hostname"] == "example.com"
    assert result.url == "https://example.com/article"
    assert result.content == b"article"


def test_redirect_to_private_host_is_blocked(monkeypatch, respx_mock):
    def resolve(host, port):
        if host == "localhost":
            raise EvidenceError("non-public")
        return "1.1.1.1"

    monkeypatch.setattr(fetching, "public_address", resolve)
    route = respx_mock.get("https://1.1.1.1/start").mock(
        return_value=httpx.Response(302, headers={"location": "http://localhost/admin"})
    )
    with pytest.raises(EvidenceError, match="non-public"):
        fetching.fetch_url("https://example.com/start")
    assert route.call_count == 1


def test_download_limit_and_error_redaction(monkeypatch, respx_mock):
    monkeypatch.setattr(fetching, "public_address", lambda *a: "1.1.1.1")
    monkeypatch.setattr(fetching, "MAX_BYTES", 5)
    respx_mock.get("https://1.1.1.1/large").mock(
        return_value=httpx.Response(200, content=b"long content")
    )
    with pytest.raises(EvidenceError, match="download limit"):
        fetching.fetch_url("https://example.com/large")
    respx_mock.get("https://1.1.1.1/error").mock(return_value=httpx.Response(403, text="secret"))
    with pytest.raises(EvidenceError) as error:
        fetching.fetch_url("https://example.com/error")
    assert "secret" not in str(error.value)


def test_public_redirect_revalidates_host_and_preserves_final_location(monkeypatch, respx_mock):
    resolved = []

    def resolve(host, port):
        resolved.append(host)
        return "1.1.1.1"

    monkeypatch.setattr(fetching, "public_address", resolve)
    respx_mock.get("https://1.1.1.1/start").mock(
        return_value=httpx.Response(302, headers={"location": "https://other.example/final"})
    )
    final = respx_mock.get("https://1.1.1.1/final").mock(
        return_value=httpx.Response(
            200, content=b"evidence", headers={"content-type": "text/plain"}
        )
    )
    result = fetching.fetch_url("https://example.com/start")
    assert resolved == ["example.com", "other.example"]
    assert result.url == "https://other.example/final"
    assert final.calls[0].request.extensions["sni_hostname"] == "other.example"
