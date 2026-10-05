"""Size-limited public URL retrieval with validated, pinned connection addresses."""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from evidencelab.evidence import MAX_BYTES, EvidenceError


def normalize_url(value: str) -> str:
    try:
        url = httpx.URL(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError
        return str(url.copy_with(fragment=None))
    except (httpx.InvalidURL, ValueError):
        raise EvidenceError(
            "Use a valid public HTTP(S) URL without embedded credentials."
        ) from None


def public_address(host: str, port: int) -> str:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ips = sorted({str(item[4][0]) for item in addresses})
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise EvidenceError("URL resolves to a non-public address; request blocked.")
        return ips[0]
    except (socket.gaierror, ValueError):
        raise EvidenceError("Could not resolve a public address for this URL.") from None


@dataclass(frozen=True)
class FetchedDocument:
    url: str
    content: bytes
    content_type: str


def fetch_url(value: str) -> FetchedDocument:
    current = normalize_url(value)
    try:
        for _ in range(6):
            # A fresh connection also verifies the new TLS hostname on same-IP redirects.
            with httpx.Client(timeout=20, trust_env=False, follow_redirects=False) as client:
                original = httpx.URL(current)
                ip = public_address(
                    original.host, original.port or (443 if original.scheme == "https" else 80)
                )
                # Pin the validated IP to prevent a second DNS lookup from changing the destination.
                target = original.copy_with(host=ip)
                headers = {"Host": original.netloc.decode(), "User-Agent": "EvidenceLab/0.1"}
                with client.stream(
                    "GET", target, headers=headers, extensions={"sni_hostname": original.host}
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise EvidenceError("Redirect did not specify a destination.")
                        current = normalize_url(urljoin(current, location))
                        continue
                    response.raise_for_status()
                    content = bytearray()
                    for chunk in response.iter_bytes(chunk_size=65536):
                        content.extend(chunk)
                        if len(content) > MAX_BYTES:
                            raise EvidenceError("Document exceeds the 20 MB download limit.")
                    return FetchedDocument(
                        current,
                        bytes(content),
                        response.headers.get("content-type", "").split(";")[0].lower(),
                    )
    except httpx.HTTPError:
        raise EvidenceError("Could not fetch URL; the request failed or timed out.") from None
    raise EvidenceError("URL exceeded the redirect limit.")
