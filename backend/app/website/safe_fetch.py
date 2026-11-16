"""SSRF-safe HTTP fetching for the Website Intelligence module.

Defences (each one is unit-tested):
  * scheme must be http/https; port must be 80 or 443; no user:password@ credentials
  * hostnames only: IP literals (incl. decimal / hex / octal disguises) are rejected
  * single-label, localhost, .local/.internal/.lan/... hostnames are rejected
  * DNS is resolved ONCE; every resolved address must be globally routable
    (private, loopback, link-local incl. 169.254.169.254, CGNAT, multicast, reserved,
    unspecified, IPv4-mapped/6to4/teredo/NAT64 tunnels to such addresses are all blocked)
  * the TCP connection is PINNED to the validated IP (no second DNS lookup, so DNS
    rebinding cannot swap in an internal address); TLS still verifies the hostname
  * redirects are followed manually (max MAX_REDIRECTS) and EVERY hop is re-validated
  * connect/read timeouts, an overall deadline, a hard response-size cap, and
    bounded gzip/deflate decompression (no decompression bombs)
  * no cookies, no auth headers, no JavaScript execution
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

USER_AGENT = "ShopSenseAI-WebsiteIntelligence/1.0 (+public-site audit; respects robots.txt)"
ALLOWED_PORTS = {"http": 80, "https": 443}
MAX_REDIRECTS = 5
CONNECT_TIMEOUT = 6.0
READ_TIMEOUT = 10.0
DEFAULT_MAX_BYTES = 2_000_000
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".intranet", ".lan", ".home", ".corp",
                    ".home.arpa", ".localdomain", ".test", ".invalid", ".example", ".onion")
_NUMERIC_HOST = re.compile(r"^(0x[0-9a-f]+|[0-9]+)(\.(0x[0-9a-f]+|[0-9]+))*$", re.IGNORECASE)


class UnsafeURLError(ValueError):
    """The URL (or a redirect target) is not an allowed public web address."""


class FetchError(RuntimeError):
    """Network/protocol failure while fetching an allowed URL."""


# ----------------------------------------------------------------- validation
def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        # unwrap tunnelling/mapping formats that embed an IPv4 address
        for embedded in (ip.ipv4_mapped, ip.sixtofour, (ip.teredo[1] if ip.teredo else None)):
            if embedded is not None and not _ip_is_public(embedded):
                return False
        if ip in ipaddress.ip_network("64:ff9b::/96"):            # NAT64 well-known prefix
            return _ip_is_public(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified or not ip.is_global):
        return False
    if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10"):
        return False                                             # carrier-grade NAT
    return True


def normalise_url(raw: str) -> str:
    """Trim, default to https://, and return a canonical URL string (no fragment)."""
    raw = (raw or "").strip()
    if not raw:
        raise UnsafeURLError("Enter a website URL.")
    if len(raw) > 2048:
        raise UnsafeURLError("URL is too long.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "https://" + raw
    parts = urlsplit(raw)
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path or "/", parts.query, ""))


def validate_url(url: str) -> tuple[str, str, int]:
    """Static checks. Returns (scheme, ascii hostname, port). Raises UnsafeURLError."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError(f"Malformed URL: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_PORTS:
        raise UnsafeURLError("Only http:// and https:// URLs can be analysed.")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UnsafeURLError("URLs with embedded credentials are not allowed.")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeURLError("The URL has no hostname.")
    if port is not None and port != ALLOWED_PORTS[scheme]:
        raise UnsafeURLError("Only the standard ports (80 for http, 443 for https) are allowed.")
    try:
        ipaddress.ip_address(host.strip("[]"))
        raise UnsafeURLError("IP addresses are not accepted; enter the site's domain name.")
    except ValueError:
        pass
    if _NUMERIC_HOST.match(host):
        raise UnsafeURLError("Numeric host forms are not accepted; enter the site's domain name.")
    if "." not in host or host == "localhost" or host.endswith(BLOCKED_SUFFIXES):
        raise UnsafeURLError("Only public internet domain names can be analysed.")
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeURLError("The hostname is not a valid domain name.") from exc
    if not re.fullmatch(r"[a-z0-9.-]+", ascii_host) or len(ascii_host) > 253:
        raise UnsafeURLError("The hostname is not a valid domain name.")
    return scheme, ascii_host, ALLOWED_PORTS[scheme]


Resolver = Callable[[str, int], list[str]]


def system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"Could not resolve {host} (DNS lookup failed).") from exc
    return sorted({info[4][0] for info in infos})


def resolve_public(host: str, port: int, resolver: Resolver) -> str:
    """Resolve once; ALL addresses must be public. Returns the IP to pin the connection to."""
    addresses = resolver(host, port)
    if not addresses:
        raise FetchError(f"{host} did not resolve to any address.")
    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError as exc:
            raise UnsafeURLError(f"{host} resolved to an invalid address.") from exc
        if not _ip_is_public(ip):
            raise UnsafeURLError(f"{host} resolves to a non-public network address; it cannot be analysed.")
    # prefer IPv4 for wider reachability, otherwise the first address
    return next((a for a in addresses if ":" not in a), addresses[0])


# ------------------------------------------------------------------ transport
class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, ip: str, port: int, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), timeout=CONNECT_TIMEOUT)
        self.sock.settimeout(self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, ip: str, port: int, timeout: float, context: ssl.SSLContext):
        super().__init__(host, port, timeout=timeout, context=context)
        self._ip = ip

    def connect(self):
        raw = socket.create_connection((self._ip, self.port), timeout=CONNECT_TIMEOUT)
        raw.settimeout(self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)   # verifies cert for host


def _tls_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


ConnectionFactory = Callable[[str, str, str, int, float], http.client.HTTPConnection]


def default_connection_factory(scheme: str, host: str, ip: str, port: int, timeout: float):
    if scheme == "https":
        return _PinnedHTTPSConnection(host, ip, port, timeout, _tls_context())
    return _PinnedHTTPConnection(host, ip, port, timeout)


# ---------------------------------------------------------------------- fetch
@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    headers: dict[str, str]
    body: bytes
    elapsed_ms: float
    redirects: list[dict] = field(default_factory=list)
    truncated: bool = False
    redirect_location: str | None = None     # set when a redirect was NOT followed (max_redirects reached)

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    def text(self) -> str:
        m = re.search(r"charset=([\w-]+)", self.headers.get("content-type", ""), re.I)
        enc = m.group(1) if m else "utf-8"
        try:
            return self.body.decode(enc, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


def _read_limited(resp: http.client.HTTPResponse, max_bytes: int, deadline: float) -> tuple[bytes, bool]:
    chunks, total = [], 0
    while True:
        if time.monotonic() > deadline:
            raise FetchError("The site took too long to respond.")
        chunk = resp.read(65536)
        if not chunk:
            return b"".join(chunks), False
        total += len(chunk)
        if total > max_bytes:
            chunks.append(chunk[: max_bytes - (total - len(chunk))])
            return b"".join(chunks), True
        chunks.append(chunk)


def _decode_body(body: bytes, encoding: str, max_bytes: int) -> tuple[bytes, bool]:
    encoding = (encoding or "").lower().strip()
    if encoding in ("", "identity"):
        return body, False
    if encoding not in ("gzip", "deflate", "x-gzip"):
        raise FetchError(f"Unsupported content encoding: {encoding}")
    d = zlib.decompressobj(16 + zlib.MAX_WBITS if "gzip" in encoding else zlib.MAX_WBITS)
    try:
        out = d.decompress(body, max_bytes + 1)          # bounded: never more than max_bytes + 1
    except zlib.error:
        out = zlib.decompressobj(-zlib.MAX_WBITS).decompress(body, max_bytes + 1)  # raw deflate
    if len(out) > max_bytes:
        return out[:max_bytes], True
    return out, False


@dataclass
class SafeFetcher:
    resolver: Resolver = system_resolver
    connection_factory: ConnectionFactory = default_connection_factory
    timeout: float = READ_TIMEOUT
    total_deadline_s: float = 20.0

    def fetch(self, url: str, *, max_bytes: int = DEFAULT_MAX_BYTES, accept: str = "text/html,*/*;q=0.5",
              max_redirects: int = MAX_REDIRECTS) -> FetchResult:
        """GET a public URL safely, following up to ``max_redirects`` validated redirects.

        With max_redirects=0 a redirect is returned (status 3xx, ``redirect_location`` set)
        instead of followed, so callers can apply their own checks (e.g. robots.txt) per hop.
        """
        start = time.monotonic()
        deadline = start + self.total_deadline_s
        current = normalise_url(url)
        redirects: list[dict] = []
        for _ in range(min(max_redirects, MAX_REDIRECTS) + 1):
            scheme, host, port = validate_url(current)
            ip = resolve_public(host, port, self.resolver)
            parts = urlsplit(current)
            path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~") + \
                (f"?{quote(parts.query, safe='=&%:@!$()*+,;/?-._~')}" if parts.query else "")
            conn = self.connection_factory(scheme, host, ip, port, self.timeout)
            try:
                conn.request("GET", path, headers={
                    "Host": host if port == ALLOWED_PORTS[scheme] else f"{host}:{port}",
                    "User-Agent": USER_AGENT,
                    "Accept": accept,
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "close",
                })
                resp = conn.getresponse()
                headers = {k.lower(): v for k, v in resp.getheaders()}
                if resp.status in (301, 302, 303, 307, 308) and headers.get("location"):
                    nxt = normalise_url(urljoin(current, headers["location"]))
                    if max_redirects == 0:                   # hop-by-hop mode: caller handles this hop
                        return FetchResult(url, current, resp.status, headers, b"",
                                           round((time.monotonic() - start) * 1000, 1), redirects,
                                           redirect_location=nxt)
                    if len(redirects) >= min(max_redirects, MAX_REDIRECTS):
                        raise FetchError(f"Too many redirects (more than {MAX_REDIRECTS}).")
                    redirects.append({"from": current, "to": nxt, "status": resp.status})
                    current = nxt
                    continue
                raw, truncated = _read_limited(resp, max_bytes, deadline)
                body, trunc2 = _decode_body(raw, headers.get("content-encoding", ""), max_bytes)
                return FetchResult(url, current, resp.status, headers, body,
                                   round((time.monotonic() - start) * 1000, 1), redirects, truncated or trunc2)
            except (OSError, http.client.HTTPException) as exc:
                if isinstance(exc, ssl.SSLCertVerificationError):
                    raise FetchError(f"TLS certificate for {host} could not be verified.") from exc
                if isinstance(exc, (socket.timeout, TimeoutError)):
                    raise FetchError(f"{host} did not respond in time.") from exc
                raise FetchError(f"Could not connect to {host}.") from exc
            finally:
                conn.close()
        raise FetchError(f"Too many redirects (more than {MAX_REDIRECTS}).")
