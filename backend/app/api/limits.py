"""Per-client request limits shared by the API routers.

Client identity is ``request.client.host``. Behind a reverse proxy (Render, Nginx, ...) Uvicorn
must run with ``--proxy-headers`` and ``FORWARDED_ALLOW_IPS`` set to the proxy's addresses. Uvicorn
then takes the right-most address in ``X-Forwarded-For`` that is NOT a trusted proxy, i.e. the
address the proxy itself saw. A client cannot spoof it by sending its own X-Forwarded-For.
(Never use ``FORWARDED_ALLOW_IPS=*``: Uvicorn then trusts the LEFT-most, client-supplied entry.)
"""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.config import get_settings


class RateLimiter:
    """In-process sliding window per client key (use a shared store when running many replicas)."""

    def __init__(self, limit: int, window_s: float):
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """None if allowed (and recorded), else seconds until the next slot frees up."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return max(1.0, self.window - (now - q[0]))
            q.append(now)
            if len(self._hits) > 10_000:            # bound memory: drop idle clients
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return None


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def masked_ip(ip: str) -> str:
    """Coarse client address for logs (IPv4 /24, IPv6 /48): enough to check proxy setup, not to identify."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "unknown"
    prefix = 24 if addr.version == 4 else 48
    return str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))


def too_many(limiter: RateLimiter, wait: float, what: str) -> HTTPException:
    return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                         detail=f"Rate limit: at most {limiter.limit} {what} per {int(limiter.window)} s. Try again later.",
                         headers={"Retry-After": str(int(wait) + 1)})


_limit = get_settings().api_rate_limit
# General limit for the model/dashboard endpoints. None = off (the development default).
api_limiter: RateLimiter | None = RateLimiter(_limit, 60.0) if _limit else None


def api_rate_limit(request: Request) -> None:
    if api_limiter is None:
        return
    wait = api_limiter.check(client_ip(request))
    if wait is not None:
        raise too_many(api_limiter, wait, "API requests")
