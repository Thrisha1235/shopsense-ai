"""POST /api/analyze-website: Website Intelligence (separate from purchase prediction).

Abuse controls, on top of the SSRF-safe fetcher:
  * per-client sliding-window rate limit (WEBSITE_RATE_LIMIT per WEBSITE_RATE_WINDOW_S)
  * global concurrency cap (WEBSITE_MAX_CONCURRENCY); extra requests get 429
  * analyses are synchronous and bounded (<= 5 pages, ~45 s budget); nothing is stored
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import get_settings
from app.website.analyzer import analyze
from app.website.pagespeed import run_pagespeed
from app.website.safe_fetch import SafeFetcher, UnsafeURLError
from app.website.schemas import AnalyzeWebsiteRequest, AnalyzeWebsiteResponse

router = APIRouter(prefix="/api", tags=["website intelligence"])
log = logging.getLogger("shopsense.website")


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


_settings = get_settings()
limiter = RateLimiter(_settings.website_rate_limit, _settings.website_rate_window_s)
_slots = threading.BoundedSemaphore(_settings.website_max_concurrency)


def get_fetcher() -> SafeFetcher:
    return SafeFetcher()


def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    wait = limiter.check(client)
    if wait is not None:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Rate limit: at most {limiter.limit} analyses per {int(limiter.window)} s. Try again later.",
                            headers={"Retry-After": str(int(wait) + 1)})


def concurrency_slot():
    if not _slots.acquire(blocking=False):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="The analyser is busy with other requests. Try again shortly.",
                            headers={"Retry-After": "10"})
    try:
        yield
    finally:
        _slots.release()


@router.post(
    "/analyze-website",
    response_model=AnalyzeWebsiteResponse,
    summary="Evidence-based technical, SEO, usability and conversion-readiness analysis of a public website",
    responses={400: {"description": "URL not allowed (non-public, credentials, IP literal, bad port)"},
               429: {"description": "Rate limit or concurrency limit"}, 503: {"description": "Feature disabled"}},
    dependencies=[Depends(rate_limit), Depends(concurrency_slot)],
)
def analyze_website(body: AnalyzeWebsiteRequest, fetcher: SafeFetcher = Depends(get_fetcher)) -> AnalyzeWebsiteResponse:
    settings = get_settings()
    if not settings.website_analysis_enabled:
        raise HTTPException(503, detail="Website analysis is disabled on this server.")
    psi = None
    if body.include_performance:
        key = settings.pagespeed_api_key.get_secret_value() if settings.pagespeed_api_key else ""
        psi = lambda u: run_pagespeed(u, key)  # noqa: E731
    try:
        report = analyze(body.url, fetcher=fetcher, pagespeed=psi)
    except UnsafeURLError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    log.info("website analysis: %s pages, %s issues", report["summary"]["pages_analyzed"], len(report["issues"]))
    return AnalyzeWebsiteResponse(**report)
