"""POST /api/analyze-website: Website Intelligence (separate from purchase prediction).

Abuse controls, on top of the SSRF-safe fetcher:
  * per-client sliding-window rate limit (WEBSITE_RATE_LIMIT per WEBSITE_RATE_WINDOW_S)
  * global concurrency cap (WEBSITE_MAX_CONCURRENCY); extra requests get 429
  * analyses are synchronous and bounded (<= 5 pages, ~45 s budget); nothing is stored
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.limits import RateLimiter, client_ip, masked_ip, too_many
from app.config import get_settings
from app.website.analyzer import analyze
from app.website.pagespeed import run_pagespeed
from app.website.safe_fetch import SafeFetcher, UnsafeURLError
from app.website.schemas import AnalyzeWebsiteRequest, AnalyzeWebsiteResponse

router = APIRouter(prefix="/api", tags=["website intelligence"])
log = logging.getLogger("shopsense.website")


_settings = get_settings()
limiter = RateLimiter(_settings.website_rate_limit, _settings.website_rate_window_s)
_slots = threading.BoundedSemaphore(_settings.website_max_concurrency)


def get_fetcher() -> SafeFetcher:
    return SafeFetcher()


def rate_limit(request: Request) -> None:
    wait = limiter.check(client_ip(request))
    if wait is not None:
        raise too_many(limiter, wait, "analyses")


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
def analyze_website(body: AnalyzeWebsiteRequest, request: Request,
                    fetcher: SafeFetcher = Depends(get_fetcher)) -> AnalyzeWebsiteResponse:
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
    # client = coarse network of the caller: after deploying, check it shows your public network,
    # not the proxy's private address (that would mean FORWARDED_ALLOW_IPS is wrong).
    log.info("website analysis: %s pages, %s issues, client=%s", report["summary"]["pages_analyzed"],
             len(report["issues"]), masked_ip(client_ip(request)))
    return AnalyzeWebsiteResponse(**report)
