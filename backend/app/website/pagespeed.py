"""Optional Google PageSpeed Insights (PSI) integration.

Only used when PAGESPEED_API_KEY is configured on the server (an authorised, documented
Google API). The key is never sent to the browser. Results are passed through exactly as
returned and labelled:
  lab   - Lighthouse run by Google on that request (simulated device/network)
  field - Chrome UX Report (CrUX) real-user data, ONLY if Google has enough data for the URL
If PSI is not configured or fails, the result is "unavailable" with a reason. Nothing is estimated.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
LAB_AUDITS = {
    "largest-contentful-paint": "Largest Contentful Paint",
    "cumulative-layout-shift": "Cumulative Layout Shift",
    "total-blocking-time": "Total Blocking Time",
    "first-contentful-paint": "First Contentful Paint",
    "speed-index": "Speed Index",
}
FIELD_METRICS = {
    "LARGEST_CONTENTFUL_PAINT_MS": "Largest Contentful Paint",
    "INTERACTION_TO_NEXT_PAINT": "Interaction to Next Paint",
    "CUMULATIVE_LAYOUT_SHIFT_SCORE": "Cumulative Layout Shift",
}


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "source": "Google PageSpeed Insights (Lighthouse)", "reason": reason}


def parse_psi(data: dict, strategy: str) -> dict:
    """Convert a PSI v5 response into the report format (pure function; unit-tested)."""
    lh = data.get("lighthouseResult") or {}
    perf = (lh.get("categories") or {}).get("performance") or {}
    audits = lh.get("audits") or {}
    lab = [{"metric": label, "value": audits[k].get("displayValue"), "numeric": audits[k].get("numericValue")}
           for k, label in LAB_AUDITS.items() if k in audits]
    exp = data.get("loadingExperience") or {}
    field = [{"metric": label, "percentile": m.get("percentile"), "category": m.get("category")}
             for k, label in FIELD_METRICS.items() if (m := (exp.get("metrics") or {}).get(k))]
    return {
        "status": "available",
        "source": "Google PageSpeed Insights (Lighthouse)",
        "strategy": strategy,
        "lighthouse_version": lh.get("lighthouseVersion"),
        "fetched_at": lh.get("fetchTime"),
        "performance_score": round(perf["score"] * 100) if isinstance(perf.get("score"), (int, float)) else None,
        "lab": lab,
        "field": field,
        "field_overall": exp.get("overall_category"),
        "field_available": bool(field),
        "note": "Lab = one simulated Lighthouse run by Google. Field = real Chrome users (CrUX), shown only when Google has enough data.",
    }


def run_pagespeed(url: str, api_key: str, strategy: str = "mobile", timeout: float = 60.0) -> dict:
    if not api_key:
        return unavailable("No PAGESPEED_API_KEY configured on the server.")
    q = urllib.parse.urlencode({"url": url, "strategy": strategy, "category": "performance", "key": api_key})
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(f"{PSI_ENDPOINT}?{q}", timeout=timeout, context=ctx) as r:
            return parse_psi(json.loads(r.read(5_000_000)), strategy)
    except urllib.error.HTTPError as exc:
        return unavailable(f"PageSpeed Insights returned HTTP {exc.code}.")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return unavailable(f"PageSpeed Insights request failed ({type(exc).__name__}).")
