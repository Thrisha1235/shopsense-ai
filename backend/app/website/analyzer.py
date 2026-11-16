"""Crawl a few public pages politely and assemble the Website Intelligence report.

robots.txt follows RFC 9309:
  2xx  -> rules are applied to EVERY request, including redirect hops
  4xx  -> robots.txt "unavailable": crawling allowed
  5xx / network error -> "unreachable": treated as full disallow (nothing is fetched)
Crawl-delay is honoured (pages are spaced by max(1 s, crawl-delay)); if it exceeds
MAX_CRAWL_DELAY only the start page is analysed. At most MAX_PAGES same-site pages are fetched.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from app.website import checks
from app.website.html_extract import extract, same_site
from app.website.safe_fetch import FetchError, SafeFetcher, USER_AGENT, UnsafeURLError, normalise_url, validate_url

ROBOTS_AGENT = "ShopSenseAI-WebsiteIntelligence"
MAX_PAGES = 5
MIN_DELAY_S = 1.0
MAX_CRAWL_DELAY = 5.0
TIME_BUDGET_S = 45.0
ROBOTS_MAX_BYTES = 500_000
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".mp4", ".mp3", ".xml", ".json",
            ".css", ".js", ".ico", ".doc", ".docx", ".xls", ".xlsx", ".avif")
PRIORITY = ("product", "shop", "store", "collection", "categor", "catalog", "pricing", "plans", "services",
            "about", "contact")

DISCLAIMERS = [
    "This analysis reads the HTML that the site's server returns. It does not run JavaScript, so "
    "content added in the browser is not seen.",
    "Conversion-readiness items are heuristics about what is publicly visible. HTML alone cannot "
    "measure real visitors, conversions, revenue or sales.",
    "Response times are single measurements from the analysis server, not real-user speed or Core Web Vitals.",
    "Only a few public pages are analysed, and robots.txt rules are respected. Results describe "
    "those pages, not the whole site.",
]


class RobotsDisallowed(RuntimeError):
    pass


def _origin(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc.lower(), "", "", ""))


@dataclass
class RobotsInfo:
    url: str
    status: str            # found | not_found | unreachable
    http_status: int | None
    allows_start: bool | None = None
    crawl_delay: float | None = None
    sitemaps: list[str] | None = None
    note: str = ""


class RobotsCache:
    def __init__(self, fetcher: SafeFetcher):
        self.fetcher = fetcher
        self._parsers: dict[str, RobotFileParser | str] = {}
        self.info: dict[str, RobotsInfo] = {}

    def _load(self, origin: str):
        robots_url = origin + "/robots.txt"
        try:
            # no robots check while fetching robots.txt itself (that would recurse)
            r = self.fetcher.fetch(robots_url, max_bytes=ROBOTS_MAX_BYTES, accept="text/plain,*/*;q=0.5")
        except UnsafeURLError:
            raise                      # the host itself is not public: report that, not a robots problem
        except FetchError as exc:
            self._parsers[origin] = "disallow"
            self.info[origin] = RobotsInfo(robots_url, "unreachable", None,
                                           note=f"robots.txt could not be fetched ({exc}); treated as disallow-all per RFC 9309.")
            return
        if 200 <= r.status < 300:
            rp = RobotFileParser()
            rp.parse(r.text().splitlines())
            self._parsers[origin] = rp
            self.info[origin] = RobotsInfo(robots_url, "found", r.status,
                                           crawl_delay=rp.crawl_delay(ROBOTS_AGENT), sitemaps=list(rp.site_maps() or []))
        elif 400 <= r.status < 500:
            self._parsers[origin] = "allow"
            self.info[origin] = RobotsInfo(robots_url, "not_found", r.status, note="No robots.txt (4xx): crawling allowed per RFC 9309.")
        else:
            self._parsers[origin] = "disallow"
            self.info[origin] = RobotsInfo(robots_url, "unreachable", r.status,
                                           note=f"robots.txt returned HTTP {r.status}; treated as disallow-all per RFC 9309.")

    def allowed(self, url: str) -> bool:
        origin = _origin(url)
        if origin not in self._parsers:
            self._load(origin)
        rule = self._parsers[origin]
        if rule == "allow":
            return True
        if rule == "disallow":
            return False
        return rule.can_fetch(ROBOTS_AGENT, url)

    def check(self, url: str) -> None:
        if not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")


class _PoliteFetcher:
    """Wraps SafeFetcher so EVERY hop (incl. redirects) is checked against robots.txt."""

    def __init__(self, fetcher: SafeFetcher, robots: RobotsCache):
        self.fetcher, self.robots = fetcher, robots

    def fetch_page(self, url: str):
        # resolve redirects hop by hop so each target is robots-checked before it is requested
        current, hops = url, []
        for _ in range(6):
            self.robots.check(current)
            r = self.fetcher.fetch(current, max_redirects=0)
            if r.redirect_location:
                hops.append({"from": current, "to": r.redirect_location, "status": r.status})
                current = r.redirect_location
                continue
            r.redirects = hops
            r.requested_url = url
            return r
        raise FetchError("Too many redirects.")


def _candidate_links(home, base_host: str, limit: int) -> list[str]:
    seen, scored = set(), []
    for l in home.facts.links if home.facts else []:
        href = l["href"]
        p = urlsplit(href)
        if p.scheme not in ("http", "https") or not same_site(href, base_host):
            continue
        clean = urlunsplit((p.scheme, p.netloc.lower(), p.path or "/", "", ""))
        if clean in seen or clean.rstrip("/") == home.final_url.split("?")[0].split("#")[0].rstrip("/"):
            continue
        if p.path.lower().endswith(SKIP_EXT) or "cart" in p.path.lower() or "checkout" in p.path.lower() \
                or "account" in p.path.lower() or "login" in p.path.lower():
            continue          # never crawl carts, checkouts or account pages
        seen.add(clean)
        text = (p.path + " " + (l["text"] or "")).lower()
        # whole-word-ish match at a word start ("shop" matches /shop/, /shopping, not /workshops/)
        score = next((i for i, k in enumerate(PRIORITY) if re.search(rf"(?<![a-z]){k}", text)), len(PRIORITY))
        scored.append((score, len(scored), clean))
    return [u for _, _, u in sorted(scored)[:limit]]


def analyze(url: str, *, fetcher: SafeFetcher | None = None, max_pages: int = MAX_PAGES,
            pagespeed: Callable[[str], dict] | None = None, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Run the full analysis. Raises UnsafeURLError for disallowed input URLs."""
    started = time.monotonic()
    start_url = normalise_url(url)
    validate_url(start_url)                      # static checks before any network activity
    fetcher = fetcher or SafeFetcher()
    robots = RobotsCache(fetcher)
    polite = _PoliteFetcher(fetcher, robots)
    pages: list[checks.PageResult] = []
    skipped: list[dict] = []

    def record(r) -> checks.PageResult:
        is_html = r.content_type in ("text/html", "application/xhtml+xml") or (not r.content_type and r.body[:200].lstrip().startswith(b"<"))
        facts = extract(r.final_url, r.text()) if is_html and r.status < 400 else None
        pr = checks.PageResult(r.requested_url, r.final_url, r.status, r.elapsed_ms, r.content_type, len(r.body),
                               r.truncated, r.headers, r.redirects, facts)
        pages.append(pr)
        return pr

    fatal: str | None = None
    try:
        home = record(polite.fetch_page(start_url))
    except RobotsDisallowed as exc:
        home, fatal = None, f"Not analysed: {exc}. ShopSense AI respects robots.txt."
    except FetchError as exc:
        home, fatal = None, f"The site could not be reached: {exc}"

    base_host = urlsplit(home.final_url).hostname if home else urlsplit(start_url).hostname
    robots_info = robots.info.get(_origin(home.final_url if home else start_url)) or \
        RobotsInfo(_origin(start_url) + "/robots.txt", "unreachable", None, note="Not checked.")
    delay = max(MIN_DELAY_S, robots_info.crawl_delay or 0)
    if home and home.facts and home.status < 400:
        if delay > MAX_CRAWL_DELAY:
            skipped.append({"url": home.final_url, "reason": f"robots.txt crawl-delay {delay:.0f}s: only the start page analysed"})
        else:
            for link in _candidate_links(home, base_host, max_pages - 1):
                if time.monotonic() - started > TIME_BUDGET_S:
                    skipped.append({"url": link, "reason": "analysis time budget reached"})
                    continue
                try:
                    if not robots.allowed(link):
                        skipped.append({"url": link, "reason": "disallowed by robots.txt"})
                        continue
                except UnsafeURLError as exc:
                    skipped.append({"url": link, "reason": str(exc)})
                    continue
                sleep(delay)
                try:
                    record(polite.fetch_page(link))
                except RobotsDisallowed:
                    skipped.append({"url": link, "reason": "redirect target disallowed by robots.txt"})
                except (FetchError, UnsafeURLError) as exc:
                    skipped.append({"url": link, "reason": str(exc)})

    issues = checks.sort_issues([i for p in pages for i in checks.page_issues(p)] +
                                checks.site_issues(pages, asdict(robots_info)))
    checklist = checks.conversion_checklist(pages, base_host or "") if pages else []

    perf = {"status": "unavailable", "source": "Google PageSpeed Insights (Lighthouse)",
            "reason": "Not requested, or no PAGESPEED_API_KEY configured on the server. No performance or "
                      "Core Web Vitals figures are estimated."}
    if pagespeed and home and home.status < 400 and not fatal:
        perf = pagespeed(home.final_url)

    return {
        "input_url": url,
        "analyzed_url": start_url,
        "final_url": home.final_url if home else None,
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_ms": round((time.monotonic() - started) * 1000),
        "user_agent": USER_AGENT,
        "fatal_error": fatal,
        "robots": asdict(robots_info),
        "pages": [_page_summary(p) for p in pages],
        "skipped": skipped,
        "issues": [asdict(i) for i in issues],
        "summary": {
            "pages_analyzed": len(pages),
            "pages_skipped": len(skipped),
            "issues_by_severity": {s: sum(i.severity == s for i in issues) for s in checks.SEVERITY_ORDER},
            "issues_by_category": dict(Counter(i.category for i in issues)),
            "checklist_observed": sum(c.status == "observed" for c in checklist),
            "checklist_total": len(checklist),
        },
        "conversion_checklist": [asdict(c) for c in checklist],
        "performance": perf,
        "data_availability": [
            {"item": "Actual conversions, revenue, traffic", "status": "unavailable",
             "reason": "Requires analytics data (e.g. GA4) or store data (e.g. Shopify). Not connected."},
            {"item": "Real-user speed / Core Web Vitals", "status": "available" if perf["status"] == "available" else "unavailable",
             "reason": "From PageSpeed Insights when configured; otherwise not measured."},
            {"item": "JavaScript-rendered content", "status": "unavailable", "reason": "JavaScript is not executed."},
            {"item": "Pages behind login, cart and checkout", "status": "unavailable",
             "reason": "Not crawled by design (no accounts, no sessions)."},
        ],
        "disclaimers": DISCLAIMERS,
    }


def _page_summary(p: checks.PageResult) -> dict:
    f = p.facts
    base = {"url": p.url, "final_url": p.final_url, "status": p.status, "elapsed_ms": p.elapsed_ms,
            "content_type": p.content_type, "bytes": p.bytes, "truncated": p.truncated,
            "https": p.final_url.startswith("https://"), "hsts": "strict-transport-security" in p.headers,
            "redirects": len(p.redirects), "is_html": f is not None}
    if f is None:
        return base
    types, errors = f.jsonld()
    return base | {
        "title": f.title, "title_length": len(f.title or ""),
        "meta_description": f.meta.get("description"), "meta_description_length": len(f.meta.get("description", "")),
        "canonical": f.canonical, "robots_meta": f.meta.get("robots"), "lang": f.lang,
        "viewport": f.meta.get("viewport"), "h1": f.h1[:5],
        "headings": {f"h{i}": sum(1 for lvl, _ in f.headings if lvl == i) for i in range(1, 7)},
        "images_total": len(f.images), "images_missing_alt": sum(i["alt"] is None for i in f.images),
        "images_empty_alt": sum(i["alt"] == "" for i in f.images),
        "structured_data_types": types + sorted(set(f.microdata_types)), "structured_data_errors": len(errors),
        "open_graph": {k: f.meta.get(k) for k in ("og:title", "og:description", "og:image") if f.meta.get(k)},
        "links_internal": sum(1 for l in f.links if same_site(l["href"], urlsplit(p.final_url).hostname or "")),
        "links_external": sum(1 for l in f.links if l["href"].startswith("http") and
                              not same_site(l["href"], urlsplit(p.final_url).hostname or "")),
        "word_count": f.word_count, "likely_js_rendered": f.likely_js_rendered(),
    }
