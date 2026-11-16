"""Website Intelligence tests: SSRF protection, robots.txt, crawling limits, analysis, API.

A local HTTP server plays the role of public websites. A fake resolver maps test domains to
public-looking IPs and a test connection factory routes those "pinned" connections to the
local server, so the real validation/redirect/limit logic runs without touching the internet.
"""

import gzip
import http.client
import http.server
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.api import website as website_api
from app.main import app
from app.website import pagespeed
from app.website.analyzer import analyze
from app.website.html_extract import extract
from app.website.safe_fetch import (FetchError, SafeFetcher, UnsafeURLError, normalise_url, resolve_public,
                                    validate_url)

PUBLIC_IP = "93.184.215.14"


# ---------------------------------------------------------------- test "internet"
class Site:
    """Routes: path -> (status, headers, body) or callable(handler) for special behaviour."""

    def __init__(self):
        self.routes: dict[str, object] = {}
        self.requests: list[str] = []


SITES: dict[str, Site] = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        host = self.headers.get("Host", "").split(":")[0]
        site = SITES.get(host)
        if site is None:
            self.send_response(404); self.end_headers(); return
        site.requests.append(self.path)
        route = site.routes.get(self.path.split("?")[0])
        if route is None:
            self.send_response(404); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(b"<html><title>Not found</title></html>")
            return
        if callable(route):
            return route(self)
        status, headers, body = route
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def web(server):
    SITES.clear()
    resolved: dict[str, list[str]] = {}
    calls: list[tuple] = []

    def resolver(host, port):
        resolved.setdefault(host, [])
        resolved[host].append(port)
        if host.startswith("private-"):
            return ["10.0.0.5"]
        if host.startswith("meta-"):
            return ["169.254.169.254"]
        if host in SITES:
            return [PUBLIC_IP]
        raise FetchError(f"Could not resolve {host} (DNS lookup failed).")

    def factory(scheme, host, ip, port, timeout):
        calls.append((scheme, host, ip, port))
        assert ip == PUBLIC_IP, "connection must be pinned to the validated IP"
        return http.client.HTTPConnection("127.0.0.1", server, timeout=timeout)   # test-only routing

    def add(host):
        SITES[host] = Site()
        return SITES[host]

    fetcher = SafeFetcher(resolver=resolver, connection_factory=factory, timeout=2.0, total_deadline_s=5.0)
    return type("Web", (), {"add": staticmethod(add), "fetcher": fetcher, "calls": calls, "resolved": resolved})


def html(body: str, head: str = "") -> tuple:
    return (200, {"Content-Type": "text/html; charset=utf-8"},
            f"<!doctype html><html lang='en'><head>{head}</head><body>{body}</body></html>".encode())


GOOD_HEAD = ("<title>Acme Outdoor Gear – Tents and Backpacks</title>"
             "<meta name='description' content='Quality tents, backpacks and outdoor gear with free shipping on orders over $50 and easy returns.'>"
             "<meta name='viewport' content='width=device-width, initial-scale=1'>"
             "<link rel='canonical' href='http://shop.example.com/'><link rel='icon' href='/favicon.ico'>"
             "<meta property='og:title' content='Acme'><meta property='og:image' content='/og.png'>"
             "<script type='application/ld+json'>{\"@context\":\"https://schema.org\",\"@type\":\"Product\","
             "\"name\":\"Tent\",\"offers\":{\"@type\":\"Offer\",\"price\":\"199\"}}</script>")
GOOD_BODY = ("<header><nav><a href='/'>Home</a><a href='/products/tent'>Tents</a><a href='/collections/all'>Shop all</a>"
             "<a href='/about'>About</a><a href='/contact'>Contact us</a><a href='/cart'>Cart</a></nav></header>"
             "<h1>Outdoor gear</h1><h2>Bestsellers</h2><p>Tent only $199.00. " + "Great gear for every trip. " * 20 + "</p>"
             "<img src='/tent.jpg' alt='Green tent'><button>Add to cart</button>"
             "<form role='search' action='/search'><input type='search' name='q'></form>"
             "<footer><a href='/policies/privacy-policy'>Privacy</a><a href='/policies/refund-policy'>Refunds</a>"
             "<a href='mailto:help@example.com'>Email</a><a href='/account/login'>Log in</a></footer>")


# ======================================================== URL validation (SSRF)
@pytest.mark.parametrize("url", [
    "ftp://example.com/", "file:///etc/passwd", "gopher://example.com", "javascript:alert(1)",
    "http://user:pass@example.com/", "https://example.com@evil.com/",
    "http://127.0.0.1/", "http://[::1]/", "http://10.0.0.1/", "http://169.254.169.254/latest/meta-data/",
    "http://2130706433/", "http://0x7f000001/", "http://0177.0.0.1/", "http://127.1/",
    "http://localhost/", "http://LOCALHOST./", "http://intranet/", "http://printer.local/", "http://db.internal/",
    "http://example.com:8080/", "https://example.com:22/", "", "   ",
])
def test_unsafe_urls_rejected_before_any_network_access(url):
    with pytest.raises(UnsafeURLError):
        validate_url(normalise_url(url))


@pytest.mark.parametrize("url, expected", [
    ("example.com", "https://example.com/"),
    ("  https://www.example.com/shop?x=1#frag ", "https://www.example.com/shop?x=1"),
    ("HTTP://Example.com", "http://Example.com/"),
])
def test_valid_urls_normalised(url, expected):
    assert normalise_url(url) == expected
    validate_url(normalise_url(url))


@pytest.mark.parametrize("addresses", [
    ["127.0.0.1"], ["10.1.2.3"], ["172.16.0.9"], ["192.168.1.1"], ["169.254.169.254"], ["100.64.0.1"],
    ["0.0.0.0"], ["224.0.0.1"], ["240.0.0.1"], ["255.255.255.255"], ["::1"], ["fe80::1"], ["fc00::1"],
    ["::ffff:127.0.0.1"], ["2002:7f00:1::"], ["64:ff9b::7f00:1"], ["2001:db8::1"],
    [PUBLIC_IP, "10.0.0.5"],                    # ANY private address in the answer blocks the host
])
def test_non_public_resolutions_blocked(addresses):
    with pytest.raises(UnsafeURLError):
        resolve_public("site.example.com", 443, lambda h, p: addresses)


def test_public_resolution_allowed_and_ipv4_preferred():
    assert resolve_public("site.example.com", 443, lambda h, p: ["2606:2800:21f:cb07:6820:80da:af6b:8b2c", PUBLIC_IP]) == PUBLIC_IP


# ======================================================== fetcher behaviour
def test_redirect_to_private_host_is_blocked_before_connecting(web):
    s = web.add("shop.example.com")
    s.routes["/"] = (302, {"Location": "http://private-admin.example.com/"}, b"")
    with pytest.raises(UnsafeURLError):
        web.fetcher.fetch("http://shop.example.com/")
    assert all(host == "shop.example.com" for _, host, _, _ in web.calls)    # never connected to the private host


def test_redirect_to_metadata_ip_literal_blocked(web):
    s = web.add("shop.example.com")
    s.routes["/"] = (301, {"Location": "http://169.254.169.254/latest/meta-data/"}, b"")
    with pytest.raises(UnsafeURLError):
        web.fetcher.fetch("http://shop.example.com/")


def test_redirect_loop_limited(web):
    s = web.add("shop.example.com")
    s.routes["/"] = (302, {"Location": "/a"}, b"")
    s.routes["/a"] = (302, {"Location": "/"}, b"")
    with pytest.raises(FetchError, match="redirects"):
        web.fetcher.fetch("http://shop.example.com/")


def test_dns_resolved_once_per_hop_and_connection_pinned(web):
    s = web.add("shop.example.com")
    s.routes["/"] = html("<p>hi</p>")
    web.fetcher.fetch("http://shop.example.com/")
    assert web.resolved["shop.example.com"] == [80] and web.calls[0][2] == PUBLIC_IP


def test_response_size_capped(web):
    s = web.add("shop.example.com")
    s.routes["/"] = (200, {"Content-Type": "text/html"}, b"<p>" + b"x" * 3_000_000)
    r = web.fetcher.fetch("http://shop.example.com/", max_bytes=100_000)
    assert r.truncated and len(r.body) == 100_000


def test_gzip_bomb_is_bounded(web):
    s = web.add("shop.example.com")
    bomb = gzip.compress(b"\0" * 50_000_000)        # ~50 KB compressed, 50 MB expanded
    s.routes["/"] = (200, {"Content-Type": "text/html", "Content-Encoding": "gzip"}, bomb)
    r = web.fetcher.fetch("http://shop.example.com/", max_bytes=200_000)
    assert r.truncated and len(r.body) == 200_000


def test_slow_server_times_out(web):
    s = web.add("shop.example.com")
    def slow(h):
        time.sleep(3.5)
        h.send_response(200); h.end_headers(); h.wfile.write(b"late")
    s.routes["/"] = slow
    with pytest.raises(FetchError):
        web.fetcher.fetch("http://shop.example.com/")


def test_unresolvable_host_is_a_fetch_error(web):
    with pytest.raises(FetchError, match="DNS"):
        web.fetcher.fetch("http://nonexistent.example.com/")


# ======================================================== robots.txt + crawling
def test_robots_disallow_all_prevents_any_page_fetch(web):
    s = web.add("shop.example.com")
    s.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
    s.routes["/"] = html(GOOD_BODY, GOOD_HEAD)
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=lambda s: None)
    assert r["fatal_error"] and "robots.txt" in r["fatal_error"]
    assert s.requests == ["/robots.txt"] and r["pages"] == []


def test_robots_5xx_treated_as_disallow(web):
    s = web.add("shop.example.com")
    s.routes["/robots.txt"] = (503, {"Content-Type": "text/plain"}, b"down")
    s.routes["/"] = html(GOOD_BODY, GOOD_HEAD)
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=lambda s: None)
    assert r["robots"]["status"] == "unreachable" and r["pages"] == [] and "/" not in s.requests


def test_robots_404_allows_crawl_and_specific_rules_respected(web):
    s = web.add("shop.example.com")
    s.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"},
                               b"User-agent: *\nDisallow: /about\nSitemap: http://shop.example.com/sitemap.xml\n")
    s.routes["/"] = html(GOOD_BODY, GOOD_HEAD)
    for p in ("/products/tent", "/collections/all", "/contact"):
        s.routes[p] = html("<h1>Page</h1>" + "<p>text</p>" * 5, GOOD_HEAD.replace("Acme Outdoor", p))
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=lambda s: None)
    fetched = set(s.requests)
    assert "/about" not in fetched                                  # robots rule respected
    assert any(sk["url"].endswith("/about") and "robots" in sk["reason"] for sk in r["skipped"])
    assert not fetched & {"/cart", "/account/login"}                # carts/accounts never crawled
    assert r["robots"]["sitemaps"] == ["http://shop.example.com/sitemap.xml"]
    assert 1 < r["summary"]["pages_analyzed"] <= 5


def test_large_crawl_delay_limits_to_start_page(web):
    s = web.add("shop.example.com")
    s.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nCrawl-delay: 30\n")
    s.routes["/"] = html(GOOD_BODY, GOOD_HEAD)
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=lambda s: None)
    assert r["summary"]["pages_analyzed"] == 1 and "crawl-delay" in r["skipped"][0]["reason"]


def test_redirect_into_disallowed_path_is_not_followed(web):
    s = web.add("shop.example.com")
    s.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /private\n")
    s.routes["/"] = (302, {"Location": "/private/home"}, b"")
    s.routes["/private/home"] = html("<p>secret</p>")
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=lambda s: None)
    assert "/private/home" not in s.requests and "robots.txt" in (r["fatal_error"] or "")


def test_link_priority_uses_word_starts(web):
    s = web.add("shop.example.com")
    s.routes["/"] = html("<h1>Home</h1><a href='/community/workshops/'>Workshops</a><a href='/blog/x'>Blog</a>"
                         "<a href='/shop/'>Shop</a>", GOOD_HEAD)
    s.routes["/shop/"] = html("<h1>Shop</h1>", GOOD_HEAD)
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, max_pages=2, sleep=lambda s: None)
    assert r["pages"][1]["final_url"].endswith("/shop/")       # "workshops" must not outrank a real /shop/ page


def test_page_limit_and_politeness_delay(web):
    s = web.add("shop.example.com")
    links = "".join(f"<a href='/products/p{i}'>Product {i}</a>" for i in range(20))
    s.routes["/"] = html(links + GOOD_BODY, GOOD_HEAD)
    for i in range(20):
        s.routes[f"/products/p{i}"] = html("<h1>P</h1>", GOOD_HEAD)
    delays = []
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=delays.append)
    assert r["summary"]["pages_analyzed"] == 5 and len([p for p in s.requests if p.startswith("/products/")]) == 4
    assert delays and all(d >= 1.0 for d in delays)


# ======================================================== analysis content
def test_issues_have_severity_url_evidence_and_fix(web):
    s = web.add("shop.example.com")
    s.routes["/"] = html("<h2>No h1</h2><h4>skip</h4><img src='/a.png'><img src='/b.png' alt=''>"
                         "<script type='application/ld+json'>{bad json</script>",
                         "<title>Hi</title><script src='http://cdn.example.com/x.js'></script>")
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=lambda s: None)
    ids = {i["id"] for i in r["issues"]}
    assert {"no-https", "meta-description-missing", "viewport-missing", "h1-missing", "heading-skip",
            "img-alt-missing", "jsonld-invalid", "title-length", "canonical-missing"} <= ids
    for i in r["issues"]:
        assert i["severity"] in ("high", "medium", "low", "info") and i["url"] and i["evidence"] and i["recommendation"]
    alt = next(i for i in r["issues"] if i["id"] == "img-alt-missing")
    assert alt["evidence_type"] == "observed" and "1 of 2" in alt["title"]          # empty alt="" is decorative, not missing
    assert next(i for i in r["issues"] if i["id"] == "title-length")["evidence_type"] == "heuristic"


def test_conversion_checklist_observed_and_labelled_heuristic(web):
    s = web.add("shop.example.com")
    s.routes["/"] = html(GOOD_BODY, GOOD_HEAD)
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, max_pages=1, sleep=lambda s: None)
    c = {i["id"]: i for i in r["conversion_checklist"]}
    for k in ("cta", "navigation", "product-info", "pricing", "cart-checkout", "search", "contact", "policies", "mobile"):
        assert c[k]["status"] == "observed", k
        assert c[k]["evidence"]
    assert c["cta"]["evidence_type"] == "heuristic" and c["https"]["status"] == "not_observed"
    assert any("conversions" in d for d in r["disclaimers"])
    assert next(a for a in r["data_availability"] if a["item"].startswith("Actual conversions"))["status"] == "unavailable"


def test_bare_page_reports_not_observed_not_absent(web):
    s = web.add("shop.example.com")
    s.routes["/"] = html("<div id='root'></div><script src='/app.js'></script>", "<title>App shell page here</title>")
    r = analyze("http://shop.example.com/", fetcher=web.fetcher, sleep=lambda s: None)
    c = {i["id"]: i for i in r["conversion_checklist"]}
    assert c["cta"]["status"] == "not_observed" and "JavaScript" in c["cta"]["note"]
    assert any(i["id"] == "js-rendered" for i in r["issues"])


def test_extractor_handles_hostile_markup():
    f = extract("https://x.example.com/", "<html><title>a</title><p>" + "<div>" * 5000 + "<a href='javascript:alert(1)'>x</a>"
                "<img src=x onerror=alert(1)><script>document.write('<h1>fake</h1>')</script>")
    assert f.title == "a" and all(not l["href"].startswith("javascript:") for l in f.links) and f.h1 == []


# ======================================================== PageSpeed (no invented numbers)
def test_pagespeed_unavailable_without_key():
    out = pagespeed.run_pagespeed("https://example.com/", "")
    assert out["status"] == "unavailable" and "PAGESPEED_API_KEY" in out["reason"]


def test_pagespeed_parsing_passes_values_through():
    sample = {"lighthouseResult": {"lighthouseVersion": "12.0.0", "fetchTime": "2026-01-01T00:00:00Z",
                                   "categories": {"performance": {"score": 0.87}},
                                   "audits": {"largest-contentful-paint": {"displayValue": "2.1 s", "numericValue": 2100.5}}},
              "loadingExperience": {"overall_category": "AVERAGE",
                                    "metrics": {"CUMULATIVE_LAYOUT_SHIFT_SCORE": {"percentile": 5, "category": "FAST"}}}}
    out = pagespeed.parse_psi(sample, "mobile")
    assert out["performance_score"] == 87 and out["lab"] == [{"metric": "Largest Contentful Paint", "value": "2.1 s", "numeric": 2100.5}]
    assert out["field"][0]["category"] == "FAST" and out["field_available"]
    empty = pagespeed.parse_psi({"lighthouseResult": {}}, "mobile")
    assert empty["performance_score"] is None and empty["lab"] == [] and not empty["field_available"]


# ======================================================== API
@pytest.fixture
def client(web):
    app.dependency_overrides[website_api.get_fetcher] = lambda: web.fetcher
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def fresh_limits(monkeypatch):
    monkeypatch.setattr(website_api, "limiter", website_api.RateLimiter(100, 60))


def test_api_returns_full_report(client, web):
    s = web.add("shop.example.com")
    s.routes["/"] = html(GOOD_BODY, GOOD_HEAD)
    r = client.post("/api/analyze-website", json={"url": "http://shop.example.com"})
    assert r.status_code == 200
    b = r.json()
    assert b["summary"]["pages_analyzed"] >= 1 and b["performance"]["status"] == "unavailable"
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://localhost:8000/health", "http://user:pw@example.com/",
                                 "http://example.com:8000/", "file:///etc/passwd", "http://private-x.example.com/",
                                 "http://meta-data.example.com/latest/meta-data/"])
def test_api_rejects_unsafe_targets_with_400(client, web, url):
    r = client.post("/api/analyze-website", json={"url": url})
    assert r.status_code == 400, r.text
    assert r.json()["detail"]
    assert web.calls == []                       # no connection was ever opened


def test_private_dns_answer_reported_as_such(client):
    r = client.post("/api/analyze-website", json={"url": "http://private-x.example.com/"})
    assert r.status_code == 400 and "non-public" in r.json()["detail"]


@pytest.mark.parametrize("body", [{}, {"url": "x" * 3000}, {"url": "example.com", "extra": 1}, {"url": 123}])
def test_api_validates_request_body(client, body):
    assert client.post("/api/analyze-website", json=body).status_code == 422


def test_api_rate_limit(client, web, monkeypatch):
    monkeypatch.setattr(website_api, "limiter", website_api.RateLimiter(2, 60))
    s = web.add("shop.example.com")
    s.routes["/"] = html(GOOD_BODY, GOOD_HEAD)
    codes = [client.post("/api/analyze-website", json={"url": "http://shop.example.com"}).status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429


def test_api_concurrency_limit(client, monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(website_api, "_slots", sem)
    sem.acquire()
    try:
        r = client.post("/api/analyze-website", json={"url": "http://shop.example.com"})
        assert r.status_code == 429 and "retry-after" in r.headers
    finally:
        sem.release()


def test_prediction_endpoints_unchanged(client):
    from app.schemas import EXAMPLE_SESSION
    r = client.post("/api/predict", json=EXAMPLE_SESSION)
    if r.status_code == 200:
        assert round(r.json()["purchase_probability"], 4) == 0.2256
