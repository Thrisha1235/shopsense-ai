"""Turn observed page facts into an issues report and a conversion-readiness checklist.

Evidence types:
  observed   - a direct fact about the delivered HTML/HTTP response (e.g. "no <title> element")
  heuristic  - a judgement based on common guidelines or keyword matching (e.g. "title longer
               than ~60 characters may be truncated"; "a link containing 'cart' looks like a
               cart entry point"). Heuristics can be wrong and say so.
Nothing here measures traffic, conversions, revenue or real-user speed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from app.website.html_extract import PageFacts, _clip, same_site

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class PageResult:
    url: str
    final_url: str
    status: int
    elapsed_ms: float
    content_type: str
    bytes: int
    truncated: bool
    headers: dict
    redirects: list
    facts: PageFacts | None          # None when the response was not HTML


@dataclass
class Issue:
    id: str
    severity: str
    category: str                    # technical | seo | usability
    title: str
    url: str
    evidence: str
    recommendation: str
    evidence_type: str = "observed"


@dataclass
class ChecklistItem:
    id: str
    label: str
    status: str                      # observed | not_observed | unavailable
    evidence: list[str] = field(default_factory=list)
    note: str = ""
    evidence_type: str = "heuristic"


# ------------------------------------------------------------------- issues
def page_issues(p: PageResult) -> list[Issue]:
    out: list[Issue] = []
    u = p.final_url
    add = lambda *a, **k: out.append(Issue(*a, url=u, **k))  # noqa: E731

    if p.status >= 400:
        add("http-error", "high", "technical", f"Page returned HTTP {p.status}",
            evidence=f"GET {u} → {p.status}", recommendation="Fix or redirect the URL; broken pages lose visitors and rankings.")
    if urlsplit(u).scheme != "https":
        add("no-https", "high", "technical", "Page is served over plain HTTP",
            evidence=f"Final URL: {u}", recommendation="Serve every page over HTTPS and redirect HTTP to HTTPS.")
    elif "strict-transport-security" not in p.headers:
        add("no-hsts", "low", "technical", "No HSTS header",
            evidence="Response has no Strict-Transport-Security header",
            recommendation="Add Strict-Transport-Security (e.g. max-age=31536000; includeSubDomains) once HTTPS is stable.")
    if len(p.redirects) > 1:
        chain = " → ".join([p.redirects[0]["from"]] + [r["to"] for r in p.redirects])
        add("redirect-chain", "low", "technical", f"Redirect chain of {len(p.redirects)} hops",
            evidence=_clip(chain, 300), recommendation="Link and redirect directly to the final URL.")
    if p.elapsed_ms > 2500:
        add("slow-response", "low", "technical", f"Slow response from the analysis server ({p.elapsed_ms:.0f} ms)",
            evidence=f"Time to fetch the full HTML from our server: {p.elapsed_ms:.0f} ms",
            recommendation="Check server/CDN caching. This is one measurement from one location, not real-user speed.",
            evidence_type="heuristic")
    if p.truncated:
        add("large-html", "low", "technical", "HTML document is very large",
            evidence="HTML exceeded the 2 MB analysis limit and was truncated",
            recommendation="Reduce inline data/markup; large HTML slows the first render.")

    f = p.facts
    if f is None:
        if p.status < 400:
            add("not-html", "info", "technical", "Response is not an HTML page",
                evidence=f"Content-Type: {p.content_type or 'missing'}", recommendation="No HTML checks were possible for this URL.")
        return out

    # ---- SEO
    if not f.title:
        add("title-missing", "high", "seo", "Missing <title>", evidence="No non-empty <title> element",
            recommendation="Add a unique, descriptive title (roughly 30–60 characters).")
    elif not 15 <= len(f.title) <= 65:
        add("title-length", "low", "seo", f"Title length {len(f.title)} characters", evidence=f'"{_clip(f.title, 90)}"',
            recommendation="Aim for roughly 30–60 characters so search results show it in full.", evidence_type="heuristic")
    desc = f.meta.get("description", "").strip()
    if not desc:
        add("meta-description-missing", "medium", "seo", "Missing meta description",
            evidence='No <meta name="description"> with content',
            recommendation="Add a 70–160 character summary that encourages the click.")
    elif not 50 <= len(desc) <= 170:
        add("meta-description-length", "low", "seo", f"Meta description length {len(desc)} characters",
            evidence=f'"{_clip(desc, 120)}"', recommendation="Aim for roughly 70–160 characters.", evidence_type="heuristic")
    if not f.canonical:
        add("canonical-missing", "low", "seo", "No canonical link", evidence='No <link rel="canonical">',
            recommendation="Add a self-referencing canonical URL to avoid duplicate-content ambiguity.")
    elif not same_site(f.canonical, urlsplit(u).hostname or ""):
        add("canonical-offsite", "medium", "seo", "Canonical points to another site",
            evidence=f"canonical = {f.canonical}", recommendation="Confirm this is intentional; it tells search engines to index the other URL.")
    robots_meta = f.meta.get("robots", "").lower()
    if "noindex" in robots_meta:
        add("noindex", "high", "seo", "Page is marked noindex", evidence=f'<meta name="robots" content="{robots_meta}">',
            recommendation="Remove noindex if this page should appear in search results.")
    if not f.h1:
        add("h1-missing", "medium", "seo", "No <h1> heading", evidence="0 <h1> elements in the HTML",
            recommendation="Add one clear <h1> describing the page.")
    elif len(f.h1) > 1:
        add("h1-multiple", "low", "seo", f"{len(f.h1)} <h1> headings",
            evidence="; ".join(_clip(t, 50) for t in f.h1[:4]), recommendation="Usually keep a single <h1> per page.",
            evidence_type="heuristic")
    skips = f.heading_skips()
    if skips:
        add("heading-skip", "low", "usability", "Heading levels are skipped", evidence="; ".join(skips[:4]),
            recommendation="Use headings in order (h1 → h2 → h3) so screen-reader users can navigate.")
    types, errors = f.jsonld()
    for e in errors[:3]:
        add("jsonld-invalid", "medium", "seo", "Invalid JSON-LD structured data", evidence=e,
            recommendation="Fix the JSON syntax; search engines ignore invalid structured data.")
    if not types and not f.microdata_types:
        add("no-structured-data", "low", "seo", "No structured data found",
            evidence="No JSON-LD or microdata (itemtype) in the HTML",
            recommendation="Add schema.org markup (e.g. Organization, Product, BreadcrumbList) where relevant.")
    if not f.meta.get("og:title") or not f.meta.get("og:image"):
        missing = [k for k in ("og:title", "og:image") if not f.meta.get(k)]
        add("open-graph", "low", "seo", "Incomplete Open Graph tags", evidence=f"Missing: {', '.join(missing)}",
            recommendation="Add og:title, og:description and og:image for better link previews.")

    # ---- usability / accessibility
    vp = f.meta.get("viewport", "").lower().replace(" ", "")
    if not vp:
        add("viewport-missing", "high", "usability", "No mobile viewport meta tag", evidence='No <meta name="viewport">',
            recommendation='Add <meta name="viewport" content="width=device-width, initial-scale=1">.')
    else:
        if "width=device-width" not in vp:
            add("viewport-width", "medium", "usability", "Viewport not set to device width", evidence=f'viewport="{vp}"',
                recommendation="Use width=device-width so the page fits mobile screens.")
        m = re.search(r"maximum-scale=([\d.]+)", vp)
        if "user-scalable=no" in vp or "user-scalable=0" in vp or (m and float(m.group(1)) < 2):
            add("zoom-disabled", "medium", "usability", "Pinch-zoom is restricted", evidence=f'viewport="{vp}"',
                recommendation="Allow zoom (remove user-scalable=no / maximum-scale<2) for low-vision users.")
    if not f.lang:
        add("lang-missing", "low", "usability", "No language declared", evidence="<html> has no lang attribute",
            recommendation='Add lang (e.g. <html lang="en">) for screen readers and translation.')
    missing_alt = [i["src"] for i in f.images if i["alt"] is None]
    if missing_alt:
        add("img-alt-missing", "medium", "usability", f"{len(missing_alt)} of {len(f.images)} images have no alt attribute",
            evidence="e.g. " + ", ".join(_clip(s.rsplit("/", 1)[-1] or s, 50) for s in missing_alt[:3]),
            recommendation='Add descriptive alt text (or alt="" for purely decorative images).')
    if urlsplit(u).scheme == "https":
        mixed = [r for r in f.resource_urls if r.startswith("http://")]
        if mixed:
            add("mixed-content", "medium", "technical", f"{len(mixed)} insecure (http://) resources on an HTTPS page",
                evidence="e.g. " + ", ".join(_clip(r, 70) for r in mixed[:2]),
                recommendation="Load every script, style, image and frame over HTTPS.")
    if not f.has_favicon:
        add("favicon", "info", "usability", "No favicon link", evidence='No <link rel="icon">',
            recommendation="Add a favicon for recognisable browser tabs and bookmarks.")
    if f.likely_js_rendered():
        add("js-rendered", "info", "technical", "Content appears to be rendered by JavaScript",
            evidence=f"Only {f.word_count} words and {len(f.links)} links in the server HTML",
            recommendation="Findings for this page are limited. Server-side rendering also helps search engines.",
            evidence_type="heuristic")
    return out


def site_issues(pages: list[PageResult], robots: dict) -> list[Issue]:
    out: list[Issue] = []
    html_pages = [p for p in pages if p.facts and p.facts.title]
    titles = Counter(p.facts.title for p in html_pages)
    for title, n in titles.items():
        if n > 1:
            urls = [p.final_url for p in html_pages if p.facts.title == title]
            out.append(Issue("title-duplicate", "medium", "seo", f"Same title on {n} pages", url=urls[0],
                             evidence=f'"{_clip(title, 80)}" on: ' + ", ".join(urls[:3]),
                             recommendation="Give every page a unique title."))
    if robots.get("status") == "not_found":
        out.append(Issue("robots-missing", "info", "technical", "No robots.txt", url=robots.get("url", ""),
                         evidence=f"{robots.get('url')} returned {robots.get('http_status')}",
                         recommendation="Add a robots.txt (and reference your sitemap in it)."))
    if robots.get("status") == "found" and not robots.get("sitemaps"):
        out.append(Issue("sitemap-not-referenced", "low", "seo", "robots.txt does not reference a sitemap",
                         url=robots.get("url", ""), evidence="No 'Sitemap:' line in robots.txt",
                         recommendation="Add 'Sitemap: https://…/sitemap.xml' to robots.txt."))
    return out


def sort_issues(issues: list[Issue]) -> list[Issue]:
    return sorted(issues, key=lambda i: (SEVERITY_ORDER[i.severity], i.category, i.id, i.url))


# --------------------------------------------------------- conversion checklist
CTA = re.compile(r"\b(add to (cart|bag|basket)|buy( now)?|shop( now)?|order( now)?|get started|sign ?up|subscribe|"
                 r"book( now| a demo)?|request (a )?(quote|demo)|contact us|get (a )?quote|start (free )?trial|"
                 r"try (it )?free|checkout|enquire|inquire|downloads?|donate|join( now| us)?|register|"
                 r"learn more|request (a )?(call|callback))\b", re.I)
CART = re.compile(r"(/cart|/checkout|/basket|/bag\b|cart\.php|/trolley)|\b(cart|checkout|basket)\b", re.I)
PRICE = re.compile(r"(?:[$€£¥₹]|\b(?:USD|EUR|GBP|OMR|AED|SAR|INR|QAR|KWD|BHD|RO|R\.O\.)\s?)\s?\d[\d,.]*", re.I)
CONTACT = re.compile(r"\b(contact|support|help|get in touch)\b", re.I)
POLICY = {
    "privacy": re.compile(r"privacy", re.I),
    "terms": re.compile(r"terms|conditions", re.I),
    "returns/refunds": re.compile(r"return|refund", re.I),
    "shipping/delivery": re.compile(r"shipping|delivery", re.I),
}


def _link_label(link: dict) -> str:
    return f'"{link["text"] or "(no text)"}" → {link["href"]}'


def conversion_checklist(pages: list[PageResult], base_host: str) -> list[ChecklistItem]:
    html = [p for p in pages if p.facts is not None and p.status < 400]
    if not html:
        return [ChecklistItem("all", "Conversion checks", "unavailable", note="No HTML page could be analysed.")]
    links = [l for p in html for l in p.facts.links]
    buttons = [b for p in html for b in p.facts.buttons]
    texts = " ".join(p.facts.text_sample for p in html)
    types = {t for p in html for t in p.facts.jsonld()[0]} | {t for p in html for t in p.facts.microdata_types}
    items: list[ChecklistItem] = []

    def item(id_, label, found: list[str], note_not: str, note_found: str = ""):
        found = list(dict.fromkeys(found))          # same link on several pages -> show it once
        items.append(ChecklistItem(id_, label, "observed" if found else "not_observed", found[:3],
                                   note_found if found else note_not))

    cta_hits = [f'button "{b}"' for b in buttons if CTA.search(b)] + \
               [_link_label(l) for l in links if CTA.search(l["text"] or "")]
    item("cta", "Clear call-to-action (buy, shop, contact, sign up…)", cta_hits,
         "No CTA-like button or link text found in the server HTML (it may be rendered by JavaScript or use other wording).")
    nav_links = max((len([l for l in p.facts.links if same_site(l["href"], base_host)]) for p in html), default=0)
    nav_found = [f"<nav> element on {sum(p.facts.has_nav for p in html)} page(s)"] if any(p.facts.has_nav for p in html) else []
    if nav_links >= 5:
        nav_found.append(f"{nav_links} internal links on the richest page")
    item("navigation", "Navigation menu", nav_found, "No <nav> landmark and few internal links found.")
    product = [f"schema.org {t}" for t in sorted(types) if t.lower() in ("product", "offer", "aggregateoffer", "productgroup")]
    item("product-info", "Product information (Product / Offer structured data)", product,
         "No Product/Offer structured data observed. Product details may exist but are not machine-readable.")
    prices = sorted(set(m.group(0).strip() for m in PRICE.finditer(texts)))[:3]
    item("pricing", "Prices visible in page text", [f"price-like text: {p}" for p in prices],
         "No price-like text (currency + number) in the server HTML.")
    cart = [_link_label(l) for l in links if CART.search(l["href"] or "") or CART.search(l["text"] or "")]
    item("cart-checkout", "Cart or checkout entry point", cart,
         "No link to a cart/checkout/basket found. Normal for non-shop sites; JavaScript carts are not visible here.")
    search = [f"search form/input on {sum(p.facts.has_search_input for p in html)} page(s)"] if any(p.facts.has_search_input for p in html) else []
    item("search", "Site search", search, "No search input observed.")
    contact = [_link_label(l) for l in links if CONTACT.search(l["text"] or "") or l["href"].startswith(("tel:", "mailto:"))]
    item("contact", "Contact options (contact page, phone or email link)", contact, "No contact link, tel: or mailto: link observed.")
    pol = [f"{name}: {_link_label(l)}" for name, rx in POLICY.items()
           for l in links[:400] if rx.search(l["text"] or "") or rx.search(l["href"] or "")]
    item("policies", "Trust & policy pages (privacy, terms, returns, shipping)", list(dict.fromkeys(pol)),
         "No links to privacy, terms, returns or shipping pages observed.")
    https_all = all(urlsplit(p.final_url).scheme == "https" for p in html)
    items.append(ChecklistItem("https", "Secure connection (HTTPS) on analysed pages",
                               "observed" if https_all else "not_observed",
                               [f"{sum(urlsplit(p.final_url).scheme == 'https' for p in html)}/{len(html)} pages on HTTPS"],
                               evidence_type="observed"))
    vp = [p.final_url for p in html if "width=device-width" in p.facts.meta.get("viewport", "").replace(" ", "").lower()]
    items.append(ChecklistItem("mobile", "Mobile-friendly viewport", "observed" if len(vp) == len(html) else "not_observed",
                               [f"{len(vp)}/{len(html)} pages declare width=device-width"], evidence_type="observed"))
    return items
