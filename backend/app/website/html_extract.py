"""Extract observable facts from one HTML document (stdlib html.parser; no JS execution).

Everything here is a direct observation of the server-delivered HTML. Content injected later
by JavaScript is invisible to this parser; ``likely_js_rendered`` flags pages where that is
probably the case so findings can be labelled as limited.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

MAX_ITEMS = 400          # cap list sizes so huge pages cannot blow up memory/response size
SNIPPET = 120


def _clip(s: str, n: int = SNIPPET) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class PageFacts:
    url: str
    title: str | None = None
    lang: str | None = None
    meta: dict[str, str] = field(default_factory=dict)          # name/property -> content (lower-cased keys)
    canonical: str | None = None
    has_favicon: bool = False
    headings: list[tuple[int, str]] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)              # {src, alt (None if absent)}
    links: list[dict] = field(default_factory=list)               # {href (absolute), text, rel}
    buttons: list[str] = field(default_factory=list)
    forms: list[dict] = field(default_factory=list)               # {action, method, input_types}
    jsonld_raw: list[str] = field(default_factory=list)
    microdata_types: list[str] = field(default_factory=list)
    resource_urls: list[str] = field(default_factory=list)        # src/href of scripts, styles, iframes, images
    has_nav: bool = False
    has_header: bool = False
    has_footer: bool = False
    has_search_input: bool = False
    text_chars: int = 0
    word_count: int = 0
    script_count: int = 0
    text_sample: str = ""          # first ~4,000 visible characters (heuristic keyword checks only)

    # ------------------------------------------------------------ derived
    @property
    def h1(self) -> list[str]:
        return [t for lvl, t in self.headings if lvl == 1]

    def heading_skips(self) -> list[str]:
        out, prev = [], 0
        for lvl, text in self.headings:
            if prev and lvl > prev + 1:
                out.append(f"h{prev} → h{lvl} ('{_clip(text, 60)}')")
            prev = lvl
        return out[:10]

    def jsonld(self) -> tuple[list[str], list[str]]:
        """(schema.org @types found, parse errors)."""
        types, errors = [], []
        for raw in self.jsonld_raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON-LD: {exc.msg} at line {exc.lineno}")
                continue
            stack = [data]
            while stack:
                node = stack.pop()
                if isinstance(node, list):
                    stack.extend(node)
                elif isinstance(node, dict):
                    t = node.get("@type")
                    if isinstance(t, str):
                        types.append(t)
                    elif isinstance(t, list):
                        types.extend(str(x) for x in t)
                    stack.extend(v for k, v in node.items() if k in ("@graph", "mainEntity", "itemListElement", "offers"))
        return sorted(set(types)), errors

    def likely_js_rendered(self) -> bool:
        return self.word_count < 60 and len(self.links) < 5 and self.script_count >= 1


class _Extractor(HTMLParser):
    def __init__(self, url: str):
        super().__init__(convert_charrefs=True)
        self.f = PageFacts(url=url)
        self._in_title = False
        self._title_parts: list[str] = []
        self._heading: int | None = None
        self._heading_parts: list[str] = []
        self._link: dict | None = None
        self._button_parts: list[str] | None = None
        self._jsonld: list[str] | None = None
        self._skip_text = 0          # inside <script>/<style>/<noscript>/<template>
        self._text: list[str] = []

    def _abs(self, href: str | None) -> str | None:
        if not href:
            return None
        href = href.strip()
        if href.startswith(("javascript:", "data:")):
            return None
        return urljoin(self.f.url, href)

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        f = self.f
        if tag == "html" and a.get("lang"):
            f.lang = a["lang"].strip()
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (a.get("name") or a.get("property") or a.get("http-equiv") or "").lower()
            if key and "content" in a and len(f.meta) < MAX_ITEMS:
                f.meta.setdefault(key, a["content"])
        elif tag == "link":
            rel = a.get("rel", "").lower().split()
            if "canonical" in rel and f.canonical is None:
                f.canonical = self._abs(a.get("href"))
            if "icon" in rel:
                f.has_favicon = True
            if "stylesheet" in rel and a.get("href") and len(f.resource_urls) < MAX_ITEMS:
                f.resource_urls.append(self._abs(a["href"]) or "")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._heading, self._heading_parts = int(tag[1]), []
        elif tag == "img":
            if len(f.images) < MAX_ITEMS:
                f.images.append({"src": self._abs(a.get("src") or a.get("data-src")) or "",
                                 "alt": a["alt"] if "alt" in a else None})
            if a.get("src"):
                f.resource_urls.append(self._abs(a["src"]) or "")
        elif tag == "a":
            href = self._abs(a.get("href"))
            self._link = {"href": href, "text": "", "rel": a.get("rel", ""),
                          "aria": a.get("aria-label", "") or a.get("title", "")}
        elif tag == "button" or (tag == "input" and a.get("type", "").lower() in ("submit", "button")):
            if tag == "input":
                if len(f.buttons) < MAX_ITEMS:
                    f.buttons.append(_clip(a.get("value", "") or a.get("aria-label", "")))
            else:
                self._button_parts = [a.get("aria-label", "")]
        elif tag == "form":
            if len(f.forms) < 50:
                f.forms.append({"action": self._abs(a.get("action")) or f.url, "method": a.get("method", "get").lower(),
                                "role": a.get("role", ""), "input_types": []})
        elif tag == "input":
            t = a.get("type", "text").lower()
            if f.forms:
                f.forms[-1]["input_types"].append(t)
            name = (a.get("name", "") + " " + a.get("placeholder", "") + " " + a.get("aria-label", "")).lower()
            if t == "search" or re.search(r"\bsearch\b|\bq\b", name):
                f.has_search_input = True
        elif tag == "script":
            f.script_count += 1
            self._skip_text += 1
            if a.get("type", "").lower() == "application/ld+json":
                self._jsonld = []
            elif a.get("src") and len(f.resource_urls) < MAX_ITEMS:
                f.resource_urls.append(self._abs(a["src"]) or "")
        elif tag in ("style", "noscript", "template"):
            self._skip_text += 1
        elif tag == "iframe" and a.get("src") and len(f.resource_urls) < MAX_ITEMS:
            f.resource_urls.append(self._abs(a["src"]) or "")
        elif tag == "nav":
            f.has_nav = True
        elif tag == "header":
            f.has_header = True
        elif tag == "footer":
            f.has_footer = True
        if a.get("role") == "navigation":
            f.has_nav = True
        if a.get("role") == "search":
            f.has_search_input = True
        if a.get("itemtype") and len(f.microdata_types) < 50:
            f.microdata_types.append(a["itemtype"].rsplit("/", 1)[-1])

    def handle_endtag(self, tag):
        f = self.f
        if tag == "title":
            self._in_title = False
            if f.title is None:
                f.title = _clip("".join(self._title_parts), 300)
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._heading is not None:
            if len(f.headings) < MAX_ITEMS:
                f.headings.append((self._heading, _clip("".join(self._heading_parts), 200)))
            self._heading = None
        elif tag == "a" and self._link is not None:
            if self._link["href"] and len(f.links) < MAX_ITEMS:
                self._link["text"] = _clip(self._link["text"] or self._link["aria"], 80)
                f.links.append(self._link)
            self._link = None
        elif tag == "button" and self._button_parts is not None:
            if len(f.buttons) < MAX_ITEMS:
                f.buttons.append(_clip(" ".join(self._button_parts)))
            self._button_parts = None
        elif tag == "script":
            self._skip_text = max(0, self._skip_text - 1)
            if self._jsonld is not None:
                if len(f.jsonld_raw) < 20:
                    f.jsonld_raw.append("".join(self._jsonld)[:200_000])
                self._jsonld = None
        elif tag in ("style", "noscript", "template"):
            self._skip_text = max(0, self._skip_text - 1)

    def handle_data(self, data):
        if self._jsonld is not None:
            self._jsonld.append(data)
            return
        if self._in_title:
            self._title_parts.append(data)
        if self._skip_text:
            return
        if self._heading is not None:
            self._heading_parts.append(data)
        if self._link is not None:
            self._link["text"] += data
        if self._button_parts is not None:
            self._button_parts.append(data)
        if len(self._text) < 20000:
            self._text.append(data)

    def close(self):
        super().close()
        text = re.sub(r"\s+", " ", " ".join(self._text)).strip()
        self.f.text_chars = len(text)
        self.f.word_count = len(text.split()) if text else 0
        self.f.text_sample = text[:4000]          # used only for heuristic keyword checks


def extract(url: str, html: str) -> PageFacts:
    p = _Extractor(url)
    try:
        p.feed(html)
        p.close()
    except Exception:            # malformed HTML: keep whatever was parsed so far
        pass
    return p.f


def same_site(url: str, base_host: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    bare = lambda h: h[4:] if h.startswith("www.") else h  # noqa: E731
    return bare(host) == bare(base_host.lower())
