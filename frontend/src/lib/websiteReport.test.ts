// The exported report contains text from third-party websites: it must be inert.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { buildHtmlReport, escapeHtml, reportFileName, safeHref } from "./websiteReport.ts";
import type { WebsiteAnalysis } from "./websiteTypes.ts";

const evil = `<script>alert("x")</script><img src=x onerror=alert(1)>`;

function sample(): WebsiteAnalysis {
  return {
    input_url: "evil.example.com", analyzed_url: "https://evil.example.com/", final_url: "https://evil.example.com/",
    analyzed_at: "2026-11-15T10:00:00+00:00", duration_ms: 1200, user_agent: "ShopSenseAI-WebsiteIntelligence/1.0",
    fatal_error: null,
    robots: { url: "https://evil.example.com/robots.txt", status: "not_found", http_status: 404, note: "" },
    pages: [{ url: "https://evil.example.com/", final_url: "https://evil.example.com/", status: 200, elapsed_ms: 300,
      content_type: "text/html", bytes: 100, truncated: false, https: true, hsts: false, redirects: 0, is_html: true,
      title: evil, h1: [evil], word_count: 3, images_missing_alt: 0, meta_description_length: 0 }],
    skipped: [],
    issues: [{ id: "x", severity: "high", category: "seo", title: evil, url: "javascript:alert(1)", evidence: evil,
      recommendation: evil, evidence_type: "observed" }],
    summary: { pages_analyzed: 1, pages_skipped: 0, issues_by_severity: { high: 1, medium: 0, low: 0, info: 0 },
      issues_by_category: { seo: 1 }, checklist_observed: 0, checklist_total: 1 },
    conversion_checklist: [{ id: "cta", label: "CTA", status: "observed", evidence: [evil], note: "", evidence_type: "heuristic" }],
    performance: { status: "unavailable", source: "PSI", reason: "not configured" },
    data_availability: [{ item: "Conversions", status: "unavailable", reason: "needs analytics" }],
    disclaimers: ["HTML only"],
  };
}

describe("website report export", () => {
  it("escapes HTML special characters", () => {
    assert.equal(escapeHtml(`<a href="x">'&'</a>`), "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;");
  });

  it("only allows http(s) links", () => {
    assert.equal(safeHref("https://shop.example.com/a?b=1"), "https://shop.example.com/a?b=1");
    assert.equal(safeHref("javascript:alert(1)"), null);
    assert.equal(safeHref("data:text/html,hi"), null);
    assert.equal(safeHref("not a url"), null);
  });

  it("produces an inert HTML document with a script-blocking CSP", () => {
    const html = buildHtmlReport(sample());
    assert.ok(!html.includes("<script>alert"), "raw script from the analysed site must not survive");
    assert.ok(!html.includes("<img src=x"), "raw img tag must not survive");
    assert.ok(!html.includes('href="javascript:'), "javascript: URLs must not become links");
    assert.ok(html.includes("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"));
    assert.ok(html.includes(`content="default-src 'none'; style-src 'unsafe-inline'"`));
    assert.ok(html.includes("heuristic") && html.includes("unavailable"));
  });

  it("builds a safe file name", () => {
    assert.equal(reportFileName(sample(), "html"), "shopsense-website-report_evil.example.com_2026-11-15.html");
  });
});
