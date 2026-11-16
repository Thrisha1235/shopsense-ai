// Exportable Website Intelligence report (HTML / JSON), built in the browser.
// All text comes from third-party websites, so EVERY value is HTML-escaped and only
// http(s) URLs become links. The exported file also carries a CSP that blocks scripts.
import type { WebsiteAnalysis } from "./websiteTypes";

export function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Only absolute http(s) URLs are treated as links (blocks javascript:, data:, etc.). */
export function safeHref(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
  } catch {
    return null;
  }
}

const link = (url: string | null | undefined) => {
  const href = safeHref(url);
  return href ? `<a href="${escapeHtml(href)}" rel="noopener noreferrer nofollow">${escapeHtml(url)}</a>` : escapeHtml(url);
};

export function reportFileName(r: WebsiteAnalysis, ext: string): string {
  let host = "website";
  try {
    host = new URL(r.final_url ?? r.analyzed_url).hostname;
  } catch {
    /* keep default */
  }
  const day = r.analyzed_at.slice(0, 10);
  return `shopsense-website-report_${host.replace(/[^a-z0-9.-]/gi, "_")}_${day}.${ext}`;
}

export function buildHtmlReport(r: WebsiteAnalysis): string {
  const sev = r.summary.issues_by_severity;
  const rows = (cells: string[]) => `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
  const issues = r.issues
    .map((i) => rows([escapeHtml(i.severity), escapeHtml(i.category),
      `<strong>${escapeHtml(i.title)}</strong><br><small>Evidence (${escapeHtml(i.evidence_type)}): ${escapeHtml(i.evidence)}</small>`,
      escapeHtml(i.recommendation), link(i.url)]))
    .join("");
  const pages = r.pages
    .map((p) => rows([link(p.final_url), escapeHtml(p.status), escapeHtml(Math.round(p.elapsed_ms)),
      escapeHtml(p.title ?? "—"), escapeHtml(p.meta_description_length ?? "—"), escapeHtml(p.h1?.length ?? "—"),
      escapeHtml(p.images_missing_alt ?? "—"), escapeHtml(p.word_count ?? "—")]))
    .join("");
  const checklist = r.conversion_checklist
    .map((c) => rows([escapeHtml(c.label), escapeHtml(c.status.replace("_", " ")), escapeHtml(c.evidence_type),
      c.evidence.length ? c.evidence.map(escapeHtml).join("<br>") : escapeHtml(c.note)]))
    .join("");
  const perf = r.performance.status === "available"
    ? `<p>PageSpeed Insights (${escapeHtml(r.performance.strategy)}): performance score ${escapeHtml(r.performance.performance_score ?? "n/a")}.</p>
       <ul>${(r.performance.lab ?? []).map((m) => `<li>Lab ${escapeHtml(m.metric)}: ${escapeHtml(m.value)}</li>`).join("")}
       ${(r.performance.field ?? []).map((m) => `<li>Field ${escapeHtml(m.metric)}: p75 ${escapeHtml(m.percentile)} (${escapeHtml(m.category)})</li>`).join("")}</ul>`
    : `<p>Unavailable: ${escapeHtml(r.performance.reason)}</p>`;

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Website analysis – ${escapeHtml(r.final_url ?? r.analyzed_url)}</title>
<style>
body{font:14px/1.5 system-ui,"Segoe UI",sans-serif;color:#0f172a;max-width:1100px;margin:32px auto;padding:0 16px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #e4e6eb;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e4e6eb;padding:6px 8px;text-align:left;vertical-align:top}
th{color:#556070;font-weight:600}small{color:#556070}.note{background:#f6f7f9;border:1px solid #e4e6eb;padding:10px 12px;border-radius:8px}
</style></head><body>
<h1>Website intelligence report</h1>
<p><strong>Analysed:</strong> ${link(r.final_url ?? r.analyzed_url)}<br>
<strong>Date:</strong> ${escapeHtml(r.analyzed_at)} · <strong>Pages:</strong> ${r.summary.pages_analyzed} analysed, ${r.summary.pages_skipped} skipped ·
<strong>robots.txt:</strong> ${escapeHtml(r.robots.status)}<br>
<strong>Issues:</strong> ${sev.high} high, ${sev.medium} medium, ${sev.low} low, ${sev.info} info ·
<strong>Conversion checklist:</strong> ${r.summary.checklist_observed}/${r.summary.checklist_total} observed</p>
${r.fatal_error ? `<p class="note"><strong>Not analysed:</strong> ${escapeHtml(r.fatal_error)}</p>` : ""}
<p class="note">Evidence types: <strong>observed</strong> = a direct fact in the delivered HTML/HTTP response;
<strong>heuristic</strong> = a guideline or keyword-based judgement that can be wrong; <strong>unavailable</strong> = not measurable from public HTML.</p>
<h2>Technical, SEO and usability issues (${r.issues.length})</h2>
<table><thead><tr><th>Severity</th><th>Category</th><th>Issue and evidence</th><th>Suggested fix</th><th>Affected URL</th></tr></thead><tbody>${issues}</tbody></table>
<h2>Pages analysed</h2>
<table><thead><tr><th>URL</th><th>Status</th><th>Fetch ms</th><th>Title</th><th>Description length</th><th>H1</th><th>Images without alt</th><th>Words</th></tr></thead><tbody>${pages}</tbody></table>
<h2>Conversion-readiness checklist (heuristic)</h2>
<table><thead><tr><th>Item</th><th>Status</th><th>Evidence type</th><th>Evidence / note</th></tr></thead><tbody>${checklist}</tbody></table>
<h2>Performance</h2>${perf}
<h2>Data availability</h2><ul>${r.data_availability.map((d) => `<li><strong>${escapeHtml(d.item)}</strong>: ${escapeHtml(d.status)}. ${escapeHtml(d.reason)}</li>`).join("")}</ul>
<h2>Limitations</h2><ul>${r.disclaimers.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}</ul>
<p><small>Generated by ShopSense AI Website Intelligence (user agent: ${escapeHtml(r.user_agent)}).</small></p>
</body></html>`;
}

export function downloadFile(name: string, content: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: `${mime};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
