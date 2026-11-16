"use client";

import { useMemo, useRef, useState, type SubmitEvent } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { IconAlert, IconCheck, IconDownload, IconGlobe, IconInfo, IconPlug } from "@/components/icons";
import { Badge, Card, CardHeader, Notice, PageHeader } from "@/components/ui/Card";
import ChartCard, { DataTable, type Column } from "@/components/ui/ChartCard";
import KpiCard from "@/components/ui/KpiCard";
import { EmptyState, ErrorState } from "@/components/ui/States";
import { ApiError, analyzeWebsite } from "@/lib/api";
import { axisTick, chart } from "@/lib/chartTheme";
import { fmtInt } from "@/lib/format";
import { buildHtmlReport, downloadFile, reportFileName, safeHref } from "@/lib/websiteReport";
import type { ChecklistItem, Severity, WebsiteAnalysis, WebsiteIssue, WebsitePage } from "@/lib/websiteTypes";

// Severity is a status: fixed status colours, always paired with a text label.
const SEVERITY: Record<Severity, { label: string; color: string; badge: string }> = {
  high: { label: "High", color: "#d03b3b", badge: "bg-bad-bg text-bad" },
  medium: { label: "Medium", color: "#ec835a", badge: "bg-[#fdf0e8] text-[#9a3f12]" },
  low: { label: "Low", color: "#fab219", badge: "bg-warn-bg text-warn" },
  info: { label: "Info", color: "#94a3b8", badge: "bg-slate-100 text-ink-2" },
};
const SEVERITIES: Severity[] = ["high", "medium", "low", "info"];
const CATEGORY_LABEL: Record<string, string> = { technical: "Technical", seo: "SEO", usability: "Usability" };

function ExternalLink({ url, children }: { url: string | null | undefined; children?: React.ReactNode }) {
  const href = safeHref(url);
  if (!href) return <span className="break-all">{url ?? "—"}</span>;
  return (
    <a href={href} target="_blank" rel="noopener noreferrer nofollow" className="break-all text-brand-600 underline-offset-2 hover:underline">
      {children ?? url}
    </a>
  );
}

function SeverityBadge({ s }: { s: Severity }) {
  return <span className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold ${SEVERITY[s].badge}`}>{SEVERITY[s].label}</span>;
}

function EvidenceBadge({ type }: { type: "observed" | "heuristic" | "unavailable" }) {
  const tone = type === "observed" ? "brand" : type === "heuristic" ? "warn" : "neutral";
  return <Badge tone={tone}>{type[0].toUpperCase() + type.slice(1)}</Badge>;
}

function EvidenceLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-ink-2" aria-label="Evidence types">
      <span className="flex items-center gap-2"><EvidenceBadge type="observed" /> A direct fact in the page&rsquo;s HTML or HTTP response</span>
      <span className="flex items-center gap-2"><EvidenceBadge type="heuristic" /> A guideline or keyword-based judgement; can be wrong</span>
      <span className="flex items-center gap-2"><EvidenceBadge type="unavailable" /> Cannot be measured from public HTML</span>
    </div>
  );
}

// -------------------------------------------------------------------------- charts
function IssuesByCategoryChart({ issues }: { issues: WebsiteIssue[] }) {
  const rows = ["technical", "seo", "usability"].map((cat) => ({
    category: CATEGORY_LABEL[cat],
    ...Object.fromEntries(SEVERITIES.map((s) => [s, issues.filter((i) => i.category === cat && i.severity === s).length])),
  }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }} barCategoryGap="30%">
        <CartesianGrid horizontal={false} />
        <XAxis type="number" allowDecimals={false} tick={axisTick} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="category" width={84} tick={axisTick} axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: "#f1f5f9" }} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
        <Legend iconType="circle" iconSize={8} itemSorter={null} wrapperStyle={{ fontSize: 12, color: chart.ink2 }} />
        {SEVERITIES.map((s) => (
          <Bar key={s} dataKey={s} name={SEVERITY[s].label} stackId="sev" fill={SEVERITY[s].color} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function FetchTimeChart({ pages }: { pages: WebsitePage[] }) {
  const rows = pages.map((p) => {
    let label = p.final_url;
    try {
      label = new URL(p.final_url).pathname || "/";
    } catch {
      /* keep full URL */
    }
    // keep the meaningful END of long paths ("…/category/arts/"), which fits the 150px axis
    return { label: label.length > 22 ? "…" + label.slice(-21) : label, ms: Math.round(p.elapsed_ms), url: p.final_url };
  });
  return (
    <ResponsiveContainer width="100%" height={Math.max(180, rows.length * 36 + 40)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 0, left: 8 }} barCategoryGap="30%">
        <CartesianGrid horizontal={false} />
        <XAxis type="number" tick={axisTick} axisLine={false} tickLine={false} unit=" ms" />
        <YAxis type="category" dataKey="label" width={150} tick={{ ...axisTick, fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: "#f1f5f9" }} formatter={(v) => [`${v} ms`, "Fetch time"]} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
        <Bar dataKey="ms" name="Fetch time" fill={chart.primary} radius={[0, 4, 4, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// -------------------------------------------------------------------------- sections
const issueColumns: Column<WebsiteIssue>[] = [
  { key: "s", header: "Severity", render: (i) => <SeverityBadge s={i.severity} /> },
  {
    key: "t", header: "Issue and evidence", render: (i) => (
      <div className="min-w-64">
        <p className="font-medium text-ink">{i.title}</p>
        <p className="mt-0.5 text-xs text-ink-3"><span className="font-medium">Evidence:</span> {i.evidence}</p>
        <p className="mt-1 text-xs text-ink-2"><span className="font-medium">Fix:</span> {i.recommendation}</p>
      </div>
    ),
  },
  { key: "c", header: "Category", render: (i) => CATEGORY_LABEL[i.category] },
  { key: "e", header: "Evidence type", render: (i) => <EvidenceBadge type={i.evidence_type} /> },
  { key: "u", header: "Affected URL", render: (i) => <span className="text-xs"><ExternalLink url={i.url} /></span> },
];

const pageColumns: Column<WebsitePage>[] = [
  { key: "u", header: "Page", render: (p) => <span className="text-xs"><ExternalLink url={p.final_url} /></span> },
  { key: "s", header: "HTTP", numeric: true, render: (p) => String(p.status) },
  { key: "ms", header: "Fetch ms", numeric: true, render: (p) => fmtInt(p.elapsed_ms) },
  { key: "t", header: "Title chars", numeric: true, render: (p) => (p.is_html ? String(p.title_length ?? 0) : "—") },
  { key: "d", header: "Description chars", numeric: true, render: (p) => (p.is_html ? String(p.meta_description_length ?? 0) : "—") },
  { key: "h", header: "H1", numeric: true, render: (p) => (p.is_html ? String(p.h1?.length ?? 0) : "—") },
  { key: "i", header: "Images w/o alt", numeric: true, render: (p) => (p.is_html ? `${p.images_missing_alt}/${p.images_total}` : "—") },
  { key: "w", header: "Words", numeric: true, render: (p) => (p.is_html ? fmtInt(p.word_count ?? 0) : "—") },
  { key: "sd", header: "Structured data", render: (p) => (p.structured_data_types?.length ? p.structured_data_types.join(", ") : "—") },
];

function IssuesSection({ issues }: { issues: WebsiteIssue[] }) {
  const [severity, setSeverity] = useState<"all" | Severity>("all");
  const [category, setCategory] = useState("all");
  const shown = issues.filter((i) => (severity === "all" || i.severity === severity) && (category === "all" || i.category === category));
  const selectCls = "rounded-lg border border-line-strong bg-surface px-2.5 py-1.5 text-xs text-ink focus:outline-none focus:ring-2 focus:ring-brand-500/40";
  return (
    <Card className="mt-4">
      <CardHeader
        title={`Technical issues report (${issues.length})`}
        description="Every finding lists the affected URL, the evidence observed and a suggested fix. Sorted by severity."
        action={
          <div className="flex flex-wrap gap-2">
            <label className="flex items-center gap-1.5 text-xs text-ink-3">
              Severity
              <select className={selectCls} value={severity} onChange={(e) => setSeverity(e.target.value as "all" | Severity)}>
                <option value="all">All</option>
                {SEVERITIES.map((s) => <option key={s} value={s}>{SEVERITY[s].label}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-xs text-ink-3">
              Category
              <select className={selectCls} value={category} onChange={(e) => setCategory(e.target.value)}>
                <option value="all">All</option>
                {Object.entries(CATEGORY_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </label>
          </div>
        }
      />
      <div className="px-5 pb-5 pt-4">
        {issues.length === 0 ? (
          <EmptyState title="No issues found" icon={<IconCheck className="text-good" width={24} height={24} />}>
            None of the checks flagged a problem on the analysed pages.
          </EmptyState>
        ) : shown.length === 0 ? (
          <p className="text-sm text-ink-3">No issues match these filters.</p>
        ) : (
          <DataTable columns={issueColumns} rows={shown} caption="Technical issues report" />
        )}
      </div>
    </Card>
  );
}

function ChecklistSection({ items, observed, total }: { items: ChecklistItem[]; observed: number; total: number }) {
  return (
    <Card className="mt-4">
      <CardHeader
        title={`Conversion-readiness checklist (${observed}/${total} observed)`}
        description="What a visitor can publicly see that supports buying or enquiring. These are heuristics from the HTML: they do not measure actual conversions."
        action={<EvidenceBadge type="heuristic" />}
      />
      <ul className="divide-y divide-line px-5 pb-3 pt-2">
        {items.map((c) => (
          <li key={c.id} className="flex flex-col gap-1.5 py-3 sm:flex-row sm:items-start sm:gap-4">
            <span className="w-32 shrink-0">
              {c.status === "observed" ? (
                <Badge tone="good"><IconCheck width={13} height={13} /> Observed</Badge>
              ) : c.status === "not_observed" ? (
                <Badge tone="neutral">Not observed</Badge>
              ) : (
                <Badge tone="neutral">Unavailable</Badge>
              )}
            </span>
            <div className="min-w-0 flex-1 text-sm">
              <p className="font-medium text-ink">
                {c.label} {c.evidence_type === "observed" && <span className="ml-1 align-middle"><EvidenceBadge type="observed" /></span>}
              </p>
              {c.evidence.length > 0 ? (
                <ul className="mt-1 space-y-0.5 text-xs text-ink-3">
                  {c.evidence.map((e, i) => <li key={i} className="break-all">{e}</li>)}
                </ul>
              ) : (
                <p className="mt-1 text-xs text-ink-3">{c.note}</p>
              )}
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function PerformanceSection({ r }: { r: WebsiteAnalysis }) {
  const p = r.performance;
  return (
    <Card>
      <CardHeader title="Performance (PageSpeed Insights)" description="Lab and real-user speed data from Google, only when configured on the server."
                  action={p.status === "available" ? <EvidenceBadge type="observed" /> : <EvidenceBadge type="unavailable" />} />
      <div className="px-5 pb-5 pt-3 text-sm text-ink-2">
        {p.status === "available" ? (
          <>
            <p>Performance score ({p.strategy}): <strong className="text-ink">{p.performance_score ?? "n/a"}</strong> · Lighthouse {p.lighthouse_version}</p>
            <ul className="mt-2 space-y-1">
              {(p.lab ?? []).map((m) => <li key={m.metric}>Lab · {m.metric}: <strong className="text-ink">{m.value ?? "n/a"}</strong></li>)}
              {(p.field ?? []).map((m) => <li key={m.metric}>Real users · {m.metric}: p75 {m.percentile ?? "n/a"} ({m.category ?? "n/a"})</li>)}
            </ul>
            {!p.field_available && <p className="mt-2 text-xs text-ink-3">No real-user (field) data: Google has too little traffic data for this URL.</p>}
            <p className="mt-2 text-xs text-ink-3">{p.note}</p>
          </>
        ) : (
          <p><strong className="text-ink">Unavailable.</strong> {p.reason} Lighthouse and Core Web Vitals figures are never estimated.</p>
        )}
      </div>
    </Card>
  );
}

function FutureIntegrations() {
  return (
    <Card>
      <CardHeader title="Future integrations" description="Planned. Not connected." action={<Badge tone="warn"><IconPlug width={13} height={13} /> Not connected</Badge>} />
      <div className="space-y-3 px-5 pb-5 pt-3 text-sm text-ink-2">
        {[
          ["Shopify", "Real orders, products and checkout funnel for your own store."],
          ["Google Analytics 4", "Real sessions, traffic sources and conversion events."],
        ].map(([name, text]) => (
          <div key={name} className="flex items-start justify-between gap-3 rounded-lg border border-dashed border-line-strong p-3">
            <div>
              <p className="font-medium text-ink">{name}</p>
              <p className="text-xs text-ink-3">{text}</p>
            </div>
            <button type="button" disabled aria-disabled="true" className="shrink-0 cursor-not-allowed rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink-3">
              Coming later
            </button>
          </div>
        ))}
        <p className="text-xs text-ink-3">No credentials are requested or stored, and no store or analytics accounts are connected. Actual conversion rates need data like this; they cannot be measured from public HTML.</p>
      </div>
    </Card>
  );
}

function Results({ r }: { r: WebsiteAnalysis }) {
  const sev = r.summary.issues_by_severity;
  const exportHtml = () => downloadFile(reportFileName(r, "html"), buildHtmlReport(r), "text/html");
  const exportJson = () => downloadFile(reportFileName(r, "json"), JSON.stringify(r, null, 2), "application/json");
  return (
    <div className="mt-6" aria-label="Analysis results">
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4 p-5">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-3">Analysed site</p>
            <p className="mt-1 text-lg font-semibold text-ink"><ExternalLink url={r.final_url ?? r.analyzed_url} /></p>
            <p className="mt-1 text-xs text-ink-3">
              {new Date(r.analyzed_at).toLocaleString()} · {(r.duration_ms / 1000).toFixed(1)} s · robots.txt: {r.robots.status.replace("_", " ")}
              {r.robots.http_status ? ` (HTTP ${r.robots.http_status})` : ""} · {r.summary.pages_analyzed} page(s) analysed
              {r.summary.pages_skipped ? `, ${r.summary.pages_skipped} skipped` : ""}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={exportHtml} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-2 text-xs font-medium text-ink hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-brand-500">
              <IconDownload width={14} height={14} /> Download report (HTML)
            </button>
            <button type="button" onClick={exportJson} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-2 text-xs font-medium text-ink hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-brand-500">
              <IconDownload width={14} height={14} /> Download JSON
            </button>
          </div>
        </div>
        <div className="border-t border-line px-5 py-3"><EvidenceLegend /></div>
      </Card>

      {r.fatal_error && (
        <div className="mt-4"><Notice tone="warn" title="This site was not analysed">{r.fatal_error}</Notice></div>
      )}

      <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-5">
        <KpiCard label="Pages analysed" value={String(r.summary.pages_analyzed)} detail={`max 5 · robots.txt respected`} />
        {SEVERITIES.map((s) => (
          <KpiCard key={s} label={`${SEVERITY[s].label} issues`} value={String(sev[s])} accent={s === "high" && sev[s] > 0} />
        ))}
      </div>

      {r.pages.length > 0 && (
        <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
          <ChartCard title="Issues by category and severity" description="Counts of findings on the analysed pages."
                     columns={[{ key: "c", header: "Category", render: (x: { c: string }) => x.c },
                       ...SEVERITIES.map((s) => ({ key: s, header: SEVERITY[s].label, numeric: true,
                         render: (x: { c: string }) => String(r.issues.filter((i) => CATEGORY_LABEL[i.category] === x.c && i.severity === s).length) }))]}
                     rows={["Technical", "SEO", "Usability"].map((c) => ({ c }))}>
            <IssuesByCategoryChart issues={r.issues} />
          </ChartCard>
          <ChartCard title="Server fetch time per page" description="Time for the analysis server to download each page's HTML. One measurement from one location, not real-user speed."
                     columns={pageColumns.slice(0, 3)} rows={r.pages}>
            <FetchTimeChart pages={r.pages} />
          </ChartCard>
        </div>
      )}

      <IssuesSection issues={r.issues} />

      {r.conversion_checklist.length > 0 && (
        <ChecklistSection items={r.conversion_checklist} observed={r.summary.checklist_observed} total={r.summary.checklist_total} />
      )}

      {r.pages.length > 0 && (
        <Card className="mt-4">
          <CardHeader title="Pages analysed" description="Observed facts per page. Carts, checkouts and account pages are never crawled." />
          <div className="px-5 pb-5 pt-4">
            <DataTable columns={pageColumns} rows={r.pages} caption="Pages analysed" />
            {r.skipped.length > 0 && (
              <div className="mt-3 text-xs text-ink-3">
                <p className="font-medium text-ink-2">Skipped:</p>
                <ul className="mt-1 list-disc space-y-0.5 pl-5">
                  {r.skipped.map((s, i) => <li key={i} className="break-all">{s.url}: {s.reason}</li>)}
                </ul>
              </div>
            )}
          </div>
        </Card>
      )}

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
        <PerformanceSection r={r} />
        <Card>
          <CardHeader title="Data availability" description="What this analysis can and cannot tell you." />
          <ul className="space-y-2 px-5 pb-5 pt-3 text-sm">
            {r.data_availability.map((d) => (
              <li key={d.item}>
                <span className="font-medium text-ink">{d.item}</span>{" "}
                <EvidenceBadge type={d.status === "available" ? "observed" : "unavailable"} />
                <p className="text-xs text-ink-3">{d.reason}</p>
              </li>
            ))}
          </ul>
        </Card>
        <FutureIntegrations />
      </div>

      <div className="mt-4">
        <Notice tone="neutral" title="Limitations">
          <ul className="mt-1 list-disc space-y-1 pl-5">{r.disclaimers.map((d) => <li key={d}>{d}</li>)}</ul>
        </Notice>
      </div>
    </div>
  );
}

export default function WebsiteIntelligenceView() {
  const [url, setUrl] = useState("");
  const [includePerf, setIncludePerf] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [state, setState] = useState<{ status: "idle" | "loading" | "success" | "error"; data?: WebsiteAnalysis; error?: Error }>({ status: "idle" });
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const loading = state.status === "loading";

  async function onSubmit(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    const value = url.trim();
    if (!value) {
      setInputError("Enter a website address, for example example.com.");
      inputRef.current?.focus();
      return;
    }
    if (value.length > 2048) {
      setInputError("That address is too long.");
      return;
    }
    setInputError(null);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setState({ status: "loading" });
    try {
      setState({ status: "success", data: await analyzeWebsite(value, includePerf, controller.signal) });
    } catch (err) {
      if (controller.signal.aborted) return;
      const error = err instanceof Error ? err : new Error(String(err));
      if (err instanceof ApiError && err.status === 400) setInputError(error.message);
      setState({ status: "error", error });
    }
  }

  const issueTotal = useMemo(() => state.data?.issues.length ?? 0, [state.data]);

  return (
    <>
      <PageHeader
        title="Website intelligence"
        description="Evidence-based technical SEO, usability and conversion-readiness checks for any publicly accessible website."
      >
        <Badge tone="brand"><IconGlobe width={13} height={13} /> Separate from purchase prediction</Badge>
      </PageHeader>

      <Card>
        <form onSubmit={onSubmit} noValidate className="p-5" aria-label="Analyse a website">
          <label htmlFor="site-url" className="block text-sm font-medium text-ink">Website URL</label>
          <div className="mt-1.5 flex flex-col gap-3 sm:flex-row">
            <input
              id="site-url"
              ref={inputRef}
              type="text"
              inputMode="url"
              autoComplete="url"
              spellCheck={false}
              placeholder="example.com or https://www.example.com"
              value={url}
              onChange={(e) => { setUrl(e.target.value); if (inputError) setInputError(null); }}
              aria-invalid={!!inputError}
              aria-describedby="site-url-help site-url-error"
              maxLength={2048}
              className={`block w-full rounded-lg border bg-surface px-3 py-2.5 text-sm text-ink shadow-sm placeholder:text-ink-3 focus:outline-none focus:ring-2 focus:ring-brand-500/40 ${inputError ? "border-bad" : "border-line-strong"}`}
            />
            <button
              type="submit"
              disabled={loading}
              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-brand-700 disabled:cursor-wait disabled:opacity-70 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
            >
              {loading ? "Analysing…" : "Analyze Website"}
            </button>
          </div>
          <p id="site-url-error" className="mt-1.5 text-xs font-medium text-bad" role={inputError ? "alert" : undefined}>{inputError ?? ""}</p>
          <p id="site-url-help" className="mt-1 text-xs leading-5 text-ink-3">
            Up to 5 public pages are fetched. robots.txt is respected; carts, checkouts and logins are never visited. Only public
            web addresses are accepted (no IP addresses, internal hosts or non-standard ports).
          </p>
          <label className="mt-3 inline-flex items-center gap-2 text-xs text-ink-2">
            <input type="checkbox" checked={includePerf} onChange={(e) => setIncludePerf(e.target.checked)} className="h-4 w-4 rounded border-line-strong accent-brand-600" />
            Include Google PageSpeed Insights (only if an API key is configured on the server; can take up to a minute)
          </label>
        </form>
      </Card>

      <div aria-live="polite" aria-busy={loading}>
        {loading && (
          <Card className="mt-6">
            <div className="flex items-center gap-4 p-6" role="status">
              <span className="h-6 w-6 shrink-0 animate-spin rounded-full border-2 border-brand-200 border-t-brand-600" aria-hidden="true" />
              <div>
                <p className="text-sm font-semibold text-ink">Analysing {url.trim()}…</p>
                <p className="text-xs text-ink-3">Checking robots.txt, fetching up to 5 public pages politely, then running the checks. This usually takes 5–30 seconds.</p>
              </div>
            </div>
          </Card>
        )}
        {state.status === "error" && state.error && !(state.error instanceof ApiError && state.error.status === 400) && (
          <div className="mt-6">
            <ErrorState error={state.error} />
          </div>
        )}
        {state.status === "error" && state.error instanceof ApiError && state.error.status === 400 && (
          <div className="mt-6"><Notice tone="warn" title="This address cannot be analysed">{state.error.message}</Notice></div>
        )}
        {state.status === "success" && state.data && (
          <>
            <p className="sr-only">Analysis complete: {issueTotal} issues found.</p>
            <Results r={state.data} />
          </>
        )}
        {state.status === "idle" && (
          <div className="mt-6 grid grid-cols-1 gap-4 xl:grid-cols-3">
            <div className="xl:col-span-2">
              <EmptyState title="Enter a website to analyse" icon={<IconInfo className="text-ink-3" width={24} height={24} />}>
                You will get a technical issues report with evidence and fixes, a conversion-readiness checklist and an exportable
                report. Every finding is labelled as observed, heuristic or unavailable.
              </EmptyState>
            </div>
            <FutureIntegrations />
          </div>
        )}
      </div>
      {state.status === "success" && state.data?.fatal_error && (
        <p className="mt-2 flex items-center gap-1.5 text-xs text-ink-3"><IconAlert width={14} height={14} /> Partial result: see the notice above.</p>
      )}
    </>
  );
}
