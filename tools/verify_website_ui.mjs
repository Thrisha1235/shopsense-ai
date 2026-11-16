// Phase 8: browser verification of /website-intelligence (headless Edge/Chrome via CDP).
// Usage: node tools/verify_website_ui.mjs <outDir> <browserPath> <profileDir>
// Needs the backend (8000) and frontend (UI_BASE, default http://localhost:3002) running, plus
// internet access (it analyses example.com and www.python.org, both public sites).
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const [outDir, browser, profile] = process.argv.slice(2);
const BASE = process.env.UI_BASE ?? "http://localhost:3002";
const API = "http://127.0.0.1:8000";
const PORT = 9336;
const AXE = join(dirname(fileURLToPath(import.meta.url)), "..", "frontend", "node_modules", "axe-core", "axe.min.js");
const downloads = join(outDir, `downloads-${Date.now()}`);   // fresh per run (same-name files would be overwritten)
mkdirSync(downloads, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const proc = spawn(browser, ["--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`, "--disable-extensions", "about:blank"], { stdio: "ignore" });
let target;
for (let i = 0; i < 60 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page"); } catch { await sleep(250); }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 1;
const pending = new Map();
const appErrors = [];
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result); }
  if (m.method === "Runtime.exceptionThrown") appErrors.push(m.params.exceptionDetails.text);
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
    const t = m.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
    if (!/chrome-extension:|Failed to load resource/.test(t)) appErrors.push("console.error: " + t.slice(0, 200));
  }
});
const send = (method, params = {}) => new Promise((resolve, reject) => { const i = id++; pending.set(i, { resolve, reject }); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result.value;
const waitFor = async (expr, timeout = 90000) => { const t0 = Date.now(); while (Date.now() - t0 < timeout) { if (await ev(expr)) return true; await sleep(250); } return false; };
const results = [];
const assert = (name, ok, detail = "") => results.push({ name, ok: !!ok, detail });
let vp = { width: 1440, height: 900, mobile: false };
const viewport = async (width, height = 900, mobile = width < 768) => { vp = { width, height, mobile }; await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile }); };
const shot = async (name) => {
  const { height } = (await send("Page.getLayoutMetrics")).cssContentSize;
  await send("Emulation.setDeviceMetricsOverride", { ...vp, height: Math.min(Math.ceil(height), 9000), deviceScaleFactor: 1 });
  await sleep(400);
  writeFileSync(join(outDir, `${name}.png`), Buffer.from((await send("Page.captureScreenshot", { format: "png" })).data, "base64"));
  await send("Emulation.setDeviceMetricsOverride", { ...vp, deviceScaleFactor: 1 });
};
const setInput = (value) => ev(`(() => { const el = document.getElementById('site-url');
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, ${JSON.stringify(value)});
  el.dispatchEvent(new Event('input', { bubbles: true })); return el.value; })()`);
const submit = () => ev(`document.querySelector('form[aria-label="Analyse a website"] button[type=submit]').click()`);
const done = `!!document.querySelector('[aria-busy="false"]') && !document.querySelector('[role=status] .animate-spin')`;
const apiAnalyze = async (url) => (await fetch(API + "/api/analyze-website", { method: "POST",
  headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) })).json();

try {
  await send("Page.enable"); await send("Runtime.enable");
  await send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: downloads });
  await viewport(1440);

  // 1) page loads, sidebar entry, form controls
  await send("Page.navigate", { url: BASE + "/website-intelligence" });
  await waitFor(`!!document.querySelector('main h1')`, 30000);
  await waitFor(`/API connected/.test(document.querySelector('header')?.innerText ?? '')`, 30000);   // hydrated
  const s = await ev(`({ h1: document.querySelector('main h1')?.textContent, title: document.title,
    active: document.querySelector('aside nav a[aria-current="page"]')?.getAttribute('href'),
    label: document.querySelector('label[for="site-url"]')?.textContent, button: document.querySelector('form button[type=submit]')?.textContent,
    future: /Shopify/.test(document.body.innerText) && /Google Analytics 4/.test(document.body.innerText) })`);
  assert("page loads with heading, sidebar entry active, URL field + Analyze button",
    s.h1 === "Website intelligence" && s.active === "/website-intelligence" && s.label === "Website URL" && s.button === "Analyze Website",
    JSON.stringify(s));
  assert("Shopify / GA4 shown as future, not connected", s.future && await ev(`[...document.querySelectorAll('button')].filter(b => b.textContent.trim() === 'Coming later').every(b => b.disabled)`));
  assert("document title", /Website intelligence · ShopSense AI/.test(s.title), s.title);

  // 2) validation: empty and unsafe URLs
  await submit(); await sleep(300);
  assert("empty submit shows an inline error", /Enter a website address/.test(await ev(`document.getElementById('site-url-error').textContent`)));
  await setInput("http://127.0.0.1:8000/health"); await submit();
  await waitFor(done, 20000);
  const unsafe = await ev(`document.querySelector('main').innerText`);
  assert("unsafe URL (local IP + port) refused with explanation", /cannot be analysed/i.test(unsafe) && /(IP address|port)/i.test(unsafe), unsafe.slice(0, 200));

  // 3) real analysis: example.com. Live sites are not deterministic between runs (e.g. the
  //    slow-response heuristic), so the UI is compared with the exact analysis it rendered:
  //    the JSON export of that same result.
  await setInput("example.com"); await submit();
  const ok = await waitFor(`!!document.querySelector('[aria-label="Analysis results"]')`, 90000);
  await waitFor(done, 30000);
  const before = new Set(readdirSync(downloads));
  await ev(`[...document.querySelectorAll('button')].find(b => /Download report \\(HTML\\)/.test(b.textContent)).click()`);
  await ev(`[...document.querySelectorAll('button')].find(b => /Download JSON/.test(b.textContent)).click()`);
  for (let i = 0; i < 40 && readdirSync(downloads).filter((f) => !before.has(f) && !f.endsWith(".crdownload")).length < 2; i++) await sleep(250);
  const files = readdirSync(downloads).filter((f) => !before.has(f));
  const htmlFile = files.find((f) => f.endsWith(".html"));
  const jsonFile = files.find((f) => f.endsWith(".json"));
  const htmlText = htmlFile ? readFileSync(join(downloads, htmlFile), "utf8") : "";
  const api = jsonFile ? JSON.parse(readFileSync(join(downloads, jsonFile), "utf8")) : { summary: { issues_by_severity: {} }, issues: [], conversion_checklist: [] };
  assert("HTML report downloads, contains findings, is script-free with CSP",
    !!htmlFile && htmlText.includes("Technical, SEO and usability issues") && !/<script/i.test(htmlText) && htmlText.includes("default-src 'none'"), String(files));
  const fresh = await apiAnalyze("example.com");      // independent API call: same structure, same site
  assert("JSON report downloads and is the full analysis (same shape as the API)",
    !!jsonFile && Object.keys(api).sort().join() === Object.keys(fresh).sort().join() && api.final_url === fresh.final_url, String(files));
  const kpis = await ev(`[...document.querySelectorAll('[aria-label="Analysis results"] .text-3xl')].map(e => e.textContent)`);
  const sev = api.summary.issues_by_severity;
  assert("example.com: results rendered and KPI counts match the analysis data",
    ok && JSON.stringify(kpis) === JSON.stringify([String(api.summary.pages_analyzed), String(sev.high), String(sev.medium), String(sev.low), String(sev.info)]),
    `${kpis} vs API ${JSON.stringify(sev)}`);
  const issueRows = await ev(`[...document.querySelectorAll('main section')].find(x => /Technical issues report/.test(x.innerText))?.querySelectorAll('tbody tr').length ?? 0`);
  assert("issues table lists every finding with evidence + fix", issueRows === api.issues.length &&
    await ev(`/Evidence:/.test(document.body.innerText) && /Fix:/.test(document.body.innerText)`), `${issueRows} vs ${api.issues.length}`);
  const checklist = await ev(`[...document.querySelectorAll('main section')].find(x => /Conversion-readiness checklist/.test(x.innerText))?.querySelectorAll('li').length ?? 0`);
  assert("conversion checklist shows all items, labelled heuristic", checklist >= api.conversion_checklist.length &&
    await ev(`/do not measure actual conversions/.test(document.body.innerText)`), String(checklist));
  assert("charts render (issues by category, fetch time)", (await ev(`document.querySelectorAll('svg.recharts-surface').length`)) >= 2);
  assert("performance shown as unavailable (no API key) - not invented",
    await ev(`/Unavailable\\./.test(document.body.innerText) && /never estimated/.test(document.body.innerText)`));
  assert("evidence legend present", await ev(`/A direct fact/.test(document.body.innerText) && /can be wrong/.test(document.body.innerText)`));
  // filter
  await ev(`(() => { const sel = [...document.querySelectorAll('select')].find(x => x.parentElement.textContent.includes('Severity'));
    const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(sel, 'medium');
    sel.dispatchEvent(new Event('change', { bubbles: true })); })()`);
  await sleep(300);
  const filtered = await ev(`[...document.querySelectorAll('main section')].find(x => /Technical issues report/.test(x.innerText))?.querySelectorAll('tbody tr').length ?? 0`);
  assert("severity filter works", filtered === api.issues.filter((i) => i.severity === "medium").length, String(filtered));

  await shot("website-intelligence-example");

  // 5) a richer real site: multiple pages, robots.txt found
  await setInput("https://www.python.org"); await submit();
  await waitFor(`(document.querySelector('[aria-label="Analysis results"]')?.innerText ?? '').includes('python.org') && ${done}`, 120000);
  const pyRows = await ev(`[...document.querySelectorAll('main section')].find(x => /Pages analysed/.test(x.querySelector('h2')?.textContent ?? ''))?.querySelectorAll('tbody tr').length ?? 0`);
  const robotsText = await ev(`document.querySelector('[aria-label="Analysis results"]').innerText`);
  assert("python.org: up to 5 pages analysed, robots.txt found", pyRows >= 2 && pyRows <= 5 && /robots\.txt: found/.test(robotsText), `${pyRows} rows`);
  await shot("website-intelligence-python");

  // 6) accessibility (axe-core) on the results page
  await send("Runtime.evaluate", { expression: readFileSync(AXE, "utf8") });
  const vio = await ev(`axe.run(document, { runOnly: { type: "tag", values: ["wcag2a","wcag2aa","wcag21a","wcag21aa"] } })
    .then(r => r.violations.map(v => ({ id: v.id, impact: v.impact, nodes: v.nodes.length, sample: v.nodes[0]?.target?.join(' ') })))`);
  assert("a11y: no WCAG 2.1 A/AA violations on the results page", vio.length === 0, JSON.stringify(vio));

  // 7) mobile layout with results
  await viewport(390, 844, true);
  await sleep(600);
  const o = await ev(`({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth })`);
  assert("390px: no horizontal overflow with results shown", o.sw <= o.cw, JSON.stringify(o));
  await shot("website-intelligence-mobile");
  assert("no app exceptions or console errors (incl. React key warnings)", appErrors.filter((t) => !/chrome-extension/.test(t)).length === 0, appErrors.join(" | "));
} catch (e) {
  assert("script completed", false, e.message);
} finally {
  ws.close(); proc.kill();
}
const failed = results.filter((r) => !r.ok);
for (const r of results) console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.ok ? "" : "  -> " + r.detail}`);
console.log(`\n${results.length - failed.length}/${results.length} website-intelligence checks passed`);
process.exitCode = failed.length ? 1 : 0;
