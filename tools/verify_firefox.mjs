// Firefox compatibility check via WebDriver BiDi (Firefox's standard automation protocol).
// Usage: node tools/verify_firefox.mjs <screenshotDir> <firefoxPath> <profileDir>
// Needs the backend (8000) and a frontend (UI_BASE, default http://localhost:3002) running.
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [outDir, firefox, profile] = process.argv.slice(2);
const BASE = process.env.UI_BASE ?? "http://localhost:3002";
const API = "http://localhost:8000";
const PORT = 9335;
mkdirSync(outDir, { recursive: true });
mkdirSync(profile, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const proc = spawn(firefox, ["--headless", "--no-remote", "--profile", profile, `--remote-debugging-port=${PORT}`],
  { stdio: "ignore" });

let ws;
for (let i = 0; i < 80 && !ws; i++) {
  try {
    const sock = new WebSocket(`ws://127.0.0.1:${PORT}/session`);
    await new Promise((res, rej) => { sock.addEventListener("open", res, { once: true }); sock.addEventListener("error", rej, { once: true }); });
    ws = sock;
  } catch { await sleep(300); }
}
if (!ws) { console.error("could not connect to Firefox WebDriver BiDi"); proc.kill(); process.exit(1); }

let id = 1;
const pending = new Map();
const errors = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.type === "error" ? p.reject(new Error(`${m.error}: ${m.message}`)) : p.resolve(m.result); }
  if (m.method === "log.entryAdded" && m.params.level === "error") errors.push(m.params.text.slice(0, 200));
});
const send = (method, params = {}) => new Promise((resolve, reject) => { const i = id++; pending.set(i, { resolve, reject }); ws.send(JSON.stringify({ id: i, method, params })); });

const results = [];
const assert = (name, ok, detail = "") => results.push({ name, ok: !!ok, detail });

try {
  await send("session.new", { capabilities: {} });
  await send("session.subscribe", { events: ["log.entryAdded"] });
  const context = (await send("browsingContext.getTree", {})).contexts[0].context;
  const ev = async (expr) => {
    const r = await send("script.evaluate", { expression: `(async () => JSON.stringify(await (async () => (${expr}))()))()`,
      target: { context }, awaitPromise: true, resultOwnership: "none" });
    if (r.type === "exception") throw new Error(r.exceptionDetails.text);
    return r.result.value === undefined ? undefined : JSON.parse(r.result.value);
  };
  const viewport = (width, height = 900) => send("browsingContext.setViewport", { context, viewport: { width, height } });
  const settle = async () => {
    for (let i = 0; i < 100; i++) {
      if (await ev(`!!document.querySelector('main h1') && document.querySelectorAll('.animate-pulse, [aria-busy="true"]').length === 0`)) return true;
      await sleep(200);
    }
    return false;
  };
  const go = async (path) => { await send("browsingContext.navigate", { context, url: BASE + path, wait: "complete" }); await sleep(500); return settle(); };
  const shot = async (name) => writeFileSync(join(outDir, `${name}.png`),
    Buffer.from((await send("browsingContext.captureScreenshot", { context, origin: "document" })).data, "base64"));

  const pages = [["/", "Overview", 3], ["/analytics", "Customer behaviour", 12], ["/predict", "Purchase prediction", 0],
                 ["/performance", "Model performance", 3], ["/explainability", "Explainable AI", 2],
                 ["/website-intelligence", "Website intelligence", 0]];
  await viewport(1440);
  for (const [path, h1, minCharts] of pages) {
    errors.length = 0;
    const settled = await go(path);
    // Recharts draws after measuring its container: wait (max 15 s) for the expected charts
    for (let i = 0; i < 75 && (await ev(`document.querySelectorAll('svg.recharts-surface').length`)) < minCharts; i++) await sleep(200);
    const s = await ev(`({ h1: document.querySelector('main h1')?.textContent, charts: document.querySelectorAll('svg.recharts-surface').length,
      alerts: document.querySelectorAll('[role=alert]').length })`);
    assert(`firefox ${path}: loads (heading, ${s.charts} charts, no errors)`,
      settled && s.h1 === h1 && s.charts >= minCharts && s.alerts === 0 && errors.length === 0, JSON.stringify({ s, errors }));
  }

  // prediction + explanation agree with the API
  const session = { Administrative: 2, Administrative_Duration: 45, Informational: 0, Informational_Duration: 0,
    ProductRelated: 30, ProductRelated_Duration: 1200, BounceRates: 0.01, ExitRates: 0.03, SpecialDay: 0, Month: "Nov",
    OperatingSystems: 2, Browser: 2, Region: 1, TrafficType: 2, VisitorType: "Returning_Visitor", Weekend: false };
  const api = await (await fetch(API + "/api/explain", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(session) })).json();
  await go("/predict");
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Predict purchase').click()`);
  for (let i = 0; i < 50 && !(await ev(`/Calibrated purchase probability/.test(document.querySelector('[aria-live="polite"]')?.innerText ?? '')`)); i++) await sleep(200);
  const res = await ev(`document.querySelector('[aria-live="polite"]').innerText`);
  assert("firefox: prediction shows 22.6% calibrated probability", res.includes((api.prediction.purchase_probability * 100).toFixed(1) + "%"), res.slice(0, 120));
  await shot("firefox-predict");
  await ev(`[...document.querySelectorAll('a')].find(a => a.textContent.trim() === 'Explain this prediction').click()`);
  for (let i = 0; i < 100 && !(await ev(`location.pathname === '/explainability' && !!document.querySelector('[aria-live="polite"][aria-busy="false"] [data-tile-value]')`)); i++) await sleep(200);
  const tiles = await ev(`[...document.querySelectorAll('main [data-tile-value]')].map(e => e.textContent)`);
  assert("firefox: SHAP explanation tiles match /api/explain", tiles[1] === api.explanation.model_score.toFixed(3) && tiles[0] === "0.500", String(tiles));
  await shot("firefox-explainability");

  // Website Intelligence: URL input -> analysis results (Phase 8)
  await go("/website-intelligence");
  for (let i = 0; i < 100 && !(await ev(`/API connected/.test(document.querySelector('header')?.innerText ?? '')`)); i++) await sleep(200);
  await ev(`(() => { const el = document.getElementById('site-url');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, 'example.com');
    el.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  await ev(`document.querySelector('form[aria-label="Analyse a website"] button[type=submit]').click()`);
  for (let i = 0; i < 300 && !(await ev(`!!document.querySelector('[aria-label="Analysis results"]')`)); i++) await sleep(200);
  const wi = await ev(`({ results: !!document.querySelector('[aria-label="Analysis results"]'),
    report: /Technical issues report/.test(document.body.innerText), checklist: /Conversion-readiness checklist/.test(document.body.innerText),
    downloads: [...document.querySelectorAll('button')].filter(b => /Download (report|JSON)/.test(b.textContent)).length,
    charts: document.querySelectorAll('svg.recharts-surface').length })`);
  assert("firefox: website intelligence URL -> results (issues, checklist, charts, export buttons)",
    wi.results && wi.report && wi.checklist && wi.downloads === 2 && wi.charts >= 2 && errors.length === 0, JSON.stringify({ wi, errors }));
  await shot("firefox-website-intelligence");

  // mobile width: no horizontal overflow
  await viewport(390, 844);
  for (const [path] of pages) {
    await go(path);
    const o = await ev(`({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth })`);
    assert(`firefox 390px ${path}: no horizontal overflow`, o.sw <= o.cw, JSON.stringify(o));
  }
  await send("session.end", {});
} catch (e) {
  assert("firefox script completed", false, e.message);
} finally {
  ws.close(); proc.kill();
}
const failed = results.filter((r) => !r.ok);
for (const r of results) console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.ok ? "" : "  -> " + r.detail}`);
console.log(`\n${results.length - failed.length}/${results.length} Firefox checks passed`);
process.exitCode = failed.length ? 1 : 0;
