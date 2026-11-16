// Phase 5 UI verification via the Chrome DevTools Protocol (headless Edge).
// Usage: node verify_ui.mjs <screenshotDir> <edgePath> <profileDir>
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// axe-core is a frontend devDependency (npm install in frontend/).
const AXE_PATH = join(dirname(fileURLToPath(import.meta.url)), "..", "frontend", "node_modules", "axe-core", "axe.min.js");
const axeReport = {};

const [outDir, edge, profile] = process.argv.slice(2);
const BASE = process.env.UI_BASE ?? "http://localhost:3002";
const API = "http://localhost:8000";
const PORT = 9334;
mkdirSync(outDir, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const proc = spawn(edge, ["--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`, "--disable-extensions", "about:blank"], { stdio: "ignore" });
let target;
for (let i = 0; i < 60 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page"); }
  catch { await sleep(250); }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 1;
const pending = new Map();
const appErrors = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result); }
  if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails; const text = `${d.text} ${d.exception?.description ?? ""}`;
    if (!/chrome-extension:/.test(text)) appErrors.push(text.slice(0, 300));
  }
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
    const text = m.params.args.map((a) => a.value ?? a.description).join(" ");
    if (!/chrome-extension:|Download the React DevTools/.test(text)) appErrors.push("console.error: " + text.slice(0, 300));
  }
});
const send = (method, params = {}) => new Promise((resolve, reject) => { const i = id++; pending.set(i, { resolve, reject }); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result.value;
await send("Page.enable"); await send("Runtime.enable");

const results = [];
const assert = (name, ok, detail = "") => { results.push({ name, ok: !!ok, detail }); };

async function viewport(width, height = 900, mobile = width < 768) {
  current = { width, height, mobile };
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile });
}
async function settle(timeout = 20000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const busy = await ev(`document.querySelectorAll('[role=status][aria-label^="Loading"], .animate-pulse').length`);
    const h1 = await ev(`!!document.querySelector('main h1')`);
    if (h1 && busy === 0) return true;
    await sleep(300);
  }
  return false;
}
async function waitFor(expr, timeout = 20000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    if (await ev(expr)) return true;
    await sleep(200);
  }
  return false;
}
// Explanation panel finished loading (it keeps old results visible while busy).
const explainIdle = () => waitFor(`!!document.querySelector('[aria-live="polite"][aria-busy="false"] [data-tile-value]')`);
async function go(path) { await send("Page.navigate", { url: BASE + path }); await sleep(800); return settle(); }
let current = { width: 1440, height: 900, mobile: false };
async function shot(name, full = true) {
  if (full) {
    // Grow the viewport to the page height so fixed elements (sidebar) render full-length.
    const { height } = (await send("Page.getLayoutMetrics")).cssContentSize;
    await send("Emulation.setDeviceMetricsOverride", { width: current.width, height: Math.min(Math.ceil(height), 7000), deviceScaleFactor: 1, mobile: current.mobile });
    await sleep(500);
  }
  writeFileSync(join(outDir, `${name}.png`), Buffer.from((await send("Page.captureScreenshot", { format: "png" })).data, "base64"));
  if (full) await send("Emulation.setDeviceMetricsOverride", { width: current.width, height: current.height, deviceScaleFactor: 1, mobile: current.mobile });
}
const key = async (k) => {
  const map = { Tab: 9, Escape: 27 };
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: k, code: k, windowsVirtualKeyCode: map[k] });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: k, code: k, windowsVirtualKeyCode: map[k] });
};
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const int = (n) => n.toLocaleString("en-US");

const PAGES = [
  { path: "/", h1: "Overview", minCharts: 3 },
  { path: "/analytics", h1: "Customer behaviour", minCharts: 12 },
  { path: "/predict", h1: "Purchase prediction", minCharts: 0 },
  { path: "/performance", h1: "Model performance", minCharts: 3 },
  { path: "/explainability", h1: "Explainable AI", minCharts: 2 },
  { path: "/website-intelligence", h1: "Website intelligence", minCharts: 0 },   // Phase 8 (detailed checks: verify_website_ui.mjs)
];
// Same sessions the frontend sends for its two samples (fields.ts EXAMPLE / QUICK_BOUNCE).
const ENGAGED = { Administrative: 2, Administrative_Duration: 45, Informational: 0, Informational_Duration: 0,
  ProductRelated: 30, ProductRelated_Duration: 1200, BounceRates: 0.01, ExitRates: 0.03, SpecialDay: 0, Month: "Nov",
  OperatingSystems: 2, Browser: 2, Region: 1, TrafficType: 2, VisitorType: "Returning_Visitor", Weekend: false };
const BOUNCE = { ...ENGAGED, Administrative: 0, Administrative_Duration: 0, ProductRelated: 1, ProductRelated_Duration: 0,
  BounceRates: 0.2, ExitRates: 0.2, Month: "May" };
const apiExplain = async (s) => (await fetch(API + "/api/explain", { method: "POST",
  headers: { "Content-Type": "application/json" }, body: JSON.stringify(s) })).json();
const analytics = await (await fetch(API + "/api/analytics")).json();
const perf = await (await fetch(API + "/api/model-performance")).json();

try {
  // 1) Every page loads (desktop) ------------------------------------------------
  await viewport(1440);
  for (const p of PAGES) {
    appErrors.length = 0;
    const settled = await go(p.path);
    const info = await ev(`({ h1: document.querySelector('main h1')?.textContent, charts: document.querySelectorAll('svg.recharts-surface').length,
      alerts: document.querySelectorAll('[role=alert]').length, title: document.title })`);
    assert(`load ${p.path}: settled (no loading skeletons left)`, settled);
    assert(`load ${p.path}: heading "${p.h1}"`, info.h1 === p.h1, info.h1);
    assert(`load ${p.path}: ${info.charts} charts (>= ${p.minCharts})`, info.charts >= p.minCharts);
    assert(`load ${p.path}: no error alerts`, info.alerts === 0, String(info.alerts));
    assert(`load ${p.path}: no app exceptions/console errors`, appErrors.length === 0, appErrors.join(" | "));
    assert(`load ${p.path}: document title`, /ShopSense AI/.test(info.title), info.title);
  }

  // 2) Rendered numbers equal the API -------------------------------------------
  await go("/");
  const kpis = await ev(`[...document.querySelectorAll('main .text-3xl')].map(e => e.textContent)`);
  const o = analytics.overview;
  const expected = [int(o.sessions), pct(o.conversion_rate, 2), int(o.purchases), int(o.no_purchases)];
  assert("overview KPIs match /api/analytics", JSON.stringify(kpis) === JSON.stringify(expected), `${kpis} vs ${expected}`);
  await go("/performance");
  const served = perf.models.find((m) => m.feature_set === perf.default_variant && m.selected);
  const pk = await ev(`[...document.querySelectorAll('main .text-3xl')].map(e => e.textContent)`);
  const pe = [served.test.pr_auc.toFixed(3), served.test.roc_auc.toFixed(3), served.test.f1.toFixed(3)];
  assert("performance KPIs match /api/model-performance", JSON.stringify(pk.slice(0, 3)) === JSON.stringify(pe), `${pk} vs ${pe}`);
  const cm = await ev(`[...document.querySelectorAll('main table td span.text-xl')].map(e => e.textContent)`);
  const c = served.confusion_matrix;
  assert("confusion matrix matches API", JSON.stringify(cm) === JSON.stringify([c.tn, c.fp, c.fn, c.tp].map(int)), String(cm));
  await shot("desktop-performance");

  // 3) Navigation ------------------------------------------------------------------
  await go("/");
  const tab = await (async () => { await key("Tab"); return ev(`document.activeElement?.textContent?.trim()`); })();
  assert("keyboard: first Tab focuses the skip link", tab === "Skip to content", tab);
  for (const p of PAGES) {
    await ev(`document.querySelector('aside nav a[href="${p.path}"]').click()`);
    await sleep(600); await settle();
    const s = await ev(`({ path: location.pathname, current: document.querySelector('aside nav a[aria-current="page"]')?.getAttribute('href'), h1: document.querySelector('main h1')?.textContent })`);
    assert(`sidebar link ${p.path}: navigates + aria-current`, s.path === p.path && s.current === p.path && s.h1 === p.h1, JSON.stringify(s));
  }

  // 4) Charts: hover tooltip + table toggle ---------------------------------------
  await go("/");
  const box = await ev(`(() => { const r = document.querySelector('svg.recharts-surface').getBoundingClientRect(); return { x: r.x + r.width * 0.55, y: r.y + r.height * 0.5 }; })()`);
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: box.x, y: box.y });
  await sleep(400);
  const tip = await ev(`document.querySelector('.recharts-tooltip-wrapper')?.innerText ?? ''`);
  assert("chart hover shows tooltip with conversion + CI", /Conversion/.test(tip) && /95% CI/.test(tip), tip.replace(/\n/g, " "));
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim().toLowerCase() === 'table').click()`);
  await sleep(300);
  const rows = await ev(`document.querySelector('main table tbody')?.rows.length ?? 0`);
  assert("table toggle shows one row per month", rows === analytics.monthly.length, String(rows));
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim().toLowerCase() === 'chart').click()`);

  // 5) Prediction form ----------------------------------------------------------
  await go("/predict");
  const form = await ev(`(() => {
    const ctrls = [...document.querySelectorAll('form input, form select')];
    const labelled = ctrls.filter(c => document.querySelector('label[for="' + c.id + '"]'));
    const described = ctrls.filter(c => (c.getAttribute('aria-describedby') || '').split(' ').every(i => document.getElementById(i)));
    const sw = document.querySelector('form [role=switch]');
    return { n: ctrls.length, labelled: labelled.length, described: described.length,
             switchLabelled: !!(sw && document.getElementById(sw.getAttribute('aria-labelledby'))), groups: document.querySelectorAll('form fieldset legend').length };
  })()`);
  assert(`form: ${form.n} inputs/selects + 1 switch = 16 model features`, form.n + 1 === 16, JSON.stringify(form));
  assert("form: every control has a <label>", form.labelled === form.n && form.switchLabelled, JSON.stringify(form));
  assert("form: every control has help text (aria-describedby)", form.described === form.n);
  assert("form: 5 grouped fieldsets with legends", form.groups === 5, String(form.groups));
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Predict purchase').click()`);
  await sleep(2500);
  const res = await ev(`document.querySelector('[aria-live="polite"]')?.innerText ?? ''`);
  assert("form: valid submit shows calibrated probability + class + raw score",
    /Calibrated purchase probability/.test(res) && /(Likely|Unlikely) to purchase/.test(res) && /Raw model score/.test(res), res.slice(0, 160).replace(/\n/g, " "));
  await shot("desktop-predict");
  await ev(`(() => { const lab = [...document.querySelectorAll('label')].find(l => l.textContent.trim() === 'Exit rate');
    const el = document.getElementById(lab.htmlFor); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '-4');
    el.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Predict purchase').click()`);
  await sleep(500);
  const inv = await ev(`({ invalid: document.querySelectorAll('[aria-invalid="true"]').length, summary: document.activeElement?.innerText?.slice(0, 80) })`);
  assert("form: invalid value flagged + focus moves to error summary", inv.invalid === 1 && /Please fix/.test(inv.summary ?? ""), JSON.stringify(inv));

  // 5b) Explainable AI (Phase 6) ----------------------------------------------
  const tiles = () => ev(`[...document.querySelectorAll('main [data-tile-value]')].map(e => e.textContent)`);
  const expEngaged = await apiExplain(ENGAGED);
  const expBounce = await apiExplain(BOUNCE);
  await send("Runtime.evaluate", { expression: "sessionStorage.clear()" });
  await go("/explainability");
  await explainIdle();
  let t = await tiles();
  assert("XAI: sample explanation tiles match /api/explain (baseline, score, probability)",
    t[0] === expEngaged.explanation.baseline_score.toFixed(3) && t[1] === expEngaged.explanation.model_score.toFixed(3)
      && t[3] === pct(expEngaged.prediction.purchase_probability), `${t} vs API ${expEngaged.explanation.model_score}`);
  assert("XAI: 'Last prediction' disabled when nothing was predicted",
    await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Last prediction')?.disabled === true`));
  const xaiText = await ev(`document.querySelector('main').innerText`);
  assert("XAI: page states it explains the raw score, not causes",
    /raw model score/i.test(xaiText) && /not causes|does not establish cause/i.test(xaiText));
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Sample: quick bounce').click()`);
  await sleep(300); await explainIdle();
  t = await tiles();
  assert("XAI: switching sample re-explains (quick bounce matches API)", t[1] === expBounce.explanation.model_score.toFixed(3), String(t));
  await ev(`[...document.querySelectorAll('main section')].find(s => /pushed this session/.test(s.innerText))
    .querySelectorAll('button').forEach(b => { if (b.textContent.trim().toLowerCase() === 'table') b.click(); })`);
  await sleep(300);
  const nRows = await ev(`[...document.querySelectorAll('main section')].find(s => /pushed this session/.test(s.innerText)).querySelector('tbody')?.rows.length ?? 0`);
  assert("XAI: contribution table lists all 16 model features", nRows === 16, String(nRows));
  // prediction page -> "Explain this prediction" explains the exact session just scored
  await go("/predict");
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Sample: quick bounce').click()`);
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Predict purchase').click()`);
  await sleep(2500);
  await ev(`[...document.querySelectorAll('a')].find(a => a.textContent.trim() === 'Explain this prediction').click()`);
  await waitFor(`location.pathname === '/explainability'`); await sleep(300); await explainIdle();
  t = await tiles();
  const lastPressed = await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Last prediction')?.getAttribute('aria-pressed')`);
  assert("XAI: 'Explain this prediction' explains the session just scored",
    lastPressed === "true" && t[1] === expBounce.explanation.model_score.toFixed(3), `${lastPressed} ${t}`);
  await shot("desktop-explainability");
  // API failure -> error state with retry (explain endpoint blocked)
  await send("Network.enable");
  await send("Network.setBlockedURLs", { urls: ["*/api/explain"] });
  await go("/explainability");
  await sleep(1500);
  assert("XAI: explanation API failure shows an error with retry",
    await ev(`!!document.querySelector('[role=alert]') && [...document.querySelectorAll('button')].some(b => /Try again/.test(b.textContent))`));
  await send("Network.setBlockedURLs", { urls: [] });

  // Remaining desktop screenshots
  for (const [name, path] of [["desktop-overview", "/"], ["desktop-analytics", "/analytics"]]) {
    await go(path); await shot(name);
  }

  // 5c) End-to-end user journey: Overview -> Analytics -> Prediction -> SHAP -> Performance -----
  await send("Runtime.evaluate", { expression: "sessionStorage.clear()" });
  await go("/");
  const step = async (href) => {
    await ev(`document.querySelector('aside nav a[href="${href}"]').click()`);
    await waitFor(`location.pathname === '${href}'`); await sleep(300); await settle();
    return ev(`({ path: location.pathname, h1: document.querySelector('main h1')?.textContent,
      alerts: document.querySelectorAll('[role=alert]').length, charts: document.querySelectorAll('svg.recharts-surface').length })`);
  };
  const j1 = await step("/analytics");
  const j2 = await step("/predict");
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Predict purchase').click()`);
  await sleep(2500);
  const jPred = await ev(`document.querySelector('[aria-live="polite"]')?.innerText ?? ''`);
  await ev(`[...document.querySelectorAll('a')].find(a => a.textContent.trim() === 'Explain this prediction').click()`);
  await waitFor(`location.pathname === '/explainability'`); await sleep(300); await explainIdle();
  const jTiles = await ev(`[...document.querySelectorAll('main [data-tile-value]')].map(e => e.textContent)`);
  const j4 = await step("/performance");
  assert("journey: Overview -> Analytics -> Prediction -> SHAP explanation -> Model performance",
    j1.path === "/analytics" && j1.charts >= 12 && j1.alerts === 0 &&
    j2.path === "/predict" && /22\.6%/.test(jPred) &&
    jTiles[1] === expEngaged.explanation.model_score.toFixed(3) && jTiles[3] === "22.6%" &&
    j4.path === "/performance" && j4.charts >= 3 && j4.alerts === 0,
    JSON.stringify({ j1, j2, jTiles, j4 }));

  // 5d) Accessibility audit with axe-core (WCAG 2.1 A + AA) on every page --------------------
  const axeSrc = readFileSync(AXE_PATH, "utf8");
  for (const p of PAGES) {
    await go(p.path);
    if (p.path === "/predict") {   // audit the result state too
      await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Predict purchase').click()`);
      await sleep(2500);
    }
    await send("Runtime.evaluate", { expression: axeSrc });
    const res = await ev(`axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } })
      .then(r => r.violations.map(v => ({ id: v.id, impact: v.impact, nodes: v.nodes.length,
        sample: v.nodes[0]?.target?.join(" ") })))`);
    axeReport[p.path] = res;
    const serious = res.filter((v) => v.impact === "critical" || v.impact === "serious");
    assert(`a11y ${p.path}: no critical/serious axe violations`, serious.length === 0, JSON.stringify(serious));
  }

  // 6) Responsiveness -----------------------------------------------------------
  for (const w of [390, 768, 1024, 1440]) {
    await viewport(w);
    for (const p of PAGES) {
      await go(p.path);
      const s = await ev(`({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth,
        aside: getComputedStyle(document.querySelector('aside')).display, menu: getComputedStyle(document.querySelector('[aria-label="Open navigation"]')).display })`);
      const navOk = w >= 1024 ? s.aside !== "none" && s.menu === "none" : s.aside === "none" && s.menu !== "none";
      assert(`responsive ${w}px ${p.path}: no horizontal overflow, correct nav`, s.sw <= s.cw && navOk, JSON.stringify(s));
    }
  }

  // 7) Mobile drawer + mobile prediction ---------------------------------------
  await viewport(390, 844, true);
  await go("/");
  await shot("mobile-overview");
  await ev(`document.querySelector('[aria-label="Open navigation"]').click()`); await sleep(300);
  assert("mobile: menu opens a navigation dialog", await ev(`!!document.querySelector('[role=dialog][aria-label=Navigation]')`));
  await shot("mobile-navigation", false);
  await key("Escape"); await sleep(200);
  assert("mobile: Escape closes the menu", await ev(`!document.querySelector('[role=dialog]')`));
  await ev(`document.querySelector('[aria-label="Open navigation"]').click()`); await sleep(200);
  await ev(`document.querySelector('[role=dialog] a[href="/performance"]').click()`); await sleep(800); await settle();
  assert("mobile: drawer link navigates and closes drawer",
    await ev(`location.pathname === '/performance' && !document.querySelector('[role=dialog]')`));
  await go("/predict");
  await ev(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Predict purchase').click()`);
  await sleep(3000);
  const inView = await ev(`(() => { const r = document.querySelector('[aria-live="polite"]').getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0; })()`);
  assert("mobile: result scrolled into view after predicting", inView);
  await shot("mobile-prediction-result", false);
  await go("/explainability");
  const plotW = await ev(`(() => { const s = [...document.querySelectorAll('main section')].find(x => /pushed this session/.test(x.innerText));
    const g = s?.querySelector('.recharts-bar-rectangles, .recharts-bar'); const svg = s?.querySelector('svg.recharts-surface');
    const grid = s?.querySelector('.recharts-cartesian-grid'); return grid ? grid.getBoundingClientRect().width : 0; })()`);
  assert("mobile: XAI contribution chart keeps a usable plot width (>= 150px)", plotW >= 150, String(plotW));
  await shot("mobile-explainability");
} catch (e) {
  assert("script completed", false, e.message);
} finally {
  ws.close(); proc.kill();
}

writeFileSync(join(outDir, "axe-report.json"), JSON.stringify(axeReport, null, 2));
for (const [page, vs] of Object.entries(axeReport)) {
  if (vs.length) console.log(`axe ${page}: ${vs.map((v) => `${v.id}(${v.impact}, ${v.nodes})`).join(", ")}`);
}
const failed = results.filter((r) => !r.ok);
for (const r of results) console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.ok ? "" : "  -> " + r.detail}`);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exitCode = failed.length ? 1 : 0;
