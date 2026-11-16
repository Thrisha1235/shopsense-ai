# tools/

Verification scripts used to sign off Phases 5–8. All of them need the backend (port 8000) running,
except `verify_rate_limits.py`, which uses its own production-mode server.

## `verify_dashboard_data.py`: are the dashboard numbers real?

Recomputes the dashboard statistics **directly from the raw CSV** with plain pandas, independent
of the project's own `ml/` code. It compares them with what `/api/analytics` serves, and compares
`/api/model-performance` and `/api/model-info` with the Phase 3–4 report files. It also re-scores
the saved models on the test split to check the served ROC curves.

```powershell
cd backend
.\.venv\Scripts\python.exe ..\tools\verify_dashboard_data.py   # -> "148 checks, 0 failures"
```

## `verify_ui.mjs`: does every page work?

Drives headless Microsoft Edge through the Chrome DevTools Protocol (Node 24, no extra packages).
It checks:
- every page loads with no errors;
- KPI values match the API;
- sidebar and mobile-drawer navigation, and keyboard skip link;
- no horizontal overflow at 390, 768, 1024 and 1440 px;
- chart tooltips and the table toggle;
- prediction form labels, validation and results;
- Explainable AI: tiles match `/api/explain`, sample switching, 16-row contribution table, the
  "Explain this prediction" hand-off, API-error state, and mobile chart width.

It writes screenshots to the folder you pass in.

```powershell
# frontend running on 3002 (npm run dev) or set UI_BASE, e.g. a production build on 3005
$env:UI_BASE = "http://localhost:3002"
node tools\verify_ui.mjs docs\screenshots "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" "$env:TEMP\shopsense-edge"
# -> "79/79 checks passed" (71 Phase 5 checks + 8 Explainable AI checks added in Phase 6)
```

Phase 7 added an end-to-end user-journey check (Overview → Analytics → Prediction → SHAP →
Performance) and an **axe-core WCAG 2.1 A/AA audit** of every page (results in
`<outDir>/axe-report.json`). It now reports **85/85**. It also runs in Google Chrome: pass
`"$env:ProgramFiles\Google\Chrome\Application\chrome.exe"` as the browser.

## `verify_firefox.mjs`: Firefox compatibility

Uses Firefox's standard **WebDriver BiDi** protocol (Node 24, no extra packages). It checks every
page loads without errors, a prediction (22.6%), that the SHAP tiles match the API, and no
horizontal overflow at 390 px.

```powershell
node tools\verify_firefox.mjs docs\screenshots "$env:ProgramFiles\Mozilla Firefox\firefox.exe" "$env:TEMP\shopsense-ff"
# -> "12/12 Firefox checks passed"
```

## `verify_predictions.py`: live API vs saved artifacts

Sends all 2,466 held-out test sessions to `/api/predict`. For each one it checks the score against
the saved pipeline, the probability against the saved calibrator, and the class against the
threshold. It then recomputes the Phase 3 test metrics from the API's answers (they must match
exactly) and checks SHAP additivity on 100 `/api/explain` calls. It takes ~10 minutes and uses
`127.0.0.1`, because "localhost" adds ~2 s per new connection on Windows.

```powershell
cd backend
.\.venv\Scripts\python.exe ..\tools\verify_predictions.py   # -> "0 failures"
```

## Phase 8: Website Intelligence

**Current results:**
- `verify_ui.mjs`: **97/97** in Edge, covering 6 pages including `/website-intelligence`, with
  0 axe violations.
- `verify_firefox.mjs`: **15/15**. It now includes a URL → results flow on the Website
  Intelligence page.

### `verify_website_ui.mjs`: URL input → analysis → report downloads

Drives Edge or Chrome through the DevTools protocol against the **live** backend. It makes real
requests to `example.com` and `www.python.org`, and respects their robots.txt.

It checks:
- the page, the URL field and the button;
- Shopify and GA4 shown as disabled "Coming later" items;
- an empty URL shows an inline error, and a local IP with a port is refused with an explanation;
- a real analysis, with the KPI tiles and issue and checklist counts compared with the
  downloaded JSON report;
- performance shown as unavailable when no key is set;
- the charts and the severity filter;
- the HTML report download: no scripts, a restrictive CSP, and all content escaped;
- the JSON download;
- a 5-page analysis of python.org;
- axe-core, a 390 px layout, and no console errors.

```powershell
$env:UI_BASE = "http://localhost:3002"
node tools\verify_website_ui.mjs "$env:TEMP\wi-shots" "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" "$env:TEMP\shopsense-wi"
# -> "18/18 checks passed" (also 18/18 with Chrome)
```

Each run uses about 3–4 website analyses. With the default limit (10 per 10 minutes), repeated
runs hit HTTP 429. For repeated local testing, start the dev backend with
`$env:WEBSITE_RATE_LIMIT=100` (an environment variable only; do not change the defaults).

### `verify_rate_limits.py`: the **default** production limits

Start a separate production-mode server with **no** `WEBSITE_*` overrides, then run the check:

```powershell
cd backend
$env:ENVIRONMENT = "production"; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8011
# second terminal
.\.venv\Scripts\python.exe ..\tools\verify_rate_limits.py http://127.0.0.1:8011
# -> concurrency [200, 200, 429]; 429 at request #11 with Retry-After; /api/predict 200 (0.2256); "0 failures"
```

## `verify_deployment.py`: check the live Render + Vercel deployment

It runs read-only checks against the **deployed** addresses and uses only Python's standard
library. It checks:
- HTTPS, HSTS and docs hidden;
- every endpoint the six pages use (prediction 0.2256, SHAP additivity, 422 on bad input);
- CORS for the frontend origin and refusal of other origins;
- 7 SSRF targets refused with 400;
- one real analysis of example.com;
- all six frontend pages.

It uses 8 of the 10 website analyses allowed per 10 minutes.

```powershell
backend\.venv\Scripts\python.exe tools\verify_deployment.py --api https://<service>.onrender.com --web https://<app>.vercel.app
# -> "28/28 deployment checks passed" (verified against a local production simulation)
```

---

Run `verify_predictions.py` against a quiet server. If files change during a run, the dev server reloads pages and
some navigation checks can fail.
