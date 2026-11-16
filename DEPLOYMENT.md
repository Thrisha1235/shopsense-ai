# ShopSense AI: Deployment Guide

> **Status:** prepared, **not deployed**. Nothing has been published or provisioned. Choose a
> provider and approve any cost before running the commands below.

ShopSense AI has two deployable parts:

| Part | Tech | Serves | Needs at runtime |
|---|---|---|---|
| **Frontend** | Next.js 16 (static-rendered pages, client-side data fetching) | The dashboard | Node 24 (or any static/Node host); the public API URL at **build** time |
| **Backend** | FastAPI + Uvicorn, scikit-learn 1.9.1, SHAP 0.54.1 | `/api/*` JSON endpoints | Python 3.13, the model artifacts in `backend/models/`, and the JSON reports in `backend/reports/` |

```
Browser ──HTTPS──▶ Frontend (Next.js) ──(JS in the browser)──HTTPS──▶ Backend (FastAPI)
                                                                         ├─ models/*.joblib   (pickled pipeline, calibrator)
                                                                         └─ reports/*.json    (dashboard + SHAP global data)
```

The browser calls the API directly. The API must therefore be publicly reachable over HTTPS,
and its `CORS_ORIGINS` must contain the frontend's URL.

**Chosen plan:** backend on **Render** and frontend on **Vercel**. The step-by-step guide is
[section 7](#7-render--vercel-step-by-step). Sections 1–6 explain the reasons and the alternatives.

---

## 1. Model artifacts (not in git)

`backend/models/*.joblib` (about 13 MB in total) are excluded from git: they are binary pickles
and can be regenerated. The committed `backend/models/model_manifest.json` records a **SHA-256
for every artifact**. The API verifies each file **before unpickling it** and refuses mismatches.

| Option | How | When to use |
|---|---|---|
| **A. Rebuild in CI** (deterministic) | Run the pipeline in section 5 (~10 min on 12 cores). Verified: a clean rebuild reproduces every reported metric | Simplest; no storage needed |
| **B. GitHub Release asset** *(recommended; used for Render)* | Attach the 6 `.joblib` files to a release (`models-v1`) of your own repository. At build time, `python -m ml.fetch_artifacts` downloads them, then `python -m ml.integrity` checks them | Fast builds; exact artifacts; nothing to pay for |
| **C. Object storage** | Private S3/GCS/Azure Blob; download in the build with a token kept in the platform's secret store | If the models must stay private |
| **D. Git LFS** | `git lfs track "backend/models/*.joblib"` | Only if LFS quota and bandwidth suit you |

**How the download stays secure (`ml/fetch_artifacts.py`):**
- It accepts **https only** (plain http only for localhost testing).
- Each file must match the **SHA-256 recorded in `model_manifest.json`, which lives in git**.
- A wrong or tampered file is discarded **before** it is saved, so it can never be unpickled.
- File names must be plain names inside `models/`, and each file has a size cap.
- Anyone who wanted to swap a model would need write access to your repository to change the
  manifest as well. That is the same trust boundary as the code itself.

**Is a public release OK?** Yes:
- The files contain no secrets or personal data. They are trained on the public UCI dataset
  (CC BY 4.0), and anyone can rebuild them with option A.
- The risk with pickles is **tampering**, not confidentiality. The checksum check guards against
  tampering.

Security rule for every option: **only load pickles from storage you control, and only after the
checksum check.** The build fails on any missing or mismatched file.

After retraining, always run `ml.calibration` and then `ml.explain`, and commit the updated
manifest. Otherwise the API serves score-only predictions (calibration mismatch) and
`/api/explain` returns 503.

---

## 2. Backend hosting

**Measured requirements** (Windows, Python 3.13, production mode, fresh install from
`requirements-api.txt`, SHAP explainer preloaded, `OPENBLAS_NUM_THREADS=1`):

| Metric | Value |
|---|---|
| Memory after startup | ~230 MB resident |
| Peak memory during the full browser regression (all 6 pages, predictions, explanations, 4 website analyses) | **~241 MB** resident (Windows committed memory ~204 MB). **512 MB is enough**, with about 50% headroom |
| Without the `*_NUM_THREADS=1` settings | Resident memory is about the same, but Windows committed memory reached ~910 MB, because numeric libraries size their thread buffers by CPU count. Keep these settings on small instances |
| `/api/predict` | tens of ms server time |
| `/api/explain` | ~0.2 s SHAP time, ~0.7 s end-to-end warm. The first call is ~5–7 s unless `PRELOAD_EXPLAINER` is on (default in production) |
| CPU | 1 vCPU is enough for demo traffic. SHAP is CPU-bound; at most `EXPLAIN_MAX_CONCURRENCY` (4) run at once, and further requests get 429 |

Run **one Uvicorn worker per container** (each worker holds its own model copy) and scale out
with more containers.

| Option | Notes |
|---|---|
| **Container platform**: Render, Railway, Fly.io, Google Cloud Run, Azure Container Apps, AWS App Runner | Use `backend/Dockerfile`. Set the env vars below. Prefer a minimum of 1 instance (avoid scale-to-zero cold starts of ~5–10 s) |
| **VM / VPS** | `pip install -r requirements-api.txt`, then run `uvicorn app.main:app --host 127.0.0.1 --port 8000` behind Nginx/Caddy for TLS, under systemd |
| Serverless functions (Lambda/Vercel Python) | **Not recommended**: package size (pandas + scikit-learn + SHAP) and cold starts |

### Backend environment variables

| Variable | Production value | Purpose |
|---|---|---|
| `ENVIRONMENT` | `production` | Turns off docs, turns off the localhost CORS wildcard, preloads the explainer |
| `CORS_ORIGINS` | `["https://<your-frontend-domain>"]` | **Required.** Exact frontend origin(s), as a JSON list |
| `PREDICTION_MODEL_VARIANT` | *(unset)* | Default `without_page_values` (leakage-safe). Do not switch without reading MODEL_REPORT §4 |
| `ENABLE_DOCS` | *(unset)* or `false` | Set `true` only if you want public `/docs` |
| `EXPLAIN_MAX_CONCURRENCY` | `4` (use `2` on a small instance) | Per-process cap on simultaneous SHAP explanations |
| `PRELOAD_EXPLAINER` | *(unset)* | Defaults to `true` in production |
| `API_RATE_LIMIT_PER_MINUTE` | *(unset)* = `120` in production | Per-client requests per minute to the model and dashboard endpoints (429 + `Retry-After`). `0` turns it off. Off in development |
| `FORWARDED_ALLOW_IPS` | `10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10,127.0.0.1,::1` | Read by Uvicorn (with `--proxy-headers`): which proxies' `X-Forwarded-For` to believe. **Never `*`**: Uvicorn would then believe the client's own forged entry |
| `MODEL_ARTIFACTS_URL` | `https://github.com/<user>/shopsense-ai/releases/download/models-v1` | **Build time.** Where `ml.fetch_artifacts` downloads the model files from |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` | `1` | Keep numeric thread pools small on 1-CPU instances |
| `PORT` | set by the platform | Port to listen on (Render sets it) |
| `WEBSITE_ANALYSIS_ENABLED` | `true` | Set `false` to turn off `/api/analyze-website` (503) |
| `WEBSITE_RATE_LIMIT` / `WEBSITE_RATE_WINDOW_S` | `10` / `600` | Website analyses per client IP per window (429 + `Retry-After`) |
| `WEBSITE_MAX_CONCURRENCY` | `2` | Simultaneous website analyses per process (each one takes up to ~45 s) |
| `PAGESPEED_API_KEY` | *(unset)* | **Secret, optional.** Google PageSpeed Insights key. Without it, performance is reported as "unavailable" |

The prediction API needs no secrets: it has no database, keys or user accounts. The only optional
secret is `PAGESPEED_API_KEY`. Keep it in the platform's secret store, never in git or in the
frontend.

### Website Intelligence: network and proxy notes

- **Outbound traffic:** the backend needs outbound HTTPS/HTTP (ports 443/80) and DNS. SSRF
  protection is built into the application: it blocks non-public IPs and metadata addresses, pins
  connections to the checked IP, and re-checks every redirect. As defence in depth, also block
  egress from the container to private ranges and `169.254.169.254` where the platform allows it.
- **Client IP:** rate limits are keyed on the client IP. Behind a proxy, run Uvicorn with
  `--proxy-headers` and set `FORWARDED_ALLOW_IPS` to the proxy's (private) address ranges.
  - Uvicorn then uses the right-most address the proxy added to `X-Forwarded-For`, which a client
    cannot forge.
  - Verified locally: 11 requests that each carried a different fake `X-Forwarded-For` were still
    limited at #11, and a second client was unaffected.
  - Each website analysis logs `client=<network>/24`. After deploying, check that it shows your
    public network and not a `10.x` proxy address.
- **Timeouts:** an analysis can take up to ~45 s. Set the proxy or platform request timeout to at
  least **60 s**.
- **Replicas:** limits are per process. With more than one replica, add a rate limit at the proxy as
  well.

---

## 3. Frontend hosting

`NEXT_PUBLIC_API_URL` is **inlined at build time**, so build once per environment.

| Option | Notes |
|---|---|
| **Vercel** (zero-config for Next.js) | Import the repo, set the root directory to `frontend`, set `NEXT_PUBLIC_API_URL=https://<api-domain>` |
| **Netlify / Cloudflare Pages / other Node hosts** | `npm ci && npm run build`, then `npm start` (or the platform's Next.js adapter) |
| **Container** | `frontend/Dockerfile` (multi-stage, production dependencies only, non-root) |

All six routes are pre-rendered as static pages, and data is fetched in the browser, so the
frontend holds no secrets.

---

## 4. Production checklist

- [ ] HTTPS on both domains. Set `CORS_ORIGINS` to the exact frontend origin.
- [ ] `ENVIRONMENT=production` (docs off; check that `GET /docs` returns 404).
- [ ] Model artifacts pass `python -m ml.integrity`. scikit-learn is **1.9.1** (enforced at load).
- [ ] Health check on `GET /health`. Logs are collected (a 503 logs its reason server-side; clients get a generic message).
- [ ] Rate limits in place. The app limits each client IP to 120 requests per minute on the model and dashboard endpoints, caps concurrent explanations, and limits `/api/analyze-website` per IP and per process. With more than one replica, add a limit at the proxy or WAF as well.
- [ ] Website Intelligence: proxy timeout ≥ 60 s, `--proxy-headers` set, egress to private ranges blocked where possible, and `PAGESPEED_API_KEY` (if used) in the secret store.
- [ ] Monitoring: calibration drift. Probabilities describe the training population (one year of sessions from one shop; UCI dataset published 2018).
- [ ] Data notice: the UCI dataset is CC BY 4.0. Credit Sakar & Kastro (2018).

---

## 5. Exact commands

### Rebuild artifacts from scratch (option A)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -I ml/download_data.py       # official UCI zip; SHA-256 checked against data/*.sha256
python -m ml.eda                    # optional: EDA figures/stats
python -m ml.train                  # ~5-7 min (12 cores)
python -m ml.calibration            # training-only calibration (~1 min)
python -m ml.explain                # SHAP record + global importance (~1 min)
python -m ml.dashboard_data         # dashboard JSON
python -m ml.integrity              # verify checksums
pytest                              # full backend suite
```

### Local production-like stack (needs Docker; not run on the development machine)

```powershell
docker compose up --build           # web: http://localhost:3000, api: http://localhost:8000
```

### Backend container on a platform

```powershell
docker build -t shopsense-api ./backend
docker run -p 8000:8000 -e CORS_ORIGINS='["https://shopsense.example.com"]' shopsense-api
```

### Frontend build for a domain

```powershell
cd frontend
$env:NEXT_PUBLIC_API_URL = "https://api.shopsense.example.com"
npm ci; npm run build; npm start
```

---

## 6. What has and has not been verified

| Verified on the development machine | Not verified (no Docker installed / not deployed) |
|---|---|
| Production-mode backend: docs 404, CORS restricted, headers present, ~260 MB RAM | `docker build` of both Dockerfiles and `docker compose up` |
| Production-mode default website limits: 3 simultaneous analyses → `[200, 200, 429]`; 11th request in 10 min → 429 (`Retry-After` 589 s); `/api/predict` unaffected (`tools/verify_rate_limits.py`) | Rate limiting behind Render's real proxy (check the `client=` log line after deploying) |
| **Render/Vercel simulation:** a copy of only the files to be committed, set up as Render's build and start commands would (fresh environment, artifacts downloaded and checksum-verified, a tampered file repaired, production start command). The frontend was built with a production `NEXT_PUBLIC_API_URL`. Edge **97/97**, Firefox **15/15**, website UI **18/18**, `tools/verify_deployment.py` **28/28** | A real Render build on Linux, and a real Vercel build |
| SSRF on the production setup: loopback, metadata, IPv6 loopback, decimal IP and Render-style internal names → 400. Public names that resolve to private IPs (`localtest.me`, `*.nip.io`, `*.sslip.io`) → 400 | Platform-level egress filtering (the app's own SSRF checks do not depend on it) |

---

## 7. Render + Vercel: step by step

**Before you start:**
- Nothing here costs money if you choose the **free** plans.
- The free Render plan has **512 MB RAM**, which is enough (section 2). It **sleeps after
  15 minutes without visitors**, so the first visit afterwards takes about a minute to wake up.
  Render's paid plans stay awake. Check current prices on render.com before choosing.
- Vercel's free **Hobby** plan is for personal, non-commercial projects.
- Sign in to both services with your **personal GitHub account (Thrisha1235)**. When they ask for
  repository access, allow only `shopsense-ai`.

### Step 1: push the prepared changes (needs your approval)

The deployment files (`render.yaml`, `ml/fetch_artifacts.py`, the updated `requirements-api.txt`,
etc.) must be on GitHub first, because Render and Vercel build from your repository.

```powershell
# in the project folder (shopsense-ai)
git add -A
git commit -m "Prepare Render + Vercel deployment"
git push
```

### Step 2: upload the model files as a GitHub Release (needs your approval)

This creates a release called `models-v1` with the 6 `.joblib` files (~13 MB). Render downloads
them during each build and checks every file against the checksum in git.

```powershell
$gh = "$env:LOCALAPPDATA\Programs\gh-cli\bin\gh.exe"
cd backend\models                     # from the project folder
& $gh release create models-v1 --repo Thrisha1235/shopsense-ai `
  --title "Model artifacts v1" `
  --notes "Trained model files for ShopSense AI. Verified at deploy time against backend/models/model_manifest.json (SHA-256)." `
  purchase_model_without_page_values.joblib preprocessor_without_page_values.joblib calibrator_without_page_values.joblib `
  purchase_model_with_page_values.joblib preprocessor_with_page_values.joblib calibrator_with_page_values.joblib
```

The files will then be downloadable from
`https://github.com/Thrisha1235/shopsense-ai/releases/download/models-v1/<file name>`.

### Step 3: create the backend on Render

**Option A, Blueprint (uses `render.yaml`):**
1. Go to Render → **New** → **Blueprint**.
2. Connect GitHub (Thrisha1235) and choose `shopsense-ai`. Render reads `render.yaml` and shows a
   web service called `shopsense-api`.
3. It asks for **`CORS_ORIGINS`**. Enter the frontend address you plan to use, as a JSON list:
   `["https://shopsense-ai.vercel.app"]`. You can correct it in step 5.
4. Click **Apply**. The build takes a few minutes. It runs:
   1. `pip install -r requirements-api.txt`;
   2. `python -m ml.fetch_artifacts`, which downloads and verifies the model files;
   3. `python -m ml.integrity`.
   If any model file is missing or different, **the build fails on purpose**.

**Option B, manual (same settings):** Render → **New** → **Web Service** → choose the repo, then:

| Setting | Value |
|---|---|
| Language / Runtime | Python 3 |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements-api.txt && python -m ml.fetch_artifacts && python -m ml.integrity` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers` |
| Health Check Path | `/health` |
| Instance type | Free (or a paid plan to avoid sleeping) |

Environment variables (Render → service → **Environment**):

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.13.2` |
| `ENVIRONMENT` | `production` |
| `MODEL_ARTIFACTS_URL` | `https://github.com/Thrisha1235/shopsense-ai/releases/download/models-v1` |
| `CORS_ORIGINS` | `["https://shopsense-ai.vercel.app"]` (your real Vercel address) |
| `FORWARDED_ALLOW_IPS` | `10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10,127.0.0.1,::1` |
| `EXPLAIN_MAX_CONCURRENCY` | `2` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `OMP_NUM_THREADS` | `1` |
| `PAGESPEED_API_KEY` | *optional, secret*: only if you want Google PageSpeed data |

When the deploy is live, open `https://<your-service>.onrender.com/health`. It should show
`"environment":"production"`. Write down this address; Vercel needs it.

**HTTPS:** Render issues the certificate automatically. The API also sends
`Strict-Transport-Security`, so browsers stay on HTTPS.

### Step 4: create the frontend on Vercel

1. Go to Vercel → **Add New** → **Project** → import `Thrisha1235/shopsense-ai`.
2. Fill in these settings:

| Setting | Value |
|---|---|
| Framework Preset | Next.js (detected automatically) |
| **Root Directory** | **`frontend`** |
| Build / Install / Output | leave the defaults (`next build`, `npm install`, `.next`) |
| Node.js version | 24.x (also set by `"engines"` in `frontend/package.json`) |
| Environment variable | **`NEXT_PUBLIC_API_URL`** = `https://<your-service>.onrender.com` (the Render address from step 3, without a trailing `/`) |

3. Click **Deploy**.

`NEXT_PUBLIC_API_URL` is baked in at **build time**. If you change it, redeploy on Vercel. This
value is **not a secret**: it is the public API address that every visitor's browser uses. Never
put real secrets in a `NEXT_PUBLIC_` variable.

### Step 5: connect the two (CORS)

- **Check the address:** Vercel shows the final address, for example
  `https://shopsense-ai.vercel.app`, or `…-thrisha1235.vercel.app` if the name was taken.
- **If it differs from step 3:** in Render → **Environment**, set
  `CORS_ORIGINS=["https://<exact vercel address>"]` and save. Render restarts the service.
- **Format:** the value must be the exact origin, with `https://`, no path and no trailing `/`.
- **Preview deployments:** Vercel *preview* addresses are deliberately **not** allowed, so preview
  builds show "API offline". Use the production address.

### Step 6: verify the live deployment

```powershell
# in the project folder
backend\.venv\Scripts\python.exe tools\verify_deployment.py --api https://<your-service>.onrender.com --web https://<your-app>.vercel.app
```

This expects **28/28**. It covers:
- HTTPS, HSTS, docs hidden;
- every endpoint used by the six pages, including the same 0.2256 prediction and SHAP additivity;
- CORS;
- 7 SSRF refusals;
- one real analysis of example.com;
- all six pages.

Then open the Vercel address and click through the six pages. In Render → **Logs**, find a line
like `website analysis: … client=203.0.113.0/24`:

| Log shows | Meaning |
|---|---|
| **your own public network** | Proxy handling is correct |
| **a `10.x` address** | Render's proxy address isn't trusted. Every user is sharing one rate limit, so `FORWARDED_ALLOW_IPS` needs adjusting |

### What stays private

| Item | Where it lives |
|---|---|
| `PAGESPEED_API_KEY` (optional) | Render environment only, never in git or Vercel |
| `.env` files | git-ignored, never uploaded |
| GitHub sign-in | Your PC's credential store. Render and Vercel use their own GitHub app connections |
| Model files | Public release assets, which is fine (see section 1); integrity enforced by checksums in git |
| Clean rebuild from the tracked files reproduces all reported metrics | Any cloud provider deployment, TLS and custom domains |
| Frontend production build served with **production-only** dependencies (`npm ci --omit=dev` + `next start`) | Behaviour behind a real reverse proxy or CDN |
