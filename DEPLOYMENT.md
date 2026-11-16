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

---

## 1. Model artifacts (not in git)

`backend/models/*.joblib` (about 13 MB in total) are excluded from git: they are binary pickles
and can be regenerated. The committed `backend/models/model_manifest.json` records a **SHA-256
for every artifact**. The API verifies each file **before unpickling it** and refuses mismatches.

| Option | How | When to use |
|---|---|---|
| **A. Rebuild in CI** (deterministic) | Run the pipeline in section 5 (~10 min on 12 cores). Verified: a clean rebuild reproduces every reported metric | Simplest; no storage needed |
| **B. Object storage / release asset** *(recommended)* | Upload the 6 `.joblib` files to private S3/GCS/Azure Blob, or attach them to a GitHub Release. Download them into `backend/models/` at build time, then run `python -m ml.integrity` | Fast builds; exact artifacts |
| **C. Git LFS** | `git lfs track "backend/models/*.joblib"` | Only if LFS quota and bandwidth suit you |

Security rule for every option: **only load pickles from storage you control.** The committed
manifest's checksums plus the Docker build gate (`RUN python -m ml.integrity`) mean a swapped
file breaks the build instead of being executed.

After retraining, always run `ml.calibration` and then `ml.explain`, and commit the updated
manifest. Otherwise the API serves score-only predictions (calibration mismatch) and
`/api/explain` returns 503.

---

## 2. Backend hosting

**Measured requirements** (Windows, Python 3.13, production mode):

| Metric | Value |
|---|---|
| Memory after startup | ~200 MB |
| Memory after predictions + explanations | ~260 MB, so **512 MB minimum, 1 GB recommended** |
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
| `EXPLAIN_MAX_CONCURRENCY` | `4` | Per-process cap on simultaneous SHAP explanations |
| `PRELOAD_EXPLAINER` | *(unset)* | Defaults to `true` in production |
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
- **Client IP:** the rate limit is keyed on the client IP. Behind a proxy or load balancer, run
  `uvicorn ... --proxy-headers --forwarded-allow-ips=<proxy IP>`; otherwise every user shares one
  limit.
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
- [ ] Rate limiting / WAF at the proxy or platform for `/api/explain` and `/api/predict`. The app caps concurrent explanations, and limits `/api/analyze-website` per IP and per process.
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
| Production-mode default website limits: 3 simultaneous analyses → `[200, 200, 429]`; 11th request in 10 min → 429 (`Retry-After` 589 s); `/api/predict` unaffected (`tools/verify_rate_limits.py`) | Rate limiting behind a real proxy (`--proxy-headers`) |
| Clean rebuild from the tracked files reproduces all reported metrics | Any cloud provider deployment, TLS and custom domains |
| Frontend production build served with **production-only** dependencies (`npm ci --omit=dev` + `next start`) | Behaviour behind a real reverse proxy or CDN |
