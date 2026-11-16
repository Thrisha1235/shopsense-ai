# ShopSense AI: Prediction API

Base URL (local): `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`. The
docs are **development only**; with `ENVIRONMENT=production`, `/docs`, `/redoc` and
`/openapi.json` return 404 unless `ENABLE_DOCS=true`.

**Cross-cutting behaviour (Phase 7 hardening):**
- **CORS:** only the origins in `CORS_ORIGINS` (plus any `localhost` port in development), methods
  GET/POST/OPTIONS, header `Content-Type`, no credentials.
- **Security headers:** every response carries `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY` and `Referrer-Policy: no-referrer`.
- **503 errors:** the `detail` is a generic message such as "Prediction model unavailable. The
  server log has details." The specific reason is written to the server log only, because it can
  contain file paths.
- **429 on `/api/explain`:** returned when `EXPLAIN_MAX_CONCURRENCY` (default 4) explanations are
  already running in this process. Retry after the `Retry-After` header (1 s).
- **Model integrity:** model and calibrator files are verified against the SHA-256 checksums in
  `model_manifest.json` before loading. A mismatched model returns 503. A mismatched calibrator
  falls back to score-only responses.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check (unchanged since Phase 1) |
| POST | `/api/predict` | Score one browsing session |
| GET | `/api/model-info` | Model, threshold, calibration status, input schema, limitations |
| GET | `/api/analytics` | Dataset statistics for the dashboard (read-only; from `python -m ml.dashboard_data`) |
| GET | `/api/model-performance` | Phase 3 metrics, test-set ROC/PR curve points, confusion matrices, calibration results (read-only) |
| POST | `/api/explain` | SHAP explanation of one session's **raw model score**, plus the unchanged prediction |
| GET | `/api/explain/global` | Global feature importance (mean \|SHAP\| over 400 held-out sessions) |
| POST | `/api/analyze-website` | Website Intelligence: robots-respecting, SSRF-safe audit of a public website (separate from prediction) |

The two dashboard endpoints serve `reports/dashboard/*.json`. They return `503` until
`python -m ml.dashboard_data` has been run, and they reload automatically when the files change.

The API serves the **Random Forest model without PageValues** (`without_page_values`), the
leakage-safe default. To serve the PageValues model instead, set
`PREDICTION_MODEL_VARIANT=with_page_values` in `backend/.env`. Only do this if PageValues comes
from historical page values; see [MODEL_REPORT.md](reports/MODEL_REPORT.md) §4.

---

## POST `/api/predict`

### Request body
A JSON object describing **one completed session**. All fields are required except `PageValues`.
Unknown fields are rejected, including `Revenue`.

| Field | Type | Allowed values | Meaning |
|---|---|---|---|
| `Administrative` | integer | 0 – 10,000 | Administrative pages visited |
| `Administrative_Duration` | number | 0 – 86,400 s | Time on administrative pages |
| `Informational` | integer | 0 – 10,000 | Informational pages visited |
| `Informational_Duration` | number | 0 – 86,400 s | Time on informational pages |
| `ProductRelated` | integer | 0 – 10,000 | Product pages visited |
| `ProductRelated_Duration` | number | 0 – 86,400 s | Time on product pages |
| `BounceRates` | number | 0 – 1 (training max 0.2) | Avg. Google Analytics bounce rate of pages visited |
| `ExitRates` | number | 0 – 1 (training max 0.2) | Avg. Google Analytics exit rate of pages visited |
| `SpecialDay` | number | 0 – 1 | Closeness to a special day |
| `Month` | string | `Jan` … `Dec`. Case-insensitive; full names accepted (`june`, `November`) | Month of the visit |
| `OperatingSystems` | integer | ≥ 1 (training: 1–8) | OS code |
| `Browser` | integer | ≥ 1 (training: 1–13) | Browser code |
| `Region` | integer | ≥ 1 (training: 1–9) | Region code |
| `TrafficType` | integer | ≥ 1 (training: 1–20) | Traffic-source code |
| `VisitorType` | string | `Returning_Visitor`, `New_Visitor`, `Other` | Visitor type |
| `Weekend` | boolean | `true` / `false` | Weekend visit |
| `PageValues` | number, optional | ≥ 0 | **Ignored** by the default model (a warning is returned) |

The upper limits (10,000 pages, 24 h per page type) are sanity checks well above the training maxima
(705 pages and 63,974 s).

### Example (PowerShell)

```powershell
$body = @{
  Administrative = 2; Administrative_Duration = 45.0
  Informational = 0; Informational_Duration = 0.0
  ProductRelated = 30; ProductRelated_Duration = 1200.0
  BounceRates = 0.01; ExitRates = 0.03; SpecialDay = 0.0
  Month = "Nov"; OperatingSystems = 2; Browser = 2; Region = 1; TrafficType = 2
  VisitorType = "Returning_Visitor"; Weekend = $false
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/predict `
  -ContentType "application/json" -Body $body
```

### Response `200`
Actual output for the request above:

```json
{
  "predicted_purchase": true,
  "purchase_probability": 0.2256,
  "model_score": 0.4643,
  "score_type": "calibrated_probability",
  "probability_calibrated": true,
  "decision_threshold": { "model_score": 0.4503, "purchase_probability": 0.2146 },
  "model": { "variant": "without_page_values", "algorithm": "random_forest",
             "trained_at": "2026-11-14T06:09:19+00:00" },
  "warnings": []
}
```
(Values rounded here; the API returns full precision.)

| Field | Meaning |
|---|---|
| `predicted_purchase` | `model_score >= decision_threshold.model_score`. This is the saved Phase 3 threshold, tuned for F1 on training data |
| `purchase_probability` | **Calibrated** probability of purchase, or `null` if calibration is not validated |
| `model_score` | Raw Random Forest output. A *ranking score*, **not a probability**. It is inflated by class weighting (average 0.32 on training data, where the actual purchase rate is 0.155) |
| `score_type` | Which number to display: `calibrated_probability` or `model_score` |
| `probability_calibrated` | `true` only when a validated calibrator matching this exact model file is loaded |
| `decision_threshold` | The threshold as a score and as the equivalent calibrated probability (0.2146) |
| `warnings` | E.g. PageValues ignored, a category not seen in training (such as `Month = "Jan"`), or calibration unavailable |

The example shows why calibration matters: a raw score of 0.46 would read as "46%", but the
calibrated probability is 0.23. The raw scores of this model run about twice the observed purchase
rate (see the calibration table below).

**When calibration is unavailable** (not run, failed its criteria, or the model file changed),
`purchase_probability` and `decision_threshold.purchase_probability` are `null`, `score_type` is
`"model_score"`, and a warning says the score is not a probability. `predicted_purchase` still
works, because it uses the raw score and the saved threshold.

### Errors

| Status | When | Body |
|---|---|---|
| `422` | Missing field, wrong type, out-of-range value, unknown label or extra field | FastAPI validation detail, e.g. `{"detail":[{"loc":["body","BounceRates"],"msg":"Input should be less than or equal to 1", ...}]}` |
| `503` | No trained model or manifest in `backend/models/` | `{"detail":"Prediction model unavailable: ..."}` |

---

## GET `/api/model-info`

Returns: `variant`, `is_default`, `algorithm`, `best_params`, `decision_threshold`,
`required_features` (16), `ignored_features` (`["PageValues"]`), `known_categories` (the values each
categorical feature had in training), `test_metrics_at_threshold`, `calibration` (status, method,
acceptance criteria and training-CV metrics), `valid_when`, `limitations`, `trained_at`,
`sklearn_version`, `dataset_sha256`, and `input_schema` (the JSON Schema of the request body).

---

## POST `/api/explain`

**Request:** exactly the same body and validation as `/api/predict` (16 fields; PageValues
optional and ignored; unknown fields rejected with 422).

**Response** (real output for the documented example session, values rounded):

```json
{
  "prediction": { "...": "identical to the /api/predict response",
                  "predicted_purchase": true, "purchase_probability": 0.2256, "model_score": 0.4643 },
  "explanation": {
    "explains": "raw_model_score",
    "output_units": "Raw Random Forest score for 'purchase' (average of tree votes, 0-1, uncalibrated). ...",
    "method": "SHAP TreeExplainer (exact TreeSHAP, tree_path_dependent)",
    "baseline_score": 0.5,
    "baseline_description": "The model's average raw score over its training data with purchases and non-purchases weighted equally ...",
    "typical_session_score": 0.311,
    "model_score": 0.4643,
    "sum_of_contributions": -0.0357,
    "additivity_error": 1.7e-13,
    "aggregation": "One-hot columns are summed back to their original feature; ..."
  },
  "contributions": [
    { "feature": "ExitRates", "label": "Exit rate", "value": 0.03, "contribution": -0.0721,
      "direction": "decreases", "aggregated_from": 1,
      "components": [ { "column": "num__ExitRates", "transformed_value": -0.2723, "contribution": -0.0721 } ] },
    { "feature": "Month", "label": "Month", "value": "Nov", "contribution": 0.0422,
      "direction": "increases", "aggregated_from": 10, "components": [ "... 10 one-hot columns ..." ] }
  ],
  "calibrated_view": { "available": true, "probability_at_baseline": 0.2556, "purchase_probability": 0.2256,
                       "note": "... Individual SHAP contributions are NOT converted to probability units." },
  "model": { "variant": "without_page_values", "algorithm": "random_forest", "trained_at": "...",
             "sklearn_version": "1.9.1", "shap_version": "0.54.1", "explainer_created_at": "..." },
  "interpretation_notes": [ "... plain-language notes, including: not causes ..." ],
  "warnings": []
}
```

| Field | Meaning |
|---|---|
| `explanation.explains` | Always `raw_model_score`. SHAP does **not** explain the calibrated probability |
| `baseline_score` | TreeSHAP reference (0.5 for this class-balanced forest). Not an average session; see `typical_session_score` |
| `contributions[]` | One entry per model input (16), sorted by \|contribution\|. Sum + baseline = `model_score` |
| `aggregated_from` / `components` | One-hot inputs are the sum of their columns; components show each column |
| `calibrated_view` | Calibrated probability at the baseline and for the session (monotone map). `null` if calibration is unavailable |

**Errors:**
- `422`: invalid input (same rules as `/api/predict`).
- `503`: no model, no explainer setup, or the model file changed since `python -m ml.explain`.
- `500`: a consistency guard tripped (explained score ≠ predicted score, or additivity broken). A wrong
  explanation is never returned.

**Latency:** about 0.2 s of SHAP time per session. The first request also loads the explainer (a
few seconds).

## GET `/api/explain/global`

Returns `features[]` with `mean_abs_contribution`, `share_of_total`, `mean_contribution`, and the
shares of sessions where each input raised or lowered the score. It also returns the baseline,
typical session score, sample description and maximum additivity error. It returns `503` if
`reports/explanations/global_shap.json` is missing or was computed for a different model file.
Details and results are in [reports/EXPLAINABILITY.md](reports/EXPLAINABILITY.md).

---

## POST `/api/analyze-website` (Website Intelligence, Phase 8)

Runs an evidence-based audit of a **public** website: technical checks, SEO, usability and
conversion readiness. It is completely separate from purchase prediction: it uses no model and
changes nothing about `/api/predict` or `/api/explain`. Nothing is stored.

### Request body

```json
{ "url": "example.com", "include_performance": false }
```

| Field | Type | Notes |
|---|---|---|
| `url` | string, 3–2048 chars | A domain or an `http(s)` URL. A missing scheme becomes `https://`. Unknown fields → 422 |
| `include_performance` | bool, default `false` | Calls Google PageSpeed Insights **only** if the server has `PAGESPEED_API_KEY`. Otherwise performance is reported as *unavailable* |

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/analyze-website `
  -ContentType 'application/json' -Body '{"url":"example.com"}'
```

### What the crawler does

- **robots.txt first** (RFC 9309), checked for the start page and for every redirect hop:
  - 2xx: its rules apply;
  - 4xx: everything is allowed;
  - 5xx or unreachable: nothing is fetched.
- `Crawl-delay` is honoured (1–5 s between requests).
- **At most 5 HTML pages:** the home page plus same-site links, with product, shop, pricing and
  contact pages first. Cart, checkout, account and login URLs and file downloads are never fetched.
- **Limits:**
  - 6 s connect timeout, 10 s read timeout, 20 s per fetch, about 45 s per analysis;
  - 2 MB per page;
  - gzip decompression is bounded;
  - at most 5 redirects.
- **User-Agent:** `ShopSenseAI-WebsiteIntelligence/1.0 (+public-site audit; respects robots.txt)`.

### SSRF protection (returns `400` with a reason)

- **URL rules:**
  - only `http`/`https` on ports 80 and 443;
  - no user name or password in the URL;
  - no IP literals, including decimal, hex and octal forms;
  - no internal host names such as `localhost`, `*.local` or `*.internal`.
- **DNS:** the host is resolved once, and **every** address must be public. That excludes
  private, loopback, link-local and metadata addresses (169.254.169.254), CGNAT, multicast, and
  reserved ranges. IPv4 addresses inside IPv6 (mapped, 6to4, Teredo, NAT64) are unwrapped and
  checked too.
- **Pinned connection:** the connection goes to the IP address that was checked, which prevents
  DNS rebinding. TLS still verifies the certificate against the host name.
- **Redirects:** each redirect target goes through the same checks.

### Response `200` (abridged)

| Field | Meaning |
|---|---|
| `analyzed_url`, `final_url`, `analyzed_at`, `duration_ms`, `user_agent` | What was fetched and when |
| `fatal_error` | Set when the home page could not be analysed (robots.txt disallows it, or it could not be reached). An HTTP error status is reported as an issue instead; the rest of the report is then mostly empty |
| `robots` | `status` (`found` / `not_found` / `unreachable`), `allows_start`, `crawl_delay`, `sitemaps` |
| `pages[]` | Per page: status, fetch time, size, title, meta description, canonical, H1s, word count, images without alt text, structured-data types, and more |
| `skipped[]` | Links not fetched and why (robots.txt, page limit, time budget) |
| `issues[]` | `id`, `severity` (high / medium / low / info), `category` (technical / seo / usability), `title`, `url`, `evidence`, `recommendation`, `evidence_type` |
| `summary` | Counts by severity and category; checklist score |
| `conversion_checklist[]` | 10 items (CTA, navigation, product info, pricing, cart/checkout, search, contact, policies, HTTPS, mobile), each `observed` / `not_observed` / `unavailable` with evidence |
| `performance` | `status: "unavailable"` with a reason, or PageSpeed lab and field data **labelled as such** |
| `data_availability[]`, `disclaimers[]` | What this analysis cannot measure: real conversions, analytics, Core Web Vitals without PageSpeed data, content rendered only by JavaScript |

**Evidence labels:**
- `observed`: read directly from the HTTP response or HTML.
- `heuristic`: a pattern match that can miss or misfire. Examples: the "slow response" threshold,
  CTA wording, and detecting pages rendered by JavaScript.
- Nothing is estimated: there are no invented Lighthouse scores, Core Web Vitals or conversion
  rates.

### Errors

| Code | When |
|---|---|
| `400` | URL not allowed (see SSRF protection) |
| `422` | Body fails validation |
| `429` | Rate limit: `WEBSITE_RATE_LIMIT` analyses (default **10**) per client IP per `WEBSITE_RATE_WINDOW_S` (default **600 s**). Or busy: more than `WEBSITE_MAX_CONCURRENCY` (default **2**) analyses running at once. Both responses include `Retry-After` |
| `503` | `WEBSITE_ANALYSIS_ENABLED=false` |

The rate limit is per process and keyed on the client IP that Uvicorn sees. Behind a reverse proxy,
start Uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy IP>`; otherwise all users share one
limit. With several replicas, also add a limit at the proxy.

---

## How probability calibration works

Calibration is done by `python -m ml.calibration` using the **training split only**:

1. **Out-of-fold scores.** A clone of the saved pipeline (same hyperparameters) is
   cross-validated (5-fold) on the training split. Every score comes from a model that never saw
   that session.
2. **Validation.** Sigmoid (Platt) and isotonic calibrators are each validated by a second 5-fold CV
   over those out-of-fold scores. The method with the lower Brier score is chosen.
3. **Acceptance criteria**, fixed in code before running: expected calibration error (ECE) ≤ 0.05;
   |mean predicted − observed rate| ≤ 0.02; Brier skill score > 0 against the base rate; and Brier
   better than the raw score.
4. If accepted, the calibrator is fitted on all out-of-fold pairs and saved together with a SHA-256
   checksum of the model file. The API refuses to use it if the model file changes.
5. After the decision, a held-out test check is computed **for reporting only**.

**Results for the served model** (`reports/calibration_results.json`):

| | Raw score | Calibrated (sigmoid) | Base rate |
|---|---|---|---|
| Training-CV Brier score | 0.1451 | **0.1128** | 0.1308 |
| Training-CV ECE | 0.1624 | **0.0058** | — |
| Training-CV mean predicted (observed 0.1547) | 0.3171 | **0.1546** | — |
| Held-out test Brier *(reporting only)* | 0.1497 | **0.1151** | — |
| Held-out test ECE *(reporting only)* | 0.1663 | **0.0179** | — |

Status: **validated**. The sigmoid map is strictly increasing, so ranking (ROC-AUC 0.766, PR-AUC
0.362 on test) and all yes/no decisions are unchanged. See `reports/figures/12_calibration_curve.png`.

> After retraining (`python -m ml.train`), always re-run `python -m ml.calibration`. Until you do,
> the API detects the changed model file and returns only `model_score`.

---

## Limitations: when predictions are valid

- **Completed sessions only.** Features are whole-session aggregates. Scoring a session still in
  progress usually underestimates purchase likelihood.
- **Moderate accuracy.** Test PR-AUC is 0.36 (no-skill 0.155) and ROC-AUC is 0.77. At the threshold,
  precision is 0.31 and recall 0.57, so about 2 in 3 predicted purchasers do not buy.
- **One shop, one year.** UCI dataset 468 only. January and April are absent. Other sites, years or
  traffic mixes need re-validation, and calibration can drift.
- **Unseen codes** (months, browser/OS/region/traffic codes not in training) contribute nothing to
  the prediction. A warning is returned.
- **Rates above 0.2** are accepted but never occurred in training.
- **Threshold.** The threshold is tuned for F1. Pick another one, on validation data, if false
  positives and false negatives have different costs.
- **No PageValues.** The default model ignores PageValues. The PageValues model scores much higher
  (PR-AUC 0.73) but is only valid if PageValues excludes the current session's own outcome.
- **Explanations are not causes.** `/api/explain` describes the model's behaviour (raw score), not
  why a customer buys. See [reports/EXPLAINABILITY.md](reports/EXPLAINABILITY.md).
