"""Post-deployment smoke test: run it against the LIVE backend and frontend after deploying.

    python tools/verify_deployment.py --api https://shopsense-api.onrender.com --web https://shopsense-ai.vercel.app

Checks (read-only; standard library only, so any Python 3.10+ works):
  * backend: /health in production mode, HTTPS + HSTS + security headers, /docs hidden
  * every API endpoint the six pages use answers 200 with the expected numbers
    (calibrated probability 0.2256 for the example session, SHAP adds up to the score)
  * CORS: the frontend origin is allowed, another origin is not
  * Website Intelligence refuses private/internal/metadata targets (SSRF) with 400
  * one real website analysis (example.com) succeeds
  * frontend: all six pages return 200 over HTTPS
Uses 8 of the 10 website analyses allowed per IP per 10 minutes (the rest are cheap GETs/POSTs).
A free Render instance may be asleep: the first request can take ~1 minute.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

EXAMPLE = {"Administrative": 2, "Administrative_Duration": 45.0, "Informational": 0, "Informational_Duration": 0.0,
           "ProductRelated": 30, "ProductRelated_Duration": 1200.0, "BounceRates": 0.01, "ExitRates": 0.03,
           "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
           "VisitorType": "Returning_Visitor", "Weekend": False}
# Each must be refused before any connection is made (400). Internal Render-style names included.
SSRF_TARGETS = ["http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/meta-data/",
                "http://[::1]/", "http://2130706433/", "http://shopsense-api:10000/",
                "http://metadata.google.internal/"]
PAGES = ["/", "/analytics", "/predict", "/performance", "/explainability", "/website-intelligence"]

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((bool(ok), name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -> {detail}"))


def call(url: str, body: dict | None = None, method: str | None = None, headers: dict | None = None,
         timeout: float = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, r.headers, raw
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def jbody(raw: bytes):
    try:
        return json.loads(raw or b"null")
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", required=True, help="backend base URL, e.g. https://shopsense-api.onrender.com")
    ap.add_argument("--web", required=True, help="frontend URL, e.g. https://shopsense-ai.vercel.app")
    args = ap.parse_args()
    api, web = args.api.rstrip("/"), args.web.rstrip("/")
    local = "localhost" in api or "127.0.0.1" in api

    # --- backend basics (first call may wake a sleeping free instance) ---
    t = time.monotonic()
    status, headers, raw = call(api + "/health", timeout=180)
    health = jbody(raw) or {}
    check(f"/health 200 in production mode ({time.monotonic() - t:.1f} s)",
          status == 200 and health.get("environment") == "production", f"{status} {health}")
    check("API served over HTTPS", api.startswith("https://") or local, api)
    check("HSTS + security headers", headers.get("Strict-Transport-Security", "").startswith("max-age=")
          and headers.get("X-Content-Type-Options") == "nosniff" and headers.get("X-Frame-Options") == "DENY",
          str(dict(headers)))
    check("/docs hidden in production", call(api + "/docs")[0] == 404)

    # --- endpoints used by the six pages ---
    for path in ["/api/analytics", "/api/model-performance", "/api/model-info", "/api/explain/global"]:
        s, _, raw = call(api + path)
        check(f"GET {path} 200", s == 200 and isinstance(jbody(raw), dict), f"{s} {raw[:150]!r}")
    s, _, raw = call(api + "/api/predict", EXAMPLE)
    pred = jbody(raw) or {}
    check("POST /api/predict: calibrated probability 0.2256 (unchanged model)",
          s == 200 and round(pred.get("purchase_probability", -1), 4) == 0.2256, f"{s} {pred}")
    s, _, raw = call(api + "/api/explain", EXAMPLE)
    body = jbody(raw) or {}
    exp = body.get("explanation", {})
    total = exp.get("baseline_score", 0) + sum(f.get("contribution", 0) for f in body.get("contributions", []))
    check("POST /api/explain: 16 SHAP contributions add up to the model score",
          s == 200 and len(body.get("contributions", [])) == 16 and abs(total - exp.get("model_score", -9)) < 1e-6,
          f"{s} {str(exp)[:200]}")
    check("invalid input rejected (422)", call(api + "/api/predict", {**EXAMPLE, "Month": "Smarch"})[0] == 422)

    # --- CORS ---
    pre = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    _, h_ok, _ = call(api + "/api/predict", method="OPTIONS", headers={"Origin": web, **pre})
    _, h_bad, _ = call(api + "/api/predict", method="OPTIONS", headers={"Origin": "https://evil.example", **pre})
    check(f"CORS allows the frontend origin {web}", h_ok.get("access-control-allow-origin") == web,
          f"got {h_ok.get('access-control-allow-origin')!r}; set CORS_ORIGINS='[\"{web}\"]' on the backend")
    check("CORS refuses other origins", h_bad.get("access-control-allow-origin") is None,
          str(h_bad.get("access-control-allow-origin")))

    # --- Website Intelligence: SSRF refusals, then one real analysis ---
    for target in SSRF_TARGETS:
        s, _, raw = call(api + "/api/analyze-website", {"url": target})
        check(f"SSRF refused: {target}", s == 400, f"{s} {raw[:150]!r}")
    s, _, raw = call(api + "/api/analyze-website", {"url": "example.com"}, timeout=120)
    rep = jbody(raw) or {}
    check("real analysis of example.com succeeds (robots.txt read, pages analysed)",
          s == 200 and not rep.get("fatal_error") and rep.get("summary", {}).get("pages_analyzed", 0) >= 1,
          f"{s} fatal={rep.get('fatal_error')!r} robots={rep.get('robots', {}).get('note')!r}")

    # --- frontend pages ---
    check("frontend served over HTTPS", web.startswith("https://") or "localhost" in web, web)
    for page in PAGES:
        s, _, raw = call(web + page, method="GET")
        check(f"frontend {page} 200", s == 200 and b"ShopSense AI" in raw, f"{s}")

    failed = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} deployment checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
