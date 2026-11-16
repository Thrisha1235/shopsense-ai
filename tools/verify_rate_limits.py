"""Verify the DEFAULT production abuse limits of POST /api/analyze-website on a running server.

Start a server with no WEBSITE_* overrides, e.g.:
    $env:ENVIRONMENT="production"; uvicorn app.main:app --port 8011
then:  python tools/verify_rate_limits.py http://127.0.0.1:8011

Checks (defaults from app/config.py):
  1. concurrency: 3 simultaneous real analyses -> exactly one 429 "busy" (max 2 concurrent)
  2. rate limit : the 11th request from one client within 10 minutes -> 429 + Retry-After
                  (blocked URLs are used after the concurrency test: rejected with 400 without
                   any network traffic, but they still count toward the limit)
  3. separation : /api/predict still answers 200 while the website limit is exhausted
Note: uses ~3 of the 10 slots for real analyses of www.python.org (robots.txt respected).
"""
import json
import sys
import threading
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"
fails = []


def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read() or b"{}")


cfg = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read())
print(f"server: {BASE} environment={cfg['environment']}")

# 1) concurrency
codes = []
threads = [threading.Thread(target=lambda: codes.append(post("/api/analyze-website", {"url": "https://www.python.org"})[0]))
           for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("concurrency (3 simultaneous):", sorted(codes))
if sorted(codes) != [200, 200, 429]:
    fails.append(f"expected [200, 200, 429], got {sorted(codes)}")

# 2) rate limit: keep sending until the 429 (3 slots already used above)
seq = []
for i in range(4, 13):
    status, headers, body = post("/api/analyze-website", {"url": "http://127.0.0.1/"})
    seq.append(status)
    if status == 429:
        retry = headers.get("Retry-After") or headers.get("retry-after")
        print(f"request #{i}: 429 Retry-After={retry}s detail={body.get('detail')!r}")
        if i != 11:
            fails.append(f"429 at request #{i}, expected #11")
        if not retry or int(retry) < 1:
            fails.append("missing Retry-After")
        break
print("requests #4-#11 statuses:", seq)
if 429 not in seq:
    fails.append("rate limit never triggered")

# 3) prediction unaffected
example = {"Administrative": 2, "Administrative_Duration": 45.0, "Informational": 0, "Informational_Duration": 0.0,
           "ProductRelated": 30, "ProductRelated_Duration": 1200.0, "BounceRates": 0.01, "ExitRates": 0.03,
           "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
           "VisitorType": "Returning_Visitor", "Weekend": False}
status, _, body = post("/api/predict", example)
print(f"/api/predict while website limit exhausted: {status}, probability {body.get('purchase_probability')}")
if status != 200 or round(body["purchase_probability"], 4) != 0.2256:
    fails.append("prediction endpoint affected")

print(f"{len(fails)} failures" + "".join(f"\n  FAIL {f}" for f in fails))
sys.exit(1 if fails else 0)
