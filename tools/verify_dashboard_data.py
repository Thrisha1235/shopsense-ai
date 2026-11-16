"""Independent check: dashboard API numbers vs (a) raw CSV recomputed with plain pandas,
(b) Phase 3/4 report files. Run from backend/ with the venv python (backend must be running):

    .\\.venv\\Scripts\\python.exe ..\\tools\\verify_dashboard_data.py
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "http://localhost:8000"
BACKEND = Path.cwd()
fails, checks = [], 0


def get(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read())


def check(name, api_value, truth, tol=1e-9):
    global checks
    checks += 1
    ok = (api_value == truth) if isinstance(truth, (str, int)) and not isinstance(truth, bool) and tol == 0 \
        else math.isclose(float(api_value), float(truth), rel_tol=0, abs_tol=tol)
    if not ok:
        fails.append(f"{name}: api={api_value!r} truth={truth!r}")


a = get("/api/analytics")
p = get("/api/model-performance")
info = get("/api/model-info")

# ---------------------------------------------------------- (a) raw CSV, plain pandas
raw = pd.read_csv(BACKEND / "data" / "online_shoppers_intention.csv")
y = raw["Revenue"].astype(bool)
check("sessions", a["overview"]["sessions"], len(raw), 0)
check("purchases", a["overview"]["purchases"], int(y.sum()), 0)
check("no_purchases", a["overview"]["no_purchases"], int((~y).sum()), 0)
check("conversion_rate", a["overview"]["conversion_rate"], y.mean())

month = raw["Month"].replace({"June": "Jun"})
for row in a["monthly"]:
    m = month == row["month"]
    check(f"monthly {row['month']} sessions", row["sessions"], int(m.sum()), 0)
    check(f"monthly {row['month']} conversion", row["conversion_rate"], y[m].mean())
check("months absent", ",".join(a["overview"]["months_absent"]),
      ",".join(sorted(set(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) - set(month),
                      key=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].index)), 0)

for row in a["visitor_types"]:
    m = raw["VisitorType"] == row["visitor_type"]
    check(f"visitor {row['visitor_type']} sessions", row["sessions"], int(m.sum()), 0)
    check(f"visitor {row['visitor_type']} conversion", row["conversion_rate"], y[m].mean())

for row in a["weekend"]:
    m = raw["Weekend"] == (row["day_type"] == "Weekend")
    check(f"{row['day_type']} conversion", row["conversion_rate"], y[m].mean())

dur = raw["ProductRelated_Duration"]
check("median product min (purchase)", a["product_duration"]["median_purchase_minutes"], dur[y].median() / 60)
check("median product min (no purchase)", a["product_duration"]["median_no_purchase_minutes"], dur[~y].median() / 60)

for key, col in (("bounce_rate", "BounceRates"), ("exit_rate", "ExitRates")):
    check(f"{key} median purchase", a[key]["median_purchase"], raw.loc[y, col].median())
    check(f"{key} median no purchase", a[key]["median_no_purchase"], raw.loc[~y, col].median())
    edges = [(-1, 0), (0, .005), (.005, .01), (.01, .02), (.02, .05), (.05, .1), (.1, .2)]
    for row, (lo, hi) in zip(a[key]["bands"], edges):
        m = (raw[col] > lo) & (raw[col] <= hi)
        check(f"{key} band {row['band']} sessions", row["sessions"], int(m.sum()), 0)
        check(f"{key} band {row['band']} conversion", row["conversion_rate"], y[m].mean())

top = raw["TrafficType"].value_counts().head(8)
for row in a["traffic_sources"]["rows"]:
    if row["source"] == "Other":
        m = ~raw["TrafficType"].isin(top.index)
    else:
        m = raw["TrafficType"] == int(row["source"].split()[1])
    check(f"traffic {row['source']} sessions", row["sessions"], int(m.sum()), 0)
    check(f"traffic {row['source']} conversion", row["conversion_rate"], y[m].mean())
check("traffic distinct codes", a["traffic_sources"]["distinct_codes"], raw["TrafficType"].nunique(), 0)

# Wilson interval recomputed independently for one row
r = a["monthly"][0]
n, k, z = r["sessions"], r["purchases"], 1.96
ph = k / n
c = (ph + z * z / (2 * n)) / (1 + z * z / n)
h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
check("wilson low (Feb)", r["ci95_low"], c - h)
check("wilson high (Feb)", r["ci95_high"], c + h)

# ------------------------------------------------------- (b) Phase 3/4 report files
res = json.loads((BACKEND / "reports" / "model_results.json").read_text())
cal = json.loads((BACKEND / "reports" / "calibration_results.json").read_text())
man = json.loads((BACKEND / "models" / "model_manifest.json").read_text())
for m in p["models"]:
    src = res["feature_sets"][m["feature_set"]]["models"][m["model"]]
    for k2 in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        check(f"{m['feature_set']}/{m['model']} {k2}", m["test"][k2], src["test"]["at_tuned_threshold"][k2])
    check(f"{m['feature_set']}/{m['model']} cv pr_auc", m["cv_pr_auc_mean"], src["cv"]["pr_auc"]["mean"])
    check(f"{m['feature_set']}/{m['model']} threshold", m["threshold"], src["test"]["threshold"])
    check(f"{m['feature_set']}/{m['model']} selected", str(m["selected"]),
          str(res["feature_sets"][m["feature_set"]]["selected_model"] == m["model"]), 0)
c0 = p["calibration"]["without_page_values"]
check("calibration brier", c0["training_cv"]["brier"], cal["without_page_values"]["validation_on_training_split"]["sigmoid"]["brier"])
check("calibration status", c0["status"], cal["without_page_values"]["status"], 0)
check("model-info pr_auc", info["test_metrics_at_threshold"]["pr_auc"], man["models"]["without_page_values"]["test_metrics_at_threshold"]["pr_auc"])
check("model-info threshold", info["decision_threshold"]["model_score"], man["models"]["without_page_values"]["threshold"])

# Curves: re-score the saved model on the test split and compare AUC from the served points
sys.path.insert(0, str(BACKEND))
from ml.data_preprocessing import prepare_datasets  # noqa: E402
from ml.inference import load_model  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

d = prepare_datasets()
for variant, curve in p["curves"].items():
    score = load_model(variant).estimator.predict_proba(d.X_test)[:, 1]
    true_auc = roc_auc_score(d.y_test, score)
    fpr = np.array([q["fpr"] for q in curve["roc"]]); tpr = np.array([q["tpr"] for q in curve["roc"]])
    served_auc = float(np.trapezoid(tpr, fpr))   # from 150 sampled points
    check(f"{variant} ROC-AUC from served curve (±0.005)", served_auc, true_auc, 0.005)

print(f"{checks} checks, {len(fails)} failures")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
