"""End-to-end consistency of the LIVE API with the saved artifacts and reported results.

For every held-out test session (2,466):
  * /api/predict model_score == saved pipeline predict_proba (direct, in-process)
  * purchase_probability == saved calibrator(model_score)
  * predicted_purchase == (model_score >= saved threshold)
  * PR-AUC / ROC-AUC / F1 / precision / recall / accuracy recomputed from API answers
    == Phase 3 values in reports/model_results.json
For a stratified sample of 100 sessions:
  * /api/explain prediction block == /api/predict response
  * baseline + sum(contributions) == model_score (<= 1e-6)

Run from backend/ with the backend running:
    .\\.venv\\Scripts\\python.exe ..\\tools\\verify_predictions.py
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path.cwd()
sys.path.insert(0, str(BACKEND))
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)

from ml.data_preprocessing import prepare_datasets  # noqa: E402
from ml.explain import stratified_sample  # noqa: E402
from ml.inference import load_model  # noqa: E402

# 127.0.0.1, not "localhost": on Windows each new connection to localhost tries IPv6 first (~2 s).
API = "http://127.0.0.1:8000"
fails = []


def post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def payload(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if k == "PageValues":
            continue
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (np.integer,)) or (k in ("OperatingSystems", "Browser", "Region", "TrafficType")):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        else:
            out[k] = str(v) if not isinstance(v, int) else v
    return out


data = prepare_datasets()
model = load_model()                               # same files the API serves
import joblib  # noqa: E402
man = json.loads((BACKEND / "models" / "model_manifest.json").read_text())
cal = joblib.load(BACKEND / "models" / man["models"][man["default_model"]]["calibration"]["calibrator_file"])
direct = model.estimator.predict_proba(data.X_test)[:, 1]

scores, labels, probs = [], [], []
for i, (_, row) in enumerate(data.X_test.iterrows()):
    p = post("/api/predict", payload(row))
    scores.append(p["model_score"]); labels.append(p["predicted_purchase"]); probs.append(p["purchase_probability"])
    if abs(p["model_score"] - direct[i]) > 1e-12:
        fails.append(f"row {i}: api score {p['model_score']} != direct {direct[i]}")
    if abs(p["purchase_probability"] - float(cal.predict([p["model_score"]])[0])) > 1e-12:
        fails.append(f"row {i}: probability != calibrator(score)")
    if p["predicted_purchase"] != (p["model_score"] >= model.threshold):
        fails.append(f"row {i}: class inconsistent with threshold")
scores, labels = np.array(scores), np.array(labels).astype(int)
y = data.y_test.to_numpy()

reported = json.loads((BACKEND / "reports" / "model_results.json").read_text())
rep = reported["feature_sets"][man["default_model"]]["models"][man["models"][man["default_model"]]["algorithm"]]["test"]["at_tuned_threshold"]
again = {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores),
         "f1": f1_score(y, labels), "precision": precision_score(y, labels), "recall": recall_score(y, labels),
         "accuracy": accuracy_score(y, labels)}
for k, v in again.items():
    if not math.isclose(v, rep[k], abs_tol=1e-12):
        fails.append(f"metric {k}: api {v} != reported {rep[k]}")

sample = stratified_sample(data.X_test, data.y_test, 100, seed=7)
worst = 0.0
for _, row in sample.iterrows():
    body = payload(row)
    e, p = post("/api/explain", body), post("/api/predict", body)
    if e["prediction"] != p:
        fails.append("explain.prediction != predict response")
    x = e["explanation"]
    err = abs(x["baseline_score"] + sum(c["contribution"] for c in e["contributions"]) - x["model_score"])
    worst = max(worst, err)
    if err > 1e-6 or len(e["contributions"]) != 16:
        fails.append(f"explanation additivity {err:.2e} / {len(e['contributions'])} features")

print(f"/api/predict: {len(scores)} test sessions checked against the saved model + calibrator + threshold")
print("metrics from API answers: " + ", ".join(f"{k} {v:.4f}" for k, v in again.items()))
print(f"/api/explain: {len(sample)} sessions, max additivity error {worst:.2e}")
print(f"{len(fails)} failures")
for f in fails[:20]:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
