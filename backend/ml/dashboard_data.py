"""Export the real numbers the dashboard displays (no invented values).

Writes:
    reports/dashboard/analytics.json          descriptive statistics of the dataset (Phase 2 methods)
    reports/dashboard/model_performance.json  Phase 3 metrics, test-set ROC/PR curve points of the saved
                                              models, and Phase 4 calibration results

The test split is used here only to draw curves of models whose selection, threshold and calibration
were already fixed in Phases 3-4 (reporting, not selection).

Run from the backend directory (after ml.train and ml.calibration):
    python -m ml.dashboard_data
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ml.data_preprocessing import MONTH_ORDER, RAW_CSV_PATH, TARGET, clean, load_raw, prepare_datasets, sha256_of
from ml.eda import RATE_BINS, RATE_LABELS, conversion_table
from ml.inference import MODELS_DIR, load_manifest, load_model

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
OUT_DIR = REPORTS_DIR / "dashboard"
TOP_TRAFFIC_TYPES = 8
MAX_CURVE_POINTS = 150


def _records(t: pd.DataFrame, key: str) -> list[dict]:
    cols = ["sessions", "purchases", "conversion_rate", "ci95_low", "ci95_high", "share_of_sessions"]
    out = t[cols].reset_index().rename(columns={t.index.name or "index": key})
    out[key] = out[key].astype(str)
    return out.to_dict(orient="records")


def build_analytics(df: pd.DataFrame) -> dict:
    n = len(df)
    purchases = int(df[TARGET].sum())

    monthly = conversion_table(df, "Month", MONTH_ORDER)
    monthly.index.name = "month"
    visitors = conversion_table(df, "VisitorType", ["Returning_Visitor", "New_Visitor", "Other"])
    visitors.index.name = "visitor_type"
    weekend = conversion_table(df.assign(day=np.where(df["Weekend"], "Weekend", "Weekday")), "day",
                               ["Weekday", "Weekend"])
    weekend.index.name = "day_type"

    dur = df["ProductRelated_Duration"]
    q = conversion_table(df.assign(bin=pd.qcut(dur, 5, duplicates="drop")), "bin")
    q.index = [f"{max(iv.left, 0) / 60:.0f}–{iv.right / 60:.0f} min" for iv in q.index]
    q.index.name = "band"

    def rate_block(col: str) -> dict:
        t = conversion_table(df.assign(bin=pd.cut(df[col], RATE_BINS, labels=RATE_LABELS)), "bin", RATE_LABELS)
        t.index.name = "band"
        return {"median_purchase": float(df.loc[df[TARGET] == 1, col].median()),
                "median_no_purchase": float(df.loc[df[TARGET] == 0, col].median()),
                "bands": _records(t, "band")}

    # Traffic sources: anonymised integer codes. Top N by sessions; the tail is folded into "Other".
    tt = conversion_table(df, "TrafficType").sort_values("sessions", ascending=False)
    top = tt.head(TOP_TRAFFIC_TYPES).index.tolist()
    folded = df.assign(src=np.where(df["TrafficType"].isin(top), "Type " + df["TrafficType"].astype(str), "Other"))
    order = [f"Type {c}" for c in top] + ["Other"]
    traffic = conversion_table(folded, "src", order)
    traffic.index.name = "source"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"dataset": "UCI Online Shoppers Purchasing Intention Dataset (ID 468)",
                   "file": RAW_CSV_PATH.name, "sha256": sha256_of(RAW_CSV_PATH)},
        "overview": {
            "sessions": n,
            "purchases": purchases,
            "no_purchases": n - purchases,
            "conversion_rate": purchases / n,
            "months_present": [m for m in MONTH_ORDER if m in monthly.index],
            "months_absent": [m for m in MONTH_ORDER if m not in monthly.index],
            "returning_visitor_share": float((df["VisitorType"] == "Returning_Visitor").mean()),
            "weekend_share": float(df["Weekend"].mean()),
        },
        "monthly": _records(monthly, "month"),
        "visitor_types": _records(visitors, "visitor_type"),
        "weekend": _records(weekend, "day_type"),
        "product_duration": {
            "median_purchase_minutes": float(dur[df[TARGET] == 1].median() / 60),
            "median_no_purchase_minutes": float(dur[df[TARGET] == 0].median() / 60),
            "quintiles": _records(q, "band"),
        },
        "bounce_rate": rate_block("BounceRates"),
        "exit_rate": rate_block("ExitRates"),
        "traffic_sources": {
            "note": "Traffic types are anonymised integer codes in the source dataset; their real-world "
                    f"channels are not published. Top {TOP_TRAFFIC_TYPES} by sessions shown, the rest folded into Other.",
            "distinct_codes": int(df["TrafficType"].nunique()),
            "rows": _records(traffic, "source"),
        },
    }


def _sample(arr_len: int) -> np.ndarray:
    if arr_len <= MAX_CURVE_POINTS:
        return np.arange(arr_len)
    return np.unique(np.linspace(0, arr_len - 1, MAX_CURVE_POINTS).round().astype(int))


def build_model_performance(prepared, manifest: dict) -> dict:
    results = json.loads((REPORTS_DIR / "model_results.json").read_text(encoding="utf-8"))
    calib = json.loads((REPORTS_DIR / "calibration_results.json").read_text(encoding="utf-8"))
    eda = json.loads((REPORTS_DIR / "eda_stats.json").read_text(encoding="utf-8"))

    models = []
    for fs, rec in results["feature_sets"].items():
        for name, m in rec["models"].items():
            t = m["test"]["at_tuned_threshold"]
            models.append({
                "feature_set": fs, "model": name, "label": m["label"],
                "selected": name == rec["selected_model"],
                "best_params": m["best_params"],
                "cv_pr_auc_mean": m["cv"]["pr_auc"]["mean"], "cv_pr_auc_std": m["cv"]["pr_auc"]["std"],
                "cv_roc_auc_mean": m["cv"]["roc_auc"]["mean"], "cv_roc_auc_std": m["cv"]["roc_auc"]["std"],
                "threshold": m["test"]["threshold"],
                "test": {k: t[k] for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                                           "roc_auc", "pr_auc", "brier")},
                "test_pr_auc_ci95": m["test"]["bootstrap_95ci_at_tuned_threshold"]["pr_auc"],
                "confusion_matrix": t["confusion_matrix"],
            })

    y = prepared.y_test.to_numpy()
    curves = {}
    for variant in manifest["models"]:
        loaded = load_model(variant)
        score = loaded.estimator.predict_proba(prepared.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y, score)
        prec, rec, _ = precision_recall_curve(y, score)
        i_roc, i_pr = _sample(len(fpr)), _sample(len(prec))
        pred = score >= loaded.threshold
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        curves[variant] = {
            "model": loaded.metadata["algorithm"], "label": loaded.metadata["algorithm_label"],
            "roc": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in i_roc],
            "pr": [{"recall": float(rec[i]), "precision": float(prec[i])} for i in i_pr],
            "operating_point": {"threshold": loaded.threshold, "fpr": fp / (fp + tn), "tpr": tp / (tp + fn),
                                "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else 0.0},
        }

    calibration = {}
    for variant, rec in calib.items():
        tr = rec["validation_on_training_split"]
        calibration[variant] = {
            "status": rec["status"], "method": rec["method"], "criteria": rec["criteria"],
            "training_cv": {k: tr[rec["method"]][k] for k in ("brier", "brier_base_rate", "ece",
                                                              "calibration_in_the_large", "mean_predicted",
                                                              "observed_rate")},
            "raw_training_cv": {k: tr["raw_scores"][k] for k in ("brier", "ece", "mean_predicted", "observed_rate")},
            "heldout_check": rec["heldout_test_check"],
            "reliability_training_cv": rec["reliability_training_cv"],
            "decision_threshold_probability": rec["decision_threshold_probability"],
        }

    pv = eda["page_values"]
    sel = {fs: next(m for m in models if m["feature_set"] == fs and m["selected"]) for fs in results["feature_sets"]}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "default_variant": manifest["default_model"],
        "selection_metric": results["protocol"]["selection_metric"],
        "split": results["protocol"]["split"],
        "baseline": {k: results["baseline_prior"][k] for k in ("accuracy", "precision", "recall", "f1",
                                                               "roc_auc", "pr_auc", "brier")},
        "models": models,
        "curves": curves,
        "calibration": calibration,
        "page_values": {
            "share_nonzero_purchase": pv["share_nonzero_purchase"],
            "share_nonzero_no_purchase": pv["share_nonzero_no_purchase"],
            "conversion_when_nonzero": pv["conversion_when_nonzero"],
            "conversion_when_zero": pv["conversion_when_zero"],
            "selected_test_pr_auc_with": sel["with_page_values"]["test"]["pr_auc"],
            "selected_test_pr_auc_without": sel["without_page_values"]["test"]["pr_auc"],
            "selected_test_roc_auc_with": sel["with_page_values"]["test"]["roc_auc"],
            "selected_test_roc_auc_without": sel["without_page_values"]["test"]["roc_auc"],
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    analytics = build_analytics(df)
    (OUT_DIR / "analytics.json").write_text(json.dumps(analytics, indent=2), encoding="utf-8")

    perf = build_model_performance(prepare_datasets(), load_manifest(MODELS_DIR))
    (OUT_DIR / "model_performance.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")

    o = analytics["overview"]
    print(f"analytics.json: {o['sessions']:,} sessions, {o['purchases']:,} purchases, "
          f"conversion {o['conversion_rate']:.4f}, {len(analytics['traffic_sources']['rows'])} traffic rows")
    print(f"model_performance.json: {len(perf['models'])} models, curves for {list(perf['curves'])}, "
          f"calibration for {list(perf['calibration'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
