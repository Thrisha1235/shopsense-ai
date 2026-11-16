"""Probability calibration for the saved models, using the TRAINING split only.

Why: CV chose ``class_weight="balanced_subsample"`` for the default Random Forest, which
inflates its scores (Phase 3 test Brier 0.150 vs 0.131 for the base rate). Its ranking is
good, but its raw ``predict_proba`` is not a trustworthy purchase probability.

Method (no test-set leakage)
----------------------------
1. Out-of-fold (OOF) raw scores on the training split: ``cross_val_predict`` of an UNFITTED
   clone of the saved pipeline (identical hyper-parameters), stratified 5-fold. Every score
   comes from a model that never saw that session. (Same idea as
   ``CalibratedClassifierCV(ensemble=False)``.)
2. Candidate calibrators (sigmoid / isotonic) are validated with a SECOND stratified CV over
   those OOF (score, label) pairs: fit on 4/5, predict the held-out 1/5.
3. The method with the lowest validated Brier score is checked against acceptance criteria
   fixed in advance (``CRITERIA``). Only if it passes is the calibrator fitted on all OOF pairs
   and marked ``validated`` in the manifest; otherwise the API reports only a model score.
4. AFTER that decision is written, a held-out test-set check is computed and reported for
   transparency. It is never used to choose, tune or accept anything.

Both calibrators are monotone, so ranking metrics are preserved and the saved decision
threshold keeps meaning "same decisions as Phase 3".

Run from the backend directory (after ``python -m ml.train``):
    python -m ml.calibration
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml.calibrator import METHODS, ScoreCalibrator  # importable home -> safe pickles
from ml.data_preprocessing import prepare_datasets
from ml.inference import MANIFEST_NAME, MODELS_DIR, load_manifest, load_model

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
RESULTS_PATH = REPORTS_DIR / "calibration_results.json"

RANDOM_STATE = 42
CV_FOLDS = 5
N_BINS = 10

# Acceptance criteria, fixed BEFORE calibration was run. All must hold on the
# cross-validated calibrated OOF predictions of the training split.
CRITERIA = {
    "max_ece": 0.05,                       # expected calibration error (10 equal-frequency bins)
    "max_abs_calibration_in_the_large": 0.02,  # |mean predicted - observed purchase rate|
    "min_brier_skill_score": 0.0,          # must beat always predicting the base rate
    "must_beat_raw_brier": True,           # must improve on the uncalibrated scores
}


# ---------------------------------------------------------------------- metrics
def expected_calibration_error(y, p, n_bins: int = N_BINS) -> float:
    """Weighted mean |mean(p) - mean(y)| over equal-frequency bins of p."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / g["n"].sum())


def reliability_table(y, p, n_bins: int = N_BINS) -> list[dict]:
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True).to_dict(orient="records")


def calibration_metrics(y, p, baseline_p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    brier = brier_score_loss(y, p)
    brier_base = brier_score_loss(y, baseline_p)
    return {
        "brier": float(brier),
        "brier_base_rate": float(brier_base),
        "brier_skill_score": float(1 - brier / brier_base),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        "ece": expected_calibration_error(y, p),
        "calibration_in_the_large": float(p.mean() - y.mean()),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def passes_criteria(m: dict, raw_brier: float, criteria: dict = CRITERIA) -> tuple[bool, list[str]]:
    failures = []
    if m["ece"] > criteria["max_ece"]:
        failures.append(f"ECE {m['ece']:.4f} > {criteria['max_ece']}")
    if abs(m["calibration_in_the_large"]) > criteria["max_abs_calibration_in_the_large"]:
        failures.append(f"|mean predicted - observed| {abs(m['calibration_in_the_large']):.4f} > "
                        f"{criteria['max_abs_calibration_in_the_large']}")
    if m["brier_skill_score"] <= criteria["min_brier_skill_score"]:
        failures.append(f"Brier skill score {m['brier_skill_score']:.4f} <= {criteria['min_brier_skill_score']}")
    if criteria["must_beat_raw_brier"] and m["brier"] >= raw_brier:
        failures.append(f"calibrated Brier {m['brier']:.4f} >= raw Brier {raw_brier:.4f}")
    return (not failures), failures


# ------------------------------------------------------------- training-only run
def oof_scores(fitted_model, X_train, y_train, *, cv_folds=CV_FOLDS, random_state=RANDOM_STATE, n_jobs=-1):
    """OOF raw scores from an unfitted clone of the saved pipeline (same params)."""
    pipeline = clone(fitted_model.estimator_)      # TunedThresholdClassifierCV -> inner pipeline
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state + 2)
    return cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]


def validate_calibrators(scores, y, *, cv_folds=CV_FOLDS, random_state=RANDOM_STATE, methods=METHODS) -> dict:
    """Second-level CV over OOF (score, label) pairs for each calibration method."""
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state + 3)
    folds = list(cv.split(scores.reshape(-1, 1), y))

    baseline = np.empty_like(scores)
    for tr, va in folds:
        baseline[va] = y[tr].mean()               # base rate learned on the calibration-train fold
    raw = calibration_metrics(y, scores, baseline)

    out = {"raw": raw, "methods": {}}
    for method in methods:
        cal = np.empty_like(scores)
        fold_brier = []
        for tr, va in folds:
            c = ScoreCalibrator(method).fit(scores[tr], y[tr])
            cal[va] = c.predict(scores[va])
            fold_brier.append(float(brier_score_loss(y[va], cal[va])))
        m = calibration_metrics(y, cal, baseline)
        ok, failures = passes_criteria(m, raw["brier"])
        out["methods"][method] = {**m, "fold_brier": fold_brier, "passes": ok, "failures": failures,
                                  "_calibrated": cal}
    return out


def choose_method(validation: dict) -> str:
    return min(validation["methods"], key=lambda k: validation["methods"][k]["brier"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibrate_variant(variant: str, prepared, *, models_dir: Path = MODELS_DIR, n_jobs: int = -1) -> dict:
    """Run the training-only calibration for one saved model and write its calibrator."""
    loaded = load_model(variant, models_dir=models_dir)
    X_train, y_train = prepared.X_train, prepared.y_train          # <- the ONLY data used to decide

    scores = oof_scores(loaded.estimator, X_train, y_train, n_jobs=n_jobs)
    validation = validate_calibrators(scores, y_train)
    method = choose_method(validation)
    chosen = validation["methods"][method]
    status = "validated" if chosen["passes"] else "not_validated"

    record = {
        "status": status,
        "method": method,
        "criteria": CRITERIA,
        "failures": chosen["failures"],
        "model_file": loaded.metadata["model_file"],
        "model_sha256": sha256_file(Path(models_dir) / loaded.metadata["model_file"]),
        "decision_threshold_model_score": loaded.threshold,
        "validation_on_training_split": {
            "description": "OOF scores (5-fold) -> calibrators validated by a second 5-fold CV",
            "n": int(len(y_train)),
            "raw_scores": validation["raw"],
            **{m: {k: v for k, v in r.items() if k != "_calibrated"} for m, r in validation["methods"].items()},
        },
        "reliability_training_cv": {
            "raw": reliability_table(y_train, scores),
            "calibrated": reliability_table(y_train, chosen["_calibrated"]),
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }

    if status == "validated":
        calibrator = ScoreCalibrator(method).fit(scores, y_train)   # final fit: all OOF pairs
        cal_file = f"calibrator_{variant}.joblib"
        joblib.dump(calibrator, Path(models_dir) / cal_file)
        record["calibrator_file"] = cal_file
        record["calibrator_sha256"] = sha256_file(Path(models_dir) / cal_file)   # checked before unpickling
        record["decision_threshold_probability"] = float(calibrator.predict([loaded.threshold])[0])
    else:
        calibrator = None
        record["calibrator_file"] = None
        record["calibrator_sha256"] = None
        record["decision_threshold_probability"] = None

    # ---- decision is final above. Held-out check below is reporting only. ----
    record["heldout_test_check"] = heldout_check(loaded, calibrator, prepared)
    record["_plot"] = {"y": np.asarray(y_train), "raw": scores, "cal": chosen["_calibrated"]}
    return record


def heldout_check(loaded, calibrator, prepared) -> dict:
    """Reported for transparency AFTER the acceptance decision; never used for any choice."""
    y = prepared.y_test.to_numpy()
    raw = loaded.estimator.predict_proba(prepared.X_test)[:, 1]
    base = np.full_like(raw, prepared.y_train.mean())
    out = {"note": "computed after the calibration decision; not used for selection or acceptance",
           "n": int(len(y)), "raw_scores": calibration_metrics(y, raw, base)}
    if calibrator is not None:
        cal = calibrator.predict(raw)
        out["calibrated"] = calibration_metrics(y, cal, base)
        # Does "calibrated prob >= calibrated threshold" reproduce "raw score >= saved threshold"?
        # (sigmoid: always; isotonic: can differ where it creates ties around the threshold)
        cal_thr = calibrator.predict([loaded.threshold])[0]
        out["probability_threshold_reproduces_decisions"] = bool(
            np.array_equal(raw >= loaded.threshold, cal >= cal_thr))
    return out


def _jsonify(o):
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv: list[str] | None = None) -> int:
    manifest = load_manifest()
    prepared = prepare_datasets()
    records = {}
    for variant in manifest["models"]:
        print(f"\n=== Calibrating {variant} ===")
        rec = calibrate_variant(variant, prepared)
        records[variant] = rec
        v = rec["validation_on_training_split"]
        print(f"  raw OOF   : Brier {v['raw_scores']['brier']:.4f} | base-rate Brier {v['raw_scores']['brier_base_rate']:.4f}"
              f" | ECE {v['raw_scores']['ece']:.4f} | mean pred {v['raw_scores']['mean_predicted']:.4f}"
              f" vs observed {v['raw_scores']['observed_rate']:.4f}")
        for m in METHODS:
            r = v[m]
            print(f"  {m:9s} : Brier {r['brier']:.4f} | BSS {r['brier_skill_score']:.4f} | ECE {r['ece']:.4f}"
                  f" | CITL {r['calibration_in_the_large']:+.4f} | passes {r['passes']}")
        print(f"  -> method {rec['method']}, status {rec['status']} {rec['failures'] or ''}")
        if rec["decision_threshold_probability"] is not None:
            print(f"  decision threshold: score {rec['decision_threshold_model_score']:.4f} "
                  f"= calibrated probability {rec['decision_threshold_probability']:.4f}")
        h = rec["heldout_test_check"]
        if "calibrated" in h:
            print(f"  held-out check (reporting only): Brier raw {h['raw_scores']['brier']:.4f} -> "
                  f"calibrated {h['calibrated']['brier']:.4f}, ECE {h['raw_scores']['ece']:.4f} -> "
                  f"{h['calibrated']['ece']:.4f}, prob threshold reproduces decisions "
                  f"{h['probability_threshold_reproduces_decisions']}")

    # write the calibration record into the manifest (read by the API)
    manifest = load_manifest()
    for variant, rec in records.items():
        manifest["models"][variant]["calibration"] = _jsonify({
            k: rec[k] for k in ("status", "method", "calibrator_file", "calibrator_sha256", "model_sha256",
                                "criteria", "failures",
                                "decision_threshold_model_score", "decision_threshold_probability",
                                "created_at")
        } | {"training_cv_metrics": rec["validation_on_training_split"][rec["method"]],
             "raw_training_cv_metrics": rec["validation_on_training_split"]["raw_scores"]})
    (MODELS_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(_jsonify(records), indent=2), encoding="utf-8")

    from ml.model_plots import plot_calibration
    fig = plot_calibration(records)
    print(f"\nWrote {RESULTS_PATH.relative_to(BACKEND_DIR)}, models/{MANIFEST_NAME}, "
          f"{Path(fig).relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
