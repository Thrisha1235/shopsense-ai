"""Train, tune, evaluate and save the ShopSense AI purchase-prediction models.

Protocol (leakage-safe)
-----------------------
1. ``prepare_datasets()`` (Phase 2) loads + validates the CSV and makes the SAME
   stratified 80/20 split (random_state=42) used throughout the project.
2. Every candidate is a ``Pipeline(preprocess -> classifier)``, so imputation,
   log/scaling and one-hot encoding are re-fitted inside each CV training fold.
3. Limited hyper-parameter grids are searched with stratified 5-fold CV on the
   TRAINING split only. The selection metric is PR-AUC (average precision).
4. The decision threshold is tuned (max F1) with a second internal CV on the
   training split (``TunedThresholdClassifierCV``), then the model is refit on
   the full training split.
5. The held-out test split is used exactly once per model, AFTER selection,
   to report final metrics. It never influences model, parameter or threshold choice.

Two feature sets are trained because PageValues may leak the outcome
(see reports/MODEL_REPORT.md):
    with_page_values      all 17 features
    without_page_values   16 features, PageValues excluded

Run from the backend directory:
    python -m ml.train            # full run (writes models/ and reports/)
    python -m ml.train --fast     # tiny grids, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

from ml.data_preprocessing import (
    FEATURES,
    RAW_CSV_PATH,
    PreparedData,
    build_preprocessor,
    prepare_datasets,
    sha256_of,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"
RESULTS_PATH = REPORTS_DIR / "model_results.json"
MANIFEST_NAME = "model_manifest.json"

RANDOM_STATE = 42
CV_FOLDS = 5
SELECTION_METRIC = "pr_auc"
THRESHOLD_METRIC = "f1"

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "with_page_values": (),
    "without_page_values": ("PageValues",),
}
# Chosen BEFORE seeing results: the leakage-safe variant is the default for
# real-time prediction (rationale in reports/MODEL_REPORT.md).
DEFAULT_FEATURE_SET = "without_page_values"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    # zero_division=0: a fold/threshold that predicts no purchases scores 0, not a warning.
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}
THRESHOLD_SCORER = make_scorer(f1_score, zero_division=0)


# ------------------------------------------------------------------ model specs
def model_specs(fast: bool = False, random_state: int = RANDOM_STATE) -> dict[str, tuple[Any, dict]]:
    """Estimators + LIMITED grids. ``class_weight`` is tuned so CV decides whether
    re-weighting the minority class helps. sklearn's GradientBoostingClassifier has no
    class_weight; it relies on the tuned decision threshold instead."""
    if fast:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000),
                {"classifier__C": [1.0], "classifier__class_weight": [None, "balanced"]},
            ),
            "random_forest": (
                RandomForestClassifier(n_estimators=25, random_state=random_state, n_jobs=1),
                {"classifier__max_depth": [4], "classifier__class_weight": [None]},
            ),
            "gradient_boosting": (
                GradientBoostingClassifier(n_estimators=25, random_state=random_state),
                {"classifier__max_depth": [2]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1),
            {
                "classifier__max_depth": [8, 16, None],
                "classifier__min_samples_leaf": [1, 5],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state, subsample=0.8),
            {
                "classifier__n_estimators": [100, 300],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
    }


def make_pipeline(classifier, exclude: tuple[str, ...] = ()) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(exclude=exclude)),
        ("classifier", clone(classifier)),
    ])


def cv_splitter(n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def required_features(exclude: tuple[str, ...]) -> list[str]:
    return [f for f in FEATURES if f not in exclude]


# ------------------------------------------------------------------- searching
@dataclass
class SearchResult:
    name: str
    best_params: dict
    cv_metrics: dict[str, dict[str, float]]   # metric -> {mean, std} at best params
    grid: list[dict]                          # every candidate with its CV PR-AUC
    search_seconds: float


def _jsonable_params(params: dict) -> dict:
    return {k.replace("classifier__", ""): v for k, v in params.items()}


def tune_model(
    name: str,
    classifier,
    grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    exclude: tuple[str, ...] = (),
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> SearchResult:
    """Grid search with stratified CV on the TRAINING data only."""
    t0 = time.perf_counter()
    search = GridSearchCV(
        make_pipeline(classifier, exclude),
        grid,
        scoring=CV_SCORING,
        refit=False,                 # we refit later inside the threshold tuner
        cv=cv_splitter(cv_folds, random_state),
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    res = search.cv_results_
    # Highest mean CV PR-AUC; ties broken by the first (simplest) candidate.
    best = int(np.argmax(res[f"mean_test_{SELECTION_METRIC}"]))
    cv_metrics = {
        m: {"mean": float(res[f"mean_test_{m}"][best]), "std": float(res[f"std_test_{m}"][best])}
        for m in CV_SCORING
    }
    grid_rows = [
        {
            "params": _jsonable_params(p),
            "cv_pr_auc_mean": float(res["mean_test_pr_auc"][i]),
            "cv_pr_auc_std": float(res["std_test_pr_auc"][i]),
            "cv_roc_auc_mean": float(res["mean_test_roc_auc"][i]),
        }
        for i, p in enumerate(res["params"])
    ]
    return SearchResult(
        name=name,
        best_params=res["params"][best],
        cv_metrics=cv_metrics,
        grid=grid_rows,
        search_seconds=round(time.perf_counter() - t0, 2),
    )


def select_best(searches: dict[str, SearchResult], metric: str = SELECTION_METRIC) -> str:
    """Pick the model with the highest mean CV score. Uses training-CV results only."""
    return max(searches, key=lambda n: searches[n].cv_metrics[metric]["mean"])


def fit_with_tuned_threshold(
    classifier,
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    exclude: tuple[str, ...] = (),
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> TunedThresholdClassifierCV:
    """Tune the decision threshold for max F1 via CV on TRAINING data, then refit."""
    pipeline = make_pipeline(classifier, exclude).set_params(**best_params)
    tuned = TunedThresholdClassifierCV(
        pipeline,
        scoring=THRESHOLD_SCORER,
        cv=cv_splitter(cv_folds, random_state + 1),
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return tuned.fit(X_train, y_train)


# ------------------------------------------------------------------- evaluation
def classification_metrics(y_true, proba, y_pred) -> dict:
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, proba, y_pred, n_boot: int = 1000, seed: int = RANDOM_STATE) -> dict:
    """95% percentile bootstrap intervals for the headline test metrics."""
    rng = np.random.default_rng(seed)
    y_true, proba, y_pred = map(np.asarray, (y_true, proba, y_pred))
    samples: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        samples["pr_auc"].append(average_precision_score(y_true[idx], proba[idx]))
        samples["roc_auc"].append(roc_auc_score(y_true[idx], proba[idx]))
        samples["f1"].append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


@dataclass
class TrainedModel:
    feature_set: str
    name: str
    estimator: TunedThresholdClassifierCV
    search: SearchResult
    test_proba: np.ndarray
    record: dict = field(default_factory=dict)


def evaluate_on_test(model: TunedThresholdClassifierCV, X_test, y_test) -> tuple[dict, np.ndarray]:
    proba = model.predict_proba(X_test)[:, 1]
    tuned_pred = model.predict(X_test)
    default_pred = (proba >= 0.5).astype(int)
    out = {
        "threshold": float(model.best_threshold_),
        "at_tuned_threshold": classification_metrics(y_test, proba, tuned_pred),
        "at_default_threshold_0.5": classification_metrics(y_test, proba, default_pred),
        "bootstrap_95ci_at_tuned_threshold": bootstrap_ci(y_test, proba, tuned_pred),
    }
    return out, proba


def baseline_metrics(prepared: PreparedData) -> dict:
    """Majority-class / prior baseline: the floor every model must beat."""
    dummy = DummyClassifier(strategy="prior").fit(prepared.X_train, prepared.y_train)
    proba = dummy.predict_proba(prepared.X_test)[:, 1]
    return classification_metrics(prepared.y_test, proba, dummy.predict(prepared.X_test))


# ---------------------------------------------------------------------- driver
def run_training(
    prepared: PreparedData,
    *,
    fast: bool = False,
    feature_sets: dict[str, tuple[str, ...]] = FEATURE_SETS,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[dict, dict[str, dict[str, TrainedModel]]]:
    """Search, select, threshold-tune and test every model. Returns (results, models)."""
    X_train, X_test, y_train, y_test = prepared.X_train, prepared.X_test, prepared.y_train, prepared.y_test
    specs = model_specs(fast=fast, random_state=random_state)
    trained: dict[str, dict[str, TrainedModel]] = {}
    results: dict = {"feature_sets": {}}

    for fs_name, exclude in feature_sets.items():
        if verbose:
            print(f"\n=== Feature set: {fs_name} (excluded: {list(exclude) or 'none'}) ===")
        searches: dict[str, SearchResult] = {}
        for name, (clf, grid) in specs.items():
            searches[name] = tune_model(name, clf, grid, X_train, y_train, exclude=exclude,
                                        cv_folds=cv_folds, random_state=random_state, n_jobs=n_jobs)
            if verbose:
                m = searches[name].cv_metrics
                print(f"  {MODEL_LABELS[name]:20s} CV PR-AUC {m['pr_auc']['mean']:.4f} ± {m['pr_auc']['std']:.4f} "
                      f"| ROC-AUC {m['roc_auc']['mean']:.4f} | {searches[name].search_seconds:.1f}s "
                      f"| best {_jsonable_params(searches[name].best_params)}")

        # Model selection happens HERE, before any test-set prediction is made.
        selected = select_best(searches)
        if verbose:
            print(f"  -> selected by CV {SELECTION_METRIC}: {MODEL_LABELS[selected]}")

        fs_models: dict[str, TrainedModel] = {}
        fs_record: dict = {"excluded_features": list(exclude),
                           "required_features": required_features(exclude),
                           "selected_model": selected, "models": {}}
        for name, (clf, _grid) in specs.items():
            est = fit_with_tuned_threshold(clf, searches[name].best_params, X_train, y_train,
                                           exclude=exclude, cv_folds=cv_folds,
                                           random_state=random_state, n_jobs=n_jobs)
            test_eval, proba = evaluate_on_test(est, X_test, y_test)
            record = {
                "label": MODEL_LABELS[name],
                "best_params": _jsonable_params(searches[name].best_params),
                "cv": searches[name].cv_metrics,
                "cv_f1_at_tuned_threshold": float(est.best_score_),
                "grid_search": searches[name].grid,
                "search_seconds": searches[name].search_seconds,
                "test": test_eval,
            }
            fs_record["models"][name] = record
            fs_models[name] = TrainedModel(fs_name, name, est, searches[name], proba, record)
            if verbose:
                t = test_eval["at_tuned_threshold"]
                print(f"  test {MODEL_LABELS[name]:20s} thr {test_eval['threshold']:.3f} | "
                      f"PR-AUC {t['pr_auc']:.4f} ROC-AUC {t['roc_auc']:.4f} F1 {t['f1']:.4f} "
                      f"P {t['precision']:.4f} R {t['recall']:.4f} Acc {t['accuracy']:.4f}")
        results["feature_sets"][fs_name] = fs_record
        trained[fs_name] = fs_models

    results["baseline_prior"] = baseline_metrics(prepared)
    results["protocol"] = {
        "split": {"test_size": 0.2, "random_state": RANDOM_STATE, "stratified": True,
                  "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                  "train_purchase_rate": float(y_train.mean()), "test_purchase_rate": float(y_test.mean())},
        "cv": {"folds": cv_folds, "stratified": True, "shuffle": True, "random_state": random_state},
        "selection_metric": "PR-AUC (average precision), mean over CV folds on the training split",
        "threshold_tuning": f"TunedThresholdClassifierCV maximising {THRESHOLD_METRIC} with "
                            f"{cv_folds}-fold CV on the training split",
        "imbalance_handling": [
            "stratified train/test split and stratified CV folds",
            "class_weight (None vs balanced) tuned by CV for Logistic Regression and Random Forest",
            "decision threshold tuned on training data instead of the default 0.5",
            "selection and reporting with PR-AUC, F1, precision and recall rather than accuracy",
        ],
        "test_set_usage": "evaluated once per model after selection; never used for selection or tuning",
        "fast_mode": fast,
    }
    return results, trained


# ------------------------------------------------------------------- artefacts
VALIDITY = {
    "with_page_values": (
        "Valid only when PageValues is computed from HISTORICAL page values that exclude the "
        "current session's own transaction (e.g. a nightly lookup of each page's value). If "
        "PageValues is calculated after the session ends it encodes the outcome, and scores "
        "will be over-optimistic."
    ),
    "without_page_values": (
        "Does not use PageValues. Valid for sessions whose page counts, durations, bounce and "
        "exit rates are known. Trained on COMPLETED sessions, so scoring a session midway uses "
        "partial counts and durations and will usually under-estimate purchase likelihood."
    ),
}


def save_artifacts(
    trained: dict[str, dict[str, TrainedModel]],
    results: dict,
    *,
    models_dir: Path = MODELS_DIR,
    default_feature_set: str = DEFAULT_FEATURE_SET,
    dataset_sha256: str | None = None,
) -> dict:
    """Save the SELECTED model of each feature set (+ its fitted preprocessor) and a manifest."""
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "default_model": default_feature_set,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": dataset_sha256,
        "random_state": RANDOM_STATE,
        "load_with": "ml.inference.load_model()  (joblib - load only files you trust)",
        "models": {},
    }
    for fs_name, models in trained.items():
        selected = results["feature_sets"][fs_name]["selected_model"]
        tm = models[selected]
        model_file = f"purchase_model_{fs_name}.joblib"
        pre_file = f"preprocessor_{fs_name}.joblib"
        joblib.dump(tm.estimator, models_dir / model_file, compress=3)
        joblib.dump(tm.estimator.estimator_.named_steps["preprocess"], models_dir / pre_file, compress=3)
        test = tm.record["test"]["at_tuned_threshold"]
        manifest["models"][fs_name] = {
            "model_file": model_file,
            "preprocessor_file": pre_file,
            # verified by ml.inference.load_model before unpickling
            "model_sha256": sha256_of(models_dir / model_file),
            "preprocessor_sha256": sha256_of(models_dir / pre_file),
            "algorithm": selected,
            "algorithm_label": MODEL_LABELS[selected],
            "best_params": tm.record["best_params"],
            "threshold": tm.record["test"]["threshold"],
            "required_features": results["feature_sets"][fs_name]["required_features"],
            "excluded_features": results["feature_sets"][fs_name]["excluded_features"],
            "cv_pr_auc_mean": tm.record["cv"]["pr_auc"]["mean"],
            "test_metrics_at_threshold": {k: test[k] for k in
                                          ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy")},
            "valid_when": VALIDITY.get(fs_name, ""),
        }
    (models_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _jsonify(o):
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ShopSense AI purchase models")
    parser.add_argument("--fast", action="store_true", help="tiny grids (smoke test)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    prepared = prepare_datasets()  # same validated, stratified split as Phase 2
    print(f"Train {len(prepared.y_train):,} rows | Test {len(prepared.y_test):,} rows | "
          f"purchase rate train {prepared.y_train.mean():.4f} / test {prepared.y_test.mean():.4f}")

    results, trained = run_training(prepared, fast=args.fast, n_jobs=args.n_jobs)
    digest = sha256_of(RAW_CSV_PATH)
    results["dataset_sha256"] = digest
    results["default_model"] = DEFAULT_FEATURE_SET
    results["versions"] = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__}
    results["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    manifest = save_artifacts(trained, results, dataset_sha256=digest)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(_jsonify(results), indent=2), encoding="utf-8")

    figures = []
    if not args.no_plots:
        from ml.model_plots import make_all_figures
        figures = make_all_figures(results, trained, prepared)

    b = results["baseline_prior"]
    print(f"\nBaseline (always predict the prior): PR-AUC {b['pr_auc']:.4f}, ROC-AUC {b['roc_auc']:.4f}, "
          f"F1 {b['f1']:.4f}, accuracy {b['accuracy']:.4f}")
    print("\nSaved models:")
    for fs, m in manifest["models"].items():
        flag = " (default)" if fs == manifest["default_model"] else ""
        print(f"  {fs}{flag}: {m['algorithm_label']} -> models/{m['model_file']}, models/{m['preprocessor_file']}")
    print(f"Results: {RESULTS_PATH.relative_to(BACKEND_DIR)}")
    for f in figures:
        print(f"Figure:  {Path(f).relative_to(BACKEND_DIR)}")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
