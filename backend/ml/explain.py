"""SHAP explanations for the saved Random Forest (no retraining).

What is explained
-----------------
SHAP explains the **raw model score**: the Random Forest's ``predict_proba[:, 1]``, i.e. the
average of the trees' purchase votes. It is NOT the calibrated purchase probability. The sigmoid
calibrator is a monotone map applied afterwards, so the API can report the calibrated probability
at the baseline and at the session's score. Per-feature contributions are NOT in probability units.

Method and why
--------------
``shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")``: exact TreeSHAP.

* Additivity verified on 400 held-out sessions: ``baseline + sum(contributions) == raw score``
  with max error ~2e-12 (tolerance ``ADDITIVITY_TOL`` = 1e-6, checked on every request).
* The interventional mode (with a training background sample) was evaluated first and REJECTED.
  With shap 0.54.1 and this forest, it produced additivity errors up to 4.2e-3 on some sessions.
  Its own internal predictions matched scikit-learn exactly, but the attributions did not add up.
  (By default it also silently subsampled the background to 100 rows.)
* Baseline = 0.5. TreeSHAP's baseline is the model's average output over its training data as
  weighted inside the trees. Because the forest was trained with
  ``class_weight="balanced_subsample"``, purchases and non-purchases carry equal weight, so the
  baseline is 0.5. It is a "balanced reference", NOT the score of an average session. The
  average raw score of real held-out sessions is reported separately (``typical_session_score``).

Feature handling
----------------
SHAP runs on the exact 73 transformed columns the fitted ``ColumnTransformer`` produces, in its
``get_feature_names_out()`` order (verified at load time). Each transformed column maps to one
original input:
  - numeric (log1p/scaled) and boolean columns: 1 column -> 1 feature
  - one-hot columns (e.g. ``cat__Month_Nov``): summed back to their original feature (``Month``).
Summing preserves additivity exactly.

Run from the backend directory (after ml.train and ml.calibration):
    python -m ml.explain        # global importance + manifest entry (no retraining)
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_preprocessing import FEATURES, prepare_datasets
from ml.inference import MANIFEST_NAME, MODELS_DIR, load_manifest, load_model

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
GLOBAL_PATH = REPORTS_DIR / "explanations" / "global_shap.json"

DEFAULT_VARIANT = "without_page_values"
GLOBAL_N = 400
RANDOM_STATE = 42
ADDITIVITY_TOL = 1e-6
METHOD = "SHAP TreeExplainer (exact TreeSHAP, tree_path_dependent)"

OUTPUT_UNITS = ("Raw Random Forest score for 'purchase' (average of tree votes, 0-1, uncalibrated). "
                "Contributions are in the same units: +0.05 means the feature raised the raw score by 0.05.")

BASELINE_DESCRIPTION = (
    "The model's average raw score over its training data with purchases and non-purchases weighted "
    "equally (the forest was trained with balanced class weights), so it is 0.5. It is a balanced "
    "reference point, not the score of an average session."
)

FEATURE_LABELS = {
    "Administrative": "Administrative pages",
    "Administrative_Duration": "Time on administrative pages",
    "Informational": "Informational pages",
    "Informational_Duration": "Time on informational pages",
    "ProductRelated": "Product pages viewed",
    "ProductRelated_Duration": "Time on product pages",
    "BounceRates": "Bounce rate",
    "ExitRates": "Exit rate",
    "PageValues": "Page value",
    "SpecialDay": "Closeness to a special day",
    "Month": "Month",
    "OperatingSystems": "Operating system",
    "Browser": "Browser",
    "Region": "Region",
    "TrafficType": "Traffic source",
    "VisitorType": "Visitor type",
    "Weekend": "Weekend visit",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_map(preprocessor) -> list[tuple[str, str]]:
    """[(transformed column name, original feature)] in the preprocessor's output order."""
    out: list[tuple[str, str]] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if name == "cat":
            enc = transformer.named_steps["onehot"]
            for col, cats in zip(cols, enc.categories_):
                out += [(f"cat__{col}_{c}", col) for c in cats]
        else:
            out += [(f"{name}__{c}", c) for c in cols]
    names = list(preprocessor.get_feature_names_out())
    if [n for n, _ in out] != names:   # guard: mapping must match the real column order exactly
        raise RuntimeError("Transformed feature names do not match the expected column mapping.")
    return out


@dataclass
class ShapExplainer:
    """Exact TreeSHAP for the raw score of a fitted Pipeline(preprocess -> RandomForest)."""

    pipeline: object
    required_features: list[str]

    def __post_init__(self):
        import shap  # imported lazily: heavy, and only needed for explanations

        self.preprocess = self.pipeline.named_steps["preprocess"]
        self.model = self.pipeline.named_steps["classifier"]
        self.columns = feature_map(self.preprocess)
        self._explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent",
                                             model_output="raw")
        self.positive_index = list(self.model.classes_).index(1)
        self.baseline = float(np.atleast_1d(self._explainer.expected_value)[self.positive_index])
        self.shap_version = shap.__version__

    def transformed_values(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocess.transform(X[self.required_features])

    def shap_matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(transformed X, SHAP values for the positive class, raw scores)."""
        Xt = self.transformed_values(X)
        sv = np.asarray(self._explainer.shap_values(Xt, check_additivity=False))
        sv = sv[..., self.positive_index] if sv.ndim == 3 else sv
        score = self.pipeline.predict_proba(X[self.required_features])[:, 1]
        return Xt, sv, score

    def aggregate(self, sv_row: np.ndarray) -> dict[str, float]:
        totals = {f: 0.0 for f in self.required_features}
        for (_, feat), v in zip(self.columns, sv_row):
            totals[feat] += float(v)
        return totals

    def explain_row(self, X: pd.DataFrame) -> dict:
        """Explanation for exactly one session (X has one row of cleaned features)."""
        if len(X) != 1:
            raise ValueError("explain_row expects exactly one session")
        Xt, sv, score = self.shap_matrix(X)
        sv_row, xt_row, score = sv[0], Xt[0], float(score[0])
        totals = self.aggregate(sv_row)
        components: dict[str, list[dict]] = {f: [] for f in self.required_features}
        for (col, feat), v, t in zip(self.columns, sv_row, xt_row):
            components[feat].append({"column": col, "transformed_value": float(t), "contribution": float(v)})
        total = float(sv_row.sum())
        rows = []
        for feat in self.required_features:
            c = totals[feat]
            rows.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "value": _jsonable(X.iloc[0][feat]),
                "contribution": c,
                "direction": "increases" if c > 0 else "decreases" if c < 0 else "neutral",
                "aggregated_from": len(components[feat]),
                "components": components[feat],
            })
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return {
            "baseline_score": self.baseline,
            "model_score": score,
            "sum_of_contributions": total,
            "additivity_error": abs(self.baseline + total - score),
            "contributions": rows,
        }


def _jsonable(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v.item() if hasattr(v, "item") else v


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Seeded sample that keeps the purchase rate of ``y``."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls in sorted(y.unique()):
        idx = y.index[y == cls].to_numpy()
        k = max(1, round(n * len(idx) / len(y)))
        parts.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return X.loc[np.sort(np.concatenate(parts))]


def global_importance(explainer: ShapExplainer, X: pd.DataFrame) -> dict:
    Xt, sv, score = explainer.shap_matrix(X)
    agg = np.array([[row[f] for f in explainer.required_features]
                    for row in (explainer.aggregate(r) for r in sv)])
    add_err = np.abs(explainer.baseline + sv.sum(1) - score)
    mean_abs = np.abs(agg).mean(0)
    feats = []
    for i, f in enumerate(explainer.required_features):
        feats.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "mean_abs_contribution": float(mean_abs[i]),
            "share_of_total": float(mean_abs[i] / mean_abs.sum()),
            "mean_contribution": float(agg[:, i].mean()),
            "share_of_sessions_increasing": float((agg[:, i] > 0).mean()),
            "share_of_sessions_decreasing": float((agg[:, i] < 0).mean()),
        })
    feats.sort(key=lambda r: r["mean_abs_contribution"], reverse=True)
    return {"n_sessions": int(len(X)), "max_additivity_error": float(add_err.max()),
            "mean_model_score": float(score.mean()), "features": feats}


def main() -> int:
    loaded = load_model(DEFAULT_VARIANT)
    data = prepare_datasets()                       # same validated split as Phases 2-5
    explainer = ShapExplainer(loaded.estimator.estimator_, loaded.required_features)
    print(f"method: {METHOD}; baseline raw score {explainer.baseline:.6f}")

    # Global importance on a stratified sample of the held-out test split (reporting only).
    sample = stratified_sample(data.X_test, data.y_test, GLOBAL_N, seed=RANDOM_STATE + 1)
    g = global_importance(explainer, sample)
    print(f"global: {g['n_sessions']} test sessions, max additivity error {g['max_additivity_error']:.2e}, "
          f"mean raw score {g['mean_model_score']:.4f}")
    for r in g["features"][:8]:
        print(f"  {r['feature']:24s} mean|SHAP| {r['mean_abs_contribution']:.4f}  share {r['share_of_total']:.3f}")
    if g["max_additivity_error"] > ADDITIVITY_TOL:
        raise SystemExit(f"Additivity check failed ({g['max_additivity_error']:.2e} > {ADDITIVITY_TOL})")

    GLOBAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "variant": DEFAULT_VARIANT,
        "method": METHOD,
        "explains": "raw_model_score",
        "output_units": OUTPUT_UNITS,
        "baseline_score": explainer.baseline,
        "baseline_description": BASELINE_DESCRIPTION,
        "typical_session_score": g["mean_model_score"],
        "global_sample": {"source": "test split (reporting only)", "n": g["n_sessions"],
                          "random_state": RANDOM_STATE + 1},
        "aggregation": "one-hot columns summed back to their original feature; numeric and boolean "
                       "columns map 1:1. Additivity is preserved.",
        "model_sha256": sha256_file(MODELS_DIR / loaded.metadata["model_file"]),
        "shap_version": explainer.shap_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **g,
    }
    GLOBAL_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")

    manifest = load_manifest()
    manifest["models"][DEFAULT_VARIANT]["explanations"] = {
        k: record[k] for k in ("method", "explains", "output_units", "baseline_score", "baseline_description",
                               "typical_session_score", "model_sha256", "shap_version", "created_at")
    } | {"additivity_tolerance": ADDITIVITY_TOL,
         "global_file": str(GLOBAL_PATH.relative_to(BACKEND_DIR)).replace("\\", "/")}
    (MODELS_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {GLOBAL_PATH.relative_to(BACKEND_DIR)} and the manifest 'explanations' entry")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ShapExplainer", "feature_map", "global_importance", "stratified_sample", "FEATURE_LABELS",
           "OUTPUT_UNITS", "METHOD", "ADDITIVITY_TOL", "BASELINE_DESCRIPTION", "FEATURES"]
