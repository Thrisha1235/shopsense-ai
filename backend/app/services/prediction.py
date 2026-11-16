"""Loads the saved model (+ validated calibrator) once and serves predictions."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.data_preprocessing import CATEGORICAL_FEATURES
from ml.inference import (
    InputValidationError,
    LoadedModel,
    ModelNotFoundError,
    load_manifest,
    load_model,
    safe_artifact_path,
    validate_input,
    verify_artifact,
)

log = logging.getLogger("shopsense.prediction")

LIMITATIONS = [
    "Trained on 12,330 completed sessions from one online shop over one year (UCI dataset 468); "
    "other shops, years or traffic mixes need re-validation.",
    "Features are whole-session aggregates. Scoring a session that is still in progress (fewer pages, "
    "shorter durations) is outside the training distribution and usually under-estimates purchase likelihood.",
    "The default model excludes PageValues because its provenance (possible leakage of the outcome) "
    "cannot be verified; its test PR-AUC is {pr_auc:.2f} vs a no-skill 0.155 (the test purchase rate), "
    "so expect many false positives.",
    "January and April do not occur in the training data; other unseen category codes are ignored "
    "by the encoder, which weakens the prediction (a warning is returned).",
    "BounceRates and ExitRates never exceed 0.2 in the training data; larger values are accepted but extrapolated.",
    "The decision threshold was tuned for F1. Applications that weigh false positives and false negatives "
    "differently should choose their own threshold on validation data.",
    "Probabilities are calibrated on the training split (out-of-fold). They describe the training population "
    "and may drift over time; monitor calibration on new data.",
]


class ModelUnavailableError(RuntimeError):
    """Raised when the model (or its manifest) cannot be loaded."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PredictionService:
    model: LoadedModel
    manifest: dict
    calibrator: Any | None
    calibration_status: str              # validated | not_validated | missing | model_mismatch
    calibration_reason: str | None
    known_categories: dict[str, list] = field(default_factory=dict)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, models_dir: Path, variant: str | None = None) -> "PredictionService":
        try:
            manifest = load_manifest(models_dir)
            model = load_model(variant, models_dir=models_dir)
        except (ModelNotFoundError, RuntimeError, OSError) as exc:
            raise ModelUnavailableError(str(exc)) from exc

        cal = model.metadata.get("calibration")
        calibrator, status, reason = None, "missing", "no calibration has been run (python -m ml.calibration)"
        if cal:
            if cal.get("status") != "validated":
                status = "not_validated"
                reason = "calibration failed its acceptance criteria: " + "; ".join(cal.get("failures") or [])
            elif _sha256(Path(models_dir) / model.metadata["model_file"]) != cal.get("model_sha256"):
                status, reason = "model_mismatch", "model file changed since calibration; re-run calibration"
            elif not cal.get("calibrator_file") or not (Path(models_dir) / cal["calibrator_file"]).exists():
                status, reason = "missing", "calibrator file not found"
            else:
                try:
                    cal_path = safe_artifact_path(models_dir, cal["calibrator_file"])
                    verify_artifact(cal_path, cal.get("calibrator_sha256"))   # before unpickling
                    calibrator = joblib.load(cal_path)
                    status, reason = "validated", None
                except Exception as exc:  # unreadable/tampered calibrator -> fall back to model score
                    log.warning("Could not load calibrator: %s", exc)
                    status, reason = "missing", f"calibrator could not be loaded: {exc}"

        pre = model.estimator.estimator_.named_steps["preprocess"]
        known: dict[str, list] = {}
        for name, transformer, cols in pre.transformers_:
            if name == "cat":
                enc = transformer.named_steps["onehot"]
                known = {c: [v.item() if hasattr(v, "item") else v for v in cats]
                         for c, cats in zip(cols, enc.categories_)}
        log.info("Loaded model %s (calibration: %s)", model.variant, status)
        return cls(model, manifest, calibrator, status, reason, known)

    @property
    def probability_available(self) -> bool:
        return self.calibration_status == "validated" and self.calibrator is not None

    @property
    def probability_threshold(self) -> float | None:
        if not self.probability_available:
            return None
        return float(self.calibrator.predict([self.model.threshold])[0])

    # ------------------------------------------------------------- predicting
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        if features.get("PageValues") is not None and "PageValues" not in self.model.required_features:
            warnings.append("PageValues was provided but is not used by this model (leakage-safe variant).")
        for col in CATEGORICAL_FEATURES:
            known = self.known_categories.get(col)
            if known is not None and col in features and features[col] not in known:
                warnings.append(f"{col}={features[col]!r} was not seen in training; it contributes nothing "
                                f"to this prediction.")

        X = validate_input(features, self.model.required_features)   # same checks as training data
        score = float(self.model.estimator.predict_proba(X)[:, 1][0])
        decision = bool(self.model.estimator.predict(X)[0])           # saved threshold on the raw score
        probability = float(self.calibrator.predict(np.array([score]))[0]) if self.probability_available else None
        if not self.probability_available:
            warnings.append(f"Calibration not available ({self.calibration_status}); model_score is not a "
                            f"probability.")
        return {
            "predicted_purchase": decision,
            "purchase_probability": probability,
            "model_score": score,
            "score_type": "calibrated_probability" if probability is not None else "model_score",
            "probability_calibrated": probability is not None,
            "decision_threshold": {"model_score": self.model.threshold,
                                   "purchase_probability": self.probability_threshold},
            "model": {"variant": self.model.variant, "algorithm": self.model.metadata["algorithm"],
                      "trained_at": self.manifest.get("created_at")},
            "warnings": warnings,
        }

    # ------------------------------------------------------------------- info
    def info(self, input_schema: dict) -> dict[str, Any]:
        meta = self.model.metadata
        cal = meta.get("calibration") or {}
        all_features = set(self.model.required_features) | set(meta.get("excluded_features", []))
        return {
            "variant": self.model.variant,
            "is_default": self.model.variant == self.manifest.get("default_model"),
            "algorithm": meta["algorithm"],
            "algorithm_label": meta["algorithm_label"],
            "best_params": meta["best_params"],
            "decision_threshold": {"model_score": self.model.threshold,
                                   "purchase_probability": self.probability_threshold},
            "required_features": self.model.required_features,
            "ignored_features": sorted(all_features - set(self.model.required_features)),
            "known_categories": self.known_categories,
            "test_metrics_at_threshold": meta["test_metrics_at_threshold"],
            "calibration": {
                "status": self.calibration_status,
                "probability_available": self.probability_available,
                "method": cal.get("method"),
                "reason": self.calibration_reason,
                "criteria": cal.get("criteria"),
                "training_cv_metrics": cal.get("training_cv_metrics"),
                "raw_training_cv_metrics": cal.get("raw_training_cv_metrics"),
            },
            "valid_when": meta.get("valid_when", ""),
            # Metric values in the text come from the manifest, so they stay correct after retraining.
            "limitations": [line.format(pr_auc=meta["test_metrics_at_threshold"]["pr_auc"]) for line in LIMITATIONS],
            "trained_at": self.manifest.get("created_at"),
            "sklearn_version": self.manifest.get("sklearn_version"),
            "dataset_sha256": self.manifest.get("dataset_sha256"),
            "input_schema": input_schema,
        }


@lru_cache(maxsize=4)
def load_service(models_dir: str, variant: str | None) -> PredictionService:
    """Cached per (directory, variant). Failures are not cached, so a later retry can succeed."""
    return PredictionService.load(Path(models_dir), variant)


__all__ = ["PredictionService", "ModelUnavailableError", "InputValidationError", "load_service", "LIMITATIONS"]
