"""SHAP explanations served next to the existing prediction service.

Reuses the already-loaded PredictionService (same fitted pipeline, threshold and calibrator),
so prediction behaviour is unchanged. Explanations are only served when the explainer setup
written by ``python -m ml.explain`` matches the current model file (SHA-256).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.prediction import ModelUnavailableError, PredictionService, load_service
from ml.inference import validate_input

log = logging.getLogger("shopsense.explanation")

INTERPRETATION_NOTES = [
    "Each bar shows how much one piece of session information pushed this session's model score up "
    "or down from the model's reference score (the baseline, 0.5: purchases and non-purchases weighted "
    "equally). The average real session scores below this reference (see typical_session_score), so a "
    "negative total is common and does not by itself mean an unusual session.",
    "The contributions explain the model's raw score, not the calibrated purchase probability. The "
    "probability is a rescaled version of the score: it rises and falls with the score, but "
    "contribution sizes are not percentage points of probability.",
    "These are patterns the model learned from past sessions, not causes. They do not show why a "
    "customer buys, and changing a value for a real customer will not necessarily change whether they buy.",
    "Categorical inputs such as Month or Traffic source are shown as one combined contribution "
    "(their one-hot columns are summed), so contributions still add up to the model score.",
]

CALIBRATION_NOTE = (
    "The calibrator is a monotone (sigmoid) map from raw score to probability, so it is exact to say "
    "the features together move the calibrated probability from the baseline value to this session's "
    "value. Individual SHAP contributions are NOT converted to probability units."
)


class ExplanationUnavailableError(RuntimeError):
    """Raised when explanations cannot be served (missing setup, stale model, SHAP missing)."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExplanationService:
    prediction: PredictionService
    explainer: Any            # ml.explain.ShapExplainer
    record: dict              # manifest["models"][variant]["explanations"]

    @classmethod
    def load(cls, prediction: PredictionService, models_dir: Path) -> "ExplanationService":
        meta = prediction.model.metadata
        rec = meta.get("explanations")
        if not rec:
            raise ExplanationUnavailableError(
                f"No explanation setup for model '{prediction.model.variant}'. Run `python -m ml.explain`.")
        if _sha256(Path(models_dir) / meta["model_file"]) != rec.get("model_sha256"):
            raise ExplanationUnavailableError(
                "The model file changed after the explainer was built. Re-run `python -m ml.explain`.")
        try:
            from ml.explain import ShapExplainer
            explainer = ShapExplainer(prediction.model.estimator.estimator_, prediction.model.required_features)
        except ImportError as exc:
            raise ExplanationUnavailableError(f"SHAP is not installed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface as 503, never as a wrong explanation
            log.exception("Could not build SHAP explainer")
            raise ExplanationUnavailableError(f"Could not build the SHAP explainer: {exc}") from exc
        log.info("SHAP explainer ready for %s (baseline %.4f)", prediction.model.variant, explainer.baseline)
        return cls(prediction, explainer, rec)

    def explain(self, features: dict[str, Any]) -> dict[str, Any]:
        from ml.explain import ADDITIVITY_TOL

        pred = self.prediction.predict(features)                   # unchanged /api/predict behaviour
        X = validate_input(features, self.prediction.model.required_features)
        e = self.explainer.explain_row(X)

        # Consistency guards: never return an explanation of a different number.
        if abs(e["model_score"] - pred["model_score"]) > 1e-9:
            raise RuntimeError("Explained score differs from the prediction score.")
        if e["additivity_error"] > ADDITIVITY_TOL:
            raise RuntimeError(f"SHAP additivity check failed ({e['additivity_error']:.2e}).")

        cal_available = self.prediction.probability_available
        cal = self.prediction.calibrator
        calibrated_view = {
            "available": cal_available,
            "probability_at_baseline": float(cal.predict([e["baseline_score"]])[0]) if cal_available else None,
            "purchase_probability": pred["purchase_probability"],
            "note": CALIBRATION_NOTE if cal_available else
                    "Calibration is not available, so no probability is shown; only the raw score is explained.",
        }
        return {
            "prediction": pred,
            "explanation": {
                "explains": "raw_model_score",
                "output_units": self.record["output_units"],
                "method": self.record["method"],
                "baseline_score": e["baseline_score"],
                "baseline_description": self.record["baseline_description"],
                "typical_session_score": self.record["typical_session_score"],
                "model_score": e["model_score"],
                "sum_of_contributions": e["sum_of_contributions"],
                "additivity_error": e["additivity_error"],
                "aggregation": "One-hot columns are summed back to their original feature; numeric and "
                               "boolean columns map one-to-one. Sums preserve additivity.",
            },
            "contributions": e["contributions"],
            "calibrated_view": calibrated_view,
            "model": {
                "variant": self.prediction.model.variant,
                "algorithm": self.prediction.model.metadata["algorithm"],
                "trained_at": self.prediction.manifest.get("created_at"),
                "sklearn_version": self.prediction.manifest.get("sklearn_version"),
                "shap_version": self.explainer.shap_version,
                "explainer_created_at": self.record.get("created_at"),
            },
            "interpretation_notes": INTERPRETATION_NOTES,
            "warnings": pred["warnings"],
        }


@lru_cache(maxsize=4)
def load_explanation_service(models_dir: str, variant: str | None) -> ExplanationService:
    """Cached per (directory, variant); failures are not cached."""
    try:
        prediction = load_service(models_dir, variant)
    except ModelUnavailableError as exc:
        raise ExplanationUnavailableError(str(exc)) from exc
    return ExplanationService.load(prediction, Path(models_dir))
