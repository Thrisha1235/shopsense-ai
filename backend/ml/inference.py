"""Load saved models and score sessions (library code; the HTTP API comes later).

    from ml.inference import load_model, predict
    model = load_model()                 # default variant from models/model_manifest.json
    predict(model, [{"Administrative": 0, ...}])

Inputs are validated against the same schema rules used for training and cleaned with
the same stateless ``clean_features()``; the fitted pipeline then does imputation,
scaling and encoding with statistics learned from the training split only.

Security: joblib files are pickles. Only load model files you produced yourself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import pandas as pd
import sklearn

from ml.data_preprocessing import (
    BOOLEAN_FEATURES,
    CODED_CATEGORICAL_FEATURES,
    MONTH_ORDER,
    NUMERIC_FEATURES,
    RATE_FEATURES,
    SPECIAL_DAY_FEATURES,
    TARGET,
    VISITOR_TYPES,
    _as_bool,
    clean_features,
)

log = logging.getLogger("shopsense.inference")

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"
MANIFEST_NAME = "model_manifest.json"


class InputValidationError(ValueError):
    """Raised when prediction input does not match the model's expected schema."""


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass
class LoadedModel:
    variant: str
    estimator: object            # fitted TunedThresholdClassifierCV(Pipeline)
    threshold: float
    required_features: list[str]
    metadata: dict


class ArtifactIntegrityError(RuntimeError):
    """A model artifact does not match the SHA-256 recorded in the manifest."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_artifact_path(models_dir: Path, name: str) -> Path:
    """Artifact names in the manifest must be plain file names inside models_dir."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ArtifactIntegrityError(f"Invalid artifact file name in manifest: {name!r}")
    return Path(models_dir) / name


def verify_artifact(path: Path, expected_sha256: str | None) -> None:
    """Refuse to load an artifact whose checksum differs from the manifest.

    Artifacts without a recorded checksum are still loaded (older manifests); run
    ``python -m ml.integrity --record`` to add checksums.
    """
    if expected_sha256 is None:
        log.warning("No SHA-256 recorded for %s; loading without integrity check", Path(path).name)
        return
    if sha256_file(path) != expected_sha256:
        raise ArtifactIntegrityError(f"{Path(path).name} does not match its recorded SHA-256; refusing to load it.")


def load_manifest(models_dir: Path = MODELS_DIR) -> dict:
    path = Path(models_dir) / MANIFEST_NAME
    if not path.exists():
        raise ModelNotFoundError(f"No model manifest at {path}. Run `python -m ml.train` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(variant: str | None = None, models_dir: Path = MODELS_DIR) -> LoadedModel:
    manifest = load_manifest(models_dir)
    variant = variant or manifest["default_model"]
    if variant not in manifest["models"]:
        raise ModelNotFoundError(f"Unknown model variant {variant!r}; available: {list(manifest['models'])}")
    meta = manifest["models"][variant]
    path = safe_artifact_path(models_dir, meta["model_file"])
    if not path.exists():
        raise ModelNotFoundError(f"Model file missing: {path}")
    if manifest.get("sklearn_version") != sklearn.__version__:
        raise RuntimeError(
            f"Model was saved with scikit-learn {manifest.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed; retrain or install the matching version."
        )
    # joblib files are pickles: verify the bytes BEFORE unpickling.
    verify_artifact(path, meta.get("model_sha256"))
    estimator = joblib.load(path)
    return LoadedModel(
        variant=variant,
        estimator=estimator,
        threshold=float(estimator.best_threshold_),
        required_features=list(meta["required_features"]),
        metadata=meta,
    )


def validate_input(
    records: pd.DataFrame | Mapping | Iterable[Mapping],
    required_features: list[str],
) -> pd.DataFrame:
    """Check and normalise prediction input. Returns a cleaned DataFrame of features.

    * required columns must be present (extra columns, including the target, are ignored)
    * values may be missing (NaN/None): the pipeline imputes them with training statistics
    * present values must satisfy the training schema (types, ranges, known labels)
    """
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, Mapping):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame(list(records))
    if df.empty:
        raise InputValidationError("no records to score")

    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise InputValidationError(f"missing required features: {missing}")
    df = df[required_features]
    errors: list[str] = []

    for c in [c for c in NUMERIC_FEATURES + CODED_CATEGORICAL_FEATURES if c in df]:
        raw = df[c]
        num = pd.to_numeric(raw, errors="coerce")
        bad = num.isna() & raw.notna()
        if bad.any():
            errors.append(f"{c}: non-numeric values {raw[bad].tolist()[:3]}")
        if (num < 0).any():
            errors.append(f"{c}: negative values are not allowed")
        if c in CODED_CATEGORICAL_FEATURES and ((num % 1).fillna(0) != 0).any():
            errors.append(f"{c}: must be an integer code")
        df[c] = num

    for c in [c for c in RATE_FEATURES + SPECIAL_DAY_FEATURES if c in df]:
        if (df[c] > 1).any():
            errors.append(f"{c}: must be between 0 and 1")

    for c in [c for c in BOOLEAN_FEATURES if c in df]:
        coerced = _as_bool(df[c])
        if (coerced.isna() & df[c].notna()).any():
            errors.append(f"{c}: must be true/false")

    if errors:
        raise InputValidationError("; ".join(errors))

    out = clean_features(df)
    if "Month" in out:
        unknown = sorted(set(out["Month"].dropna()) - set(MONTH_ORDER))
        if unknown:
            errors.append(f"Month: unknown values {unknown} (use Jan..Dec)")
    if "VisitorType" in out:
        unknown = sorted(set(out["VisitorType"].dropna()) - set(VISITOR_TYPES))
        if unknown:
            errors.append(f"VisitorType: unknown values {unknown} (use {VISITOR_TYPES})")
    if errors:
        raise InputValidationError("; ".join(errors))
    assert TARGET not in out.columns
    return out


def predict(model: LoadedModel, records) -> pd.DataFrame:
    """Return purchase probability and the thresholded decision for each record."""
    X = validate_input(records, model.required_features)
    proba = model.estimator.predict_proba(X)[:, 1]
    label = model.estimator.predict(X).astype(int)
    return pd.DataFrame({
        "purchase_probability": proba,
        "predicted_purchase": label.astype(bool),
        "threshold": model.threshold,
        "model_variant": model.variant,
    }, index=X.index)


__all__ = ["LoadedModel", "InputValidationError", "ModelNotFoundError", "ArtifactIntegrityError",
           "load_manifest", "load_model", "validate_input", "predict", "sha256_file",
           "safe_artifact_path", "verify_artifact"]
