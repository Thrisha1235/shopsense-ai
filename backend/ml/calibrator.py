"""The calibrator class saved to models/calibrator_*.joblib.

Kept in its own importable module so pickles always reference ``ml.calibrator.ScoreCalibrator``
(never ``__main__``), whichever script created them.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("sigmoid", "isotonic")


class ScoreCalibrator:
    """Monotone map from a model score in [0, 1] to a calibrated probability."""

    _EPS = 1e-6

    def __init__(self, method: str = "sigmoid"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method

    def _logit(self, s) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), self._EPS, 1 - self._EPS)
        return np.log(s / (1 - s)).reshape(-1, 1)

    def fit(self, scores, y) -> "ScoreCalibrator":
        y = np.asarray(y).astype(int)
        if self.method == "sigmoid":       # Platt scaling on the logit of the score
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(scores), y)
        else:
            self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip",
                                             increasing=True).fit(np.asarray(scores, float), y)
        return self

    def predict(self, scores) -> np.ndarray:
        if self.method == "sigmoid":
            return self.model_.predict_proba(self._logit(scores))[:, 1]
        return np.clip(self.model_.predict(np.asarray(scores, float)), 0.0, 1.0)
