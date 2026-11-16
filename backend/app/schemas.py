"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTH_ALIASES = {
    "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr", "june": "Jun",
    "july": "Jul", "august": "Aug", "sept": "Sep", "september": "Sep", "october": "Oct",
    "november": "Nov", "december": "Dec",
}
MonthLiteral = Literal["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VisitorTypeLiteral = Literal["Returning_Visitor", "New_Visitor", "Other"]

MAX_PAGES = 10_000          # sanity limit (training max: 705 product pages)
MAX_SECONDS = 86_400.0      # 24 h per page type (training max: 63,974 s)

EXAMPLE_SESSION = {
    "Administrative": 2, "Administrative_Duration": 45.0,
    "Informational": 0, "Informational_Duration": 0.0,
    "ProductRelated": 30, "ProductRelated_Duration": 1200.0,
    "BounceRates": 0.01, "ExitRates": 0.03, "SpecialDay": 0.0,
    "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
    "VisitorType": "Returning_Visitor", "Weekend": False,
}


def _count(desc: str):
    return Field(ge=0, le=MAX_PAGES, description=desc)


def _seconds(desc: str):
    return Field(ge=0.0, le=MAX_SECONDS, allow_inf_nan=False, description=desc)


def _rate(desc: str):
    return Field(ge=0.0, le=1.0, allow_inf_nan=False, description=desc)


def _code(desc: str):
    return Field(ge=1, le=1_000, description=desc)


class SessionFeatures(BaseModel):
    """One browsing session, aggregated over the whole visit (as in the training data)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_SESSION]})

    Administrative: int = _count("Number of administrative pages visited (account, login, ...).")
    Administrative_Duration: float = _seconds("Seconds spent on administrative pages.")
    Informational: int = _count("Number of informational pages visited (about, contact, ...).")
    Informational_Duration: float = _seconds("Seconds spent on informational pages.")
    ProductRelated: int = _count("Number of product-related pages visited.")
    ProductRelated_Duration: float = _seconds("Seconds spent on product-related pages.")
    BounceRates: float = _rate("Average Google Analytics bounce rate of the pages visited (0-1; training max 0.2).")
    ExitRates: float = _rate("Average Google Analytics exit rate of the pages visited (0-1; training max 0.2).")
    PageValues: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False,
        description="Optional. IGNORED by the default (leakage-safe) model; see /api/model-info.",
    )
    SpecialDay: float = _rate("Closeness of the visit to a special day such as Mother's Day (0-1).")
    Month: MonthLiteral = Field(description="Month of the visit: Jan..Dec (case-insensitive; full names accepted).")
    OperatingSystems: int = _code("Operating-system code as recorded in the dataset (training: 1-8).")
    Browser: int = _code("Browser code (training: 1-13).")
    Region: int = _code("Region code (training: 1-9).")
    TrafficType: int = _code("Traffic-source code (training: 1-20).")
    VisitorType: VisitorTypeLiteral = Field(description="Returning_Visitor, New_Visitor or Other.")
    Weekend: bool = Field(description="Whether the visit happened at the weekend.")

    @field_validator("Month", mode="before")
    @classmethod
    def _normalise_month(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            return _MONTH_ALIASES.get(s.lower(), s.title())
        return v


class DecisionThreshold(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_score: float = Field(description="Saved threshold on the raw model score (tuned for F1 in Phase 3).")
    purchase_probability: float | None = Field(
        description="The same threshold expressed as a calibrated probability (null if uncalibrated).")


class ModelRef(BaseModel):
    variant: str
    algorithm: str
    trained_at: str | None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_purchase: bool = Field(description="model_score >= decision_threshold.model_score")
    purchase_probability: float | None = Field(
        description="Calibrated probability of purchase. null when calibration is not validated.")
    model_score: float = Field(description="Raw model output in [0, 1]. A ranking score, NOT a probability.")
    score_type: Literal["calibrated_probability", "model_score"] = Field(
        description="Which number to display: the calibrated probability if available, else the model score.")
    probability_calibrated: bool
    decision_threshold: DecisionThreshold
    model: ModelRef
    warnings: list[str] = Field(default_factory=list)


class CalibrationInfo(BaseModel):
    status: Literal["validated", "not_validated", "missing", "model_mismatch"]
    probability_available: bool
    method: str | None = None
    reason: str | None = None
    criteria: dict[str, Any] | None = None
    training_cv_metrics: dict[str, Any] | None = None
    raw_training_cv_metrics: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    variant: str
    is_default: bool
    algorithm: str
    algorithm_label: str
    best_params: dict[str, Any]
    decision_threshold: DecisionThreshold
    required_features: list[str]
    ignored_features: list[str]
    known_categories: dict[str, list[Any]]
    test_metrics_at_threshold: dict[str, float]
    calibration: CalibrationInfo
    valid_when: str
    limitations: list[str]
    trained_at: str | None
    sklearn_version: str
    dataset_sha256: str | None
    input_schema: dict[str, Any]


# ------------------------------------------------------------------ explanations
class ContributionComponent(BaseModel):
    column: str = Field(description="Transformed model column, e.g. cat__Month_Nov or num_log__ProductRelated.")
    transformed_value: float = Field(description="Value of that column after preprocessing (log1p/scaled or 0/1).")
    contribution: float = Field(description="SHAP value of this column, in raw-score units.")


class FeatureContribution(BaseModel):
    feature: str
    label: str = Field(description="Plain-language feature name.")
    value: bool | int | float | str | None = Field(description="The session's input value for this feature.")
    contribution: float = Field(description="SHAP contribution to the RAW model score (summed over its columns).")
    direction: Literal["increases", "decreases", "neutral"]
    aggregated_from: int = Field(description="Number of transformed columns summed (one-hot features > 1).")
    components: list[ContributionComponent]


class ExplanationDetails(BaseModel):
    explains: Literal["raw_model_score"]
    output_units: str
    method: str
    baseline_score: float = Field(description="TreeSHAP reference score (0.5 for this class-balanced forest).")
    baseline_description: str
    typical_session_score: float = Field(description="Average raw score of real held-out sessions (context only).")
    model_score: float = Field(description="This session's raw score (identical to /api/predict model_score).")
    sum_of_contributions: float
    additivity_error: float = Field(description="|baseline + sum(contributions) - model_score|; verified <= 1e-6.")
    aggregation: str


class CalibratedView(BaseModel):
    available: bool
    probability_at_baseline: float | None = Field(
        description="Calibrated probability of a session scoring exactly the baseline.")
    purchase_probability: float | None = Field(description="This session's calibrated probability.")
    note: str


class ExplanationModelInfo(BaseModel):
    variant: str
    algorithm: str
    trained_at: str | None
    sklearn_version: str | None
    shap_version: str
    explainer_created_at: str | None


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: PredictionResponse = Field(description="Exactly what POST /api/predict returns for this session.")
    explanation: ExplanationDetails
    contributions: list[FeatureContribution] = Field(description="One entry per model feature, largest |effect| first.")
    calibrated_view: CalibratedView
    model: ExplanationModelInfo
    interpretation_notes: list[str]
    warnings: list[str] = Field(default_factory=list)


class GlobalFeatureImportance(BaseModel):
    feature: str
    label: str
    mean_abs_contribution: float
    share_of_total: float
    mean_contribution: float
    share_of_sessions_increasing: float
    share_of_sessions_decreasing: float


class GlobalExplanationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    variant: str
    method: str
    explains: Literal["raw_model_score"]
    output_units: str
    baseline_score: float
    baseline_description: str
    typical_session_score: float
    n_sessions: int
    sample_source: str
    max_additivity_error: float
    mean_model_score: float
    aggregation: str
    features: list[GlobalFeatureImportance]
    interpretation_notes: list[str]
    created_at: str | None