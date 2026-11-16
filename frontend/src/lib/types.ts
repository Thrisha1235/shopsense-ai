// Types mirror the FastAPI responses (backend/app/schemas.py, backend/ml/dashboard_data.py).

export interface RateRow {
  sessions: number;
  purchases: number;
  conversion_rate: number;
  ci95_low: number;
  ci95_high: number;
  share_of_sessions: number;
}

export type MonthRow = RateRow & { month: string };
export type VisitorRow = RateRow & { visitor_type: string };
export type DayRow = RateRow & { day_type: string };
export type BandRow = RateRow & { band: string };
export type SourceRow = RateRow & { source: string };

export interface Analytics {
  generated_at: string;
  source: { dataset: string; file: string; sha256: string };
  overview: {
    sessions: number;
    purchases: number;
    no_purchases: number;
    conversion_rate: number;
    months_present: string[];
    months_absent: string[];
    returning_visitor_share: number;
    weekend_share: number;
  };
  monthly: MonthRow[];
  visitor_types: VisitorRow[];
  weekend: DayRow[];
  product_duration: {
    median_purchase_minutes: number;
    median_no_purchase_minutes: number;
    quintiles: BandRow[];
  };
  bounce_rate: { median_purchase: number; median_no_purchase: number; bands: BandRow[] };
  exit_rate: { median_purchase: number; median_no_purchase: number; bands: BandRow[] };
  traffic_sources: { note: string; distinct_codes: number; rows: SourceRow[] };
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface TestMetrics {
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  brier: number;
}

export interface ModelRow {
  feature_set: "with_page_values" | "without_page_values";
  model: string;
  label: string;
  selected: boolean;
  best_params: Record<string, unknown>;
  cv_pr_auc_mean: number;
  cv_pr_auc_std: number;
  cv_roc_auc_mean: number;
  cv_roc_auc_std: number;
  threshold: number;
  test: TestMetrics;
  test_pr_auc_ci95: [number, number];
  confusion_matrix: ConfusionMatrix;
}

export interface Curve {
  model: string;
  label: string;
  roc: { fpr: number; tpr: number }[];
  pr: { recall: number; precision: number }[];
  operating_point: { threshold: number; fpr: number; tpr: number; recall: number; precision: number };
}

export interface ReliabilityPoint {
  mean_predicted: number;
  observed_rate: number;
  n: number;
}

export interface CalibrationSummary {
  status: string;
  method: string;
  criteria: Record<string, number | boolean>;
  training_cv: {
    brier: number;
    brier_base_rate: number;
    ece: number;
    calibration_in_the_large: number;
    mean_predicted: number;
    observed_rate: number;
  };
  raw_training_cv: { brier: number; ece: number; mean_predicted: number; observed_rate: number };
  heldout_check: {
    raw_scores: { brier: number; ece: number };
    calibrated?: { brier: number; ece: number };
  };
  reliability_training_cv: { raw: ReliabilityPoint[]; calibrated: ReliabilityPoint[] };
  decision_threshold_probability: number | null;
}

export interface ModelPerformance {
  generated_at: string;
  default_variant: string;
  selection_metric: string;
  split: { n_train: number; n_test: number; train_purchase_rate: number; test_purchase_rate: number };
  baseline: Omit<TestMetrics, "balanced_accuracy">;
  models: ModelRow[];
  curves: Record<string, Curve>;
  calibration: Record<string, CalibrationSummary>;
  page_values: {
    share_nonzero_purchase: number;
    share_nonzero_no_purchase: number;
    conversion_when_nonzero: number;
    conversion_when_zero: number;
    selected_test_pr_auc_with: number;
    selected_test_pr_auc_without: number;
    selected_test_roc_auc_with: number;
    selected_test_roc_auc_without: number;
  };
}

export interface ModelInfo {
  variant: string;
  is_default: boolean;
  algorithm: string;
  algorithm_label: string;
  best_params: Record<string, unknown>;
  decision_threshold: { model_score: number; purchase_probability: number | null };
  required_features: string[];
  ignored_features: string[];
  known_categories: Record<string, (string | number)[]>;
  test_metrics_at_threshold: Record<string, number>;
  calibration: {
    status: "validated" | "not_validated" | "missing" | "model_mismatch";
    probability_available: boolean;
    method: string | null;
    reason: string | null;
  };
  valid_when: string;
  limitations: string[];
  trained_at: string | null;
  sklearn_version: string;
}

export interface SessionInput {
  Administrative: number;
  Administrative_Duration: number;
  Informational: number;
  Informational_Duration: number;
  ProductRelated: number;
  ProductRelated_Duration: number;
  BounceRates: number;
  ExitRates: number;
  SpecialDay: number;
  Month: string;
  OperatingSystems: number;
  Browser: number;
  Region: number;
  TrafficType: number;
  VisitorType: "Returning_Visitor" | "New_Visitor" | "Other";
  Weekend: boolean;
}

export interface PredictionResult {
  predicted_purchase: boolean;
  purchase_probability: number | null;
  model_score: number;
  score_type: "calibrated_probability" | "model_score";
  probability_calibrated: boolean;
  decision_threshold: { model_score: number; purchase_probability: number | null };
  model: { variant: string; algorithm: string; trained_at: string | null };
  warnings: string[];
}

export interface ValidationIssue {
  loc: (string | number)[];
  msg: string;
  type: string;
}

// ---------------------------------------------------------------- explanations
export interface ContributionComponent {
  column: string;
  transformed_value: number;
  contribution: number;
}

export interface FeatureContribution {
  feature: string;
  label: string;
  value: boolean | number | string | null;
  /** SHAP contribution to the RAW model score (not probability units). */
  contribution: number;
  direction: "increases" | "decreases" | "neutral";
  aggregated_from: number;
  components: ContributionComponent[];
}

export interface ExplanationResult {
  prediction: PredictionResult;
  explanation: {
    explains: "raw_model_score";
    output_units: string;
    method: string;
    baseline_score: number;
    baseline_description: string;
    typical_session_score: number;
    model_score: number;
    sum_of_contributions: number;
    additivity_error: number;
    aggregation: string;
  };
  contributions: FeatureContribution[];
  calibrated_view: {
    available: boolean;
    probability_at_baseline: number | null;
    purchase_probability: number | null;
    note: string;
  };
  model: {
    variant: string;
    algorithm: string;
    trained_at: string | null;
    sklearn_version: string | null;
    shap_version: string;
    explainer_created_at: string | null;
  };
  interpretation_notes: string[];
  warnings: string[];
}

export interface GlobalFeatureImportance {
  feature: string;
  label: string;
  mean_abs_contribution: number;
  share_of_total: number;
  mean_contribution: number;
  share_of_sessions_increasing: number;
  share_of_sessions_decreasing: number;
}

export interface GlobalExplanation {
  variant: string;
  method: string;
  explains: "raw_model_score";
  output_units: string;
  baseline_score: number;
  baseline_description: string;
  typical_session_score: number;
  n_sessions: number;
  sample_source: string;
  max_additivity_error: number;
  mean_model_score: number;
  aggregation: string;
  features: GlobalFeatureImportance[];
  interpretation_notes: string[];
  created_at: string | null;
}