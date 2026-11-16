import type {
  Analytics,
  ExplanationResult,
  GlobalExplanation,
  ModelInfo,
  ModelPerformance,
  PredictionResult,
  SessionInput,
  ValidationIssue,
} from "./types";
import type { WebsiteAnalysis } from "./websiteTypes";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

/** Error with the HTTP status and, for 422 responses, FastAPI's validation issues. */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number | null,
    public issues: ValidationIssue[] = [],
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store", ...init });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach the ShopSense API at ${API_BASE_URL}. Is the FastAPI server running?`,
      null,
    );
  }
  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = (await res.json())?.detail;
    } catch {
      /* non-JSON error body */
    }
    if (res.status === 422 && Array.isArray(detail)) {
      throw new ApiError("Some inputs are invalid.", 422, detail as ValidationIssue[]);
    }
    throw new ApiError(typeof detail === "string" ? detail : `Request failed (HTTP ${res.status}).`, res.status);
  }
  return (await res.json()) as T;
}

export const getHealth = (signal?: AbortSignal) => request<HealthResponse>("/health", { signal });
export const getAnalytics = (signal?: AbortSignal) => request<Analytics>("/api/analytics", { signal });
export const getModelPerformance = (signal?: AbortSignal) =>
  request<ModelPerformance>("/api/model-performance", { signal });
export const getModelInfo = (signal?: AbortSignal) => request<ModelInfo>("/api/model-info", { signal });

export const explain = (session: SessionInput, signal?: AbortSignal) =>
  request<ExplanationResult>("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(session),
    signal,
  });

export const getGlobalExplanation = (signal?: AbortSignal) =>
  request<GlobalExplanation>("/api/explain/global", { signal });

export const analyzeWebsite = (url: string, includePerformance: boolean, signal?: AbortSignal) =>
  request<WebsiteAnalysis>("/api/analyze-website", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, include_performance: includePerformance }),
    signal,
  });

export const predict = (session: SessionInput, signal?: AbortSignal) =>
  request<PredictionResult>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(session),
    signal,
  });
