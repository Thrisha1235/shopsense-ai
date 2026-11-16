// Mirrors backend/app/website/schemas.py (POST /api/analyze-website).

export type Severity = "high" | "medium" | "low" | "info";
export type EvidenceType = "observed" | "heuristic";

export interface WebsiteIssue {
  id: string;
  severity: Severity;
  category: "technical" | "seo" | "usability";
  title: string;
  url: string;
  evidence: string;
  recommendation: string;
  evidence_type: EvidenceType;
}

export interface ChecklistItem {
  id: string;
  label: string;
  status: "observed" | "not_observed" | "unavailable";
  evidence: string[];
  note: string;
  evidence_type: EvidenceType;
}

export interface WebsitePage {
  url: string;
  final_url: string;
  status: number;
  elapsed_ms: number;
  content_type: string;
  bytes: number;
  truncated: boolean;
  https: boolean;
  hsts: boolean;
  redirects: number;
  is_html: boolean;
  title?: string | null;
  title_length?: number;
  meta_description?: string | null;
  meta_description_length?: number;
  canonical?: string | null;
  robots_meta?: string | null;
  lang?: string | null;
  viewport?: string | null;
  h1?: string[];
  headings?: Record<string, number>;
  images_total?: number;
  images_missing_alt?: number;
  images_empty_alt?: number;
  structured_data_types?: string[];
  structured_data_errors?: number;
  open_graph?: Record<string, string>;
  links_internal?: number;
  links_external?: number;
  word_count?: number;
  likely_js_rendered?: boolean;
}

export interface PerformanceReport {
  status: "available" | "unavailable";
  source: string;
  reason?: string;
  strategy?: string;
  lighthouse_version?: string | null;
  fetched_at?: string | null;
  performance_score?: number | null;
  lab?: { metric: string; value: string | null; numeric: number | null }[];
  field?: { metric: string; percentile: number | null; category: string | null }[];
  field_overall?: string | null;
  field_available?: boolean;
  note?: string;
}

export interface WebsiteAnalysis {
  input_url: string;
  analyzed_url: string;
  final_url: string | null;
  analyzed_at: string;
  duration_ms: number;
  user_agent: string;
  fatal_error: string | null;
  robots: {
    url: string;
    status: "found" | "not_found" | "unreachable";
    http_status: number | null;
    crawl_delay?: number | null;
    sitemaps?: string[] | null;
    note: string;
  };
  pages: WebsitePage[];
  skipped: { url: string; reason: string }[];
  issues: WebsiteIssue[];
  summary: {
    pages_analyzed: number;
    pages_skipped: number;
    issues_by_severity: Record<Severity, number>;
    issues_by_category: Record<string, number>;
    checklist_observed: number;
    checklist_total: number;
  };
  conversion_checklist: ChecklistItem[];
  performance: PerformanceReport;
  data_availability: { item: string; status: "available" | "unavailable"; reason: string }[];
  disclaimers: string[];
}
