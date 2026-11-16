"""Pydantic models for POST /api/analyze-website."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EvidenceType = Literal["observed", "heuristic"]


class AnalyzeWebsiteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=3, max_length=2048, description="Public http(s) website URL or domain, e.g. example.com")
    include_performance: bool = Field(
        default=False, description="Also request Google PageSpeed Insights (only if configured on the server).")


class RobotsReport(BaseModel):
    url: str
    status: Literal["found", "not_found", "unreachable"]
    http_status: int | None
    allows_start: bool | None = None
    crawl_delay: float | None = None
    sitemaps: list[str] | None = None
    note: str = ""


class IssueOut(BaseModel):
    id: str
    severity: Literal["high", "medium", "low", "info"]
    category: Literal["technical", "seo", "usability"]
    title: str
    url: str
    evidence: str
    recommendation: str
    evidence_type: EvidenceType


class ChecklistOut(BaseModel):
    id: str
    label: str
    status: Literal["observed", "not_observed", "unavailable"]
    evidence: list[str]
    note: str
    evidence_type: EvidenceType


class Availability(BaseModel):
    item: str
    status: Literal["available", "unavailable"]
    reason: str


class Summary(BaseModel):
    pages_analyzed: int
    pages_skipped: int
    issues_by_severity: dict[str, int]
    issues_by_category: dict[str, int]
    checklist_observed: int
    checklist_total: int


class AnalyzeWebsiteResponse(BaseModel):
    input_url: str
    analyzed_url: str
    final_url: str | None
    analyzed_at: str
    duration_ms: int
    user_agent: str
    fatal_error: str | None
    robots: RobotsReport
    pages: list[dict[str, Any]]
    skipped: list[dict[str, str]]
    issues: list[IssueOut]
    summary: Summary
    conversion_checklist: list[ChecklistOut]
    performance: dict[str, Any]
    data_availability: list[Availability]
    disclaimers: list[str]
