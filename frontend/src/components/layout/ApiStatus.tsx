"use client";

import { getHealth } from "@/lib/api";
import { useApi } from "@/lib/useApi";

/** Header pill showing whether the FastAPI backend is reachable (GET /health). */
export default function ApiStatus() {
  const { state, retry } = useApi(getHealth);

  const { dot, text } =
    state.status === "success"
      ? { dot: "bg-good", text: "API connected" }
      : state.status === "error"
        ? { dot: "bg-bad", text: "API offline" }
        : { dot: "bg-amber-400 animate-pulse", text: "Connecting…" };

  return (
    <button
      type="button"
      onClick={retry}
      title={state.status === "error" ? `${state.error.message} Click to retry.` : "Backend status (click to re-check)"}
      className="flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink-2 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-brand-500"
    >
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
      <span role="status">{text}</span>
    </button>
  );
}
