import type { ReactNode } from "react";

import { IconAlert, IconInfo, IconRefresh } from "@/components/icons";

export function LoadingBlock({ height = 280, label = "Loading data" }: { height?: number; label?: string }) {
  return (
    <div role="status" aria-label={label} className="w-full animate-pulse rounded-lg bg-slate-100" style={{ height }}>
      <span className="sr-only">{label}…</span>
    </div>
  );
}

export function ErrorState({ error, onRetry, compact = false }: { error: Error; onRetry?: () => void; compact?: boolean }) {
  return (
    <div
      role="alert"
      className={`flex flex-col items-center justify-center rounded-lg border border-red-200 bg-bad-bg text-center ${
        compact ? "gap-2 p-4" : "gap-3 px-6 py-10"
      }`}
    >
      <IconAlert className="text-bad" width={24} height={24} />
      <div>
        <p className="text-sm font-semibold text-ink">Could not load data</p>
        <p className="mt-1 max-w-md text-sm text-ink-2">{error.message}</p>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm font-medium text-ink hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-brand-500"
        >
          <IconRefresh width={16} height={16} /> Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-line-strong bg-slate-50/60 px-6 py-10 text-center">
      {icon ?? <IconInfo className="text-ink-3" width={24} height={24} />}
      <div>
        <p className="text-sm font-semibold text-ink">{title}</p>
        {children && <div className="mx-auto mt-1 max-w-md text-sm leading-6 text-ink-3">{children}</div>}
      </div>
    </div>
  );
}
