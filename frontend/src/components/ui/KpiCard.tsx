import type { ReactNode } from "react";

export default function KpiCard({
  label,
  value,
  detail,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className="rounded-xl border border-line bg-surface p-5 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
      <p className="text-sm font-medium text-ink-3">{label}</p>
      <p className={`mt-2 text-3xl font-semibold tracking-tight ${accent ? "text-brand-600" : "text-ink"}`}>{value}</p>
      {detail && <p className="mt-1.5 text-sm leading-5 text-ink-3">{detail}</p>}
    </div>
  );
}

export function KpiSkeleton() {
  return (
    <div className="rounded-xl border border-line bg-surface p-5" aria-hidden="true">
      <div className="h-4 w-28 animate-pulse rounded bg-slate-100" />
      <div className="mt-3 h-8 w-24 animate-pulse rounded bg-slate-100" />
      <div className="mt-3 h-4 w-40 animate-pulse rounded bg-slate-100" />
    </div>
  );
}
