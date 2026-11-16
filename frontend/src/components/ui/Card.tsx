import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-surface shadow-[0_1px_2px_rgba(15,23,42,0.04)] ${className}`}>
      {children}
    </section>
  );
}

export function CardHeader({
  title,
  description,
  action,
  id,
}: {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  id?: string;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 px-5 pt-5">
      <div className="min-w-0">
        <h2 id={id} className="text-[15px] font-semibold text-ink">
          {title}
        </h2>
        {description && <p className="mt-1 text-sm leading-6 text-ink-3">{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({ title, description, children }: { title: string; description: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        <p className="mt-1.5 text-sm leading-6 text-ink-2">{description}</p>
      </div>
      {children}
    </div>
  );
}

type Tone = "neutral" | "brand" | "good" | "warn" | "bad";
const TONES: Record<Tone, string> = {
  neutral: "bg-slate-100 text-ink-2",
  brand: "bg-brand-50 text-brand-600",
  good: "bg-good-bg text-good",
  warn: "bg-warn-bg text-warn",
  bad: "bg-bad-bg text-bad",
};

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export function Notice({ children, tone = "neutral", title }: { children: ReactNode; tone?: Tone; title?: string }) {
  const border: Record<Tone, string> = {
    neutral: "border-line bg-slate-50",
    brand: "border-brand-100 bg-brand-50",
    good: "border-green-200 bg-good-bg",
    warn: "border-amber-200 bg-warn-bg",
    bad: "border-red-200 bg-bad-bg",
  };
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm leading-6 text-ink-2 ${border[tone]}`}>
      {title && <p className="font-semibold text-ink">{title}</p>}
      {children}
    </div>
  );
}
