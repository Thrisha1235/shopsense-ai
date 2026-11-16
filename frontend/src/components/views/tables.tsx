import type { Column } from "@/components/ui/ChartCard";
import { fmtInt, fmtPct } from "@/lib/format";
import type { RateRow } from "@/lib/types";

export type LabelledRow = RateRow & { label: string };

export function withLabel<T extends RateRow>(rows: T[], label: (r: T) => string): LabelledRow[] {
  return rows.map((r) => ({ ...r, label: label(r) }));
}

export function rateColumns(labelHeader: string): Column<LabelledRow>[] {
  return [
    { key: "label", header: labelHeader, render: (r) => r.label },
    { key: "sessions", header: "Sessions", numeric: true, render: (r) => fmtInt(r.sessions) },
    { key: "purchases", header: "Purchases", numeric: true, render: (r) => fmtInt(r.purchases) },
    { key: "share", header: "Share", numeric: true, render: (r) => fmtPct(r.share_of_sessions) },
    { key: "conv", header: "Conversion", numeric: true, render: (r) => fmtPct(r.conversion_rate) },
    { key: "ci", header: "95% CI", numeric: true, render: (r) => `${fmtPct(r.ci95_low)}–${fmtPct(r.ci95_high)}` },
  ];
}
