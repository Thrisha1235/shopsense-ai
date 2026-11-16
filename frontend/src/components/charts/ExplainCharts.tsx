"use client";

import { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";

import { axisTick, chart } from "@/lib/chartTheme";
import { fmtPct, fmtSigned, formatFeatureValue } from "@/lib/format";
import type { FeatureContribution, GlobalFeatureImportance } from "@/lib/types";

// Diverging pair: raises the score (blue, series 1) vs lowers the score (orange, series 2).
// Validated as a pair in Phase 2 (CVD dE 24.7). Sign is also shown in the tooltip and table.
export const INCREASE = "#2a78d6";
export const DECREASE = "#eb6834";

interface Row {
  label: string;
  axisLabel: string;
  value: string;
  contribution: number;
  increase: number;
  decrease: number;
  aggregatedFrom: number;
  isOther: boolean;
}

function toRows(contribs: FeatureContribution[], top: number): Row[] {
  const sorted = [...contribs].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const head = sorted.slice(0, top);
  const tail = sorted.slice(top);
  const rows: Row[] = head.map((c) => {
    const value = formatFeatureValue(c.feature, c.value);
    return {
      label: c.label,
      axisLabel: `${c.label} = ${value}`,
      value,
      contribution: c.contribution,
      increase: c.contribution > 0 ? c.contribution : 0,
      decrease: c.contribution < 0 ? c.contribution : 0,
      aggregatedFrom: c.aggregated_from,
      isOther: false,
    };
  });
  if (tail.length) {
    const sum = tail.reduce((s, c) => s + c.contribution, 0);
    rows.push({
      label: `Other ${tail.length} features`,
      axisLabel: `Other ${tail.length} features (combined)`,
      value: tail.map((c) => c.label).join(", "),
      contribution: sum,
      increase: sum > 0 ? sum : 0,
      decrease: sum < 0 ? sum : 0,
      aggregatedFrom: tail.length,
      isOther: true,
    });
  }
  return rows;
}

function ContributionTooltip({ active, payload }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload as Row;
  return (
    <div className="max-w-xs rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <p className="font-semibold text-ink">{r.label}</p>
      <p className="mt-0.5 text-ink-2">{r.isOther ? r.value : `Value: ${r.value}`}</p>
      <p className="mt-1 tabular font-medium" style={{ color: r.contribution >= 0 ? INCREASE : DECREASE }}>
        {fmtSigned(r.contribution, 4)} raw-score points ({r.contribution >= 0 ? "raises" : "lowers"} the score)
      </p>
      {!r.isOther && r.aggregatedFrom > 1 && (
        <p className="mt-1 text-ink-3">Combined from {r.aggregatedFrom} one-hot columns.</p>
      )}
    </div>
  );
}

/** Width of an element, tracked with ResizeObserver (for narrow-screen chart layout). */
function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return { ref, width };
}

const truncate = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

/** Diverging bar chart of a session's SHAP contributions (raw-score units). */
export function ContributionChart({ contributions, top = 10 }: { contributions: FeatureContribution[]; top?: number }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const narrow = width > 0 && width < 560;
  // Narrow screens: shorter label column, feature name only (value stays in tooltip + table view).
  const yWidth = narrow ? Math.max(96, Math.round(width * 0.36)) : 230;
  const maxChars = Math.floor(yWidth / 6.4);
  const rows = toRows(contributions, top);
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 0.01);
  const lim = Math.ceil((max * 1.1) / 0.02) * 0.02;
  return (
    <div ref={ref}>
    <ResponsiveContainer width="100%" height={Math.max(260, rows.length * 34 + 70)}>
      <BarChart data={rows} layout="vertical" stackOffset="sign" margin={{ top: 4, right: narrow ? 8 : 20, bottom: 4, left: narrow ? 0 : 8 }} barCategoryGap="22%">
        <CartesianGrid horizontal={false} />
        <XAxis type="number" domain={[-lim, lim]} tickFormatter={(v: number) => (v === 0 ? "0" : fmtSigned(v, 2))}
               tick={axisTick} axisLine={false} tickLine={false} tickCount={narrow ? 3 : 5} />
        <YAxis type="category" dataKey={narrow ? "label" : "axisLabel"} width={yWidth}
               tickFormatter={(v: string) => truncate(v, maxChars)}
               tick={{ ...axisTick, fontSize: narrow ? 11 : 11.5 }} axisLine={false} tickLine={false} />
        <ReferenceLine x={0} stroke={chart.ink2} strokeWidth={1} />
        <Tooltip content={(p) => <ContributionTooltip {...p} />} cursor={{ fill: "#f1f5f9" }} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: chart.ink2, paddingTop: 6 }} />
        <Bar isAnimationActive={false} dataKey="increase" name="Raises the score" stackId="c" fill={INCREASE} radius={[0, 4, 4, 0]} />
        <Bar isAnimationActive={false} dataKey="decrease" name="Lowers the score" stackId="c" fill={DECREASE} radius={[4, 0, 0, 4]} />
      </BarChart>
    </ResponsiveContainer>
    </div>
  );
}

function ImportanceTooltip({ active, payload }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload as GlobalFeatureImportance;
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <p className="font-semibold text-ink">{r.label}</p>
      <p className="tabular text-ink-2">Average |contribution|: {r.mean_abs_contribution.toFixed(4)}</p>
      <p className="tabular text-ink-2">Share of total importance: {fmtPct(r.share_of_total)}</p>
      <p className="tabular text-ink-2">Raises the score in {fmtPct(r.share_of_sessions_increasing, 0)} of sessions,
        lowers it in {fmtPct(r.share_of_sessions_decreasing, 0)}</p>
    </div>
  );
}

/** Global importance: mean |SHAP| per feature (single series). */
export function ImportanceChart({ features }: { features: GlobalFeatureImportance[] }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const narrow = width > 0 && width < 560;
  const yWidth = narrow ? Math.max(96, Math.round(width * 0.38)) : 190;
  return (
    <div ref={ref}>
    <ResponsiveContainer width="100%" height={features.length * 28 + 50}>
      <BarChart data={features} layout="vertical" margin={{ top: 4, right: narrow ? 8 : 20, bottom: 4, left: narrow ? 0 : 8 }} barCategoryGap="24%">
        <CartesianGrid horizontal={false} />
        <XAxis type="number" tickFormatter={(v: number) => v.toFixed(2)} tick={axisTick} axisLine={false} tickLine={false} tickCount={narrow ? 3 : 5} />
        <YAxis type="category" dataKey="label" width={yWidth} tickFormatter={(v: string) => truncate(v, Math.floor(yWidth / 6.4))}
               tick={{ ...axisTick, fontSize: narrow ? 11 : 11.5 }} axisLine={false} tickLine={false} />
        <Tooltip content={(p) => <ImportanceTooltip {...p} />} cursor={{ fill: "#f1f5f9" }} />
        <Bar isAnimationActive={false} dataKey="mean_abs_contribution" name="Average |contribution|" fill={chart.primary} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
    </div>
  );
}
