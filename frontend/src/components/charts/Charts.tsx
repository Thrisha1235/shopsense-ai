"use client";

import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";

import { axisTick, chart } from "@/lib/chartTheme";
import { fmtInt, fmtPct } from "@/lib/format";
import type { RateRow } from "@/lib/types";

const pctTick = (v: number) => `${Math.round(v * 100)}%`;

/** Round an axis up to an even step with at most `maxTicks` ticks (e.g. 0, 10%, 20%, 30%). */
function niceAxis(max: number, steps: number[], maxTicks: number): { domain: [number, number]; ticks: number[] } {
  const target = max * 1.05;
  const step = steps.find((s) => Math.ceil(target / s) + 1 <= maxTicks) ?? steps[steps.length - 1];
  const top = Math.max(step, Math.ceil(target / step) * step);
  const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => Number((i * step).toFixed(6)));
  return { domain: [0, top], ticks };
}

const rateAxis = (max: number, maxTicks = 7) => niceAxis(max, [0.05, 0.1, 0.2, 0.25, 0.5], maxTicks);

const countAxis = (max: number) => {
  const mag = 10 ** Math.floor(Math.log10(Math.max(1, max)));
  return niceAxis(max, [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10].map((m) => m * mag), 6);
};
const intTick = (v: number) => (v >= 1000 ? `${Number((v / 1000).toFixed(1))}k` : `${v}`);

function TooltipBox({ title, lines }: { title: string; lines: [string, string][] }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 font-semibold text-ink">{title}</p>
      {lines.map(([k, v]) => (
        <p key={k} className="flex justify-between gap-4 text-ink-2">
          <span>{k}</span>
          <span className="tabular font-medium text-ink">{v}</span>
        </p>
      ))}
    </div>
  );
}

type Labelled = RateRow & { label: string };

function RateTooltip({ active, payload }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as Labelled;
  return (
    <TooltipBox
      title={row.label}
      lines={[
        ["Conversion", fmtPct(row.conversion_rate)],
        ["95% CI", `${fmtPct(row.ci95_low)} – ${fmtPct(row.ci95_high)}`],
        ["Sessions", fmtInt(row.sessions)],
        ["Purchases", fmtInt(row.purchases)],
      ]}
    />
  );
}

/** Conversion rate per category with 95% Wilson intervals and the overall rate as reference. */
export function ConversionBars({
  rows,
  overall,
  height = 280,
  horizontal = false,
}: {
  rows: Labelled[];
  overall?: number;
  height?: number;
  horizontal?: boolean;
}) {
  const data = rows.map((r) => ({ ...r, err: [r.conversion_rate - r.ci95_low, r.ci95_high - r.conversion_rate] }));
  const axis = rateAxis(Math.max(...rows.map((r) => r.ci95_high), overall ?? 0), horizontal ? 5 : 7);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={horizontal ? "vertical" : "horizontal"}
        margin={{ top: 8, right: 16, bottom: 0, left: horizontal ? 8 : -8 }}
        barCategoryGap="28%"
      >
        <CartesianGrid vertical={horizontal} horizontal={!horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" domain={axis.domain} ticks={axis.ticks} tickFormatter={pctTick} tick={axisTick} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="label" width={130} tick={axisTick} axisLine={false} tickLine={false} />
          </>
        ) : (
          <>
            <XAxis dataKey="label" tick={axisTick} axisLine={{ stroke: chart.axis }} tickLine={false} interval={0} />
            <YAxis domain={axis.domain} ticks={axis.ticks} tickFormatter={pctTick} tick={axisTick} axisLine={false} tickLine={false} width={48} />
          </>
        )}
        <Tooltip content={(p) => <RateTooltip {...p} />} cursor={{ fill: "#f1f5f9" }} />
        {/* No in-plot label (it collides with bars); callers name the grey line in the description. */}
        {overall !== undefined &&
          (horizontal ? (
            <ReferenceLine x={overall} stroke={chart.reference} strokeWidth={1} />
          ) : (
            <ReferenceLine y={overall} stroke={chart.reference} strokeWidth={1} />
          ))}
        <Bar isAnimationActive={false} dataKey="conversion_rate" name="Conversion rate" fill={chart.primary} radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]} maxBarSize={44}>
          <ErrorBar dataKey="err" width={5} strokeWidth={1.25} stroke={chart.ink2} direction={horizontal ? "x" : "y"} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function SessionsTooltip({ active, payload }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as Labelled & { no_purchase: number };
  return (
    <TooltipBox
      title={row.label}
      lines={[
        ["Sessions", fmtInt(row.sessions)],
        ["Purchases", fmtInt(row.purchases)],
        ["No purchase", fmtInt(row.no_purchase)],
        ["Share of all sessions", fmtPct(row.share_of_sessions)],
      ]}
    />
  );
}

/** Session volume split into purchases (blue) and non-purchases (muted). */
export function SessionBars({ rows, height = 280, horizontal = false }: { rows: Labelled[]; height?: number; horizontal?: boolean }) {
  const data = rows.map((r) => ({ ...r, no_purchase: r.sessions - r.purchases }));
  const axis = countAxis(Math.max(...rows.map((r) => r.sessions)));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={horizontal ? "vertical" : "horizontal"}
        margin={{ top: 8, right: 16, bottom: 0, left: horizontal ? 8 : -4 }}
        barCategoryGap="28%"
      >
        <CartesianGrid vertical={horizontal} horizontal={!horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" domain={axis.domain} ticks={axis.ticks} tickFormatter={intTick} tick={axisTick} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="label" width={130} tick={axisTick} axisLine={false} tickLine={false} />
          </>
        ) : (
          <>
            <XAxis dataKey="label" tick={axisTick} axisLine={{ stroke: chart.axis }} tickLine={false} interval={0} />
            <YAxis domain={axis.domain} ticks={axis.ticks} tickFormatter={intTick} tick={axisTick} axisLine={false} tickLine={false} width={48} />
          </>
        )}
        <Tooltip content={(p) => <SessionsTooltip {...p} />} cursor={{ fill: "#f1f5f9" }} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: chart.ink2, paddingTop: 4 }} />
        <Bar isAnimationActive={false} dataKey="purchases" name="Purchase" stackId="s" fill={chart.primary} maxBarSize={44} />
        <Bar isAnimationActive={false} dataKey="no_purchase" name="No purchase" stackId="s" fill={chart.muted} maxBarSize={44}
             radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Conversion trend with a shaded 95% CI band (one series, one axis). */
export function ConversionTrend({ rows, overall, height = 280 }: { rows: Labelled[]; overall?: number; height?: number }) {
  const data = rows.map((r) => ({ ...r, band: [r.ci95_low, r.ci95_high] }));
  const axis = rateAxis(Math.max(...rows.map((r) => r.ci95_high)));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 12, right: 16, bottom: 0, left: -8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="label" tick={axisTick} axisLine={{ stroke: chart.axis }} tickLine={false} />
        <YAxis domain={axis.domain} ticks={axis.ticks} tickFormatter={pctTick} tick={axisTick} axisLine={false} tickLine={false} width={48} />
        <Tooltip content={(p) => <RateTooltip {...p} />} cursor={{ stroke: chart.axis }} />
        {overall !== undefined && (
          <ReferenceLine y={overall} stroke={chart.reference} label={{ value: `Overall ${fmtPct(overall)}`, position: "insideBottomLeft", fill: chart.ink2, fontSize: 11 }} />
        )}
        <Area dataKey="band" stroke="none" fill={chart.primary} fillOpacity={0.12} isAnimationActive={false} name="95% CI" />
        <Line isAnimationActive={false} dataKey="conversion_rate" name="Conversion rate" stroke={chart.primary} strokeWidth={2}
              dot={{ r: 4, fill: chart.primary, stroke: "#fff", strokeWidth: 2 }} activeDot={{ r: 6 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------- model curves
interface CurveSeries {
  key: string;
  name: string;
  points: { x: number; y: number }[];
  emphasis: boolean;
  operatingPoint?: { x: number; y: number };
  dots?: boolean;
}

function CurveTooltip({ active, payload, xLabel, yLabel }: TooltipContentProps & { xLabel: string; yLabel: string }) {
  if (!active || !payload?.length) return null;
  const pt = payload[0].payload as { x: number; y: number };
  return (
    <TooltipBox
      title={String(payload[0].name)}
      lines={[[xLabel, pt.x.toFixed(3)], [yLabel, pt.y.toFixed(3)]]}
    />
  );
}

export function CurveChart({
  series,
  xLabel,
  yLabel,
  diagonal = false,
  baselineY,
  height = 300,
}: {
  series: CurveSeries[];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  baselineY?: number;
  height?: number;
}) {
  return (
    <>
    {/* Y-axis title as text above the plot: never clipped, readable on narrow screens. */}
    <p className="mb-1 text-xs font-medium text-ink-2">↑ {yLabel}</p>
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 8, right: 16, bottom: 18, left: -4 }}>
        <CartesianGrid />
        <XAxis type="number" dataKey="x" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tick={axisTick}
               axisLine={{ stroke: chart.axis }} tickLine={false}
               label={{ value: xLabel, position: "insideBottom", offset: -10, fill: chart.ink2, fontSize: 12 }} />
        <YAxis type="number" dataKey="y" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tick={axisTick}
               axisLine={false} tickLine={false} width={44} />
        <Tooltip content={(p) => <CurveTooltip {...p} xLabel={xLabel} yLabel={yLabel} />} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: chart.ink2, paddingTop: 14 }} />
        {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={chart.axis} />}
        {baselineY !== undefined && (
          <ReferenceLine y={baselineY} stroke={chart.axis} label={{ value: `No skill ${baselineY.toFixed(3)}`, position: "insideBottomRight", fill: chart.ink2, fontSize: 11 }} />
        )}
        {series.map((s) => (
          <Line key={s.key} data={s.points} dataKey="y" name={s.name} type="linear"
                dot={s.dots ? { r: 4, fill: s.emphasis ? chart.primary : chart.reference, stroke: "#fff", strokeWidth: 2 } : false}
                stroke={s.emphasis ? chart.primary : chart.reference} strokeWidth={s.emphasis ? 2.25 : 1.5}
                isAnimationActive={false} />
        ))}
        {series.filter((s) => s.operatingPoint).map((s) => (
          <ReferenceDot key={`${s.key}-op`} x={s.operatingPoint!.x} y={s.operatingPoint!.y} r={6}
                        fill={s.emphasis ? chart.primary : chart.reference} stroke="#fff" strokeWidth={2} />
        ))}
      </LineChart>
    </ResponsiveContainer>
    </>
  );
}

// ------------------------------------------------------------ confusion matrix
export function ConfusionMatrixGrid({ tn, fp, fn, tp }: { tn: number; fp: number; fn: number; tp: number }) {
  const rows = [
    { actual: "No purchase", cells: [{ v: tn, label: "True negative" }, { v: fp, label: "False positive" }] },
    { actual: "Purchase", cells: [{ v: fn, label: "False negative" }, { v: tp, label: "True positive" }] },
  ];
  const shade = (share: number) =>
    // white text only on brand-600/700 (>= 6.6:1); dark ink on the light steps
    share > 0.7 ? "bg-brand-700 text-white" : share > 0.45 ? "bg-brand-600 text-white" : share > 0.2 ? "bg-brand-200 text-ink" : "bg-brand-50 text-ink";
  return (
    <table className="w-full border-separate border-spacing-1.5 text-sm">
      <caption className="sr-only">Confusion matrix on the test set</caption>
      <thead>
        <tr>
          <th scope="col" className="w-28" />
          <th scope="col" className="pb-1 text-xs font-medium text-ink-3">Predicted: no purchase</th>
          <th scope="col" className="pb-1 text-xs font-medium text-ink-3">Predicted: purchase</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const total = r.cells[0].v + r.cells[1].v;
          return (
            <tr key={r.actual}>
              <th scope="row" className="pr-2 text-right text-xs font-medium text-ink-3">Actual: {r.actual.toLowerCase()}</th>
              {r.cells.map((c) => {
                const share = c.v / total;
                return (
                  <td key={c.label} className={`rounded-lg px-3 py-4 text-center ${shade(share)}`}>
                    <span className="block text-xl font-semibold tabular">{fmtInt(c.v)}</span>
                    <span className="block text-xs opacity-90">
                      {c.label} · {fmtPct(share)} of row
                    </span>
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
