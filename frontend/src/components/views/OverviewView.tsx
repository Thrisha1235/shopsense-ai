"use client";

import Link from "next/link";

import { ConversionBars, ConversionTrend, SessionBars } from "@/components/charts/Charts";
import { Badge, Card, CardHeader, PageHeader } from "@/components/ui/Card";
import ChartCard from "@/components/ui/ChartCard";
import KpiCard, { KpiSkeleton } from "@/components/ui/KpiCard";
import { ErrorState, LoadingBlock } from "@/components/ui/States";
import { getAnalytics, getModelInfo } from "@/lib/api";
import { fmtInt, fmtPct, visitorLabel } from "@/lib/format";
import type { Analytics } from "@/lib/types";
import { useApi } from "@/lib/useApi";
import { rateColumns, withLabel } from "./tables";

function insights(a: Analytics): string[] {
  const busiest = a.monthly.reduce((x, y) => (y.sessions > x.sessions ? y : x));
  const best = a.monthly.reduce((x, y) => (y.conversion_rate > x.conversion_rate ? y : x));
  const worst = a.monthly.reduce((x, y) => (y.conversion_rate < x.conversion_rate ? y : x));
  const nv = a.visitor_types.find((v) => v.visitor_type === "New_Visitor");
  const rv = a.visitor_types.find((v) => v.visitor_type === "Returning_Visitor");
  const out = [
    `${best.month} has the highest conversion rate (${fmtPct(best.conversion_rate)}) and ${busiest.month} the most sessions (${fmtInt(busiest.sessions)}).`,
    `${worst.month} converts lowest: ${fmtPct(worst.conversion_rate)} from ${fmtInt(worst.sessions)} sessions.`,
  ];
  if (nv && rv) {
    out.push(
      `New visitors convert at ${fmtPct(nv.conversion_rate)} vs ${fmtPct(rv.conversion_rate)} for returning visitors, who make up ${fmtPct(rv.share_of_sessions)} of traffic.`,
    );
  }
  out.push(
    `Purchasers spend a median ${a.product_duration.median_purchase_minutes.toFixed(1)} min on product pages vs ${a.product_duration.median_no_purchase_minutes.toFixed(1)} min for non-purchasers.`,
  );
  if (a.overview.months_absent.length) {
    out.push(`No sessions are recorded for ${a.overview.months_absent.join(" or ")} in the dataset.`);
  }
  return out;
}

function ModelStatusCard() {
  const { state, retry } = useApi(getModelInfo);
  return (
    <Card className="flex flex-col">
      <CardHeader title="Prediction model" description="Currently served by the FastAPI backend." />
      <div className="flex-1 px-5 pb-5 pt-4">
        {state.status === "loading" && <LoadingBlock height={150} label="Loading model information" />}
        {state.status === "error" && <ErrorState error={state.error} onRetry={retry} compact />}
        {state.status === "success" && (
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between gap-3">
              <dt className="text-ink-3">Algorithm</dt>
              <dd className="font-medium text-ink">{state.data.algorithm_label}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-ink-3">Features</dt>
              <dd className="text-right font-medium text-ink">
                {state.data.required_features.length} (PageValues excluded)
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-ink-3">Test PR-AUC</dt>
              <dd className="tabular font-medium text-ink">{state.data.test_metrics_at_threshold.pr_auc.toFixed(3)}</dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-ink-3">Probabilities</dt>
              <dd>
                {state.data.calibration.probability_available ? (
                  <Badge tone="good">Calibrated ({state.data.calibration.method})</Badge>
                ) : (
                  <Badge tone="warn">Score only</Badge>
                )}
              </dd>
            </div>
          </dl>
        )}
        {state.status === "success" && (
          <Link
            href="/predict"
            className="mt-5 inline-flex w-full items-center justify-center rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
          >
            Score a session
          </Link>
        )}
      </div>
    </Card>
  );
}

export default function OverviewView() {
  const { state, retry } = useApi(getAnalytics);
  const a = state.data;
  const header = (
    <PageHeader
      title="Overview"
      description="Shopping-session behaviour and purchase conversion from the UCI Online Shoppers Purchasing Intention Dataset."
    />
  );

  // No data and the request failed: show only the error, not perpetual skeletons.
  if (state.status === "error" && !a) {
    return (
      <>
        {header}
        <ErrorState error={state.error} onRetry={retry} />
      </>
    );
  }

  return (
    <>
      {header}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {a ? (
          <>
            <KpiCard label="Total sessions" value={fmtInt(a.overview.sessions)} detail={`${a.overview.months_present.length} months of browsing data`} />
            <KpiCard accent label="Conversion rate" value={fmtPct(a.overview.conversion_rate, 2)} detail="Sessions ending in a purchase" />
            <KpiCard label="Purchases" value={fmtInt(a.overview.purchases)} detail={`${fmtPct(a.overview.conversion_rate)} of sessions`} />
            <KpiCard label="No purchase" value={fmtInt(a.overview.no_purchases)} detail={`${fmtPct(1 - a.overview.conversion_rate)} of sessions`} />
          </>
        ) : (
          state.status === "loading" && Array.from({ length: 4 }, (_, i) => <KpiSkeleton key={i} />)
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
        <ChartCard
          className="xl:col-span-2"
          title="Conversion trend by month"
          description="Share of sessions ending in a purchase. Shaded band: 95% confidence interval."
          columns={rateColumns("Month")}
          rows={a ? withLabel(a.monthly, (r) => r.month) : undefined}
          footnote={a?.overview.months_absent.length ? `${a.overview.months_absent.join(" and ")} are not present in the dataset.` : undefined}
        >
          {a ? <ConversionTrend rows={withLabel(a.monthly, (r) => r.month)} overall={a.overview.conversion_rate} /> : <LoadingBlock />}
        </ChartCard>

        <ChartCard
          title="Visitor insights"
          description="Conversion rate by visitor type (95% CI). Grey line: overall rate."
          columns={rateColumns("Visitor type")}
          rows={a ? withLabel(a.visitor_types, (r) => visitorLabel(r.visitor_type)) : undefined}
        >
          {a ? (
            <ConversionBars horizontal height={240} rows={withLabel(a.visitor_types, (r) => visitorLabel(r.visitor_type))} overall={a.overview.conversion_rate} />
          ) : (
            <LoadingBlock height={240} />
          )}
        </ChartCard>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
        <ChartCard
          className="xl:col-span-2"
          title="Sessions per month"
          description="Session volume, split into purchases and non-purchases."
          columns={rateColumns("Month")}
          rows={a ? withLabel(a.monthly, (r) => r.month) : undefined}
        >
          {a ? <SessionBars rows={withLabel(a.monthly, (r) => r.month)} /> : <LoadingBlock />}
        </ChartCard>
        <ModelStatusCard />
      </div>

      {a && (
        <Card className="mt-4">
          <CardHeader title="Key insights" description="Computed from the dataset; associations, not causal effects." />
          <ul className="list-disc space-y-1.5 px-5 pb-5 pl-10 pt-3 text-sm leading-6 text-ink-2">
            {insights(a).map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </Card>
      )}
    </>
  );
}
