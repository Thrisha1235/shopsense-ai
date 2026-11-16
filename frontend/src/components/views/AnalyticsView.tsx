"use client";

import { ConversionBars, SessionBars } from "@/components/charts/Charts";
import { PageHeader } from "@/components/ui/Card";
import ChartCard from "@/components/ui/ChartCard";
import KpiCard from "@/components/ui/KpiCard";
import { ErrorState, LoadingBlock } from "@/components/ui/States";
import { getAnalytics } from "@/lib/api";
import { fmtPct, visitorLabel } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import { rateColumns, withLabel } from "./tables";

function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <section className="mt-8 first:mt-0" aria-label={title}>
      <h2 className="text-lg font-semibold tracking-tight text-ink">{title}</h2>
      <p className="mt-1 text-sm text-ink-3">{description}</p>
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">{children}</div>
    </section>
  );
}

export default function AnalyticsView() {
  const { state, retry } = useApi(getAnalytics);
  const a = state.data;

  if (!a) {
    return (
      <>
        <PageHeader title="Customer behaviour" description="How browsing behaviour relates to purchasing." />
        {state.status === "error" ? (
          <ErrorState error={state.error} onRetry={retry} />
        ) : (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <LoadingBlock /> <LoadingBlock />
            <LoadingBlock /> <LoadingBlock />
          </div>
        )}
      </>
    );
  }

  const overall = a.overview.conversion_rate;
  const months = withLabel(a.monthly, (r) => r.month);
  const visitors = withLabel(a.visitor_types, (r) => visitorLabel(r.visitor_type));
  const duration = withLabel(a.product_duration.quintiles, (r) => r.band);
  const bounce = withLabel(a.bounce_rate.bands, (r) => r.band);
  const exit = withLabel(a.exit_rate.bands, (r) => r.band);
  const traffic = withLabel(a.traffic_sources.rows, (r) => r.source);
  const days = withLabel(a.weekend, (r) => r.day_type);
  const ciNote = "Error bars: 95% Wilson confidence intervals. Grey line: overall conversion rate.";

  return (
    <>
      <PageHeader
        title="Customer behaviour"
        description={`How browsing behaviour relates to purchasing across ${a.overview.sessions.toLocaleString("en-US")} sessions. All figures are descriptive associations, not causal effects.`}
      />

      <Section title="Monthly patterns" description="Traffic volume and conversion by month of visit.">
        <ChartCard title="Sessions per month" description="Purchases vs non-purchases." columns={rateColumns("Month")} rows={months}
                   footnote={`${a.overview.months_absent.join(" and ")} are absent from the dataset.`}>
          <SessionBars rows={months} />
        </ChartCard>
        <ChartCard title="Monthly conversion rate" description={ciNote} columns={rateColumns("Month")} rows={months}>
          <ConversionBars rows={months} overall={overall} />
        </ChartCard>
      </Section>

      <Section title="New vs returning visitors" description="Session volume and conversion rate by visitor type.">
        <ChartCard title="Sessions by visitor type" columns={rateColumns("Visitor type")} rows={visitors}>
          <SessionBars horizontal rows={visitors} height={220} />
        </ChartCard>
        <ChartCard title="Conversion by visitor type" description={ciNote} columns={rateColumns("Visitor type")} rows={visitors}
                   footnote={`'Other' has only ${visitors.find((v) => v.label === "Other")?.sessions ?? 0} sessions, so its interval is wide.`}>
          <ConversionBars horizontal rows={visitors} overall={overall} height={220} />
        </ChartCard>
      </Section>

      <Section title="Product-page engagement" description="Time spent on product pages per session, in five equal-size groups.">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:col-span-2">
          <KpiCard label="Median product-page time: purchasers" value={`${a.product_duration.median_purchase_minutes.toFixed(1)} min`} accent />
          <KpiCard label="Median product-page time: non-purchasers" value={`${a.product_duration.median_no_purchase_minutes.toFixed(1)} min`} />
        </div>
        <ChartCard className="xl:col-span-2" title="Conversion by product-page duration" description={ciNote}
                   columns={rateColumns("Duration quintile")} rows={duration}>
          <ConversionBars rows={duration} overall={overall} />
        </ChartCard>
      </Section>

      <Section title="Bounce and exit rates" description="Average Google Analytics bounce and exit rates of the pages a session visited.">
        <ChartCard title="Conversion by bounce rate" description={`Median bounce rate: ${fmtPct(a.bounce_rate.median_purchase, 2)} for purchasers vs ${fmtPct(a.bounce_rate.median_no_purchase, 2)} for non-purchasers.`}
                   columns={rateColumns("Bounce rate")} rows={bounce} footnote={ciNote}>
          <ConversionBars rows={bounce} overall={overall} />
        </ChartCard>
        <ChartCard title="Conversion by exit rate" description={`Median exit rate: ${fmtPct(a.exit_rate.median_purchase, 2)} for purchasers vs ${fmtPct(a.exit_rate.median_no_purchase, 2)} for non-purchasers.`}
                   columns={rateColumns("Exit rate")} rows={exit} footnote={`${ciNote} The 0% exit band has only ${exit[0]?.sessions ?? 0} sessions.`}>
          <ConversionBars rows={exit} overall={overall} />
        </ChartCard>
      </Section>

      <Section title="Traffic sources" description={a.traffic_sources.note}>
        <ChartCard title="Sessions by traffic type" columns={rateColumns("Traffic type")} rows={traffic}
                   footnote={`${a.traffic_sources.distinct_codes} distinct traffic codes in the data.`}>
          <SessionBars horizontal rows={traffic} height={340} />
        </ChartCard>
        <ChartCard title="Conversion by traffic type" description={ciNote} columns={rateColumns("Traffic type")} rows={traffic}>
          <ConversionBars horizontal rows={traffic} overall={overall} height={340} />
        </ChartCard>
      </Section>

      <Section title="Weekday vs weekend" description="Whether the visit happened on a weekend.">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:col-span-2">
          {days.map((d) => (
            <KpiCard key={d.label} label={`${d.label} conversion`} value={fmtPct(d.conversion_rate)}
                     detail={`${d.sessions.toLocaleString("en-US")} sessions · 95% CI ${fmtPct(d.ci95_low)}–${fmtPct(d.ci95_high)}`} />
          ))}
        </div>
      </Section>
    </>
  );
}
