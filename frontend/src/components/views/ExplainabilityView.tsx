"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { ContributionChart, DECREASE, INCREASE, ImportanceChart } from "@/components/charts/ExplainCharts";
import { IconCheck } from "@/components/icons";
import { EXAMPLE, QUICK_BOUNCE, toSession } from "@/components/predict/fields";
import { Badge, Card, CardHeader, Notice, PageHeader } from "@/components/ui/Card";
import ChartCard, { type Column } from "@/components/ui/ChartCard";
import { EmptyState, ErrorState, LoadingBlock } from "@/components/ui/States";
import { explain, getGlobalExplanation } from "@/lib/api";
import { fmtPct, fmtSigned, formatFeatureValue } from "@/lib/format";
import { loadLastSession } from "@/lib/sessionHandoff";
import type { ExplanationResult, FeatureContribution, GlobalFeatureImportance, SessionInput } from "@/lib/types";
import { useApi } from "@/lib/useApi";

type Source = "last" | "engaged" | "bounce";
const SOURCES: { key: Source; label: string }[] = [
  { key: "last", label: "Last prediction" },
  { key: "engaged", label: "Sample: engaged visitor" },
  { key: "bounce", label: "Sample: quick bounce" },
];

const contributionColumns: Column<FeatureContribution>[] = [
  { key: "f", header: "Feature", render: (c) => c.label },
  { key: "v", header: "Session value", render: (c) => formatFeatureValue(c.feature, c.value) },
  {
    key: "c", header: "Contribution (raw score)", numeric: true,
    render: (c) => <span style={{ color: c.contribution >= 0 ? INCREASE : DECREASE }}>{fmtSigned(c.contribution, 4)}</span>,
  },
  { key: "d", header: "Effect", render: (c) => (c.direction === "increases" ? "Raises score" : c.direction === "decreases" ? "Lowers score" : "No effect") },
  { key: "n", header: "Columns combined", numeric: true, render: (c) => String(c.aggregated_from) },
];

const importanceColumns: Column<GlobalFeatureImportance>[] = [
  { key: "f", header: "Feature", render: (r) => r.label },
  { key: "m", header: "Avg |contribution|", numeric: true, render: (r) => r.mean_abs_contribution.toFixed(4) },
  { key: "s", header: "Share", numeric: true, render: (r) => fmtPct(r.share_of_total) },
  { key: "u", header: "Raises score in", numeric: true, render: (r) => fmtPct(r.share_of_sessions_increasing, 0) },
  { key: "d", header: "Lowers score in", numeric: true, render: (r) => fmtPct(r.share_of_sessions_decreasing, 0) },
];

function factorSentence(items: FeatureContribution[]): string {
  return items.map((c) => `${c.label.toLowerCase()} (${formatFeatureValue(c.feature, c.value)})`).join(", ");
}

function Tile({ label, value, detail, accent }: { label: string; value: string; detail?: string; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-slate-50/60 p-4">
      <p className="text-xs font-medium text-ink-3">{label}</p>
      <p data-tile-value className={`mt-1 text-2xl font-semibold tracking-tight ${accent ? "text-brand-600" : "text-ink"}`}>{value}</p>
      {detail && <p className="mt-1 text-xs leading-5 text-ink-3">{detail}</p>}
    </div>
  );
}

function SessionExplanation({ result }: { result: ExplanationResult }) {
  const e = result.explanation;
  const cv = result.calibrated_view;
  const ups = result.contributions.filter((c) => c.contribution > 0).slice(0, 3);
  const downs = result.contributions.filter((c) => c.contribution < 0).slice(0, 3);
  const net = e.model_score - e.baseline_score;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        {result.prediction.predicted_purchase ? (
          <Badge tone="good"><IconCheck width={14} height={14} /> Predicted: likely to purchase</Badge>
        ) : (
          <Badge tone="neutral">Predicted: unlikely to purchase</Badge>
        )}
        <Badge tone="brand">Explains the raw model score</Badge>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Tile label="Model baseline (reference score)" value={e.baseline_score.toFixed(3)}
              detail={`Balanced reference all explanations start from. A typical real session scores ${e.typical_session_score.toFixed(3)}.`} />
        <Tile label="This session's raw score" value={e.model_score.toFixed(3)} accent
              detail={`Decision threshold ${result.prediction.decision_threshold.model_score.toFixed(3)}`} />
        <Tile label="Net effect of all features" value={fmtSigned(net)}
              detail={`Sum of the ${result.contributions.length} contributions below`} />
        <Tile
          label="Calibrated purchase probability"
          value={cv.purchase_probability !== null ? fmtPct(cv.purchase_probability) : "n/a"}
          detail={cv.probability_at_baseline !== null ? `At the baseline score: ${fmtPct(cv.probability_at_baseline)}. Shown separately; the bars explain the score, not this.` : "Calibration unavailable"}
        />
      </div>

      <div className="rounded-lg border border-line p-4 text-sm leading-6 text-ink-2">
        <p className="font-semibold text-ink">In plain language</p>
        <p className="mt-1">
          This session&rsquo;s score is{" "}
          <strong className="text-ink">{net >= 0 ? "above" : "below"}</strong> the model&rsquo;s reference score by{" "}
          {Math.abs(net).toFixed(3)}
          {" "}({e.model_score >= e.typical_session_score ? "and above" : "and below"} the typical real session&rsquo;s {e.typical_session_score.toFixed(3)}).
          {ups.length > 0 && <> The biggest factors raising it are {factorSentence(ups)}.</>}
          {downs.length > 0 && <> The biggest factors lowering it are {factorSentence(downs)}.</>}
        </p>
        {cv.available && cv.probability_at_baseline !== null && cv.purchase_probability !== null && (
          <p className="mt-2">
            Together these move the calibrated purchase probability from {fmtPct(cv.probability_at_baseline)} at the
            reference score to {fmtPct(cv.purchase_probability)} for this session.
          </p>
        )}
      </div>

      {result.warnings.length > 0 && (
        <Notice tone="warn" title={`Warnings (${result.warnings.length})`}>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {result.warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
        </Notice>
      )}
    </div>
  );
}

function ExplainSession() {
  const [source, setSource] = useState<Source>("engaged");
  const [hasLast, setHasLast] = useState(false);
  const [state, setState] = useState<{ status: "idle" | "loading" | "success" | "error"; data?: ExplanationResult; error?: Error }>({ status: "idle" });
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (session: SessionInput) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setState((s) => ({ status: "loading", data: s.data }));
    try {
      const data = await explain(session, controller.signal);
      setState({ status: "success", data });
    } catch (error) {
      if (controller.signal.aborted) return;
      setState((s) => ({ status: "error", error: error instanceof Error ? error : new Error(String(error)), data: s.data }));
    }
  }, []);

  const select = useCallback((key: Source) => {
    setSource(key);
    const session = key === "last" ? loadLastSession() : toSession(key === "engaged" ? EXAMPLE : QUICK_BOUNCE);
    if (session) void run(session);
  }, [run]);

  // On first load: explain the last prediction if there is one, otherwise the sample session.
  useEffect(() => {
    const last = loadLastSession();
    setHasLast(!!last);
    select(last ? "last" : "engaged");
    return () => abortRef.current?.abort();
  }, [select]);

  const data = state.data;
  return (
    <>
      <Card>
        <CardHeader
          title="Explain a session"
          description={<>Why the model gave this session its score. Change the inputs on the <Link href="/predict" className="font-medium text-brand-600 hover:underline">prediction page</Link>, then come back.</>}
          action={
            <div role="group" aria-label="Session to explain" className="flex flex-wrap gap-2">
              {SOURCES.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  aria-pressed={source === s.key}
                  disabled={s.key === "last" && !hasLast}
                  title={s.key === "last" && !hasLast ? "Score a session on the prediction page first" : undefined}
                  onClick={() => select(s.key)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium focus-visible:outline-2 focus-visible:outline-brand-500 disabled:cursor-not-allowed disabled:opacity-50 ${
                    source === s.key ? "border-brand-200 bg-brand-50 text-brand-600" : "border-line text-ink-2 hover:bg-slate-50"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          }
        />
        <div className="px-5 pb-5 pt-4" aria-live="polite" aria-busy={state.status === "loading"}>
          {state.status === "error" && <div className="mb-4"><ErrorState error={state.error!} onRetry={() => select(source)} compact /></div>}
          {data ? (
            <div className={state.status === "loading" ? "opacity-60 transition-opacity" : "transition-opacity"}>
              <SessionExplanation result={data} />
            </div>
          ) : state.status === "loading" || state.status === "idle" ? (
            <LoadingBlock height={220} label="Computing explanation" />
          ) : (
            <EmptyState title="No explanation yet">Choose a session above.</EmptyState>
          )}
        </div>
      </Card>

      {data && (
        <ChartCard
          className="mt-4"
          title="What pushed this session's score up or down"
          description={<>Each bar is one input&rsquo;s SHAP contribution to the <strong>raw model score</strong>, relative to the model&rsquo;s reference score (the baseline). Blue raises the score, orange lowers it. Bars add up exactly to the session&rsquo;s score minus the baseline.</>}
          columns={contributionColumns}
          rows={data.contributions}
          footnote={`Top 10 features shown; the remaining ${Math.max(0, data.contributions.length - 10)} are combined. Units: raw-score points, not percentage points of probability. Categorical inputs (e.g. Month) are their one-hot columns summed.`}
        >
          <ContributionChart contributions={data.contributions} />
        </ChartCard>
      )}
    </>
  );
}

function GlobalImportance() {
  const { state, retry } = useApi(getGlobalExplanation);
  if (state.status === "loading") return <Card className="mt-4"><div className="p-5"><LoadingBlock height={420} label="Loading feature importance" /></div></Card>;
  if (state.status === "error") return <Card className="mt-4"><div className="p-5"><ErrorState error={state.error} onRetry={retry} compact /></div></Card>;
  const g = state.data;
  return (
    <ChartCard
      className="mt-4"
      title="Which inputs matter most overall"
      description={`Average size of each input's contribution across ${g.n_sessions} held-out sessions from the test split (used for reporting only). Larger bars mean the input usually moves the score more, in either direction.`}
      columns={importanceColumns}
      rows={g.features}
      footnote={`Units: raw-score points. Reference (baseline) score ${g.baseline_score.toFixed(3)}; typical real session score ${g.typical_session_score.toFixed(3)}. Contributions add up to each session's score (max error ${g.max_additivity_error.toExponential(1)}).`}
    >
      <ImportanceChart features={g.features} />
    </ChartCard>
  );
}

function HowToRead() {
  return (
    <Card>
      <CardHeader title="How to read these explanations" />
      <ul className="space-y-2 px-5 pb-5 pt-3 text-sm leading-6 text-ink-2">
        <li><strong className="text-ink">Baseline:</strong> the model&rsquo;s reference score (0.5) before it looks at any session details. The model was trained with purchases and non-purchases weighted equally, so the reference sits above a typical real session&rsquo;s score. Every explanation starts there.</li>
        <li><strong className="text-ink">Bars:</strong> how much each piece of session information moved the score up (blue) or down (orange) from the baseline.</li>
        <li><strong className="text-ink">Score vs probability:</strong> the bars explain the model&rsquo;s raw score. The calibrated purchase probability is that score rescaled; it moves in the same direction, but the bars are not percentage points of probability.</li>
        <li><strong className="text-ink">Not causes:</strong> the model learned patterns from past sessions. A bar does not say why a customer buys, and changing one value for a real visitor will not necessarily change what they do.</li>
      </ul>
    </Card>
  );
}

function TechnicalDetails() {
  const { state } = useApi(getGlobalExplanation);
  if (state.status !== "success") return null;
  const g = state.data;
  return (
    <details className="mt-4 rounded-xl border border-line bg-surface px-5 py-4 text-sm text-ink-2">
      <summary className="cursor-pointer font-semibold text-ink">Technical details</summary>
      <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-[max-content_1fr]">
        <dt className="text-ink-3">Method</dt><dd>{g.method}</dd>
        <dt className="text-ink-3">Explains</dt><dd>Raw Random Forest score (<code className="text-xs">predict_proba</code>), not the calibrated probability</dd>
        <dt className="text-ink-3">Units</dt><dd>{g.output_units}</dd>
        <dt className="text-ink-3">Baseline</dt><dd>{g.baseline_score.toFixed(3)}: {g.baseline_description}</dd>
        <dt className="text-ink-3">Feature aggregation</dt><dd>{g.aggregation}</dd>
        <dt className="text-ink-3">Model</dt><dd>Random Forest without PageValues (variant <code className="text-xs">{g.variant}</code>)</dd>
        <dt className="text-ink-3">Computed</dt><dd>{g.created_at ?? "—"}</dd>
      </dl>
    </details>
  );
}

export default function ExplainabilityView() {
  return (
    <>
      <PageHeader
        title="Explainable AI"
        description="SHAP explanations of how the served Random Forest arrives at a session's purchase score."
      >
        <Badge tone="good">Connected · SHAP</Badge>
      </PageHeader>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <ExplainSession />
        </div>
        <div className="space-y-4">
          <HowToRead />
          <Notice tone="warn" title="Model behaviour, not customer motives">
            SHAP describes what this model does with its inputs. It does not establish cause and effect, and it
            inherits the model&rsquo;s limitations (completed sessions only, one shop, PageValues excluded).
          </Notice>
        </div>
      </div>
      <GlobalImportance />
      <TechnicalDetails />
    </>
  );
}
