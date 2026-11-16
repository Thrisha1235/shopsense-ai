"use client";

import { ConfusionMatrixGrid, CurveChart } from "@/components/charts/Charts";
import { Badge, Card, CardHeader, Notice, PageHeader } from "@/components/ui/Card";
import ChartCard, { DataTable, type Column } from "@/components/ui/ChartCard";
import KpiCard, { KpiSkeleton } from "@/components/ui/KpiCard";
import { ErrorState, LoadingBlock } from "@/components/ui/States";
import { getModelInfo, getModelPerformance } from "@/lib/api";
import { featureSetLabel, fmtInt, fmtNum, fmtPct } from "@/lib/format";
import type { ModelPerformance, ModelRow } from "@/lib/types";
import { useApi } from "@/lib/useApi";

const modelColumns: Column<ModelRow>[] = [
  {
    key: "model",
    header: "Model",
    render: (m) => (
      <span className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-ink">{m.label}</span>
        {m.selected && <Badge tone="brand">Selected</Badge>}
      </span>
    ),
  },
  { key: "fs", header: "Features", render: (m) => featureSetLabel(m.feature_set) },
  { key: "cv", header: "CV PR-AUC", numeric: true, render: (m) => `${fmtNum(m.cv_pr_auc_mean)} ± ${fmtNum(m.cv_pr_auc_std)}` },
  { key: "pr", header: "Test PR-AUC (95% CI)", numeric: true, render: (m) => `${fmtNum(m.test.pr_auc)} (${fmtNum(m.test_pr_auc_ci95[0], 2)}–${fmtNum(m.test_pr_auc_ci95[1], 2)})` },
  { key: "roc", header: "ROC-AUC", numeric: true, render: (m) => fmtNum(m.test.roc_auc) },
  { key: "f1", header: "F1", numeric: true, render: (m) => fmtNum(m.test.f1) },
  { key: "p", header: "Precision", numeric: true, render: (m) => fmtNum(m.test.precision) },
  { key: "r", header: "Recall", numeric: true, render: (m) => fmtNum(m.test.recall) },
  { key: "acc", header: "Accuracy", numeric: true, render: (m) => fmtNum(m.test.accuracy) },
  { key: "thr", header: "Threshold", numeric: true, render: (m) => fmtNum(m.threshold) },
];

function Curves({ p }: { p: ModelPerformance }) {
  const def = p.curves[p.default_variant];
  const bench = p.curves["with_page_values"];
  const roc = [
    { key: "def", name: `${def.label} without PageValues (served)`, emphasis: true,
      points: def.roc.map((d) => ({ x: d.fpr, y: d.tpr })), operatingPoint: { x: def.operating_point.fpr, y: def.operating_point.tpr } },
    ...(bench ? [{ key: "bench", name: `${bench.label} with PageValues (benchmark)`, emphasis: false,
      points: bench.roc.map((d) => ({ x: d.fpr, y: d.tpr })), operatingPoint: { x: bench.operating_point.fpr, y: bench.operating_point.tpr } }] : []),
  ];
  const pr = [
    { key: "def", name: `${def.label} without PageValues (served)`, emphasis: true,
      points: def.pr.map((d) => ({ x: d.recall, y: d.precision })), operatingPoint: { x: def.operating_point.recall, y: def.operating_point.precision } },
    ...(bench ? [{ key: "bench", name: `${bench.label} with PageValues (benchmark)`, emphasis: false,
      points: bench.pr.map((d) => ({ x: d.recall, y: d.precision })), operatingPoint: { x: bench.operating_point.recall, y: bench.operating_point.precision } }] : []),
  ];
  const sel = (fs: string) => p.models.find((m) => m.feature_set === fs && m.selected)!;
  const curveRows = [p.default_variant, "with_page_values"].filter((v) => p.curves[v]).map((v) => ({ v, m: sel(v) }));
  const curveCols: Column<{ v: string; m: ModelRow }>[] = [
    { key: "v", header: "Model", render: (r) => `${r.m.label} ${featureSetLabel(r.v).toLowerCase()}` },
    { key: "roc", header: "ROC-AUC", numeric: true, render: (r) => fmtNum(r.m.test.roc_auc) },
    { key: "pr", header: "PR-AUC", numeric: true, render: (r) => fmtNum(r.m.test.pr_auc) },
    { key: "op", header: "Recall / precision at threshold", numeric: true, render: (r) => `${fmtNum(r.m.test.recall)} / ${fmtNum(r.m.test.precision)}` },
  ];

  return (
    <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
      <ChartCard title="ROC curve" description={`Test set (${fmtInt(p.split.n_test)} sessions). Dots mark each model's decision threshold. Diagonal: no skill.`}
                 columns={curveCols} rows={curveRows}>
        <CurveChart series={roc} xLabel="False positive rate" yLabel="True positive rate" diagonal height={320} />
      </ChartCard>
      <ChartCard title="Precision–recall curve" description="Better suited to imbalanced data. Grey line: no-skill precision (the purchase rate)."
                 columns={curveCols} rows={curveRows}>
        <CurveChart series={pr} xLabel="Recall" yLabel="Precision" baselineY={p.baseline.pr_auc} height={320} />
      </ChartCard>
    </div>
  );
}

function Calibration({ p }: { p: ModelPerformance }) {
  const c = p.calibration[p.default_variant];
  if (!c) return null;
  const series = [
    { key: "cal", name: `Calibrated (${c.method})`, emphasis: true, dots: true,
      points: c.reliability_training_cv.calibrated.map((r) => ({ x: r.mean_predicted, y: r.observed_rate })) },
    { key: "raw", name: "Raw model score", emphasis: false, dots: true,
      points: c.reliability_training_cv.raw.map((r) => ({ x: r.mean_predicted, y: r.observed_rate })) },
  ];
  const rows = [
    { k: "Brier score (lower is better)", raw: c.raw_training_cv.brier, cal: c.training_cv.brier },
    { k: "Expected calibration error", raw: c.raw_training_cv.ece, cal: c.training_cv.ece },
    { k: "Mean predicted (observed " + fmtPct(c.training_cv.observed_rate) + ")", raw: c.raw_training_cv.mean_predicted, cal: c.training_cv.mean_predicted },
  ];
  return (
    <ChartCard
      title="Probability calibration"
      description="Training-split cross-validation (test data not used). Points on the diagonal are perfectly calibrated."
      columns={[
        { key: "k", header: "Metric", render: (r: (typeof rows)[number]) => r.k },
        { key: "raw", header: "Raw score", numeric: true, render: (r: (typeof rows)[number]) => fmtNum(r.raw, 4) },
        { key: "cal", header: "Calibrated", numeric: true, render: (r: (typeof rows)[number]) => fmtNum(r.cal, 4) },
      ]}
      rows={rows}
      footnote={
        <>
          Status: <strong className="text-ink">{c.status}</strong>. Brier {fmtNum(c.raw_training_cv.brier)} → {fmtNum(c.training_cv.brier)} (base rate {fmtNum(c.training_cv.brier_base_rate)}).
          {c.heldout_check.calibrated && <> Held-out check, reported after the decision: Brier {fmtNum(c.heldout_check.raw_scores.brier)} → {fmtNum(c.heldout_check.calibrated.brier)}.</>}
        </>
      }
    >
      <CurveChart series={series} xLabel="Mean predicted value" yLabel="Observed purchase rate" diagonal height={300} />
    </ChartCard>
  );
}

export default function PerformanceView() {
  const perf = useApi(getModelPerformance);
  const info = useApi(getModelInfo);
  const p = perf.state.data;
  const served = p?.models.find((m) => m.feature_set === p.default_variant && m.selected);

  return (
    <>
      <PageHeader
        title="Model performance"
        description="How well the served Random Forest separates purchasers from non-purchasers on a held-out test set it never saw during training, selection or calibration."
      />

      {perf.state.status === "error" && !p && <ErrorState error={perf.state.error} onRetry={perf.retry} />}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {p && served ? (
          <>
            <KpiCard accent label="PR-AUC (selection metric)" value={fmtNum(served.test.pr_auc)} detail={`No-skill baseline ${fmtNum(p.baseline.pr_auc)}`} />
            <KpiCard label="ROC-AUC" value={fmtNum(served.test.roc_auc)} detail="No-skill baseline 0.500" />
            <KpiCard label="F1 score" value={fmtNum(served.test.f1)} detail={`At threshold ${fmtNum(served.threshold)}`} />
            <KpiCard label="Recall · Precision" value={`${fmtPct(served.test.recall, 0)} · ${fmtPct(served.test.precision, 0)}`}
                     detail="Purchasers found · flagged sessions that bought" />
          </>
        ) : (
          perf.state.status === "loading" && Array.from({ length: 4 }, (_, i) => <KpiSkeleton key={i} />)
        )}
      </div>

      {p && served && (
        <>
          <div className="mt-4">
            <Notice tone="brand" title="Why not accuracy?">
              Only {fmtPct(p.split.test_purchase_rate)} of sessions end in a purchase, so always predicting &ldquo;no purchase&rdquo; scores{" "}
              {fmtPct(p.baseline.accuracy)} accuracy while finding no buyers. The served model&rsquo;s accuracy is {fmtPct(served.test.accuracy)}, but it
              finds {fmtPct(served.test.recall)} of purchasers. Models were selected by cross-validated PR-AUC on the training split.
            </Notice>
          </div>

          <Curves p={p} />

          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader title="Confusion matrix" description={`Served model on ${fmtInt(p.split.n_test)} test sessions at threshold ${fmtNum(served.threshold)}.`} />
              <div className="px-5 pb-5 pt-4">
                <ConfusionMatrixGrid {...served.confusion_matrix} />
                <p className="mt-3 text-sm leading-6 text-ink-2">
                  Of {fmtInt(served.confusion_matrix.tp + served.confusion_matrix.fn)} actual purchasers, the model flagged{" "}
                  {fmtInt(served.confusion_matrix.tp)}. Of {fmtInt(served.confusion_matrix.tp + served.confusion_matrix.fp)} flagged sessions,{" "}
                  {fmtInt(served.confusion_matrix.tp)} purchased, so roughly {Math.round(served.confusion_matrix.fp / served.confusion_matrix.tp)} false
                  alarms for every correct one.
                </p>
              </div>
            </Card>
            <Calibration p={p} />
          </div>

          <Card className="mt-4">
            <CardHeader title="Model comparison" description="All candidates trained in Phase 3. Test metrics at each model's tuned threshold; selection used CV PR-AUC only." />
            <div className="px-5 pb-5 pt-4">
              <DataTable columns={modelColumns} rows={p.models} caption="Model comparison" />
              <p className="mt-3 text-xs text-ink-3">
                Baseline (always &ldquo;no purchase&rdquo;): accuracy {fmtNum(p.baseline.accuracy)}, PR-AUC {fmtNum(p.baseline.pr_auc)}, F1 {fmtNum(p.baseline.f1)}.
              </p>
            </div>
          </Card>

          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader title="Why PageValues is excluded" />
              <div className="space-y-3 px-5 pb-5 pt-3 text-sm leading-6 text-ink-2">
                <p>
                  PageValues is defined as the average value of pages a user visited <em>before completing a transaction</em>. If it was
                  computed after a session ended, it partly encodes the purchase itself, which is outcome leakage.
                </p>
                <ul className="list-disc space-y-1 pl-5">
                  <li>{fmtPct(p.page_values.share_nonzero_purchase)} of purchasing sessions have PageValues &gt; 0, vs {fmtPct(p.page_values.share_nonzero_no_purchase)} of non-purchasing sessions.</li>
                  <li>Conversion is {fmtPct(p.page_values.conversion_when_nonzero)} when PageValues &gt; 0 and {fmtPct(p.page_values.conversion_when_zero)} when it is 0.</li>
                  <li>Including it raises test PR-AUC from {fmtNum(p.page_values.selected_test_pr_auc_without)} to {fmtNum(p.page_values.selected_test_pr_auc_with)}.</li>
                </ul>
                <p>
                  The dataset cannot show whether this is genuine signal or leakage, so the served model excludes it. The PageValues model is a
                  benchmark, valid only if PageValues comes from historical page values that exclude the current session.
                </p>
              </div>
            </Card>
            <Card>
              <CardHeader title="Limitations" description="When predictions are valid." />
              <div className="px-5 pb-5 pt-3">
                {info.state.status === "loading" && <LoadingBlock height={180} label="Loading limitations" />}
                {info.state.status === "error" && <ErrorState error={info.state.error} onRetry={info.retry} compact />}
                {info.state.status === "success" && (
                  <ul className="list-disc space-y-1.5 pl-5 text-sm leading-6 text-ink-2">
                    {info.state.data.limitations.map((l) => (
                      <li key={l}>{l}</li>
                    ))}
                  </ul>
                )}
              </div>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
