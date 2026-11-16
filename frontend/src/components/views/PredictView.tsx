"use client";

import Link from "next/link";
import { useId, useRef, useState, type SubmitEvent } from "react";

import { IconAlert, IconCheck } from "@/components/icons";
import {
  ALL_FIELDS,
  EMPTY,
  EXAMPLE,
  FIELD_GROUPS,
  QUICK_BOUNCE,
  toSession,
  validateField,
  type Field,
  type FieldKey,
  type FormValues,
} from "@/components/predict/fields";
import { Badge, Card, CardHeader, Notice, PageHeader } from "@/components/ui/Card";
import { EmptyState, ErrorState, LoadingBlock } from "@/components/ui/States";
import { ApiError, getModelInfo, predict } from "@/lib/api";
import { featureSetLabel, fmtPct, fmtSecondsHint } from "@/lib/format";
import type { ModelInfo, PredictionResult } from "@/lib/types";
import { saveLastSession } from "@/lib/sessionHandoff";
import { useApi } from "@/lib/useApi";

const inputCls =
  "block w-full rounded-lg border bg-surface px-3 py-2 text-sm text-ink shadow-sm placeholder:text-ink-3 focus:outline-none focus:ring-2 focus:ring-brand-500/40";

function FieldControl({
  field,
  value,
  error,
  info,
  onChange,
  onBlur,
}: {
  field: Field;
  value: string;
  error: string | null;
  info: ModelInfo;
  onChange: (v: string) => void;
  onBlur: () => void;
}) {
  const id = useId();
  const helpId = `${id}-help`;
  const errId = `${id}-err`;
  const describedBy = [helpId, error ? errId : null].filter(Boolean).join(" ");
  const border = error ? "border-bad" : "border-line-strong";

  if (field.kind === "toggle") {
    const on = value === "true";
    return (
      <div className="sm:col-span-1">
        <span id={`${id}-label`} className="block text-sm font-medium text-ink">{field.label}</span>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-labelledby={`${id}-label`}
          aria-describedby={helpId}
          onClick={() => onChange(on ? "false" : "true")}
          className="mt-1.5 inline-flex items-center gap-3 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
        >
          <span className={`relative h-6 w-11 rounded-full transition-colors ${on ? "bg-brand-500" : "bg-slate-300"}`}>
            <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${on ? "translate-x-5" : "translate-x-0.5"}`} />
          </span>
          <span className="text-sm text-ink-2">{on ? "Yes" : "No"}</span>
        </button>
        <p id={helpId} className="mt-1.5 text-xs leading-5 text-ink-3">{field.help}</p>
      </div>
    );
  }

  let control;
  if (field.kind === "select") {
    let options = field.options === "known" ? [] : field.options;
    if (field.options === "known") {
      const known = info.known_categories[field.key] ?? [];
      options = known.map((v) => ({ value: String(v), label: field.key === "Month" ? String(v) : `Code ${v}` }));
      if (field.key === "Month") {
        const order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        options.sort((a, b) => order.indexOf(a.value) - order.indexOf(b.value));
      } else {
        options.sort((a, b) => Number(a.value) - Number(b.value));
      }
    }
    control = (
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
        aria-invalid={!!error}
        aria-describedby={describedBy}
        className={`${inputCls} ${border}`}
      >
        <option value="" disabled>
          Select…
        </option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    );
  } else {
    const unit = field.kind === "seconds" ? "s" : field.kind === "percent" ? "%" : null;
    control = (
      <div className="relative">
        <input
          id={id}
          type="number"
          inputMode={field.kind === "int" ? "numeric" : "decimal"}
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onBlur={onBlur}
          aria-invalid={!!error}
          aria-describedby={describedBy}
          className={`${inputCls} ${border} ${unit ? "pr-9" : ""} tabular`}
        />
        {unit && <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-sm text-ink-3">{unit}</span>}
      </div>
    );
  }

  const hint = field.kind === "seconds" ? fmtSecondsHint(Number(value)) : "";
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-ink">
        {field.label}
      </label>
      <div className="mt-1.5">{control}</div>
      {error ? (
        <p id={errId} className="mt-1.5 text-xs font-medium text-bad">{error}</p>
      ) : null}
      <p id={helpId} className="mt-1.5 text-xs leading-5 text-ink-3">
        {field.help}
        {hint && <span className="ml-1 text-ink-2">({hint})</span>}
      </p>
    </div>
  );
}

function ProbabilityMeter({ value, threshold }: { value: number; threshold: number | null }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="mt-4" aria-hidden="true">
      <div className="relative h-2.5 rounded-full bg-slate-100">
        <div className="absolute inset-y-0 left-0 rounded-full bg-brand-500" style={{ width: `${pct}%` }} />
        {threshold !== null && (
          <div className="absolute -top-1 h-4.5 w-0.5 rounded bg-ink" style={{ left: `${threshold * 100}%` }} title="Decision threshold" />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-ink-3 tabular">
        <span>0%</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function ResultCard({ result, pending }: { result: PredictionResult; pending: boolean }) {
  const prob = result.purchase_probability;
  const thr = result.decision_threshold;
  return (
    <div className={pending ? "opacity-60 transition-opacity" : "transition-opacity"}>
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-ink-3">Predicted class</p>
        {result.predicted_purchase ? (
          <Badge tone="good"><IconCheck width={14} height={14} /> Likely to purchase</Badge>
        ) : (
          <Badge tone="neutral">Unlikely to purchase</Badge>
        )}
      </div>

      {prob !== null ? (
        <div className="mt-5">
          <p className="text-sm font-medium text-ink-3">Calibrated purchase probability</p>
          <p className="mt-1 text-4xl font-semibold tracking-tight text-ink">{fmtPct(prob)}</p>
          <ProbabilityMeter value={prob} threshold={thr.purchase_probability} />
          <p className="mt-2 text-xs leading-5 text-ink-3">
            The black mark is the decision threshold ({thr.purchase_probability !== null ? fmtPct(thr.purchase_probability) : "n/a"}).
            Sessions at or above it are classed as likely purchases.
          </p>
        </div>
      ) : (
        <div className="mt-5">
          <p className="text-sm font-medium text-ink-3">Model score</p>
          <p className="mt-1 text-4xl font-semibold tracking-tight text-ink">{result.model_score.toFixed(3)}</p>
          <Notice tone="warn" title="Not a probability">
            Calibration is not available, so only the raw model score is shown. Use it to rank sessions, not as a
            chance of purchase.
          </Notice>
        </div>
      )}

      <dl className="mt-5 divide-y divide-line border-y border-line text-sm">
        <div className="flex justify-between gap-4 py-2.5">
          <dt className="text-ink-3">Raw model score</dt>
          <dd className="tabular font-medium text-ink">{result.model_score.toFixed(4)}</dd>
        </div>
        <div className="flex justify-between gap-4 py-2.5">
          <dt className="text-ink-3">Threshold (score)</dt>
          <dd className="tabular font-medium text-ink">{thr.model_score.toFixed(4)}</dd>
        </div>
        <div className="flex justify-between gap-4 py-2.5">
          <dt className="text-ink-3">Model</dt>
          <dd className="text-right font-medium text-ink">
            {result.model.algorithm.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())} ·{" "}
            {featureSetLabel(result.model.variant).replace(/^W/, "w")}
          </dd>
        </div>
      </dl>
      <p className="mt-2 text-xs leading-5 text-ink-3">
        The raw score ranks sessions but is not a probability; class weighting inflates it.
      </p>
      <Link
        href="/explainability"
        className="mt-4 inline-flex w-full items-center justify-center rounded-lg border border-brand-200 bg-brand-50 px-4 py-2 text-sm font-medium text-brand-600 hover:bg-brand-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
      >
        Explain this prediction
      </Link>

      {result.warnings.length > 0 && (
        <div className="mt-4">
          <Notice tone="warn" title={`Warnings (${result.warnings.length})`}>
            <ul className="mt-1 list-disc space-y-1 pl-5">
              {result.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </Notice>
        </div>
      )}
    </div>
  );
}

function PredictionForm({ info }: { info: ModelInfo }) {
  const [values, setValues] = useState<FormValues>(EXAMPLE);
  const [errors, setErrors] = useState<Partial<Record<FieldKey, string>>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [pending, setPending] = useState(false);
  const [failure, setFailure] = useState<Error | null>(null);
  const summaryRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const setValue = (key: FieldKey, v: string) => {
    setValues((prev) => ({ ...prev, [key]: v }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: undefined }));
  };

  const blur = (field: Field) => {
    const err = validateField(field, values[field.key]);
    setErrors((e) => ({ ...e, [field.key]: err ?? undefined }));
  };

  const load = (preset: FormValues) => {
    setValues(preset);
    setErrors({});
    setFailure(null);
  };

  async function onSubmit(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    const found: Partial<Record<FieldKey, string>> = {};
    for (const f of ALL_FIELDS) {
      const err = validateField(f, values[f.key]);
      if (err) found[f.key] = err;
    }
    setErrors(found);
    if (Object.keys(found).length) {
      summaryRef.current?.focus();
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setPending(true);
    setFailure(null);
    try {
      const session = toSession(values);
      setResult(await predict(session, controller.signal));
      saveLastSession(session); // lets the Explainable AI page explain this exact session
      // On narrow screens the result card sits below the form: bring it into view.
      if (window.matchMedia("(max-width: 1279px)").matches) {
        requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      if (err instanceof ApiError && err.status === 422) {
        const serverErrors: Partial<Record<FieldKey, string>> = {};
        for (const issue of err.issues) {
          const key = issue.loc[issue.loc.length - 1] as FieldKey;
          serverErrors[key] = issue.msg;
        }
        setErrors(serverErrors);
      }
      setFailure(err instanceof Error ? err : new Error(String(err)));
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }

  const errorCount = Object.values(errors).filter(Boolean).length;
  const labelOf = (k: FieldKey) => ALL_FIELDS.find((f) => f.key === k)?.label ?? k;

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
      <form onSubmit={onSubmit} noValidate className="xl:col-span-2" aria-label="Session features">
        <Card>
          <CardHeader
            title="Session details"
            description="Enter the aggregated activity of one completed browsing session."
            action={
              <div className="flex flex-wrap gap-2">
                <button type="button" onClick={() => load(EXAMPLE)} className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink-2 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-brand-500">
                  Sample: engaged visitor
                </button>
                <button type="button" onClick={() => load(QUICK_BOUNCE)} className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink-2 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-brand-500">
                  Sample: quick bounce
                </button>
                <button type="button" onClick={() => load(EMPTY)} className="rounded-lg px-3 py-1.5 text-xs font-medium text-ink-3 hover:bg-slate-50 hover:text-ink focus-visible:outline-2 focus-visible:outline-brand-500">
                  Clear
                </button>
              </div>
            }
          />

          <div ref={summaryRef} tabIndex={-1} className="px-5 pt-4 focus:outline-none" aria-live="assertive">
            {errorCount > 0 && (
              <Notice tone="bad" title={`Please fix ${errorCount} field${errorCount > 1 ? "s" : ""}`}>
                <ul className="mt-1 list-disc pl-5">
                  {(Object.entries(errors) as [FieldKey, string | undefined][])
                    .filter(([, v]) => v)
                    .map(([k, v]) => (
                      <li key={k}>
                        {labelOf(k)}: {v}
                      </li>
                    ))}
                </ul>
              </Notice>
            )}
          </div>

          <div className="space-y-8 px-5 pb-5 pt-4">
            {FIELD_GROUPS.map((g) => (
              <fieldset key={g.title}>
                <legend className="text-sm font-semibold text-ink">{g.title}</legend>
                <p className="mt-0.5 text-xs text-ink-3">{g.description}</p>
                <div className="mt-4 grid grid-cols-1 gap-x-5 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
                  {g.fields.map((f) => (
                    <FieldControl
                      key={f.key}
                      field={f}
                      value={values[f.key]}
                      error={errors[f.key] ?? null}
                      info={info}
                      onChange={(v) => setValue(f.key, v)}
                      onBlur={() => blur(f)}
                    />
                  ))}
                </div>
              </fieldset>
            ))}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-4">
            <p className="text-xs text-ink-3">PageValues is not needed: the default model excludes it.</p>
            <button
              type="submit"
              disabled={pending}
              className="inline-flex items-center justify-center rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-brand-700 disabled:cursor-wait disabled:opacity-70 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
            >
              {pending ? "Predicting…" : "Predict purchase"}
            </button>
          </div>
        </Card>
      </form>

      <div ref={resultRef} className="scroll-mt-20 xl:sticky xl:top-22 xl:self-start">
        <Card>
          <CardHeader title="Prediction result" />
          <div className="px-5 pb-5 pt-4" aria-live="polite" aria-busy={pending}>
            {failure && !(failure instanceof ApiError && failure.status === 422) && (
              <div className="mb-4">
                <ErrorState error={failure} compact />
              </div>
            )}
            {result ? (
              <ResultCard result={result} pending={pending} />
            ) : pending ? (
              <LoadingBlock height={260} label="Scoring session" />
            ) : (
              <EmptyState title="No prediction yet">
                Fill in the session details and select <strong>Predict purchase</strong>. A sample session is pre-filled.
              </EmptyState>
            )}
          </div>
        </Card>
        <p className="mt-3 px-1 text-xs leading-5 text-ink-3">
          Valid for completed sessions similar to the training data. See{" "}
          <Link href="/performance" className="font-medium text-brand-600 underline-offset-2 hover:underline">
            model performance and limitations
          </Link>
          .
        </p>
      </div>
    </div>
  );
}

export default function PredictView() {
  const { state, retry } = useApi(getModelInfo);
  return (
    <>
      <PageHeader
        title="Purchase prediction"
        description="Score a browsing session with the calibrated Random Forest model served by the FastAPI backend."
      >
        {state.status === "success" && (
          state.data.calibration.probability_available ? (
            <Badge tone="good">Probabilities calibrated</Badge>
          ) : (
            <Badge tone="warn"><IconAlert width={14} height={14} /> Score only (uncalibrated)</Badge>
          )
        )}
      </PageHeader>
      {state.status === "loading" && <LoadingBlock height={520} label="Loading model information" />}
      {state.status === "error" && <ErrorState error={state.error} onRetry={retry} />}
      {state.status === "success" && <PredictionForm info={state.data} />}
    </>
  );
}
