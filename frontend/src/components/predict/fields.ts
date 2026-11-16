import type { SessionInput } from "@/lib/types";

export type FieldKey = keyof SessionInput;

export interface NumberField {
  kind: "int" | "seconds" | "percent";
  key: FieldKey;
  label: string;
  help: string;
  min: number;
  max: number;
  step?: number;
}

export interface SelectField {
  kind: "select";
  key: FieldKey;
  label: string;
  help: string;
  /** "known" = values come from /api/model-info known_categories */
  options: { value: string; label: string }[] | "known";
}

export interface ToggleField {
  kind: "toggle";
  key: FieldKey;
  label: string;
  help: string;
}

export type Field = NumberField | SelectField | ToggleField;

const MAX_PAGES = 10_000;
const MAX_SECONDS = 86_400;

export const FIELD_GROUPS: { title: string; description: string; fields: Field[] }[] = [
  {
    title: "Pages visited",
    description: "How many pages of each type the session viewed.",
    fields: [
      { kind: "int", key: "Administrative", label: "Administrative pages", help: "Account, login and settings pages.", min: 0, max: MAX_PAGES },
      { kind: "int", key: "Informational", label: "Informational pages", help: "About, contact, shipping and policy pages.", min: 0, max: MAX_PAGES },
      { kind: "int", key: "ProductRelated", label: "Product pages", help: "Product listing and detail pages.", min: 0, max: MAX_PAGES },
    ],
  },
  {
    title: "Time on pages",
    description: "Total seconds spent on each page type during the session.",
    fields: [
      { kind: "seconds", key: "Administrative_Duration", label: "Administrative time", help: "Seconds on administrative pages.", min: 0, max: MAX_SECONDS, step: 1 },
      { kind: "seconds", key: "Informational_Duration", label: "Informational time", help: "Seconds on informational pages.", min: 0, max: MAX_SECONDS, step: 1 },
      { kind: "seconds", key: "ProductRelated_Duration", label: "Product-page time", help: "Seconds on product pages.", min: 0, max: MAX_SECONDS, step: 1 },
    ],
  },
  {
    title: "Engagement",
    description: "Google Analytics metrics averaged over the pages visited.",
    fields: [
      { kind: "percent", key: "BounceRates", label: "Bounce rate", help: "Share of visits to these pages that left after one page. Training data ranges 0–20%.", min: 0, max: 100, step: 0.01 },
      { kind: "percent", key: "ExitRates", label: "Exit rate", help: "Share of page views that were the last in a session. Training data ranges 0–20%.", min: 0, max: 100, step: 0.01 },
      {
        kind: "select", key: "SpecialDay", label: "Closeness to a special day",
        help: "How close the visit was to a special day such as Mother's Day (0 = not close, 1 = the day itself).",
        options: [0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => ({ value: String(v), label: v === 0 ? "0 — not near a special day" : v === 1 ? "1.0 — on the special day" : v.toFixed(1) })),
      },
    ],
  },
  {
    title: "Visit context",
    description: "When the visit happened and who the visitor was.",
    fields: [
      { kind: "select", key: "Month", label: "Month", help: "Month of the visit.", options: "known" },
      {
        kind: "select", key: "VisitorType", label: "Visitor type", help: "Whether the visitor had been to the site before.",
        options: [
          { value: "Returning_Visitor", label: "Returning visitor" },
          { value: "New_Visitor", label: "New visitor" },
          { value: "Other", label: "Other" },
        ],
      },
      { kind: "toggle", key: "Weekend", label: "Weekend visit", help: "The session took place on a Saturday or Sunday." },
    ],
  },
  {
    title: "Technical and traffic",
    description: "Anonymised codes from the source dataset; only codes seen in training are offered.",
    fields: [
      { kind: "select", key: "OperatingSystems", label: "Operating system code", help: "Anonymised operating-system code.", options: "known" },
      { kind: "select", key: "Browser", label: "Browser code", help: "Anonymised browser code.", options: "known" },
      { kind: "select", key: "Region", label: "Region code", help: "Anonymised visitor region.", options: "known" },
      { kind: "select", key: "TrafficType", label: "Traffic source code", help: "Anonymised traffic source (referral, campaign, ...).", options: "known" },
    ],
  },
];

export type FormValues = Record<FieldKey, string>;

/** Example session (same values as the API documentation example). */
export const EXAMPLE: FormValues = {
  Administrative: "2", Administrative_Duration: "45", Informational: "0", Informational_Duration: "0",
  ProductRelated: "30", ProductRelated_Duration: "1200", BounceRates: "1", ExitRates: "3", SpecialDay: "0",
  Month: "Nov", OperatingSystems: "2", Browser: "2", Region: "1", TrafficType: "2",
  VisitorType: "Returning_Visitor", Weekend: "false",
};

/** A single-page visit that leaves immediately (sample input, not a result). */
export const QUICK_BOUNCE: FormValues = {
  ...EXAMPLE,
  Administrative: "0", Administrative_Duration: "0", ProductRelated: "1", ProductRelated_Duration: "0",
  BounceRates: "20", ExitRates: "20", Month: "May",
};

export const EMPTY: FormValues = Object.fromEntries(
  Object.keys(EXAMPLE).map((k) => [k, ""]),
) as FormValues;

export function validateField(field: Field, raw: string): string | null {
  if (field.kind === "toggle") return null;
  if (raw.trim() === "") return "Required.";
  if (field.kind === "select") return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return "Enter a number.";
  if (field.kind === "int" && !Number.isInteger(n)) return "Enter a whole number.";
  if (n < field.min) return `Must be at least ${field.min}.`;
  if (n > field.max) return `Must be at most ${field.max.toLocaleString("en-US")}${field.kind === "percent" ? "%" : ""}.`;
  return null;
}

export function toSession(values: FormValues): SessionInput {
  const num = (k: FieldKey) => Number(values[k]);
  return {
    Administrative: num("Administrative"),
    Administrative_Duration: num("Administrative_Duration"),
    Informational: num("Informational"),
    Informational_Duration: num("Informational_Duration"),
    ProductRelated: num("ProductRelated"),
    ProductRelated_Duration: num("ProductRelated_Duration"),
    BounceRates: num("BounceRates") / 100,
    ExitRates: num("ExitRates") / 100,
    SpecialDay: num("SpecialDay"),
    Month: values.Month,
    OperatingSystems: num("OperatingSystems"),
    Browser: num("Browser"),
    Region: num("Region"),
    TrafficType: num("TrafficType"),
    VisitorType: values.VisitorType as SessionInput["VisitorType"],
    Weekend: values.Weekend === "true",
  };
}

export const ALL_FIELDS: Field[] = FIELD_GROUPS.flatMap((g) => g.fields);
