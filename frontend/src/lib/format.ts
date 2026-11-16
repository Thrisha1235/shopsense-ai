const intFmt = new Intl.NumberFormat("en-US");

export const fmtInt = (n: number) => intFmt.format(Math.round(n));

export const fmtPct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;

export const fmtNum = (x: number, digits = 3) => x.toFixed(digits);

export const fmtMinutes = (m: number) => `${m.toFixed(1)} min`;

export function fmtSecondsHint(s: number): string {
  if (!Number.isFinite(s) || s <= 0) return "";
  if (s < 60) return `${s.toFixed(0)} s`;
  if (s < 3600) return `≈ ${(s / 60).toFixed(1)} min`;
  return `≈ ${(s / 3600).toFixed(1)} h`;
}

export const visitorLabel = (v: string) =>
  ({ Returning_Visitor: "Returning visitor", New_Visitor: "New visitor", Other: "Other" })[v] ?? v;

/** Human-readable input value for a model feature (used in explanation charts and tables). */
export function formatFeatureValue(feature: string, value: boolean | number | string | null): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (feature.endsWith("_Duration") && typeof value === "number") {
    const hint = fmtSecondsHint(value);
    return `${fmtInt(value)} s${hint && value >= 60 ? ` (${hint.replace("≈ ", "")})` : ""}`;
  }
  if ((feature === "BounceRates" || feature === "ExitRates") && typeof value === "number") return fmtPct(value, 2);
  if (feature === "VisitorType") return visitorLabel(String(value));
  if (["OperatingSystems", "Browser", "Region", "TrafficType"].includes(feature)) return `Code ${value}`;
  if (typeof value === "number") return Number.isInteger(value) ? fmtInt(value) : String(value);
  return String(value);
}

/** Signed raw-score points, e.g. +0.034 */
export const fmtSigned = (x: number, digits = 3) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;

export const featureSetLabel = (fs: string) =>
  fs === "with_page_values" ? "With PageValues" : "Without PageValues";
