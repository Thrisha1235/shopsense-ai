import type { SessionInput } from "./types";

// Passes the last scored session from the prediction page to the Explainable AI page.
// sessionStorage only: it stays in this browser tab and is never sent anywhere else.
const KEY = "shopsense:last-session";

export function saveLastSession(session: SessionInput): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(session));
  } catch {
    /* storage unavailable (private mode): the explanation page falls back to samples */
  }
}

export function loadLastSession(): SessionInput | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as SessionInput) : null;
  } catch {
    return null;
  }
}
