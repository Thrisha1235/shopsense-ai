"use client";

import { useCallback, useEffect, useState } from "react";

export type ApiState<T> =
  | { status: "loading"; data?: T }
  | { status: "success"; data: T }
  | { status: "error"; error: Error; data?: T };

/**
 * Fetch on mount with abort-on-unmount and a retry function.
 * On refetch the previous data is kept (no skeleton flash).
 */
export function useApi<T>(fetcher: (signal: AbortSignal) => Promise<T>) {
  const [state, setState] = useState<ApiState<T>>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetcher(controller.signal)
      .then((data) => setState({ status: "success", data }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState((prev) => ({
          status: "error",
          error: error instanceof Error ? error : new Error(String(error)),
          data: prev.data,
        }));
      });
    return () => controller.abort();
  }, [fetcher, attempt]);

  const retry = useCallback(() => {
    setState((prev) => ({ status: "loading", data: prev.data }));
    setAttempt((a) => a + 1);
  }, []);

  return { state, retry };
}
