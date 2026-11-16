// Chart colours. Series 1 (blue) is the validated reference palette's first slot.
// "Rest of the data" uses a muted neutral (emphasis pattern: highlight one, grey the rest).
export const chart = {
  primary: "#2a78d6",
  primaryLight: "#9ec5f4",
  muted: "#c4c9d2",
  reference: "#94a3b8",
  grid: "#e9ebef",
  axis: "#cbd2dc",
  ink: "#0f172a",
  ink2: "#475569",
} as const;

// Sequential blue ramp (light -> dark) for the confusion matrix.
export const blueRamp = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"];

export const axisTick = { fill: "#64748b", fontSize: 12 };
