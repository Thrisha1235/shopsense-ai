import type { Metadata } from "next";

import PerformanceView from "@/components/views/PerformanceView";

export const metadata: Metadata = { title: "Model performance" };

export default function PerformancePage() {
  return <PerformanceView />;
}
