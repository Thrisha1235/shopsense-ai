import type { Metadata } from "next";

import AnalyticsView from "@/components/views/AnalyticsView";

export const metadata: Metadata = { title: "Customer behaviour" };

export default function AnalyticsPage() {
  return <AnalyticsView />;
}
