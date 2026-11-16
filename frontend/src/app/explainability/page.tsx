import type { Metadata } from "next";

import ExplainabilityView from "@/components/views/ExplainabilityView";

export const metadata: Metadata = { title: "Explainable AI" };

export default function ExplainabilityPage() {
  return <ExplainabilityView />;
}
