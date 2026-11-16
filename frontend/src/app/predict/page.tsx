import type { Metadata } from "next";

import PredictView from "@/components/views/PredictView";

export const metadata: Metadata = { title: "Purchase prediction" };

export default function PredictPage() {
  return <PredictView />;
}
