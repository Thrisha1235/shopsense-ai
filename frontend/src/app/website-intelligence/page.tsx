import type { Metadata } from "next";

import WebsiteIntelligenceView from "@/components/views/WebsiteIntelligenceView";

export const metadata: Metadata = { title: "Website intelligence" };

export default function WebsiteIntelligencePage() {
  return <WebsiteIntelligenceView />;
}
