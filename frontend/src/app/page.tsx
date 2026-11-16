import type { Metadata } from "next";

import OverviewView from "@/components/views/OverviewView";

// The layout's title template only applies to child segments, so the home page sets its full title.
export const metadata: Metadata = { title: { absolute: "Overview · ShopSense AI" } };

export default function Home() {
  return <OverviewView />;
}
