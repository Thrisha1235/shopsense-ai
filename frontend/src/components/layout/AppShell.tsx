"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import {
  IconAnalytics,
  IconClose,
  IconExplain,
  IconGlobe,
  IconLogo,
  IconMenu,
  IconOverview,
  IconPerformance,
  IconPredict,
} from "@/components/icons";
import ApiStatus from "./ApiStatus";

const NAV = [
  { href: "/", label: "Overview", icon: IconOverview },
  { href: "/analytics", label: "Customer behaviour", icon: IconAnalytics },
  { href: "/predict", label: "Purchase prediction", icon: IconPredict },
  { href: "/performance", label: "Model performance", icon: IconPerformance },
  { href: "/explainability", label: "Explainable AI", icon: IconExplain },
  { href: "/website-intelligence", label: "Website intelligence", icon: IconGlobe },
] as const;

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <ul className="space-y-1">
      {NAV.map((item) => {
        const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
        const Icon = item.icon;
        return (
          <li key={item.href}>
            <Link
              href={item.href}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              className={`group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500 ${
                active
                  ? "bg-brand-50 text-brand-600"
                  : "text-ink-2 hover:bg-slate-100 hover:text-ink"
              }`}
            >
              <Icon className={active ? "text-brand-500" : "text-ink-3 group-hover:text-ink-2"} />
              <span className="flex-1">{item.label}</span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2.5 rounded-md focus-visible:outline-2 focus-visible:outline-brand-500">
      <IconLogo />
      <span className="text-base font-semibold tracking-tight text-ink">ShopSense AI</span>
    </Link>
  );
}

function SidebarFooter() {
  return (
    <div className="border-t border-line px-5 py-4 text-xs leading-5 text-ink-3">
      Data: UCI Online Shoppers Purchasing Intention Dataset (ID 468).
    </div>
  );
}

const TITLES: Record<string, string> = Object.fromEntries(NAV.map((n) => [n.href, n.label]));

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const title = TITLES[pathname] ?? "ShopSense AI";

  // Close the mobile drawer with Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <div className="min-h-screen bg-page">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-surface focus:px-4 focus:py-2 focus:text-sm focus:shadow"
      >
        Skip to content
      </a>

      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col border-r border-line bg-surface lg:flex">
        <div className="flex h-16 items-center px-5">
          <Brand />
        </div>
        <nav aria-label="Main" className="flex-1 overflow-y-auto px-3 py-4">
          <NavLinks />
        </nav>
        <SidebarFooter />
      </aside>

      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button
            type="button"
            className="absolute inset-0 bg-slate-900/30"
            aria-label="Close navigation"
            onClick={() => setOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 flex w-72 flex-col bg-surface shadow-xl">
            <div className="flex h-16 items-center justify-between px-5">
              <Brand />
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md p-2 text-ink-2 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-brand-500"
                aria-label="Close navigation"
              >
                <IconClose />
              </button>
            </div>
            <nav aria-label="Main" className="flex-1 overflow-y-auto px-3 py-4">
              <NavLinks onNavigate={() => setOpen(false)} />
            </nav>
            <SidebarFooter />
          </div>
        </div>
      )}

      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-line bg-surface/90 px-4 backdrop-blur sm:px-6 lg:px-8">
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="rounded-md p-2 text-ink-2 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-brand-500 lg:hidden"
            aria-label="Open navigation"
            aria-expanded={open}
          >
            <IconMenu />
          </button>
          <p className="truncate text-sm font-medium text-ink-2">{title}</p>
          <div className="ml-auto">
            <ApiStatus />
          </div>
        </header>
        <main id="main" className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {children}
        </main>
      </div>
    </div>
  );
}
