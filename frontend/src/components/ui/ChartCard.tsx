"use client";

import { useId, useState, type ReactNode } from "react";

import { Card, CardHeader } from "./Card";

export interface Column<Row> {
  key: string;
  header: string;
  render: (row: Row) => ReactNode;
  numeric?: boolean;
}

export function DataTable<Row>({ columns, rows, caption }: { columns: Column<Row>[]; rows: Row[]; caption: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-ink-3">
            {columns.map((c) => (
              <th key={c.key} scope="col" className={`px-3 py-2 font-medium ${c.numeric ? "text-right" : ""}`}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-line/70 last:border-0">
              {columns.map((c) => (
                <td key={c.key} className={`px-3 py-2 text-ink-2 ${c.numeric ? "tabular whitespace-nowrap text-right" : ""}`}>
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * A chart with a "Table" twin: every value in the chart is also readable as text
 * (the accessible equivalent; tooltips enhance but never gate the data).
 */
export default function ChartCard<Row>({
  title,
  description,
  children,
  columns,
  rows,
  footnote,
  className = "",
}: {
  title: string;
  description?: ReactNode;
  children: ReactNode;
  columns?: Column<Row>[];
  rows?: Row[];
  footnote?: ReactNode;
  className?: string;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  const id = useId();
  const hasTable = !!columns && !!rows;

  return (
    <Card className={`flex flex-col ${className}`}>
      <CardHeader
        id={`${id}-title`}
        title={title}
        description={description}
        action={
          hasTable && (
            <div role="group" aria-label={`${title}: view`} className="flex rounded-lg border border-line p-0.5 text-xs font-medium">
              {(["chart", "table"] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  aria-pressed={view === v}
                  onClick={() => setView(v)}
                  className={`rounded-md px-2.5 py-1 capitalize focus-visible:outline-2 focus-visible:outline-brand-500 ${
                    view === v ? "bg-brand-50 text-brand-600" : "text-ink-3 hover:text-ink"
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
          )
        }
      />
      <div className="flex-1 px-5 pb-5 pt-4" aria-labelledby={`${id}-title`}>
        {view === "chart" || !hasTable ? children : <DataTable columns={columns!} rows={rows!} caption={title} />}
      </div>
      {footnote && <p className="border-t border-line px-5 py-3 text-xs leading-5 text-ink-3">{footnote}</p>}
    </Card>
  );
}
