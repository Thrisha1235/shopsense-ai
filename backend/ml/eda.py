"""Exploratory data analysis for the Online Shoppers Purchasing Intention dataset.

Outputs (all under backend/reports/):
    figures/*.png      charts
    tables/*.csv       the numbers behind every chart (table view)
    eda_stats.json     every statistic quoted in EDA_SUMMARY.md

EDA here is purely descriptive: it reads the original CSV (never writes it), applies
only the stateless clean() step, and fits no preprocessing or model. Quantile bins
used for charting are descriptive groupings, not model features.

Run from the backend directory:
    python -m ml.eda
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
from scipy import stats

from ml.data_preprocessing import (
    MONTH_ORDER,
    NUMERIC_FEATURES,
    RAW_CSV_PATH,
    TARGET,
    clean,
    load_raw,
    sha256_of,
    validate,
    verify_checksum,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

# ------------------------------------------------------------------ chart styling
# Reference palette (light mode). Purchase / No purchase validated as a pair:
# CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 contrast on the surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PURCHASE = "#2a78d6"      # categorical slot 1 (blue)
NO_PURCHASE = "#eb6834"   # categorical slot 2 (orange)
CLASS_COLORS = {"Purchase": PURCHASE, "No purchase": NO_PURCHASE}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def set_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
    })


def finish(fig: plt.Figure, name: str, title: str, subtitle: str) -> Path:
    # Position the header in inches so spacing is the same for short and tall figures.
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.08 / h, ha="left", va="top",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.42 / h, subtitle, ha="left", va="top", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / h))
    path = FIG_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def pct(x: float, _pos=None) -> str:
    return f"{x:.0%}"


# -------------------------------------------------------------------- statistics
def wilson_ci(successes: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a proportion (well-behaved for small n / extreme p)."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def conversion_table(df: pd.DataFrame, by: str | pd.Series, order=None) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[TARGET].agg(sessions="size", purchases="sum")
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    g["no_purchase"] = g["sessions"] - g["purchases"]
    g["conversion_rate"] = g["purchases"] / g["sessions"]
    g["ci95_low"], g["ci95_high"] = wilson_ci(g["purchases"].to_numpy(), g["sessions"].to_numpy())
    g["share_of_sessions"] = g["sessions"] / g["sessions"].sum()
    return g


def chi_square(df: pd.DataFrame, col: str) -> dict:
    table = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": cramers_v}


def compare_by_class(df: pd.DataFrame, col: str) -> dict:
    buy = df.loc[df[TARGET] == 1, col]
    no = df.loc[df[TARGET] == 0, col]
    u, p = stats.mannwhitneyu(buy, no, alternative="two-sided")
    # Rank-biserial correlation: effect size for Mann-Whitney U (-1..1).
    rbc = 2 * u / (len(buy) * len(no)) - 1
    return {
        "median_purchase": float(buy.median()),
        "median_no_purchase": float(no.median()),
        "mean_purchase": float(buy.mean()),
        "mean_no_purchase": float(no.mean()),
        "mann_whitney_p_value": float(p),
        "rank_biserial": float(rbc),
    }


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, float_format="%.6f")


def labelled(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].map({1: "Purchase", 0: "No purchase"})


# ------------------------------------------------------------------------ charts
def plot_conversion(df: pd.DataFrame, out: dict) -> Path:
    counts = labelled(df).value_counts().reindex(["No purchase", "Purchase"])
    shares = counts / counts.sum()
    save_table(pd.DataFrame({"sessions": counts, "share": shares}), "01_conversion_rate.csv")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    bars = ax.barh(counts.index, counts.values, height=0.5,
                   color=[CLASS_COLORS[k] for k in counts.index],
                   edgecolor=SURFACE, linewidth=2)
    for bar, n, s in zip(bars, counts.values, shares.values):
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n:,} sessions  ({s:.1%})", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Sessions")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    rate = out["dataset"]["conversion_rate"]
    return finish(fig, "01_conversion_rate.png",
                  f"Overall conversion rate: {rate:.1%}",
                  f"{counts['Purchase']:,} of {counts.sum():,} sessions ended in a purchase "
                  f"- the classes are imbalanced (~{counts['No purchase'] / counts['Purchase']:.1f} : 1).")


def plot_visitor_type(df: pd.DataFrame, out: dict) -> Path:
    order = ["Returning_Visitor", "New_Visitor", "Other"]
    t = conversion_table(df, "VisitorType", order)
    save_table(t, "02_visitor_type.csv")
    out["visitor_type"] = {"table": t.reset_index().to_dict(orient="records"),
                           "chi_square": chi_square(df, "VisitorType")}
    labels = [o.replace("_", " ") for o in t.index]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(t))
    a1.barh(y, t["no_purchase"], height=0.55, color=NO_PURCHASE, edgecolor=SURFACE, linewidth=2, label="No purchase")
    a1.barh(y, t["purchases"], left=t["no_purchase"], height=0.55, color=PURCHASE,
            edgecolor=SURFACE, linewidth=2, label="Purchase")
    for i, (n, s) in enumerate(zip(t["sessions"], t["share_of_sessions"])):
        a1.text(n + t["sessions"].max() * 0.01, i, f"{n:,} ({s:.1%})", va="center", color=INK, fontsize=9)
    a1.set_yticks(y, labels)
    a1.invert_yaxis()
    a1.set_xlim(0, t["sessions"].max() * 1.3)
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.set_title("Sessions by visitor type")
    a1.set_xlabel("Sessions")
    a1.grid(axis="y", visible=False)
    a1.legend(loc="lower right")

    overall = out["dataset"]["conversion_rate"]
    err = np.vstack([t["conversion_rate"] - t["ci95_low"], t["ci95_high"] - t["conversion_rate"]])
    a2.barh(y, t["conversion_rate"], height=0.55, color=PURCHASE, edgecolor=SURFACE, linewidth=2,
            xerr=err, error_kw={"ecolor": INK_2, "elinewidth": 1.2, "capsize": 4})
    for i, (r, hi) in enumerate(zip(t["conversion_rate"], t["ci95_high"])):
        a2.text(hi + 0.005, i, f"{r:.1%}", va="center", color=INK, fontsize=9)
    a2.axvline(overall, color=MUTED, linewidth=1)
    a2.set_yticks(y, labels)
    a2.invert_yaxis()
    a2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_xlim(0, max(t["ci95_high"].max() * 1.25, overall * 1.3))
    a2.set_title(f"Conversion rate (95% CI) - grey line: overall {overall:.1%}")
    a2.set_xlabel("Sessions ending in a purchase")
    a2.grid(axis="y", visible=False)

    best = t["conversion_rate"].idxmax()
    return finish(fig, "02_visitor_type.png", "Conversion by visitor type",
                  f"Highest conversion: {best.replace('_', ' ')} ({t.loc[best, 'conversion_rate']:.1%}). "
                  f"Error bars are 95% Wilson intervals.")


def plot_product_duration(df: pd.DataFrame, out: dict) -> Path:
    col = "ProductRelated_Duration"
    d = df.assign(cls=labelled(df), log_dur=np.log1p(df[col]))
    out["product_duration"] = compare_by_class(df, col)
    out["product_duration"]["zero_duration_sessions"] = int((df[col] == 0).sum())

    bins = pd.qcut(df[col], 5, duplicates="drop")
    t = conversion_table(df.assign(bin=bins), "bin")
    # qcut opens the first bin just below the minimum; clamp the label at 0.
    t.index = [f"{max(iv.left, 0) / 60:.0f}-{iv.right / 60:.0f} min" for iv in t.index]
    t.index.name = "duration_quintile"
    save_table(t, "03_product_duration_quintiles.csv")
    out["product_duration"]["quintiles"] = t.reset_index().to_dict(orient="records")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for cls in ["No purchase", "Purchase"]:
        sns.histplot(d.loc[d["cls"] == cls, "log_dur"], ax=a1, stat="density", bins=40,
                     element="step", fill=True, alpha=0.18, color=CLASS_COLORS[cls],
                     linewidth=1.6, label=cls)
    ticks_s = [0, 10, 60, 600, 3600, 36000]
    a1.set_xticks(np.log1p(ticks_s), ["0", "10 s", "1 min", "10 min", "1 h", "10 h"])
    a1.set_xlabel("Time on product pages per session (log scale)")
    a1.set_ylabel("Density (each class sums to 1)")
    a1.set_title("Distribution by outcome")
    a1.legend()

    x = np.arange(len(t))
    err = np.vstack([t["conversion_rate"] - t["ci95_low"], t["ci95_high"] - t["conversion_rate"]])
    a2.bar(x, t["conversion_rate"], width=0.6, color=PURCHASE, edgecolor=SURFACE, linewidth=2,
           yerr=err, error_kw={"ecolor": INK_2, "elinewidth": 1.2, "capsize": 4})
    for i, (r, hi) in enumerate(zip(t["conversion_rate"], t["ci95_high"])):
        a2.text(i, hi + 0.006, f"{r:.1%}", ha="center", color=INK, fontsize=9)
    a2.set_xticks(x, t.index, fontsize=9)
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_ylim(0, t["ci95_high"].max() * 1.2)
    a2.set_xlabel("Product-page duration quintile (equal-size groups)")
    a2.set_ylabel("Conversion rate")
    a2.set_title("Conversion rate by duration quintile (95% CI)")
    a2.grid(axis="x", visible=False)

    s = out["product_duration"]
    return finish(fig, "03_product_page_duration.png", "Product-page duration",
                  f"Median time on product pages: {s['median_purchase'] / 60:.1f} min for purchasers vs "
                  f"{s['median_no_purchase'] / 60:.1f} min for non-purchasers.")


RATE_BINS = [-1e-9, 0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
RATE_LABELS = ["0%", "0-0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", "10-20%"]


def plot_rate(df: pd.DataFrame, out: dict, col: str, key: str, fname: str, label: str) -> Path:
    out[key] = compare_by_class(df, col)
    bins = pd.cut(df[col], RATE_BINS, labels=RATE_LABELS)
    t = conversion_table(df.assign(bin=bins), "bin", RATE_LABELS)
    t.index.name = f"{col}_bin"
    save_table(t, fname.replace(".png", ".csv"))
    out[key]["bins"] = t.reset_index().astype({f"{col}_bin": str}).to_dict(orient="records")

    d = df.assign(cls=labelled(df))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for cls in ["No purchase", "Purchase"]:
        sns.ecdfplot(d.loc[d["cls"] == cls, col], ax=a1, color=CLASS_COLORS[cls], linewidth=2, label=cls)
    a1.xaxis.set_major_locator(MultipleLocator(0.05))
    a1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a1.set_xlabel(label)
    a1.set_ylabel("Share of sessions at or below value")
    a1.set_title("Cumulative distribution by outcome")
    a1.legend(loc="lower right")

    x = np.arange(len(t))
    err = np.vstack([t["conversion_rate"] - t["ci95_low"], t["ci95_high"] - t["conversion_rate"]])
    a2.bar(x, t["conversion_rate"], width=0.6, color=PURCHASE, edgecolor=SURFACE, linewidth=2,
           yerr=err, error_kw={"ecolor": INK_2, "elinewidth": 1.2, "capsize": 4})
    for i, (r, hi) in enumerate(zip(t["conversion_rate"], t["ci95_high"])):
        a2.text(i, hi + 0.006, f"{r:.1%}", ha="center", color=INK, fontsize=9)
    a2.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(t.index, t["sessions"])], fontsize=8.5)
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_ylim(0, t["ci95_high"].max() * 1.2)
    a2.set_xlabel(f"{label} band")
    a2.set_ylabel("Conversion rate")
    a2.set_title("Conversion rate by band (95% CI)")
    a2.grid(axis="x", visible=False)

    s = out[key]
    return finish(fig, fname, label,
                  f"Median {label.lower()}: {s['median_purchase']:.2%} for purchasers vs "
                  f"{s['median_no_purchase']:.2%} for non-purchasers.")


def plot_monthly(df: pd.DataFrame, out: dict) -> Path:
    t = conversion_table(df, "Month", MONTH_ORDER)
    save_table(t, "06_monthly_trends.csv")
    missing = [m for m in MONTH_ORDER if m not in t.index]
    out["monthly"] = {"table": t.reset_index().to_dict(orient="records"),
                      "months_absent": missing, "chi_square": chi_square(df, "Month")}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})
    x = np.arange(len(t))
    a1.bar(x, t["no_purchase"], width=0.62, color=NO_PURCHASE, edgecolor=SURFACE, linewidth=2, label="No purchase")
    a1.bar(x, t["purchases"], bottom=t["no_purchase"], width=0.62, color=PURCHASE,
           edgecolor=SURFACE, linewidth=2, label="Purchase")
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.set_ylabel("Sessions")
    a1.set_title("Sessions per month")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="upper left")
    busiest = t["sessions"].idxmax()
    i_b = list(t.index).index(busiest)
    a1.text(i_b, t.loc[busiest, "sessions"] * 1.02, f"{t.loc[busiest, 'sessions']:,}",
            ha="center", va="bottom", color=INK, fontsize=9)
    a1.set_ylim(0, t["sessions"].max() * 1.15)

    a2.fill_between(x, t["ci95_low"], t["ci95_high"], color=PURCHASE, alpha=0.15, linewidth=0)
    a2.plot(x, t["conversion_rate"], color=PURCHASE, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    overall = out["dataset"]["conversion_rate"]
    a2.axhline(overall, color=MUTED, linewidth=1)
    a2.text(-0.45, overall + 0.004, f"overall {overall:.1%}", color=INK_2, fontsize=8, va="bottom", ha="left")
    hi, lo = t["conversion_rate"].idxmax(), t["conversion_rate"].idxmin()
    for m in {hi, lo}:
        i = list(t.index).index(m)
        a2.annotate(f"{m}: {t.loc[m, 'conversion_rate']:.1%}", (i, t.loc[m, "conversion_rate"]),
                    xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=9)
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_ylim(0, t["ci95_high"].max() * 1.2)
    a2.set_ylabel("Conversion rate")
    a2.set_title("Conversion rate per month (shaded: 95% CI)")
    a2.set_xticks(x, t.index)
    a2.grid(axis="x", visible=False)

    return finish(fig, "06_monthly_trends.png", "Monthly sessions and purchase trends",
                  f"Busiest month: {busiest}. Highest conversion: {hi}. "
                  f"Months absent from the dataset: {', '.join(missing) or 'none'}.")


def plot_correlations(df: pd.DataFrame, out: dict) -> Path:
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].astype(float).corr(method="spearman")
    save_table(corr, "07_spearman_correlations.csv")
    with_target = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    out["spearman_with_revenue"] = with_target.to_dict()

    # Lower triangle without the trivial diagonal: drop the first row / last column.
    lower = corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(lower, dtype=bool), k=1)
    annot = lower.map(lambda v: f"{v:.2f}" if abs(v) >= 0.3 else "")
    fig, ax = plt.subplots(figsize=(10, 8.2))
    sns.heatmap(lower, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, square=True,
                annot=annot, fmt="", annot_kws={"fontsize": 8, "color": INK},
                linewidths=2, linecolor=SURFACE, cbar_kws={"shrink": 0.7, "label": "Spearman rho"}, ax=ax)
    ax.grid(False)
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    top = with_target.index[0]
    return finish(fig, "07_correlation_heatmap.png", "Rank correlations between numeric features and Revenue",
                  f"Strongest association with Revenue: {top} (rho = {with_target.iloc[0]:.2f}). "
                  f"Values shown where |rho| >= 0.3; full matrix in tables/07_spearman_correlations.csv.")


# -------------------------------------------------------------------------- main
def extra_stats(raw: pd.DataFrame, df: pd.DataFrame, out: dict) -> None:
    pv = df["PageValues"] > 0
    out["page_values"] = {
        **compare_by_class(df, "PageValues"),
        "share_nonzero_purchase": float(pv[df[TARGET] == 1].mean()),
        "share_nonzero_no_purchase": float(pv[df[TARGET] == 0].mean()),
        "conversion_when_zero": float(df.loc[~pv, TARGET].mean()),
        "conversion_when_nonzero": float(df.loc[pv, TARGET].mean()),
        "sessions_zero": int((~pv).sum()),
        "sessions_nonzero": int(pv.sum()),
    }
    out["weekend"] = conversion_table(df, "Weekend").reset_index().astype({"Weekend": str}).to_dict(orient="records")
    out["special_day"] = conversion_table(df, "SpecialDay").reset_index().to_dict(orient="records")

    pages = df[["Administrative", "Informational", "ProductRelated"]].sum(axis=1)
    dups = raw[raw.duplicated()]
    out["data_quality"] = {
        "sessions_with_zero_pages": int((pages == 0).sum()),
        "product_pages_gt0_but_zero_duration": int(((df["ProductRelated"] > 0) & (df["ProductRelated_Duration"] == 0)).sum()),
        "duplicate_rows": int(len(dups)),
        "duplicate_rows_purchase_rate": float(dups[TARGET].astype(bool).mean()) if len(dups) else None,
        "duplicate_rows_all_zero_duration": bool((dups[["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]] == 0).all().all()) if len(dups) else None,
        "bounce_rate_at_max_0.2": int((df["BounceRates"] == 0.2).sum()),
    }


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Interval):
        return str(o)
    return o


def main() -> None:
    for d in (FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    set_style()

    checksum_before = sha256_of(RAW_CSV_PATH)
    raw = load_raw()
    report = validate(raw)
    report.raise_if_invalid()
    df = clean(raw)  # stateless only; duplicates kept (see data_preprocessing.clean)

    out: dict = {
        "dataset": {
            "source": "UCI Machine Learning Repository, dataset 468",
            "file": RAW_CSV_PATH.name,
            "sha256": checksum_before,
            "rows": report.n_rows,
            "columns": report.n_columns,
            "missing_values_total": int(sum(report.missing_values.values())),
            "missing_values_by_column": report.missing_values,
            "duplicate_rows": report.n_duplicate_rows,
            "purchases": int(df[TARGET].sum()),
            "no_purchases": int((df[TARGET] == 0).sum()),
            "conversion_rate": float(df[TARGET].mean()),
            "validation_warnings": report.warnings,
        }
    }

    figures = [
        plot_conversion(df, out),
        plot_visitor_type(df, out),
        plot_product_duration(df, out),
        plot_rate(df, out, "BounceRates", "bounce_rate", "04_bounce_rate.png", "Bounce rate"),
        plot_rate(df, out, "ExitRates", "exit_rate", "05_exit_rate.png", "Exit rate"),
        plot_monthly(df, out),
        plot_correlations(df, out),
    ]
    extra_stats(raw, df, out)

    checksum_after = sha256_of(RAW_CSV_PATH)
    out["dataset"]["unchanged_after_eda"] = checksum_after == checksum_before
    out["dataset"]["checksum_matches_download"] = verify_checksum()

    (REPORTS_DIR / "eda_stats.json").write_text(json.dumps(to_jsonable(out), indent=2), encoding="utf-8")

    print(f"Rows {report.n_rows:,} | columns {report.n_columns} | missing {out['dataset']['missing_values_total']} "
          f"| duplicates {report.n_duplicate_rows} | conversion {out['dataset']['conversion_rate']:.2%}")
    print("Figures:")
    for f in figures:
        print(f"  {f.relative_to(BACKEND_DIR)}  ({f.stat().st_size / 1024:.0f} KB)")
    print(f"Tables: {len(list(TABLE_DIR.glob('*.csv')))} CSV files in {TABLE_DIR.relative_to(BACKEND_DIR)}")
    print(f"Stats:  {(REPORTS_DIR / 'eda_stats.json').relative_to(BACKEND_DIR)}")
    print(f"Original dataset unchanged: {out['dataset']['unchanged_after_eda']}")


if __name__ == "__main__":
    main()
