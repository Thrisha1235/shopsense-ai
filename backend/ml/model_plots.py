"""Model-comparison charts for Phase 3 (written to reports/figures/, data to reports/tables/).

Colour roles (validated with the dataviz palette checker):
  * feature-set charts: blue = with PageValues, orange = without PageValues
  * curve charts: blue / orange / aqua = Logistic Regression / Random Forest / Gradient Boosting
    (aqua is < 3:1 on the light surface, so every series is also named in a legend and all
    numbers are in tables/08_model_comparison.csv)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from ml.eda import AXIS, FIG_DIR, INK, INK_2, MUTED, SURFACE, TABLE_DIR, finish, set_style
from ml.train import MODEL_LABELS

FS_COLORS = {"with_page_values": "#2a78d6", "without_page_values": "#eb6834"}
FS_LABELS = {"with_page_values": "With PageValues", "without_page_values": "Without PageValues"}
MODEL_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "gradient_boosting": "#1baf7a"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
METRICS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
           ("f1", "F1 score"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for fs, rec in results["feature_sets"].items():
        for name, m in rec["models"].items():
            t = m["test"]["at_tuned_threshold"]
            ci = m["test"]["bootstrap_95ci_at_tuned_threshold"]
            rows.append({
                "feature_set": fs, "model": name, "selected": name == rec["selected_model"],
                "cv_pr_auc_mean": m["cv"]["pr_auc"]["mean"], "cv_pr_auc_std": m["cv"]["pr_auc"]["std"],
                "cv_roc_auc_mean": m["cv"]["roc_auc"]["mean"], "cv_roc_auc_std": m["cv"]["roc_auc"]["std"],
                "threshold": m["test"]["threshold"],
                **{f"test_{k}": t[k] for k, _ in METRICS},
                "test_balanced_accuracy": t["balanced_accuracy"], "test_brier": t["brier"],
                "test_pr_auc_ci_low": ci["pr_auc"][0], "test_pr_auc_ci_high": ci["pr_auc"][1],
                **{f"cm_{k}": v for k, v in t["confusion_matrix"].items()},
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "08_model_comparison.csv", index=False, float_format="%.6f")
    return df


def _grouped_barh(ax, df: pd.DataFrame, value: str, err: str | None, xlabel: str, baseline=None, base_label=""):
    models = list(MODEL_LABELS)
    y = np.arange(len(models))
    h = 0.36
    for k, fs in enumerate(FS_COLORS):
        sub = df[df.feature_set == fs].set_index("model").reindex(models)
        offs = y + (k - 0.5) * (h + 0.04)
        ax.barh(offs, sub[value], height=h, color=FS_COLORS[fs], edgecolor=SURFACE, linewidth=2,
                label=FS_LABELS[fs],
                xerr=None if err is None else sub[err],
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
        for yy, v, sel in zip(offs, sub[value], sub["selected"]):
            extra = (sub[err].max() if err else 0) + 0.01
            ax.text(v + extra, yy, f"{v:.3f}" + ("  selected" if sel else ""), va="center",
                    color=INK, fontsize=8.5, fontweight="bold" if sel else "normal")
    if baseline is not None:
        ax.axvline(baseline, color=MUTED, linewidth=1)   # value is named in the figure subtitle
    ax.set_yticks(y, [MODEL_LABELS[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def _header_legend(fig, ax) -> None:
    """Legend in the figure header, clear of every plot area."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.995, 1 - 0.05 / fig.get_figheight()))


def plot_cv_comparison(df: pd.DataFrame, results: dict) -> Path:
    prev = results["protocol"]["split"]["train_purchase_rate"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    _grouped_barh(a1, df, "cv_pr_auc_mean", "cv_pr_auc_std", "Mean PR-AUC across 5 CV folds (±1 SD)",
                  baseline=prev, base_label=f"no-skill = purchase rate {prev:.3f}")
    a1.set_title("Cross-validated PR-AUC (selection metric)")
    _grouped_barh(a2, df, "cv_roc_auc_mean", "cv_roc_auc_std", "Mean ROC-AUC across 5 CV folds (±1 SD)",
                  baseline=0.5, base_label="no-skill = 0.5")
    a2.set_title("Cross-validated ROC-AUC")
    _header_legend(fig, a1)
    return finish(fig, "08_cv_model_comparison.png", "Model comparison on the training split (cross-validation)",
                  f"Selection uses mean CV PR-AUC only; the test set played no part. Grey lines: no-skill "
                  f"(PR-AUC = purchase rate {prev:.3f}, ROC-AUC = 0.5).")


def plot_test_metrics(df: pd.DataFrame, results: dict) -> Path:
    base = results["baseline_prior"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    for ax, (key, label) in zip(axes.ravel(), METRICS):
        bl = {"accuracy": (base["accuracy"], f"always 'no' {base['accuracy']:.3f}"),
              "pr_auc": (base["pr_auc"], f"no-skill {base['pr_auc']:.3f}"),
              "roc_auc": (0.5, "no-skill 0.5")}.get(key)
        _grouped_barh(ax, df, f"test_{key}", None, label,
                      baseline=bl[0] if bl else None, base_label=bl[1] if bl else "")
        ax.set_title(label)
    _header_legend(fig, axes[0, 0])
    return finish(fig, "09_test_metrics.png", "Held-out test-set metrics (2,466 sessions, evaluated once)",
                  "Thresholded metrics use each model's tuned threshold. Grey lines: always predicting "
                  f"'no purchase' (accuracy {base['accuracy']:.3f}), no-skill ROC-AUC 0.5 and "
                  f"PR-AUC {base['pr_auc']:.3f}.")


def plot_curves(results: dict, trained: dict, prepared) -> Path:
    y = prepared.y_test.to_numpy()
    prev = y.mean()
    fs_names = list(trained)
    fig, axes = plt.subplots(len(fs_names), 2, figsize=(12, 5.2 * len(fs_names)), squeeze=False)
    for r, fs in enumerate(fs_names):
        a_roc, a_pr = axes[r]
        for name, tm in trained[fs].items():
            t = tm.record["test"]["at_tuned_threshold"]
            fpr, tpr, _ = roc_curve(y, tm.test_proba)
            prec, rec, _ = precision_recall_curve(y, tm.test_proba)
            a_roc.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]} (AUC {t['roc_auc']:.3f})")
            a_pr.plot(rec, prec, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]} (AP {t['pr_auc']:.3f})")
            # operating point at the tuned threshold
            a_pr.plot(t["recall"], t["precision"], "o", color=MODEL_COLORS[name], markersize=8,
                      markeredgecolor=SURFACE, markeredgewidth=2)
        a_roc.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        a_pr.axhline(prev, color=AXIS, linewidth=1)
        a_pr.text(0.99, prev + 0.01, f"no-skill {prev:.3f}", color=INK_2, fontsize=8, ha="right", va="bottom")
        a_roc.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate (recall)")
        a_pr.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
        a_roc.set_title(f"ROC - {FS_LABELS[fs]}")
        a_pr.set_title(f"Precision-recall - {FS_LABELS[fs]} (dots: tuned threshold)")
        a_roc.legend(loc="lower right", fontsize=9)
        a_pr.legend(loc="upper right", fontsize=9)
    return finish(fig, "10_roc_pr_curves.png", "ROC and precision-recall curves on the test set",
                  "Grey lines show a no-skill classifier. Removing PageValues lowers every curve.")


def plot_confusion(results: dict) -> Path:
    fs_names = list(results["feature_sets"])
    fig, axes = plt.subplots(1, len(fs_names), figsize=(6 * len(fs_names), 5), squeeze=False)
    for ax, fs in zip(axes[0], fs_names):
        rec = results["feature_sets"][fs]
        sel = rec["selected_model"]
        m = rec["models"][sel]["test"]
        cm = m["at_tuned_threshold"]["confusion_matrix"]
        arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        row_share = arr / arr.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                dark = row_share[i, j] > 0.55
                ax.text(j, i, f"{arr[i, j]:,}\n{row_share[i, j]:.1%} of row", ha="center", va="center",
                        fontsize=11, color="#ffffff" if dark else INK)
        ax.set_xticks([0, 1], ["Predicted: no purchase", "Predicted: purchase"])
        ax.set_yticks([0, 1], ["Actual: no purchase", "Actual: purchase"])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{FS_LABELS[fs]}: {MODEL_LABELS[sel]}\nthreshold {m['threshold']:.3f}", fontsize=11)
    return finish(fig, "11_confusion_matrices.png", "Confusion matrices of the selected models (test set)",
                  "Cell shade = share of the actual class (row). Counts are sessions.")


def plot_calibration(records: dict) -> Path:
    """Reliability diagrams from the TRAINING-split CV (the evidence used for acceptance)."""
    from ml.calibration import reliability_table

    set_style()
    variants = list(records)
    fig, axes = plt.subplots(1, len(variants), figsize=(6.2 * len(variants), 5.6), squeeze=False)
    for ax, v in zip(axes[0], variants):
        rec = records[v]
        d = rec["_plot"]
        tr = rec["validation_on_training_split"]
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        for key, label, color, m in [("raw", "Raw model score", "#eb6834", tr["raw_scores"]),
                                     ("cal", f"Calibrated ({rec['method']})", "#2a78d6", tr[rec["method"]])]:
            t = pd.DataFrame(reliability_table(d["y"], d[key]))
            ax.plot(t["mean_predicted"], t["observed_rate"], color=color, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2,
                    label=f"{label}: ECE {m['ece']:.3f}, Brier {m['brier']:.3f}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted value (10 equal-size bins)",
               ylabel="Observed purchase rate")
        status = rec["status"].replace("_", " ")
        ax.set_title(f"{FS_LABELS.get(v, v)}: {status}", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
    return finish(fig, "12_calibration_curve.png", "Probability calibration (training-split cross-validation)",
                  "Points on the grey diagonal are perfectly calibrated. Test data was not used.")


def make_all_figures(results: dict, trained: dict, prepared) -> list[Path]:
    for d in (FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    set_style()
    df = comparison_table(results)
    return [
        plot_cv_comparison(df, results),
        plot_test_metrics(df, results),
        plot_curves(results, trained, prepared),
        plot_confusion(results),
    ]
