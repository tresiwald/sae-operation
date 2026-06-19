"""
Stage 6 — Scaling analysis: aggregate results across sample-size runs.

Reads  : results/scale_{n}/results.json  for each scale
Outputs: results/scaling/scaling_curves.png
         results/scaling/scaling_table.csv
         results/scaling/scaling_summary.json

Run after all scale variants of the pipeline have completed:
  python -m pipeline.stage6_scaling_analysis
or with a custom results root:
  SAE_SCALING_ROOT=results python -m pipeline.stage6_scaling_analysis
"""

import csv, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import ROOT, OPS_EVAL

sns.set_theme(style="whitegrid", font_scale=0.9)

SCALING_ROOT = Path(os.getenv("SAE_SCALING_ROOT", str(ROOT / "results")))
OUT_DIR      = SCALING_ROOT / "scaling"
OUT_DIR.mkdir(exist_ok=True)

COLORS = {"add": "steelblue", "sub": "seagreen",
          "mul": "darkorange", "div": "mediumpurple"}


def load_results(scales: list[int]) -> list[dict]:
    rows = []
    for n in scales:
        p = SCALING_ROOT / f"scale_{n}" / "results.json"
        if not p.exists():
            print(f"  MISSING: {p}"); continue
        with open(p) as f:
            r = json.load(f)

        best = str(r["best_layer"])
        sweep = r.get("layer_sweep", {})

        row = dict(
            n_per_cell    = n,
            n_train       = r["n_train"],
            best_layer    = r["best_layer"],
            d_sae         = r["d_sae"],
        )
        for op in OPS_EVAL:
            row[f"fp_{op}"]        = r["fps_best"].get(op, 0)
            row[f"jaccard_{op}"]   = r["jaccard"].get(op, float("nan"))
            row[f"auc_{op}"]       = r.get("auc", {}).get("all", float("nan"))
        # best-layer SAE quality
        bl = r.get("layer_sweep", {}).get(best, {})
        row["var_expl"] = bl.get("var_expl", float("nan"))
        row["n_live"]   = bl.get("n_live",   float("nan"))
        row["auc"]      = r.get("auc", {}).get("all", float("nan"))
        # cheat proximity (mean over ops)
        cheat = r.get("cheat", {})
        if cheat:
            row["cheat_comp_sim"] = float(np.mean([v["comp_sim"] for v in cheat.values()]))
            row["cheat_copy_sim"] = float(np.mean([v["copy_sim"] for v in cheat.values()]))
        else:
            row["cheat_comp_sim"] = float("nan")
            row["cheat_copy_sim"] = float("nan")
        rows.append(row)
        print(f"  n={n:>7d}  train={r['n_train']:>7d}  best_L={r['best_layer']}  "
              f"fp_mean={np.mean([row[f'fp_{op}'] for op in OPS_EVAL]):.0f}  "
              f"J_mean={np.nanmean([row[f'jaccard_{op}'] for op in OPS_EVAL]):.3f}")
    return rows


def plot_scaling(rows: list[dict]):
    if not rows:
        print("No data to plot."); return

    xs = [r["n_train"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # A: fingerprint size per op
    ax = axes[0]
    for op in OPS_EVAL:
        ys = [r[f"fp_{op}"] for r in rows]
        ax.plot(xs, ys, marker="o", label=op, color=COLORS[op])
    ax.set(xscale="log", title="A  Fingerprint size (# features)",
           xlabel="Training examples", ylabel="# features")
    ax.legend(); ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # B: mean fingerprint size + error band
    ax = axes[1]
    means = [np.mean([r[f"fp_{op}"] for op in OPS_EVAL]) for r in rows]
    stds  = [np.std([r[f"fp_{op}"] for op in OPS_EVAL])  for r in rows]
    ax.plot(xs, means, marker="o", color="black")
    ax.fill_between(xs, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], alpha=0.2, color="black")
    ax.set(xscale="log", title="B  Mean fingerprint size ± std",
           xlabel="Training examples", ylabel="# features")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # C: Jaccard (format invariance) per op
    ax = axes[2]
    for op in OPS_EVAL:
        ys = [r[f"jaccard_{op}"] for r in rows]
        ax.plot(xs, ys, marker="o", label=op, color=COLORS[op])
    ax.set(xscale="log", ylim=(0, 1),
           title="C  Format invariance (Jaccard across sym/mix/verb)",
           xlabel="Training examples", ylabel="Jaccard")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.4)
    ax.legend(); ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # D: mean Jaccard
    ax = axes[3]
    mean_j = [np.nanmean([r[f"jaccard_{op}"] for op in OPS_EVAL]) for r in rows]
    ax.plot(xs, mean_j, marker="o", color="purple")
    ax.set(xscale="log", ylim=(0, 1), title="D  Mean Jaccard",
           xlabel="Training examples", ylabel="Jaccard")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.4)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # E: SAE quality (variance explained + live features)
    ax = axes[4]
    ax2 = ax.twinx()
    ax.plot(xs, [r["var_expl"] for r in rows], marker="o", color="seagreen", label="var_expl")
    ax2.plot(xs, [r["n_live"]  for r in rows], marker="s", color="darkorange",
             ls="--", label="n_live")
    ax.set(xscale="log", ylim=(0, 1),
           title="E  SAE quality (best layer)",
           xlabel="Training examples", ylabel="Variance explained")
    ax2.set_ylabel("Live features")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # F: cheat proximity
    ax = axes[5]
    cs = [r["cheat_comp_sim"] for r in rows]
    ks = [r["cheat_copy_sim"] for r in rows]
    ax.plot(xs, cs, marker="o", color="steelblue", label="→ compute centroid")
    ax.plot(xs, ks, marker="o", color="tomato",    label="→ copy centroid")
    ax.set(xscale="log", ylim=(0, 1),
           title="F  Cheat proximity (mean over ops)",
           xlabel="Training examples", ylabel="Cosine similarity")
    ax.legend(); ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.suptitle("Scaling study — SAE fingerprint quality vs. training set size",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scaling_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {OUT_DIR}/scaling_curves.png")


def save_table(rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(OUT_DIR / "scaling_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Table → {OUT_DIR}/scaling_table.csv")

    with open(OUT_DIR / "scaling_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON  → {OUT_DIR}/scaling_summary.json")


def main():
    # Discover available scale directories
    scale_dirs = sorted(SCALING_ROOT.glob("scale_*/results.json"))
    if not scale_dirs:
        print(f"No scale_*/results.json found under {SCALING_ROOT}")
        print("Run the pipeline for each scale first.")
        return

    scales = sorted(int(p.parent.name.split("_")[1]) for p in scale_dirs)
    print(f"Found {len(scales)} scale runs: {scales}")
    rows = load_results(scales)
    plot_scaling(rows)
    save_table(rows)


if __name__ == "__main__":
    main()
