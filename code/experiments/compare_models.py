"""
Cross-model comparison — read each model's result JSONs and contrast the
findings. The headline question: does math capability (qwen-base → qwen-math)
shift the SAE representation away from a magnitude code toward operation-specific
/ digit-structure features, and make the fingerprint correctness-aware?

Reads : results/<key>/{results.json, exp1_hard_negatives.json,
                       exp2_error_mechanism.json, gen_interpret.json}
Outputs: results/model_comparison.json, results/model_comparison.png

Usage: python -m experiments.compare_models [key1 key2 ...]
       (default: gemma qwen-base qwen-math)
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import ROOT

sns.set_theme(style="whitegrid", font_scale=0.9)
DEFAULT_KEYS = ["gemma", "qwen-base", "qwen-math"]


def load(key, name):
    p = ROOT / "results" / key / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def metrics_for(key):
    res  = load(key, "results.json") or {}
    e1   = load(key, "exp1_hard_negatives.json") or {}
    e2   = load(key, "exp2_error_mechanism.json") or {}
    gen  = load(key, "gen_interpret.json") or {}

    best = res.get("best_layer")
    best_auc = None
    if best is not None and "layer_sweep" in res:
        best_auc = res["layer_sweep"].get(str(best), {}).get("auc")

    return {
        "op_auc_matched":   e1.get("best_op_auc_matched"),       # exp1
        "accuracy":         e2.get("accuracy"),                  # exp2
        "corr_fingerprint": e2.get("auc_fingerprint_strength"),  # exp2
        "corr_probe":       e2.get("auc_hidden_probe"),          # exp2
        "corr_magnitude":   e2.get("auc_magnitude_only"),        # exp2 confound
        "gen_magnitude":    gen.get("mean_magnitude"),           # gen-interpret
        "gen_structure":    gen.get("mean_structure"),           # gen-interpret
        "gen_verdict":      gen.get("verdict"),
        "best_layer_auc":   best_auc,
    }


def main():
    keys = sys.argv[1:] or DEFAULT_KEYS
    keys = [k for k in keys if (ROOT / "results" / k).exists()]
    if not keys:
        print("No model result dirs found under results/<key>/.")
        return

    table = {k: metrics_for(k) for k in keys}

    # print comparison
    fields = ["accuracy", "op_auc_matched", "best_layer_auc",
              "corr_fingerprint", "corr_probe", "corr_magnitude",
              "gen_magnitude", "gen_structure"]
    print(f"\n{'metric':18s}" + "".join(f"{k:>14s}" for k in keys))
    for f in fields:
        row = f"{f:18s}"
        for k in keys:
            v = table[k][f]
            row += f"{v:14.3f}" if isinstance(v, (int, float)) else f"{'—':>14s}"
        print(row)
    for k in keys:
        print(f"  {k}: gen → {table[k]['gen_verdict']}")

    # plot: key contrasts side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(keys))

    def bars(ax, fld_pairs, title, ylim=(0, 1)):
        w = 0.8 / len(fld_pairs)
        for i, (fld, lab, col) in enumerate(fld_pairs):
            vals = [table[k][fld] if isinstance(table[k][fld], (int, float)) else np.nan
                    for k in keys]
            ax.bar(x + (i - (len(fld_pairs)-1)/2)*w, vals, w, label=lab, color=col)
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15)
        ax.set(title=title, ylim=ylim); ax.legend(fontsize=8)

    bars(axes[0], [("accuracy", "accuracy", "slategray"),
                   ("op_auc_matched", "op-AUC (matched)", "steelblue")],
         "Capability & operation separability")
    bars(axes[1], [("corr_fingerprint", "fingerprint", "darkorange"),
                   ("corr_probe", "hidden probe", "seagreen"),
                   ("corr_magnitude", "magnitude only", "lightgray")],
         "Correctness signal (exp2)")
    bars(axes[2], [("gen_magnitude", "magnitude", "steelblue"),
                   ("gen_structure", "digit structure", "seagreen")],
         "Gen-time: magnitude vs structure", ylim=(0, None))
    plt.tight_layout()
    out_png = ROOT / "results" / "model_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

    (ROOT / "results" / "model_comparison.json").write_text(
        json.dumps({"models": keys, "metrics": table}, indent=2))
    print(f"\nComparison → {ROOT/'results'/'model_comparison.json'}")
    print(f"Plot       → {out_png}")


if __name__ == "__main__":
    main()
