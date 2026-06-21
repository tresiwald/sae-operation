"""
Experiment 1 — Hard-negative control: do fingerprints encode the OPERATION,
or just digit/format presence?

The main pipeline reports AUC=1.0 for "arithmetic vs control", but arithmetic
prompts contain digits and operators that controls lack — so AUC=1.0 might just
be a digit detector.

Here we hold the operand pair FIXED and vary only the operator (a+b, a-b, a*b,
a/b, and a nonsense a#b). If the SAE features still separate the four real
operations, they encode the operation. If separability collapses, the original
signal was digit/format presence.

Decision rule (printed at the end):
  macro one-vs-rest AUC for 4-way op classification on matched operands
    > 0.85  → operations ARE encoded            → PROCEED to exp2
    ~ 0.50  → fingerprints were digit detectors  → pivot to format-invariance paper

Reads : results/checkpoints/acts_checkpoint.pt, sae_L{layer}.pt
Outputs: results/exp1_hard_negatives.json, results/exp1_hard_negatives.png
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gen import make_matched_probes
from pipeline.config import OPS_EVAL, OUT_DIR
import experiments._common as C

sns.set_theme(style="whitegrid", font_scale=0.9)
REAL_OPS = OPS_EVAL                       # add, sub, mul, div


def macro_ovr_auc(X, y):
    """Macro one-vs-rest ROC-AUC via cross-validated logistic regression."""
    aucs = []
    for op in REAL_OPS:
        yb = (np.array(y) == op).astype(int)
        if yb.sum() < 5 or (len(yb) - yb.sum()) < 5:
            continue
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        s  = cross_val_score(LogisticRegression(max_iter=500, C=0.1),
                             X, yb, cv=cv, scoring="roc_auc")
        aucs.append(s.mean())
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    layers_env = None
    acts_ck = C.load_acts_checkpoint()
    layers  = acts_ck["layers"]
    best    = C.best_layer_from_results(default=layers[len(layers) // 2])
    print(f"Exp 1 — hard-negative control   layers={layers}  best_layer={best}")

    # 1. Matched-operand probes (same bins the SAE was trained on)
    probes = make_matched_probes(n_per_bin=100)
    n_pairs = len({r["pair_id"] for r in probes})
    print(f"  {len(probes)} probes over {n_pairs} matched operand pairs "
          f"(5 operators each)")

    # 2. Collect activations at every sweep layer in one pass
    model, tok, device = C.load_model()
    raw = C.collect_last_token([r["prompt"] for r in probes], model, tok,
                               layers, device, desc="exp1 acts")
    del model

    ops = np.array([r["op"] for r in probes])

    # 3. Per-layer separability under matched operands
    results = {}
    for l in layers:
        sae   = C.load_sae(l)
        norm  = C.normalize(raw[l], acts_ck["mu"][l], acts_ck["sig"][l])
        feats = np.nan_to_num(C.encode(sae, norm), nan=0.0)

        # 3a. 4-way operation classification (real ops only, matched operands)
        m_real = np.isin(ops, REAL_OPS)
        auc_op = macro_ovr_auc(feats[m_real], ops[m_real])

        # 3b. sanity: arithmetic (any real op) vs nonsense '#' — should stay high
        y_math = np.isin(ops, REAL_OPS).astype(int)
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        auc_math = float(cross_val_score(
            LogisticRegression(max_iter=500, C=0.1),
            feats, y_math, cv=cv, scoring="roc_auc").mean())

        results[l] = dict(auc_op_matched=auc_op, auc_math_vs_nonsense=auc_math)
        print(f"  L{l:2d}:  op-AUC(matched)={auc_op:.3f}   "
              f"math-vs-nonsense={auc_math:.3f}")

    # 4. Verdict at the best layer
    best_auc = results[best]["auc_op_matched"]
    verdict  = ("OPERATIONS ENCODED — proceed to exp2"      if best_auc > 0.85 else
                "AMBIGUOUS — investigate"                    if best_auc > 0.65 else
                "DIGIT DETECTOR — pivot to format-invariance")
    print(f"\n  best_layer={best}  op-AUC(matched)={best_auc:.3f}  →  {verdict}")

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ls = sorted(layers)
    ax.plot(ls, [results[l]["auc_op_matched"] for l in ls],
            marker="o", label="op-AUC (matched operands)", color="steelblue")
    ax.plot(ls, [results[l]["auc_math_vs_nonsense"] for l in ls],
            marker="s", ls="--", label="math vs nonsense", color="gray")
    ax.axhline(0.85, ls=":", color="seagreen", alpha=0.7, label="encoded threshold")
    ax.axhline(0.50, ls=":", color="tomato",   alpha=0.7, label="chance")
    ax.set(xlabel="Layer", ylabel="ROC-AUC", ylim=(0.4, 1.02),
           title="Exp 1 — operation separability under matched operands")
    ax.set_xticks(ls); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp1_hard_negatives.png", dpi=150, bbox_inches="tight")
    plt.close()

    out = dict(best_layer=best, best_op_auc_matched=best_auc, verdict=verdict,
               per_layer={str(l): results[l] for l in layers})
    (OUT_DIR / "exp1_hard_negatives.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults → {OUT_DIR/'exp1_hard_negatives.json'}")
    print(f"Plot    → {OUT_DIR/'exp1_hard_negatives.png'}")


if __name__ == "__main__":
    main()
