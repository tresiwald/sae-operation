"""
Experiment 1b — Cross-format control: is the separability the OPERATION, or just
the operator TOKEN?

Exp1 showed operations are separable under matched operands (not a digit
detector). But the probes still differ in the operator token (`+` vs `*`), so a
perfect op-AUC could be lexical token identity rather than computation.

Here we train the operation classifier on one surface format and test it on
another (same operand pairs, different operator words: `+` → `plus`). If the
operation is genuinely encoded, the classifier transfers across formats. If it
was operator-token identity, cross-format AUC collapses toward chance.

This directly probes the Jaccard≈0 finding: do the disjoint per-format feature
sets nonetheless carry a transferable operation signal?

Decision rule (mean off-diagonal, i.e. cross-format, op-AUC at best layer):
    > 0.80  → operation transfers across formats → genuine encoding, proceed
    ~ 0.50  → operator-token identity → reframe toward format-specific circuits

Reads : results/checkpoints/{acts_checkpoint.pt, sae_L{layer}.pt}
Outputs: results/exp1b_cross_format.json, results/exp1b_cross_format.png
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gen import make_matched_probes
from pipeline.config import OPS_EVAL, OUT_DIR
import experiments._common as C

sns.set_theme(style="whitegrid", font_scale=0.9)
FORMATS  = ["symbolic", "mixed", "verbal"]
REAL_OPS = OPS_EVAL


def cross_format_auc(Xtr, ytr, Xte, yte):
    """Macro one-vs-rest AUC: train on (Xtr,ytr), evaluate on (Xte,yte)."""
    aucs = []
    for op in REAL_OPS:
        ytr_b = (ytr == op).astype(int)
        yte_b = (yte == op).astype(int)
        if ytr_b.sum() < 5 or yte_b.sum() < 5 or (1 - yte_b).sum() < 5:
            continue
        clf = LogisticRegression(max_iter=500, C=0.1).fit(Xtr, ytr_b)
        score = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte_b, score))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    acts_ck = C.load_acts_checkpoint()
    layers  = acts_ck["layers"]
    best    = C.best_layer_from_results(default=layers[len(layers) // 2])
    print(f"Exp 1b — cross-format control   best_layer={best}")

    # 1. Same operand pairs rendered in all three formats (verbal bins only)
    probes = make_matched_probes(n_per_bin=120, formats=FORMATS,
                                 include_nonsense=False)
    n_pairs = len({r["pair_id"] for r in probes})
    print(f"  {len(probes)} probes  ({n_pairs} operand pairs × {len(FORMATS)} "
          f"formats × {len(REAL_OPS)} ops)")

    # 2. Collect activations once, at every layer
    model, tok, device = C.load_model()
    raw = C.collect_last_token([r["prompt"] for r in probes], model, tok,
                               layers, device, desc="exp1b acts")
    del model

    fmt = np.array([r["fmt"] for r in probes])
    ops = np.array([r["op"]  for r in probes])

    # 3. Train-format × test-format AUC matrix, per layer
    per_layer = {}
    best_offdiag = None
    for l in layers:
        sae   = C.load_sae(l)
        norm  = C.normalize(raw[l], acts_ck["mu"][l], acts_ck["sig"][l])
        feats = np.nan_to_num(C.encode(sae, norm), nan=0.0)

        M = np.full((len(FORMATS), len(FORMATS)), np.nan)
        for i, ftr in enumerate(FORMATS):
            for j, fte in enumerate(FORMATS):
                mtr, mte = (fmt == ftr), (fmt == fte)
                M[i, j] = cross_format_auc(feats[mtr], ops[mtr],
                                           feats[mte], ops[mte])
        offdiag = float(np.nanmean(M[~np.eye(len(FORMATS), dtype=bool)]))
        diag    = float(np.nanmean(np.diag(M)))
        per_layer[l] = dict(matrix=M.tolist(), within_format=diag,
                            cross_format=offdiag)
        print(f"  L{l:2d}:  within-format={diag:.3f}   cross-format={offdiag:.3f}")
        if l == best:
            best_offdiag = offdiag
            best_matrix  = M

    verdict = ("OPERATION TRANSFERS — genuine encoding, proceed to exp2"
               if best_offdiag > 0.80 else
               "PARTIAL — operation transfers weakly across formats"
               if best_offdiag > 0.60 else
               "OPERATOR-TOKEN IDENTITY — reframe toward format-specific circuits")
    print(f"\n  best_layer={best}  cross-format op-AUC={best_offdiag:.3f}  →  {verdict}")

    # 4. Plots — best-layer matrix + cross-format trend across layers
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(best_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0.5, vmax=1,
                xticklabels=FORMATS, yticklabels=FORMATS, linewidths=0.5, ax=axes[0],
                cbar_kws={"label": "op-AUC"})
    axes[0].set(title=f"Train→test op-AUC (L{best})",
                xlabel="test format", ylabel="train format")
    ls = sorted(layers)
    axes[1].plot(ls, [per_layer[l]["within_format"] for l in ls],
                 marker="o", label="within-format", color="steelblue")
    axes[1].plot(ls, [per_layer[l]["cross_format"] for l in ls],
                 marker="s", ls="--", label="cross-format", color="darkorange")
    axes[1].axhline(0.80, ls=":", color="seagreen", alpha=0.7, label="transfer threshold")
    axes[1].axhline(0.50, ls=":", color="tomato",   alpha=0.7, label="chance")
    axes[1].set(xlabel="Layer", ylabel="op-AUC", ylim=(0.4, 1.02),
                title="Operation transfer across formats")
    axes[1].set_xticks(ls); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp1b_cross_format.png", dpi=150, bbox_inches="tight")
    plt.close()

    out = dict(best_layer=best, best_cross_format_auc=best_offdiag, verdict=verdict,
               formats=FORMATS,
               per_layer={str(l): {"within_format": per_layer[l]["within_format"],
                                   "cross_format": per_layer[l]["cross_format"],
                                   "matrix": per_layer[l]["matrix"]} for l in layers})
    (OUT_DIR / "exp1b_cross_format.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults → {OUT_DIR/'exp1b_cross_format.json'}")
    print(f"Plot    → {OUT_DIR/'exp1b_cross_format.png'}")


if __name__ == "__main__":
    main()
