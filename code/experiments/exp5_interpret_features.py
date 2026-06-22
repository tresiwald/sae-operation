"""
Experiment 5 — What do the fingerprint features actually encode?

Instead of treating an operation's fingerprint as an opaque feature *set*, we ask
what its top features *mean*. For each op we take the strongest fingerprint
features (highest Cohen's d) and correlate each feature's activation, across that
op's records, with hand-crafted interpretable properties of the problem:
result magnitude, parity, operand sizes, carry, digit count, operand equality.

This turns "addition uses these 90 features" into "addition's top features track
result magnitude / parity / …", a positive characterisation of the fingerprint.

Reads : results/checkpoints/{acts_checkpoint.pt, sae_L{best}.pt, data.pkl}
Outputs: results/exp5_interpret_features.json,
         results/exp5_feature_properties.csv,
         results/exp5_interpret_features.png
"""

import csv, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import OPS_EVAL, OUT_DIR
import experiments._common as C

sns.set_theme(style="whitegrid", font_scale=0.9)
TOP_N = int(os.getenv("EXP5_TOPN", "15"))   # top features per op to interpret


def properties(rec):
    """Interpretable numeric properties of an arithmetic problem."""
    a, b, c = rec.get("a"), rec.get("b"), rec.get("expected")
    if a is None or b is None or c is None:
        return None
    return {
        "result_mag":   np.log10(abs(c) + 1),
        "result_parity": float(c % 2),
        "a_mag":        np.log10(abs(a) + 1),
        "b_mag":        np.log10(abs(b) + 1),
        "operand_gap":  np.log10(abs(a - b) + 1),
        "units_carry":  float((a % 10 + b % 10) >= 10),
        "n_digits":     float(len(str(abs(c)))),
        "a_eq_b":       float(a == b),
    }


def main():
    acts_ck = C.load_acts_checkpoint()
    best    = C.best_layer_from_results(default=acts_ck["layers"][len(acts_ck["layers"])//2])
    print(f"Exp 5 — feature interpretation   best_layer={best}  top_n={TOP_N}")

    with open(C.ckpt_data(), "rb") as f:
        train_recs = C.pickle_load(f)["train_corpus"]
    sae   = C.load_sae(best)
    feats = np.nan_to_num(C.encode(sae, acts_ck["norm"][best]), nan=0.0)

    ctrl = feats[np.array([r["op"] == "ctrl" for r in train_recs])]
    prop_names = list(properties({"a": 1, "b": 1, "expected": 2}).keys())

    per_op = {}
    rows   = []   # for the CSV
    summary_mat = np.zeros((len(prop_names), len(OPS_EVAL)))   # mean |rho| per (prop, op)

    for oi, op in enumerate(OPS_EVAL):
        # symbolic records for this op + their feature matrix and properties
        idx = [i for i, r in enumerate(train_recs)
               if r["op"] == op and r.get("fmt") == "symbolic"
               and properties(r) is not None]
        if len(idx) < 20:
            continue
        op_feats = feats[idx]
        props    = {p: np.array([properties(train_recs[i])[p] for i in idx])
                    for p in prop_names}

        # fingerprint = top features by Cohen's d vs ctrl
        d, _ = C.cohens_d_fingerprint(op_feats, ctrl)
        top  = np.argsort(d)[::-1][:TOP_N]

        feat_records = []
        for f in top:
            acts_f = op_feats[:, f]
            if acts_f.std() < 1e-9:
                continue
            corrs = {}
            for p in prop_names:
                if props[p].std() < 1e-9:
                    corrs_val = 0.0
                else:
                    rho, _ = spearmanr(acts_f, props[p])
                    corrs_val = 0.0 if np.isnan(rho) else float(rho)
                corrs[p] = corrs_val
            best_p = max(corrs, key=lambda k: abs(corrs[k]))
            feat_records.append(dict(feature=int(f), cohens_d=float(d[f]),
                                     best_property=best_p,
                                     best_rho=corrs[best_p], corr=corrs))
            rows.append([op, int(f), f"{d[f]:.3f}", best_p, f"{corrs[best_p]:+.3f}"]
                        + [f"{corrs[p]:+.3f}" for p in prop_names])
            for pj, p in enumerate(prop_names):
                summary_mat[pj, oi] += abs(corrs[p])

        n_used = max(len(feat_records), 1)
        summary_mat[:, oi] /= n_used
        # which property dominates this op's fingerprint?
        dom = prop_names[int(np.argmax(summary_mat[:, oi]))]
        per_op[op] = dict(n_records=len(idx), dominant_property=dom,
                          features=feat_records)
        print(f"  {op:4s}: {len(idx):4d} recs   dominant property = {dom}"
              f"  (mean|ρ|={summary_mat[:, oi].max():.2f})")
        top3 = sorted(feat_records, key=lambda r: abs(r["best_rho"]), reverse=True)[:3]
        for r in top3:
            print(f"        feat {r['feature']:5d}  d={r['cohens_d']:.2f}  "
                  f"{r['best_property']}={r['best_rho']:+.2f}")

    # CSV dump
    csv_path = OUT_DIR / "exp5_feature_properties.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "feature", "cohens_d", "best_property", "best_rho"] + prop_names)
        w.writerows(rows)

    # summary heatmap: property × op (mean |rho| over top features)
    fig, ax = plt.subplots(figsize=(7, 5))
    ops_have = [op for op in OPS_EVAL if op in per_op]
    cols = [OPS_EVAL.index(op) for op in ops_have]
    sns.heatmap(summary_mat[:, cols], annot=True, fmt=".2f", cmap="viridis",
                xticklabels=ops_have, yticklabels=prop_names, ax=ax,
                cbar_kws={"label": "mean |Spearman ρ| over top features"})
    ax.set(title=f"Exp 5 — what each op's fingerprint encodes (L{best})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp5_interpret_features.png", dpi=150, bbox_inches="tight")
    plt.close()

    out = dict(best_layer=best, top_n=TOP_N, properties=prop_names, per_op=per_op)
    (OUT_DIR / "exp5_interpret_features.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults      → {OUT_DIR/'exp5_interpret_features.json'}")
    print(f"Feature CSV  → {csv_path}")
    print(f"Plot         → {OUT_DIR/'exp5_interpret_features.png'}")


if __name__ == "__main__":
    main()
