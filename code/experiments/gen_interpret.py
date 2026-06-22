"""
Generation-time interpretation — does producing the answer reveal digit-level
structure that the static prompt snapshot missed?

Two questions:
  1. CONTRASTIVE fingerprint (compute vs magnitude-matched copy): once magnitude
     is subtracted out, do any operation-specific features survive at generation
     time?  (exp5 showed prompt-time fingerprints were mostly magnitude.)
  2. STRUCTURE: do the surviving features track per-digit properties —
     place value, the digit being produced — rather than gross magnitude?

For each op we take the top compute-vs-copy features and correlate them with
both magnitude props (result/operand size) and generation-only props
(place_from_right, target_digit, is_leading).

Reads : results/checkpoints/{gen_acts.pt, gen_sae_L{layer}.pt}
Outputs: results/gen_interpret.json, results/gen_feature_properties.csv,
         results/gen_interpret.png
"""

import csv, json, os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import OPS_EVAL, OUT_DIR
import experiments._common as C
from experiments.gen_common import gen_acts_path, load_gen_sae, properties_gen, GEN_PROPS

sns.set_theme(style="whitegrid", font_scale=0.9)
TOP_N      = int(os.getenv("GEN_TOPN", "15"))
MAG_PROPS  = {"result_mag", "a_mag", "b_mag"}
STRUCT_PROPS = {"place_from_right", "target_digit", "is_leading"}


def main():
    ck    = torch.load(gen_acts_path(), weights_only=False)
    layer = ck["layer"]
    meta  = ck["meta"]
    sae   = load_gen_sae(layer)
    feats = np.nan_to_num(C.encode(sae, ck["acts"]), nan=0.0)
    print(f"Gen-interpret — layer {layer}  {feats.shape[0]} positions  top_n={TOP_N}")

    variant = np.array([m["variant"] for m in meta])
    op_arr  = np.array([m["op"]      for m in meta])
    copy_mask = variant == "copy"     # magnitude-matched baseline

    summary_mat = np.zeros((len(GEN_PROPS), len(OPS_EVAL)))
    per_op, rows = {}, []
    struct_scores = {}

    for oi, op in enumerate(OPS_EVAL):
        comp_mask = (variant == "compute") & (op_arr == op)
        if comp_mask.sum() < 20 or copy_mask.sum() < 20:
            continue
        comp_feats = feats[comp_mask]

        # CONTRASTIVE fingerprint: compute vs copy (magnitude matched)
        d, _ = C.cohens_d_fingerprint(comp_feats, feats[copy_mask])
        top  = np.argsort(d)[::-1][:TOP_N]

        # properties for the compute positions of this op
        props = {p: [] for p in GEN_PROPS}
        for i in np.where(comp_mask)[0]:
            pr = properties_gen(meta[i])
            for p in GEN_PROPS:
                props[p].append(pr[p] if pr else np.nan)
        props = {p: np.array(v) for p, v in props.items()}

        feat_recs = []
        for f in top:
            acts_f = comp_feats[:, f]
            if acts_f.std() < 1e-9:
                continue
            corrs = {}
            for p in GEN_PROPS:
                v = props[p]
                ok = ~np.isnan(v)
                if ok.sum() < 20 or v[ok].std() < 1e-9 or acts_f[ok].std() < 1e-9:
                    corrs[p] = 0.0
                else:
                    rho, _ = spearmanr(acts_f[ok], v[ok])
                    corrs[p] = 0.0 if np.isnan(rho) else float(rho)
            best_p = max(corrs, key=lambda k: abs(corrs[k]))
            feat_recs.append(dict(feature=int(f), cohens_d=float(d[f]),
                                  best_property=best_p, best_rho=corrs[best_p],
                                  corr=corrs))
            rows.append([op, int(f), f"{d[f]:.3f}", best_p, f"{corrs[best_p]:+.3f}"]
                        + [f"{corrs[p]:+.3f}" for p in GEN_PROPS])
            for pj, p in enumerate(GEN_PROPS):
                summary_mat[pj, oi] += abs(corrs[p])

        n_used = max(len(feat_recs), 1)
        summary_mat[:, oi] /= n_used
        mag_score    = float(np.mean([summary_mat[GEN_PROPS.index(p), oi] for p in MAG_PROPS]))
        struct_score = float(np.mean([summary_mat[GEN_PROPS.index(p), oi] for p in STRUCT_PROPS]))
        struct_scores[op] = dict(magnitude=mag_score, structure=struct_score)
        dom = GEN_PROPS[int(np.argmax(summary_mat[:, oi]))]
        per_op[op] = dict(n_positions=int(comp_mask.sum()),
                          dominant_property=dom, features=feat_recs)
        print(f"  {op:4s}: dom={dom:16s}  mag|ρ|={mag_score:.2f}  "
              f"struct|ρ|={struct_score:.2f}")

    # CSV
    csv_path = OUT_DIR / "gen_feature_properties.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "feature", "cohens_d", "best_property", "best_rho"] + GEN_PROPS)
        w.writerows(rows)

    # plot: property × op heatmap + magnitude-vs-structure summary
    ops_have = [op for op in OPS_EVAL if op in per_op]
    cols = [OPS_EVAL.index(op) for op in ops_have]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [3, 2]})
    sns.heatmap(summary_mat[:, cols], annot=True, fmt=".2f", cmap="viridis",
                xticklabels=ops_have, yticklabels=GEN_PROPS, ax=axes[0],
                cbar_kws={"label": "mean |ρ| over top compute-vs-copy features"})
    axes[0].axhline(3, color="white", lw=2)   # divide magnitude / structure
    axes[0].set(title=f"Generation-time: what features encode (L{layer})")
    x = np.arange(len(ops_have)); w = 0.38
    axes[1].bar(x - w/2, [struct_scores[o]["magnitude"] for o in ops_have], w,
                label="magnitude", color="steelblue")
    axes[1].bar(x + w/2, [struct_scores[o]["structure"] for o in ops_have], w,
                label="digit structure", color="seagreen")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ops_have)
    axes[1].set(ylabel="mean |ρ|", title="Magnitude vs. digit structure")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gen_interpret.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_struct = float(np.mean([s["structure"] for s in struct_scores.values()]))
    mean_mag    = float(np.mean([s["magnitude"] for s in struct_scores.values()]))
    verdict = ("STRUCTURE EMERGES — generation features track digits/place, "
               "not just magnitude" if mean_struct > mean_mag and mean_struct > 0.2
               else "STILL MAGNITUDE — generation features track size, not digit structure")
    print(f"\n  mean magnitude|ρ|={mean_mag:.2f}  mean structure|ρ|={mean_struct:.2f}")
    print(f"  → {verdict}")

    out = dict(layer=layer, top_n=TOP_N, properties=GEN_PROPS,
               mean_magnitude=mean_mag, mean_structure=mean_struct,
               verdict=verdict, struct_scores=struct_scores, per_op=per_op)
    (OUT_DIR / "gen_interpret.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults     → {OUT_DIR/'gen_interpret.json'}")
    print(f"Feature CSV → {csv_path}")
    print(f"Plot        → {OUT_DIR/'gen_interpret.png'}")


if __name__ == "__main__":
    main()
