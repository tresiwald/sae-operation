"""
Generation-time interpretation — does producing the answer reveal digit-level
structure that the static prompt snapshot missed?

Three questions:
  1. CONTRASTIVE fingerprint (compute vs magnitude-matched copy): once magnitude
     is subtracted out, do any operation-specific features survive at generation
     time?  (exp5 showed prompt-time fingerprints were mostly magnitude.)
  2. STRUCTURE: do the surviving features track per-digit properties —
     place value, the digit being produced — rather than gross magnitude?
  3. DECODE: can the actual digit value (0–9) be predicted from features?
     Spearman misses non-monotonic content; multiclass logistic is the right test.

For each op we take the top compute-vs-copy features and:
  - correlate them with magnitude props and generation-only props (incl. carry)
  - run a multiclass digit-decode test (5-fold CV logistic regression)

carry_add: binary carry INTO the current digit position for addition.
  carry_add[p] = 1 if (a % 10^p + b % 10^p) >= 10^p, else 0.  (p=0 → always 0)
  NaN for non-add ops.

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import OPS_EVAL, OUT_DIR
import experiments._common as C
from experiments.gen_common import gen_acts_path, load_gen_sae, properties_gen, GEN_PROPS

sns.set_theme(style="whitegrid", font_scale=0.9)
TOP_N        = int(os.getenv("GEN_TOPN", "15"))
MAG_PROPS    = {"result_mag", "a_mag", "b_mag"}
STRUCT_PROPS = {"place_from_right", "target_digit", "is_leading", "carry_add"}
ALL_PROPS    = GEN_PROPS + ["carry_add"]   # carry appended so heatmap row order is stable


def _carry_add(m):
    """Binary carry INTO digit position place_from_right for addition. NaN otherwise."""
    if m.get("op") != "add":
        return np.nan
    a, b, pfr = m.get("a"), m.get("b"), m.get("place_from_right", -1)
    if a is None or b is None or pfr < 0:
        return np.nan
    if pfr == 0:
        return 0.0
    p = int(pfr)
    return float((a % (10**p) + b % (10**p)) >= 10**p)


def _digit_decode(feat_mat, target_digit_arr):
    """5-fold CV logistic regression predicting target_digit (0–9) from features.
    Returns dict(accuracy, chance) or None if too few samples."""
    ok = ~np.isnan(target_digit_arr)
    if ok.sum() < 50:
        return None
    X = feat_mat[ok]
    y = target_digit_arr[ok].astype(int)
    n_cls = len(np.unique(y))
    if n_cls < 2:
        return None
    clf    = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                multi_class="multinomial")
    scores = cross_val_score(clf, X, y, cv=min(5, n_cls), scoring="accuracy")
    return dict(accuracy=float(scores.mean()), std=float(scores.std()),
                chance=1.0 / n_cls, n_samples=int(ok.sum()))


def main():
    ck    = torch.load(gen_acts_path(), weights_only=False)
    layer = ck["layer"]
    meta  = ck["meta"]
    sae   = load_gen_sae(layer)
    feats = np.nan_to_num(C.encode(sae, ck["acts"]), nan=0.0)
    print(f"Gen-interpret — layer {layer}  {feats.shape[0]} positions  top_n={TOP_N}")

    variant = np.array([m["variant"] for m in meta])
    op_arr  = np.array([m["op"]      for m in meta])
    # attach op label to each meta record so _carry_add can read it
    for m in meta:
        pass  # op already in meta

    copy_mask = variant == "copy"     # magnitude-matched baseline

    summary_mat  = np.zeros((len(ALL_PROPS), len(OPS_EVAL)))
    per_op, rows = {}, []
    struct_scores = {}
    decode_results = {}

    for oi, op in enumerate(OPS_EVAL):
        comp_mask = (variant == "compute") & (op_arr == op)
        if comp_mask.sum() < 20 or copy_mask.sum() < 20:
            continue
        comp_feats = feats[comp_mask]
        comp_meta  = [meta[i] for i in np.where(comp_mask)[0]]
        for m in comp_meta:
            m["op"] = op   # ensure op tag present for _carry_add

        # CONTRASTIVE fingerprint: compute vs copy (magnitude matched)
        d, _ = C.cohens_d_fingerprint(comp_feats, feats[copy_mask])
        top  = np.argsort(d)[::-1][:TOP_N]

        # properties for the compute positions of this op (ALL_PROPS incl. carry_add)
        props = {p: [] for p in ALL_PROPS}
        for i, m_i in zip(np.where(comp_mask)[0], comp_meta):
            pr = properties_gen(meta[i])
            for p in GEN_PROPS:
                props[p].append(pr[p] if pr else np.nan)
            props["carry_add"].append(_carry_add(m_i))
        props = {p: np.array(v) for p, v in props.items()}

        feat_recs = []
        for f in top:
            acts_f = comp_feats[:, f]
            if acts_f.std() < 1e-9:
                continue
            corrs = {}
            for p in ALL_PROPS:
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
                        + [f"{corrs[p]:+.3f}" for p in ALL_PROPS])
            for pj, p in enumerate(ALL_PROPS):
                summary_mat[pj, oi] += abs(corrs[p])

        n_used = max(len(feat_recs), 1)
        summary_mat[:, oi] /= n_used
        mag_score    = float(np.mean([summary_mat[ALL_PROPS.index(p), oi] for p in MAG_PROPS]))
        struct_score = float(np.mean([summary_mat[ALL_PROPS.index(p), oi] for p in STRUCT_PROPS
                                      if p in ALL_PROPS]))
        struct_scores[op] = dict(magnitude=mag_score, structure=struct_score)
        dom = ALL_PROPS[int(np.argmax(summary_mat[:, oi]))]
        per_op[op] = dict(n_positions=int(comp_mask.sum()),
                          dominant_property=dom, features=feat_recs)
        print(f"  {op:4s}: dom={dom:16s}  mag|ρ|={mag_score:.2f}  "
              f"struct|ρ|={struct_score:.2f}")

        # DECODE TEST: can we predict which digit (0–9) from top features?
        dec = _digit_decode(comp_feats[:, top], props["target_digit"])
        if dec:
            decode_results[op] = dec
            print(f"         digit-decode acc={dec['accuracy']:.2f}  "
                  f"chance={dec['chance']:.2f}  n={dec['n_samples']}")
        else:
            print(f"         digit-decode skipped (too few single-digit positions)")

    # CSV
    csv_path = OUT_DIR / "gen_feature_properties.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "feature", "cohens_d", "best_property", "best_rho"] + ALL_PROPS)
        w.writerows(rows)

    # plot: property × op heatmap | magnitude-vs-structure bar | digit-decode bar
    ops_have = [op for op in OPS_EVAL if op in per_op]
    cols     = [OPS_EVAL.index(op) for op in ops_have]
    n_panels = 3 if decode_results else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.6),
                             gridspec_kw={"width_ratios": [3] + [2] * (n_panels - 1)})

    sns.heatmap(summary_mat[:, cols], annot=True, fmt=".2f", cmap="viridis",
                xticklabels=ops_have, yticklabels=ALL_PROPS, ax=axes[0],
                cbar_kws={"label": "mean |ρ| over top compute-vs-copy features"})
    axes[0].axhline(3, color="white", lw=2)   # divide magnitude (top 3) / structure (rest)
    axes[0].set(title=f"Generation-time: what features encode (L{layer})")

    x = np.arange(len(ops_have)); bw = 0.38
    axes[1].bar(x - bw/2, [struct_scores[o]["magnitude"] for o in ops_have], bw,
                label="magnitude", color="steelblue")
    axes[1].bar(x + bw/2, [struct_scores[o]["structure"] for o in ops_have], bw,
                label="digit structure", color="seagreen")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ops_have)
    axes[1].set(ylabel="mean |ρ|", title="Magnitude vs. digit structure")
    axes[1].legend(fontsize=8)

    if decode_results and n_panels == 3:
        dec_ops = [o for o in ops_have if o in decode_results]
        dx = np.arange(len(dec_ops))
        acc    = [decode_results[o]["accuracy"] for o in dec_ops]
        chance = [decode_results[o]["chance"]   for o in dec_ops]
        axes[2].bar(dx, acc, 0.5, label="decode acc", color="mediumpurple")
        axes[2].plot(dx, chance, "k--", lw=1.5, label="chance")
        axes[2].set_xticks(dx); axes[2].set_xticklabels(dec_ops)
        axes[2].set(ylabel="accuracy", ylim=(0, 1),
                    title="Digit content decode (0–9 prediction)")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "gen_interpret.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_struct = float(np.mean([s["structure"] for s in struct_scores.values()]))
    mean_mag    = float(np.mean([s["magnitude"] for s in struct_scores.values()]))
    verdict = ("STRUCTURE EMERGES — generation features track digits/place, "
               "not just magnitude" if mean_struct > mean_mag and mean_struct > 0.2
               else "STILL MAGNITUDE — generation features track size, not digit structure")
    print(f"\n  mean magnitude|ρ|={mean_mag:.2f}  mean structure|ρ|={mean_struct:.2f}")
    if decode_results:
        mean_dec = np.mean([v["accuracy"] for v in decode_results.values()])
        mean_chc = np.mean([v["chance"]   for v in decode_results.values()])
        decode_lift = mean_dec - mean_chc
        print(f"  digit decode: acc={mean_dec:.2f}  chance={mean_chc:.2f}  "
              f"lift={decode_lift:+.2f}")
        verdict += (f" | digit-decode lift={decode_lift:+.2f}")
    print(f"  → {verdict}")

    out = dict(layer=layer, top_n=TOP_N, properties=ALL_PROPS,
               mean_magnitude=mean_mag, mean_structure=mean_struct,
               verdict=verdict, struct_scores=struct_scores, per_op=per_op,
               decode=decode_results)
    (OUT_DIR / "gen_interpret.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults     → {OUT_DIR/'gen_interpret.json'}")
    print(f"Feature CSV → {csv_path}")
    print(f"Plot        → {OUT_DIR/'gen_interpret.png'}")


if __name__ == "__main__":
    main()
