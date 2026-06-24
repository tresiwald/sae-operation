"""
Experiment 6 — Compute vs. retrieve: does the SAE fingerprint flag whether the
model COMPUTED an answer or RETRIEVED/memorised it?

Thesis: a correct answer that the model genuinely computed should be robust to a
tiny, magnitude-preserving change of the operands (347*8 → 348*8). A correct
answer the model *retrieved* (a memorised / heuristic hit) should be brittle —
the neighbour breaks. We call (1 - neighbour accuracy) the family's BRITTLENESS.

Because every member of a family shares operand magnitude, brittleness is a
magnitude-CONTROLLED proxy for retrieval. The decisive test:

    Does the compute fingerprint predict brittleness ABOVE a magnitude-only
    baseline (and within a single magnitude bin)?

If yes → a "truthful fingerprint" that separates computation from retrieval, not
just big numbers from small. If the fingerprint collapses to the magnitude
baseline → the compute/retrieve distinction is an illusion (a Nikankin-style
"bag of heuristics" reading).

Reads : results/checkpoints/{acts_checkpoint.pt, sae_L{best}.pt, data.pkl}
Outputs: results/exp6_robustness.json, results/exp6_per_family.csv,
         results/exp6_robustness.png
"""

import csv, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gen import make_robustness_families
from pipeline.config import OUT_DIR
import experiments._common as C

sns.set_theme(style="whitegrid", font_scale=0.9)

# ── regime (override via env) ─────────────────────────────────────────────────
EXP6_OPS      = os.getenv("EXP6_OPS",  "add,sub,mul").split(",")
EXP6_BINS     = os.getenv("EXP6_BINS", "2d,3d,4d").split(",")
EXP6_FAMILIES = int(os.getenv("EXP6_FAMILIES", "150"))   # families per (op,bin)
EXP6_PERTURB  = int(os.getenv("EXP6_PERTURB",  "8"))     # neighbours per family
EXP6_DELTA    = int(os.getenv("EXP6_DELTA",    "3"))     # max operand nudge


def _safe_auc(label, score):
    if label.sum() < 5 or (~label).sum() < 5:
        return float("nan")
    return float(roc_auc_score(label, score))


def main():
    acts_ck = C.load_acts_checkpoint()
    layers  = acts_ck["layers"]
    best    = C.best_layer_from_results(default=layers[len(layers) // 2])
    print(f"Exp 6 — compute vs retrieve   best_layer={best}")

    # 1. compute fingerprints from training data
    with open(C.ckpt_data(), "rb") as f:
        train_recs = C.pickle_load(f)["train_corpus"]
    sae = C.load_sae(best)
    fps, centroids, _ = C.build_training_fingerprints(best, sae, acts_ck, train_recs)
    ops_use = [op for op in EXP6_OPS if op in fps]
    print(f"  fingerprints for ops: {sorted(fps)}   using: {ops_use}")

    # 2. perturbation families
    fams = make_robustness_families(ops=ops_use, bins=EXP6_BINS,
                                    n_families=EXP6_FAMILIES,
                                    n_perturb=EXP6_PERTURB, max_delta=EXP6_DELTA)
    n_fam = len(set(r["family_id"] for r in fams))
    print(f"  {len(fams)} problems in {n_fam} families "
          f"(ops={ops_use} bins={EXP6_BINS})")

    # 3. accuracy on EVERY member (need neighbour accuracy for brittleness)
    model, tok, device = C.load_model()
    answers = C.generate_answers([r["prompt"] for r in fams], model, tok, device)
    correct = np.array([a == r["expected"] for a, r in zip(answers, fams)])
    print(f"  overall accuracy: {correct.mean():.1%}")

    # 4. per-family brittleness (keep families whose BASE is correct)
    fam_rows = []     # one row per kept family: base record + brittleness
    for fid in sorted(set(r["family_id"] for r in fams)):
        members = [(i, fams[i]) for i in range(len(fams)) if fams[i]["family_id"] == fid]
        base    = [(i, r) for i, r in members if r["role"] == "base"]
        perturb = [(i, r) for i, r in members if r["role"] == "perturb"]
        if not base or not perturb:
            continue
        bi, brec = base[0]
        if not correct[bi]:                 # base wrong → brittleness undefined
            continue
        pacc = float(np.mean([correct[pi] for pi, _ in perturb]))
        fam_rows.append(dict(base_idx=bi, rec=brec, brittleness=1.0 - pacc,
                             n_perturb=len(perturb)))
    print(f"  families with correct base: {len(fam_rows)}")
    if len(fam_rows) < 20:
        print("  WARNING: too few usable families — raise EXP6_FAMILIES or pick "
              "bins where the base is solved but neighbours sometimes break.")

    # 5. compute fingerprint at each kept base's '=' token
    base_prompts = [fr["rec"]["prompt"] for fr in fam_rows]
    raw   = C.collect_last_token(base_prompts, model, tok, [best], device,
                                 desc="exp6 base acts")[best]
    del model
    norm  = C.normalize(raw, acts_ck["mu"][best], acts_ck["sig"][best])
    feats = np.nan_to_num(C.encode(sae, norm), nan=0.0)

    strength = np.array([C.fingerprint_strength(feats[i:i+1], fps[fr["rec"]["op"]])[0]
                         for i, fr in enumerate(fam_rows)])
    cos_comp = np.array([C.cosine_to_centroid(feats[i:i+1], centroids[fr["rec"]["op"]])[0]
                         for i, fr in enumerate(fam_rows)])
    brittle  = np.array([fr["brittleness"] for fr in fam_rows])
    mags     = np.array([[np.log10(fr["rec"]["a"] + 1),
                          np.log10(fr["rec"]["b"] + 1),
                          np.log10((fr["rec"]["expected"] or 0) + 1)]
                         for fr in fam_rows])
    bins_arr = np.array([fr["rec"]["bin"] for fr in fam_rows])

    # 6. label: robust (brittleness <= median) vs brittle. Orient so a STRONG
    #    fingerprint predicting ROBUST gives AUC > 0.5.
    thr      = float(np.median(brittle))
    robust   = brittle <= thr
    print(f"  brittleness: mean={brittle.mean():.2f}  median={thr:.2f}  "
          f"robust={robust.sum()}  brittle={(~robust).sum()}")

    auc_fp   = _safe_auc(robust, strength)
    auc_cos  = _safe_auc(robust, cos_comp)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    def cv_auc(X):
        if robust.sum() < 5 or (~robust).sum() < 5:
            return float("nan")
        return float(cross_val_score(LogisticRegression(max_iter=500, C=0.1),
                                     X, robust.astype(int), cv=cv,
                                     scoring="roc_auc").mean())

    auc_feats = cv_auc(feats)            # full SAE feature probe
    auc_mag   = cv_auc(mags)             # magnitude-only baseline (the confound)
    auc_raw   = cv_auc(norm.numpy())     # raw hidden-state probe

    # continuous (rank) association, the magnitude-free read
    rho_fp,  _ = spearmanr(strength, brittle)
    rho_mag, _ = spearmanr(mags[:, 2], brittle)   # result magnitude vs brittleness

    # 7. WITHIN-BIN control — does the fingerprint still predict brittleness when
    #    magnitude is held fixed by conditioning on the operand bin?
    per_bin_auc = {}
    for bn in EXP6_BINS:
        m = bins_arr == bn
        if m.sum() >= 20 and robust[m].sum() >= 5 and (~robust[m]).sum() >= 5:
            per_bin_auc[bn] = _safe_auc(robust[m], strength[m])

    print(f"\n  predicting ROBUST (computed) families:")
    print(f"    fingerprint strength AUC : {auc_fp:.3f}")
    print(f"    cosine-to-compute    AUC : {auc_cos:.3f}")
    print(f"    full SAE-feature probe   : {auc_feats:.3f}")
    print(f"    magnitude-only baseline  : {auc_mag:.3f}   (the confound)")
    print(f"    raw hidden-state probe   : {auc_raw:.3f}")
    print(f"    Spearman(fp, brittle)    : {rho_fp:+.3f}")
    print(f"    within-bin fp AUC        : " +
          "  ".join(f"{bn}={a:.2f}" for bn, a in per_bin_auc.items()))

    beats_mag = (not np.isnan(auc_feats) and not np.isnan(auc_mag)
                 and auc_feats >= auc_mag + 0.05)
    verdict = ("TRUTHFUL FINGERPRINT — compute fingerprint predicts robustness "
               "ABOVE the magnitude baseline (computation ≠ magnitude)"
               if beats_mag else
               "MAGNITUDE COLLAPSE — fingerprint does not beat magnitude; the "
               "compute/retrieve split is not separable from operand size here")
    print(f"  → {verdict}")

    # 8. money figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    parts = axes[0].violinplot([strength[robust], strength[~robust]],
                               showmeans=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ["seagreen", "tomato"]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    axes[0].set_xticks([1, 2]); axes[0].set_xticklabels(["robust", "brittle"])
    axes[0].set(ylabel="compute fingerprint strength",
                title="Fingerprint: computed vs retrieved")

    axes[1].scatter(strength, brittle, s=14, alpha=0.5, color="slateblue")
    axes[1].set(xlabel="fingerprint strength", ylabel="brittleness",
                title=f"ρ={rho_fp:+.2f}  (magnitude-matched)")

    labels = ["fp\nstrength", "SAE\nprobe", "magnitude\nonly", "raw\nprobe"]
    vals   = [auc_fp, auc_feats, auc_mag, auc_raw]
    cols   = ["seagreen", "steelblue", "lightgray", "darkorange"]
    axes[2].bar(labels, vals, color=cols)
    axes[2].axhline(0.5, ls="--", color="gray")
    axes[2].set(ylim=(0, 1), ylabel="AUC (predict robust)",
                title="Does fingerprint beat magnitude?")
    fig.suptitle(f"Exp 6 — compute vs retrieve (L{best}, {len(fam_rows)} families)",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp6_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 9. per-family dump
    csv_path = OUT_DIR / "exp6_per_family.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family_id", "op", "bin", "a", "b", "expected",
                    "brittleness", "n_perturb", "fp_strength", "cos_compute"])
        for i, fr in enumerate(fam_rows):
            r = fr["rec"]
            w.writerow([r["family_id"], r["op"], r["bin"], r["a"], r["b"],
                        r["expected"], f"{fr['brittleness']:.4f}", fr["n_perturb"],
                        f"{strength[i]:.6f}", f"{cos_comp[i]:.6f}"])

    out = dict(
        best_layer=best, ops=ops_use, bins=EXP6_BINS,
        n_families=len(fam_rows), overall_accuracy=float(correct.mean()),
        brittleness_mean=float(brittle.mean()), brittleness_median=thr,
        auc_fingerprint=auc_fp, auc_cosine=auc_cos, auc_sae_probe=auc_feats,
        auc_magnitude_only=auc_mag, auc_raw_probe=auc_raw,
        spearman_fp_brittle=float(rho_fp), spearman_mag_brittle=float(rho_mag),
        within_bin_fp_auc=per_bin_auc, beats_magnitude=bool(beats_mag),
        verdict=verdict,
    )
    (OUT_DIR / "exp6_robustness.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults     → {OUT_DIR/'exp6_robustness.json'}")
    print(f"Per-family  → {csv_path}")
    print(f"Plot        → {OUT_DIR/'exp6_robustness.png'}")


if __name__ == "__main__":
    main()
