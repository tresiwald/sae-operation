"""
Experiment 2 — Error mechanism: is a wrong answer one the model emitted WITHOUT
turning on its arithmetic?

Hypothesis: on problems the model gets right, it fires the operation's compute
fingerprint before emitting the answer. On problems it gets wrong, it skips the
computation (pattern-completes), so the compute fingerprint is weak/absent —
resembling the copy/cheat condition.

We use larger operand bins (mul/div, 3d-5d) where Gemma-3-1B makes mistakes,
split by correctness, and compare compute-fingerprint strength. A mandatory
baseline (logistic probe on the raw hidden state) tells us whether the
fingerprint adds anything over a vanilla probe.

Reads : results/checkpoints/{acts_checkpoint.pt, sae_L{best}.pt, data.pkl}
Outputs: results/exp2_error_mechanism.json, results/exp2_error_mechanism.png
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
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gen import make_hard_problems
from pipeline.config import OUT_DIR
import experiments._common as C

sns.set_theme(style="whitegrid", font_scale=0.9)


def main():
    acts_ck = C.load_acts_checkpoint()
    layers  = acts_ck["layers"]
    best    = C.best_layer_from_results(default=layers[len(layers) // 2])
    print(f"Exp 2 — error mechanism   best_layer={best}")

    # 1. Compute fingerprints from the training data
    with open(C.ckpt_data(), "rb") as f:
        train_recs = C.pickle_load(f)["train_corpus"]
    sae = C.load_sae(best)
    fps, centroids, _ = C.build_training_fingerprints(best, sae, acts_ck, train_recs)
    print(f"  fingerprints for ops: {sorted(fps)}")

    # 2. Hard problems → correctness + last-token activations
    probs = make_hard_problems(ops=[op for op in ["mul", "div"] if op in fps],
                               bins=["3d", "4d", "5d"], n_per_cell=150)
    print(f"  {len(probs)} hard problems")

    model, tok, device = C.load_model()
    answers = C.generate_answers([r["prompt"] for r in probs], model, tok, device)
    raw     = C.collect_last_token([r["prompt"] for r in probs], model, tok,
                                   [best], device, desc="exp2 acts")[best]
    del model

    correct = np.array([a == r["expected"] for a, r in zip(answers, probs)])
    print(f"  accuracy: {correct.mean():.1%}  "
          f"({correct.sum()} correct / {(~correct).sum()} wrong)")
    if correct.sum() < 10 or (~correct).sum() < 10:
        print("  WARNING: too few of one class — increase bins or n_per_cell "
              "(need both correct and wrong for a clean comparison).")

    # 3. Fingerprint strength per problem (under its own op's fingerprint)
    norm  = C.normalize(raw, acts_ck["mu"][best], acts_ck["sig"][best])
    feats = np.nan_to_num(C.encode(sae, norm), nan=0.0)

    strength = np.zeros(len(probs))
    cos_comp = np.zeros(len(probs))
    for i, r in enumerate(probs):
        op = r["op"]
        strength[i] = C.fingerprint_strength(feats[i:i+1], fps[op])[0]
        cos_comp[i] = C.cosine_to_centroid(feats[i:i+1], centroids[op])[0]

    # 4. Does fingerprint strength predict correctness?  (AUC, oriented so that
    #    "strong fingerprint ⇒ correct" gives AUC > 0.5)
    def safe_auc(score, label):
        if label.sum() < 5 or (~label).sum() < 5:
            return float("nan")
        return float(roc_auc_score(label, score))

    auc_strength = safe_auc(strength, correct)
    auc_cosine   = safe_auc(cos_comp, correct)

    # 5. Baseline — logistic probe on the raw hidden state
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    if correct.sum() >= 5 and (~correct).sum() >= 5:
        auc_probe = float(cross_val_score(
            LogisticRegression(max_iter=500, C=0.1),
            norm.numpy(), correct.astype(int), cv=cv, scoring="roc_auc").mean())
    else:
        auc_probe = float("nan")

    print(f"\n  predicting correctness:")
    print(f"    fingerprint strength AUC : {auc_strength:.3f}")
    print(f"    cosine-to-compute    AUC : {auc_cosine:.3f}")
    print(f"    hidden-state probe   AUC : {auc_probe:.3f}  (baseline)")
    verdict = ("MECHANISM SUPPORTED — wrong answers have weaker fingerprints"
               if auc_strength > 0.65 else
               "NO FINGERPRINT SIGNAL — wrong answers fire the fingerprint normally")
    beats_baseline = (not np.isnan(auc_strength) and not np.isnan(auc_probe)
                      and auc_strength >= auc_probe - 0.03)
    print(f"  → {verdict}")
    print(f"  → fingerprint {'matches/beats' if beats_baseline else 'underperforms'} "
          f"the raw-probe baseline")

    # 6. Plot — the money figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, score, name in [(axes[0], strength, "Fingerprint strength"),
                            (axes[1], cos_comp, "Cosine to compute centroid")]:
        data = [score[correct], score[~correct]]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc, col in zip(parts["bodies"], ["seagreen", "tomato"]):
            pc.set_facecolor(col); pc.set_alpha(0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["correct", "wrong"])
        ax.set(ylabel=name, title=name)
    fig.suptitle(f"Exp 2 — compute fingerprint vs. correctness (L{best}, "
                 f"acc={correct.mean():.0%})", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp2_error_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close()

    out = dict(
        best_layer=best, n=len(probs), accuracy=float(correct.mean()),
        n_correct=int(correct.sum()), n_wrong=int((~correct).sum()),
        auc_fingerprint_strength=auc_strength, auc_cosine=auc_cosine,
        auc_hidden_probe=auc_probe, beats_baseline=bool(beats_baseline),
        verdict=verdict,
    )
    (OUT_DIR / "exp2_error_mechanism.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults → {OUT_DIR/'exp2_error_mechanism.json'}")
    print(f"Plot    → {OUT_DIR/'exp2_error_mechanism.png'}")


if __name__ == "__main__":
    main()
