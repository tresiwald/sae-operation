"""
Experiment 3 — Causal hierarchy via cross-operation ablation.

If addition is a sub-operation of multiplication, then ablating the addition
fingerprint should damage multiplication while leaving (say) division intact —
an ASYMMETRIC damage matrix is the signature of containment.

We ablate an operation's fingerprint by subtracting those SAE features'
reconstructed contribution from the residual stream at the best layer (all token
positions, every forward pass), then measure greedy-decode accuracy on each
operation. We build the full N×N "ablate A → accuracy on B" matrix plus a
no-ablation baseline row, and summarise asymmetry with a Baselga-style
nestedness vs. turnover split of the damage.

Reads : results/checkpoints/{acts_checkpoint.pt, sae_L{best}.pt, data.pkl}
Outputs: results/exp3_ablation.json, results/exp3_ablation.png
"""

import json, os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gen import make_hard_problems
from pipeline.config import OPS_EVAL, OUT_DIR
import experiments._common as C

sns.set_theme(style="whitegrid", font_scale=0.9)

N_PER_CELL = int(os.getenv("EXP3_N", "40"))      # problems per (op, bin)
EVAL_BINS  = os.getenv("EXP3_BINS", "1d,2d").split(",")


def make_ablation_hook(sae, mask_t, mu, sig):
    """Pre-hook that removes the masked features' decoded contribution from the
    residual stream at this layer (operates in the SAE's normalized space)."""
    def _h(mod, args):
        h          = args[0]
        orig_dtype = h.dtype
        shape      = h.shape
        flat       = h.float().reshape(-1, shape[-1])
        norm       = (flat - mu) / sig
        acts       = sae.encode(norm)                 # [N, d_sae]
        contrib    = (acts * mask_t) @ sae.W_dec      # [N, d_model]
        flat_new   = (norm - contrib) * sig + mu
        return (flat_new.reshape(shape).to(orig_dtype),) + tuple(args[1:])
    return _h


@torch.no_grad()
def accuracy_by_op(eval_probs, model, tok, device):
    """Greedy-decode accuracy per op for the given eval set."""
    answers = C.generate_answers([r["prompt"] for r in eval_probs], model, tok, device)
    accs = {}
    for op in OPS_EVAL:
        idx = [i for i, r in enumerate(eval_probs) if r["op"] == op]
        if not idx:
            continue
        hits = [answers[i] == eval_probs[i]["expected"] for i in idx]
        accs[op] = float(np.mean(hits))
    return accs


def main():
    acts_ck = C.load_acts_checkpoint()
    layers  = acts_ck["layers"]
    best    = C.best_layer_from_results(default=layers[len(layers) // 2])
    print(f"Exp 3 — cross-op ablation   best_layer={best}  "
          f"eval_bins={EVAL_BINS}  n/cell={N_PER_CELL}")

    # 1. Fingerprints from training data
    with open(C.ckpt_data(), "rb") as f:
        train_recs = C.pickle_load(f)["train_corpus"]
    sae = C.load_sae(best)
    fps, _, _ = C.build_training_fingerprints(best, sae, acts_ck, train_recs)
    ops = [op for op in OPS_EVAL if op in fps]
    print(f"  ablatable ops: {ops}")

    # 2. Eval problems (small bins → high baseline accuracy so damage is visible)
    eval_probs = make_hard_problems(ops=ops, bins=EVAL_BINS, n_per_cell=N_PER_CELL)
    print(f"  {len(eval_probs)} eval problems")

    # 3. Load model; move SAE + stats to device for the hook
    model, tok, device = C.load_model()
    sae = sae.to(device).float()
    mu  = acts_ck["mu"][best].to(device).float()
    sig = acts_ck["sig"][best].to(device).float()
    layer_mod = model.model.layers[best]

    # 4. Baseline (no ablation)
    print("  baseline (no ablation) …")
    baseline = accuracy_by_op(eval_probs, model, tok, device)
    print("   ", {k: f"{v:.0%}" for k, v in baseline.items()})

    # 5. Ablate each op's fingerprint, measure accuracy on every op
    matrix = {}            # matrix[ablated][eval_op] = accuracy
    for a in ops:
        mask_t = torch.tensor(fps[a], device=device)
        handle = layer_mod.register_forward_pre_hook(
            make_ablation_hook(sae, mask_t, mu, sig))
        accs = accuracy_by_op(eval_probs, model, tok, device)
        handle.remove()
        matrix[a] = accs
        drops = {b: baseline[b] - accs.get(b, 0) for b in ops}
        print(f"  ablate {a:3s}:  " +
              "  ".join(f"{b}={accs.get(b,0):.0%}(-{drops[b]*100:.0f})" for b in ops))
    del model

    # 6. Damage matrix D[a,b] = baseline[b] - acc(ablate a)[b], normalized by baseline
    D = np.zeros((len(ops), len(ops)))
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            base = baseline.get(b, 0) + 1e-9
            D[i, j] = (baseline.get(b, 0) - matrix[a].get(b, 0)) / base

    # 7. Asymmetry: self-damage (diagonal) vs cross-damage, and add⊆mul check
    self_dmg  = float(np.mean(np.diag(D)))
    cross_dmg = float((D.sum() - np.trace(D)) / (D.size - len(ops)))
    asym = {}
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            if a != b:
                asym[f"ablate_{a}_hurts_{b}"] = float(D[i, j])

    hierarchy_note = ""
    if "add" in ops and "mul" in ops:
        ia, im = ops.index("add"), ops.index("mul")
        add_into_mul = D[ia, im]    # ablating add damages mul
        mul_into_add = D[im, ia]    # ablating mul damages add
        supported = add_into_mul > mul_into_add + 0.05
        hierarchy_note = (f"add⊆mul: ablate-add→mul damage={add_into_mul:.2f} vs "
                          f"ablate-mul→add damage={mul_into_add:.2f}  "
                          f"→ {'SUPPORTED (asymmetric)' if supported else 'not supported'}")
        print(f"\n  {hierarchy_note}")
    print(f"  mean self-damage={self_dmg:.2f}  mean cross-damage={cross_dmg:.2f}")

    # 8. Plot damage matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(D, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1,
                xticklabels=ops, yticklabels=ops, linewidths=0.5, ax=ax,
                cbar_kws={"label": "relative accuracy drop"})
    ax.set(xlabel="evaluated operation", ylabel="ablated fingerprint",
           title=f"Exp 3 — cross-op ablation damage (L{best})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp3_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    out = dict(
        best_layer=best, ops=ops, eval_bins=EVAL_BINS, n_per_cell=N_PER_CELL,
        baseline=baseline,
        ablation_accuracy={a: matrix[a] for a in ops},
        damage_matrix={a: {b: float(D[i, j]) for j, b in enumerate(ops)}
                       for i, a in enumerate(ops)},
        mean_self_damage=self_dmg, mean_cross_damage=cross_dmg,
        asymmetry=asym, hierarchy_note=hierarchy_note,
    )
    (OUT_DIR / "exp3_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults → {OUT_DIR/'exp3_ablation.json'}")
    print(f"Plot    → {OUT_DIR/'exp3_ablation.png'}")


if __name__ == "__main__":
    main()
