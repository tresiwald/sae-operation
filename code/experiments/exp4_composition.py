"""
Experiment 4 — Compositional diagnostic: does a compound expression's last-token
fingerprint contain BOTH of its operations, or only the outer one?

For `(a+b)×c=`, the `=` token sits right after the OUTER operation (×); the inner
(+) was computed many tokens earlier, at a position the (last-token) SAE never
sees. So we expect the compound fingerprint to look like the OUTER op and be
blind to the INNER op — which would precisely delimit what last-token SAE
fingerprints can characterize.

Method: apply each atomic operation's symbolic fingerprint to the last-token
activations of the multi-op holdout, and bin its activation strength by the
op's ROLE in that compound — outer / inner / absent.

Prediction:
  outer ≫ inner ≈ absent   → last-token blindness (compound = outer op only)
  outer > inner > absent    → partial compositional superposition

Reads : results/checkpoints/{acts_checkpoint.pt, sae_L{best}.pt, data.pkl}
Outputs: results/exp4_composition.json, results/exp4_composition.png
"""

import json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import OPS_EVAL, OUT_DIR
import experiments._common as C

sns.set_theme(style="whitegrid", font_scale=0.9)
EXP4_N = int(os.getenv("EXP4_N", "1500"))   # cap compounds (one forward pass each)


def symbolic_fingerprints(layer, sae, acts_ck, train_recs):
    """Per-op fingerprints built from SYMBOLIC training records only (so the
    compound test isn't confounded by the format brittleness from exp1b)."""
    feats = np.nan_to_num(C.encode(sae, acts_ck["norm"][layer]), nan=0.0)
    def slc(pred):
        return feats[np.array([pred(r) for r in train_recs])]
    ctrl = slc(lambda r: r["op"] == "ctrl")
    fps = {}
    for op in OPS_EVAL:
        fo = slc(lambda r, op=op: r["op"] == op and r.get("fmt") == "symbolic")
        if len(fo) >= 5 and len(ctrl) >= 5:
            _, fps[op] = C.cohens_d_fingerprint(fo, ctrl)
    return fps


def roles_in(rec):
    """Return {op: 'outer'|'inner'} for the two operations of a compound."""
    op1, op2 = rec["ops_used"]
    if rec["structure"] == "left":      # (a op1 b) op2 c  → op2 outer
        return {op1: "inner", op2: "outer"}
    else:                               # a op1 (b op2 c)  → op1 outer
        return {op2: "inner", op1: "outer"}


def main():
    acts_ck = C.load_acts_checkpoint()
    best    = C.best_layer_from_results(default=acts_ck["layers"][len(acts_ck["layers"])//2])
    print(f"Exp 4 — compositional diagnostic   best_layer={best}")

    with open(C.ckpt_data(), "rb") as f:
        data = C.pickle_load(f)
    train_recs = data["train_corpus"]
    sae = C.load_sae(best)
    fps = symbolic_fingerprints(best, sae, acts_ck, train_recs)
    print(f"  fingerprints: {sorted(fps)}")

    # compounds: symbolic, compute, two DISTINCT operations (clear inner/outer)
    comp = [r for r in data["hold_multi_compute"]
            if r.get("fmt") == "symbolic" and r["variant"] == "compute"
            and r["ops_used"][0] != r["ops_used"][1]]
    import random
    random.Random(0).shuffle(comp)
    comp = comp[:EXP4_N]
    print(f"  {len(comp)} compound expressions (distinct-op, symbolic)")

    model, tok, device = C.load_model()
    raw = C.collect_last_token([r["prompt"] for r in comp], model, tok,
                               [best], device, desc="exp4 acts")[best]
    del model

    norm  = C.normalize(raw, acts_ck["mu"][best], acts_ck["sig"][best])
    feats = np.nan_to_num(C.encode(sae, norm), nan=0.0)

    # strength of each atomic op's fingerprint on each compound, binned by role
    by_role = {op: {"outer": [], "inner": [], "absent": []} for op in fps}
    for i, rec in enumerate(comp):
        role = roles_in(rec)
        for op in fps:
            s = C.fingerprint_strength(feats[i:i+1], fps[op])[0]
            by_role[op][role.get(op, "absent")].append(s)

    # summarise
    summary, ratios = {}, []
    print(f"\n  {'op':4s}  {'outer':>8s}  {'inner':>8s}  {'absent':>8s}   "
          f"inner-lift")
    for op in fps:
        m = {r: float(np.mean(v)) if v else float("nan")
             for r, v in by_role[op].items()}
        # how much of the outer-vs-absent gap does the inner role recover?
        denom = (m["outer"] - m["absent"]) or 1e-9
        inner_lift = (m["inner"] - m["absent"]) / denom
        ratios.append(inner_lift)
        summary[op] = dict(mean=m, n={r: len(v) for r, v in by_role[op].items()},
                           inner_lift=float(inner_lift))
        print(f"  {op:4s}  {m['outer']:8.3f}  {m['inner']:8.3f}  "
              f"{m['absent']:8.3f}   {inner_lift:+.2f}")

    mean_lift = float(np.nanmean(ratios))
    verdict = ("COMPOSITIONAL — inner op is visible at the last token"
               if mean_lift > 0.5 else
               "PARTIAL — inner op weakly visible"
               if mean_lift > 0.2 else
               "LAST-TOKEN BLIND — compound ≈ outer op only")
    print(f"\n  mean inner-lift = {mean_lift:+.2f}  →  {verdict}")
    print("  (inner-lift 1.0 = inner op as visible as outer; 0.0 = invisible)")

    # plot — grouped bars per op
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ops = list(fps); x = np.arange(len(ops)); w = 0.26
    cols = {"outer": "steelblue", "inner": "darkorange", "absent": "lightgray"}
    for k, role in enumerate(["outer", "inner", "absent"]):
        vals = [summary[op]["mean"][role] for op in ops]
        ax.bar(x + (k-1)*w, vals, w, label=role, color=cols[role], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(ops)
    ax.set(ylabel="fingerprint strength on compound",
           title=f"Exp 4 — does (a∘b)∘c show its inner op? (L{best}, "
                 f"mean inner-lift={mean_lift:+.2f})")
    ax.legend(title="op's role in compound")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp4_composition.png", dpi=150, bbox_inches="tight")
    plt.close()

    out = dict(best_layer=best, n_compounds=len(comp), mean_inner_lift=mean_lift,
               verdict=verdict, per_op=summary)
    (OUT_DIR / "exp4_composition.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults → {OUT_DIR/'exp4_composition.json'}")
    print(f"Plot    → {OUT_DIR/'exp4_composition.png'}")


if __name__ == "__main__":
    main()
