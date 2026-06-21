"""
Stage 4 — Fingerprint analysis, layer sweep, cheat holdout, plots.

Reads  : results/data.pkl
         results/acts_checkpoint.pt
         results/sae_L{layer}.pt  (one per sweep layer)
Outputs: results/results.json
         results/layer_sweep.png
         results/layer_sweep_metrics.png
         results/pca_best_layer.png
         results/fingerprints.png
         results/cheat.png
         results/summary.png
         results/sae_training_all_layers.png
"""

import json, pickle, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import (
    MODEL_NAME, FORMATS, OPS_EVAL, FP_THRESHOLD, CHEAT_SAMPLE,
    get_device, ckpt_data, ckpt_acts, ckpt_sae, ckpt_results, OUT_DIR
)
from pipeline.stage3_sae import TopKSAE

sns.set_theme(style="whitegrid", font_scale=0.9)

COLORS = {"add": "steelblue", "sub": "seagreen", "mul": "darkorange",
          "div": "mediumpurple", "copy": "tomato", "ctrl": "gray"}


# ── helpers ───────────────────────────────────────────────────────────────────
def cosine(a, b):
    na, nb = norm(a), norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


def compute_fps(fd: dict) -> tuple[dict, dict]:
    """Cohen's d fingerprint; baseline = ctrl."""
    if "ctrl" not in fd or len(fd["ctrl"]) == 0:
        return {}, {}
    bm  = fd["ctrl"].mean(0)
    bs  = fd["ctrl"].std(0) + 1e-8
    fps, effs = {}, {}
    for op in OPS_EVAL:
        if op not in fd or len(fd[op]) == 0:
            continue
        d = (fd[op].mean(0) - bm) / np.sqrt((bs**2 + fd[op].std(0)**2 + 1e-8) / 2)
        effs[op] = d
        fps[op]  = d > FP_THRESHOLD
    return fps, effs


def containment(a, b):
    return float((a & b).sum()) / (a.sum() + 1e-8)


def jaccard(fps_list: list) -> float:
    if not fps_list:
        return 0.0
    union = np.array(fps_list).any(0)
    inter = np.array(fps_list).all(0)
    return float(inter.sum()) / (union.sum() + 1e-8)


@torch.no_grad()
def encode(sae: TopKSAE, acts: torch.Tensor) -> np.ndarray:
    return np.concatenate([
        sae.encode(acts[i:i+512]).numpy()
        for i in range(0, len(acts), 512)
    ], axis=0)


def get_slice(feats_np, train_recs, op=None, fmt=None) -> np.ndarray:
    mask = np.ones(len(train_recs), dtype=bool)
    if op  is not None: mask &= np.array([r["op"]  == op  for r in train_recs])
    if fmt is not None: mask &= np.array([r["fmt"] == fmt for r in train_recs])
    return feats_np[mask]


def build_feat_dicts(feats_np, train_recs):
    fa = {op: get_slice(feats_np, train_recs, op=op) for op in OPS_EVAL}
    fa["copy"] = get_slice(feats_np, train_recs, op="copy")
    fa["ctrl"] = get_slice(feats_np, train_recs, op="ctrl")
    fb = {}
    for fmt in FORMATS:
        fd = {op: get_slice(feats_np, train_recs, op=op, fmt=fmt) for op in OPS_EVAL}
        fd["copy"] = get_slice(feats_np, train_recs, op="copy", fmt=fmt)
        fd["ctrl"] = fa["ctrl"]
        fb[fmt] = fd
    return fa, fb


# ── plotting helpers ──────────────────────────────────────────────────────────
def pca_plot(ax, fd, title, d_sae, sub=200):
    ops_p = [op for op in OPS_EVAL + ["copy", "ctrl"] if op in fd and len(fd[op]) >= 5]
    sub_f = {op: fd[op][np.random.choice(len(fd[op]), min(sub, len(fd[op])), replace=False)]
             for op in ops_p}
    X = np.nan_to_num(np.concatenate(list(sub_f.values())), nan=0.0)
    lbls = sum([[op] * len(sub_f[op]) for op in ops_p], [])
    if X.shape[0] < 5:
        ax.set_title(title + " (no data)")
        return
    pcs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    for op in ops_p:
        m = np.array(lbls) == op
        ax.scatter(pcs[m, 0], pcs[m, 1], c=COLORS.get(op, "k"),
                   alpha=0.4, s=10, label=op, rasterized=True)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title(title, fontsize=9)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    with open(ckpt_data(), "rb") as f:
        data = pickle.load(f)
    train_recs = data["train_corpus"]
    hold_cheat = data["hold_cheat"]

    # 2. Load activations
    acts_ckpt = torch.load(ckpt_acts(), weights_only=True)
    layers    = acts_ckpt["layers"]
    norm_acts = acts_ckpt["norm"]    # {layer: tensor}
    norm_mu   = acts_ckpt["mu"]
    norm_sig  = acts_ckpt["sig"]

    # 3. Load SAEs
    saes: dict[int, TopKSAE] = {}
    sae_meta: dict[int, dict] = {}
    for l in layers:
        p = ckpt_sae(l)
        if not p.exists():
            raise FileNotFoundError(f"SAE checkpoint missing: {p}  Run stage3 for layer {l}")
        ck = torch.load(p, weights_only=False)
        D_MODEL = norm_acts[l].shape[1]
        D_SAE   = ck["d_sae"]
        k       = ck["k"]
        sae     = TopKSAE(D_MODEL, D_SAE, k)
        sae.load_state_dict(ck["state_dict"])
        sae.eval()
        saes[l]    = sae
        sae_meta[l] = ck
    D_SAE = sae_meta[layers[0]]["d_sae"]

    print(f"Stage 4 — analysis  layers={layers}  d_sae={D_SAE}  n_train={len(train_recs)}")

    # 4. SAE training curves plot
    fig, axes = plt.subplots(len(layers), 2, figsize=(10, 3 * len(layers)), squeeze=False)
    for row, l in enumerate(layers):
        hist = sae_meta[l]["history"]
        eps  = [h["epoch"] for h in hist]
        axes[row, 0].plot(eps, [h["mse"]  for h in hist], marker="o", ms=2)
        axes[row, 0].set(title=f"Layer {l} MSE", xlabel="Epoch")
        axes[row, 1].plot(eps, [h["dead"] for h in hist], marker="o", ms=2, color="tomato")
        axes[row, 1].set(title=f"Layer {l} Dead/{D_SAE}", xlabel="Epoch")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sae_training_all_layers.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 5. Per-layer fingerprints
    layer_sweep = {}
    for l in layers:
        feats_np = np.nan_to_num(encode(saes[l], norm_acts[l]), nan=0.0)
        fa, fb   = build_feat_dicts(feats_np, train_recs)
        fps_all, _ = compute_fps(fa)
        fps_fmt    = {fmt: compute_fps(fb[fmt])[0] for fmt in FORMATS}
        jacs = {op: jaccard([fps_fmt[f].get(op, np.zeros(D_SAE, bool)) for f in FORMATS])
                for op in OPS_EVAL}

        # AUC
        ops_p = [op for op in OPS_EVAL if len(fa.get(op, [])) >= 10]
        auc_m = float("nan")
        if ops_p and len(fa.get("ctrl", [])) >= 10:
            X = np.concatenate([fa[op] for op in ops_p] + [fa["ctrl"]])
            y = np.array([1] * sum(len(fa[op]) for op in ops_p) + [0] * len(fa["ctrl"]))
            aucs = cross_val_score(LogisticRegression(max_iter=500, C=0.1),
                                   X, y, cv=min(5, int(y.sum())), scoring="roc_auc")
            auc_m = float(aucs.mean())

        fp_sizes = {op: int(fps_all.get(op, np.zeros(D_SAE, bool)).sum()) for op in OPS_EVAL}
        layer_sweep[l] = dict(
            feats_np=feats_np, fa=fa, fb=fb,
            fps_all=fps_all, fps_fmt=fps_fmt,
            fp_sizes=fp_sizes, jaccards=jacs,
            var_expl=sae_meta[l]["var_expl"],
            n_live=sae_meta[l]["n_live"],
            auc=auc_m,
        )
        fp_str = "  ".join(f"{op}={fp_sizes[op]}" for op in OPS_EVAL)
        print(f"  L{l:2d}: var={sae_meta[l]['var_expl']:.1%}  live={sae_meta[l]['n_live']}  "
              f"AUC={auc_m:.3f}  [{fp_str}]")

    # 6. Layer sweep plots
    fp_matrix = np.array([[layer_sweep[l]["fp_sizes"].get(op, 0)
                           for op in OPS_EVAL] for l in layers])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.heatmap(fp_matrix.T, annot=True, fmt="d", ax=axes[0], cmap="YlOrRd",
                xticklabels=[f"L{l}" for l in layers], yticklabels=OPS_EVAL, linewidths=0.5)
    axes[0].set(title="Fingerprint features — layer × op", xlabel="Layer", ylabel="Op")
    for k, op in enumerate(OPS_EVAL):
        axes[1].plot(layers, fp_matrix[:, k], marker="o", label=op,
                     color=list(COLORS.values())[k])
    axes[1].set(xlabel="Layer", ylabel="# fingerprint features",
                title="Fingerprint size across layers")
    axes[1].legend(); axes[1].set_xticks(layers)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "layer_sweep.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(layers, [layer_sweep[l]["var_expl"] for l in layers], marker="o")
    axes[0].set(title="Variance explained", xlabel="Layer"); axes[0].set_xticks(layers)
    axes[1].plot(layers, [layer_sweep[l]["n_live"]   for l in layers], marker="o", color="darkorange")
    axes[1].axhline(D_SAE, ls="--", color="gray", alpha=0.4)
    axes[1].set(title=f"Live features (/{D_SAE})", xlabel="Layer"); axes[1].set_xticks(layers)
    axes[2].plot(layers, [layer_sweep[l]["auc"]      for l in layers], marker="o", color="steelblue")
    axes[2].set(title="AUC (math vs ctrl)", xlabel="Layer", ylim=(0, 1)); axes[2].set_xticks(layers)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "layer_sweep_metrics.png", dpi=150, bbox_inches="tight"); plt.close()

    # Jaccard across layers
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for k, op in enumerate(OPS_EVAL):
        ax.plot(layers, [layer_sweep[l]["jaccards"].get(op, 0) for l in layers],
                marker="o", label=op, color=list(COLORS.values())[k])
    ax.set(xlabel="Layer", ylabel="Jaccard", title="Format invariance across layers", ylim=(0, 1))
    ax.axhline(0.5, ls="--", color="gray", alpha=0.4); ax.legend(); ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "layer_sweep_jaccard.png", dpi=150, bbox_inches="tight"); plt.close()

    # 7. Best layer
    best_layer = max(layers, key=lambda l: np.mean(list(layer_sweep[l]["fp_sizes"].values())))
    print(f"\nBest layer: {best_layer}  fp={layer_sweep[best_layer]['fp_sizes']}")

    fa     = layer_sweep[best_layer]["fa"]
    fb     = layer_sweep[best_layer]["fb"]
    fps_all = layer_sweep[best_layer]["fps_all"]
    fps_fmt = layer_sweep[best_layer]["fps_fmt"]
    jaccards = layer_sweep[best_layer]["jaccards"]
    feats_np = layer_sweep[best_layer]["feats_np"]

    # 8. Format-invariance + containment + AUC table (best layer)
    print(f"\n  FORMAT INVARIANCE (layer {best_layer}):")
    print(f"  {'op':4s}  {'all':>6s}  " + "  ".join(f"{f:>9s}" for f in FORMATS) + "  Jaccard")
    for op in OPS_EVAL:
        n_all  = int(fps_all.get(op, np.zeros(D_SAE, bool)).sum())
        by_fmt = [int(fps_fmt[f].get(op, np.zeros(D_SAE, bool)).sum()) for f in FORMATS]
        print(f"  {op:4s}  {n_all:>6d}  " +
              "  ".join(f"{n:>9d}" for n in by_fmt) +
              f"  {jaccards.get(op, 0):.3f}")

    auc_results = {}
    for sl, fd in [("all", fa)] + [(f, fb[f]) for f in FORMATS]:
        ops_p = [op for op in OPS_EVAL if op in fd and len(fd[op]) >= 10]
        if not ops_p or len(fd.get("ctrl", [])) < 10:
            continue
        X = np.concatenate([fd[op] for op in ops_p] + [fd["ctrl"]])
        y = np.array([1]*sum(len(fd[op]) for op in ops_p) + [0]*len(fd["ctrl"]))
        aucs = cross_val_score(LogisticRegression(max_iter=500, C=0.1), X, y,
                               cv=min(5, int(y.sum())), scoring="roc_auc")
        auc_results[sl] = (float(aucs.mean()), float(aucs.std()))
        print(f"  AUC [{sl:10s}]: {aucs.mean():.3f} ± {aucs.std():.3f}")

    # 9. Fingerprint + Jaccard + containment plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]; x = np.arange(len(OPS_EVAL)); w = 0.18
    sc = {"all": "#444", "symbolic": "steelblue", "mixed": "darkorange", "verbal": "seagreen"}
    for k, sl in enumerate(["all"] + FORMATS):
        fps_sl = fps_all if sl == "all" else fps_fmt[sl]
        vals   = [int(fps_sl.get(op, np.zeros(D_SAE, bool)).sum()) for op in OPS_EVAL]
        ax.bar(x + k*w, vals, w, label=sl, color=sc[sl], alpha=0.85)
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels(OPS_EVAL)
    ax.set(ylabel="# features", title=f"Fingerprint sizes (L{best_layer})"); ax.legend(fontsize=8)

    ax2 = axes[1]
    bars = ax2.bar(OPS_EVAL, [jaccards[op] for op in OPS_EVAL],
                   color=[COLORS[op] for op in OPS_EVAL], alpha=0.85)
    ax2.set(ylim=(0, 1), ylabel="Jaccard", title="Format invariance")
    ax2.axhline(0.5, ls="--", color="gray", alpha=0.5)
    for b, op in zip(bars, OPS_EVAL):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                 f"{jaccards[op]:.2f}", ha="center", fontsize=9)

    ax3 = axes[2]
    mat = np.zeros((len(OPS_EVAL), len(OPS_EVAL)))
    for i, a in enumerate(OPS_EVAL):
        for j, b in enumerate(OPS_EVAL):
            if a in fps_all and b in fps_all:
                mat[i, j] = containment(fps_all[a], fps_all[b])
    sns.heatmap(mat, annot=True, fmt=".2f", ax=ax3, cmap="Blues", vmin=0, vmax=1,
                xticklabels=OPS_EVAL, yticklabels=OPS_EVAL, linewidths=0.5)
    ax3.set(title="Containment [row ⊆ col]")
    plt.suptitle(f"Fingerprint analysis — layer {best_layer}", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fingerprints.png", dpi=150, bbox_inches="tight"); plt.close()

    # 10. PCA
    np.random.seed(0)
    fig, axes_pca = plt.subplots(1, len(FORMATS) + 1, figsize=(5*(len(FORMATS)+1), 4.5))
    pca_plot(axes_pca[0], fa, f"All formats (L{best_layer})", D_SAE)
    for i, fmt in enumerate(FORMATS):
        pca_plot(axes_pca[i+1], fb[fmt], fmt.capitalize(), D_SAE)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_best_layer.png", dpi=150, bbox_inches="tight"); plt.close()

    # 11. Cheat holdout — reload model, hook best layer only
    DEVICE = get_device()
    import os as _os
    _dtype_env = _os.getenv("SAE_DTYPE", "bfloat16" if DEVICE.type == "cuda" else "float32")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[_dtype_env]
    print(f"\nLoading model for cheat holdout (layer {best_layer}) …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model2    = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model2.eval().to(DEVICE)

    cheat_buf = []
    def _cheat_hook(mod, args):
        h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
        cheat_buf.append(h[:, -1, :].detach().float().cpu())

    cheat_handle = model2.model.layers[best_layer].register_forward_pre_hook(_cheat_hook)
    mu, sig      = norm_mu[best_layer], norm_sig[best_layer]
    best_sae     = saes[best_layer]

    cheat_feats = {}
    for op in OPS_EVAL:
        recs = [r for r in hold_cheat if r["op"] == op][:CHEAT_SAMPLE]
        if not recs:
            continue
        cheat_buf.clear()
        with torch.no_grad():
            for rec in tqdm(recs, desc=f"cheat/{op}", leave=False):
                inp = tokenizer(rec["prompt"], return_tensors="pt").to(DEVICE)
                model2(**inp)
        acts = torch.cat(cheat_buf, dim=0)
        acts = torch.nan_to_num(acts, nan=0.0)
        acts = (acts - mu.to(acts)) / sig.to(acts)
        cheat_feats[op] = np.nan_to_num(encode(best_sae, acts.cpu()), nan=0.0)

    cheat_handle.remove()
    del model2

    copy_centroid = fa["copy"].mean(0)
    cheat_res = {}
    print(f"\n  {'op':5s}  {'→compute':>10s}  {'→copy':>8s}  verdict")
    for op in OPS_EVAL:
        if op not in cheat_feats:
            continue
        cc   = cheat_feats[op].mean(0)
        comp = fa[op].mean(0)
        cs, ks = cosine(cc, comp), cosine(cc, copy_centroid)
        cheat_res[op] = dict(comp_sim=cs, copy_sim=ks)
        print(f"  {op:5s}  {cs:>10.4f}  {ks:>8.4f}  {'→copy' if ks>cs else '→compute'}")

    # Cheat plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ops_c = list(cheat_res.keys()); xc = np.arange(len(ops_c)); w = 0.28
    axes[0].bar(xc-w/2, [cheat_res[op]["comp_sim"] for op in ops_c], w,
                label="→compute", color="steelblue", alpha=0.85)
    axes[0].bar(xc+w/2, [cheat_res[op]["copy_sim"] for op in ops_c], w,
                label="→copy",    color="tomato",    alpha=0.85)
    axes[0].set_xticks(xc); axes[0].set_xticklabels(ops_c)
    axes[0].set(ylim=(0, 1), ylabel="Cosine sim", title="Cheat: compute vs copy centroid")
    axes[0].legend()

    overlaps = []
    for op in ops_c:
        bm = fa["ctrl"].mean(0); bs = fa["ctrl"].std(0) + 1e-8
        d  = (cheat_feats[op].mean(0) - bm) / np.sqrt((bs**2 + cheat_feats[op].std(0)**2 + 1e-8)/2)
        fp_ch = d > FP_THRESHOLD
        overlaps.append(containment(fp_ch, fps_all.get(op, np.zeros(D_SAE, bool))))
    bars = axes[1].bar(ops_c, overlaps, color=[COLORS[op] for op in ops_c], alpha=0.85)
    axes[1].set(ylim=(0, 1), ylabel="Containment(cheat fp ⊆ compute fp)",
                title="Cheat fingerprint overlap with compute")
    axes[1].axhline(0.5, ls="--", color="gray", alpha=0.4)
    for b, v in zip(bars, overlaps):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.2f}",
                     ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cheat.png", dpi=150, bbox_inches="tight"); plt.close()

    # 12. Summary dashboard (6-panel)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    fp_matrix2 = np.array([[layer_sweep[l]["fp_sizes"].get(op, 0)
                             for op in OPS_EVAL] for l in layers])
    sns.heatmap(fp_matrix2.T, annot=True, fmt="d", ax=ax, cmap="YlOrRd",
                xticklabels=[f"L{l}" for l in layers], yticklabels=OPS_EVAL, linewidths=0.5)
    ax.set(title="A  Layer sweep (fp features)", xlabel="Layer")

    ax = fig.add_subplot(gs[0, 1])
    for k, op in enumerate(OPS_EVAL):
        ax.plot(layers, [layer_sweep[l]["var_expl"] for l in layers],
                marker="o", label=f"var_expl", color="seagreen")
        ax.plot(layers, [layer_sweep[l]["auc"]      for l in layers],
                marker="s", label=f"AUC",      color="steelblue", ls="--")
        break
    ax.set(title="B  Var explained + AUC", xlabel="Layer", ylim=(0, 1))
    ax.legend(fontsize=8); ax.set_xticks(layers)

    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(OPS_EVAL)); w = 0.18
    for k, sl in enumerate(["all"] + FORMATS):
        fps_sl = fps_all if sl == "all" else fps_fmt[sl]
        vals = [int(fps_sl.get(op, np.zeros(D_SAE, bool)).sum()) for op in OPS_EVAL]
        ax.bar(x+k*w, vals, w, label=sl, color=sc[sl], alpha=0.85)
    ax.set_xticks(x+1.5*w); ax.set_xticklabels(OPS_EVAL)
    ax.set(title=f"C  Fingerprint sizes (L{best_layer})"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(OPS_EVAL, [jaccards.get(op, 0) for op in OPS_EVAL],
           color=[COLORS[op] for op in OPS_EVAL], alpha=0.85)
    ax.set(ylim=(0, 1), title="D  Format invariance (Jaccard)")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.4)

    ax = fig.add_subplot(gs[1, 1])
    mat = np.zeros((len(OPS_EVAL), len(OPS_EVAL)))
    for i, a in enumerate(OPS_EVAL):
        for j, b in enumerate(OPS_EVAL):
            if a in fps_all and b in fps_all:
                mat[i, j] = containment(fps_all[a], fps_all[b])
    sns.heatmap(mat, annot=True, fmt=".2f", ax=ax, cmap="Blues", vmin=0, vmax=1,
                xticklabels=OPS_EVAL, yticklabels=OPS_EVAL, linewidths=0.5)
    ax.set(title=f"E  Containment (L{best_layer})")

    ax = fig.add_subplot(gs[1, 2])
    if cheat_res:
        ax.bar(xc-w/2, [cheat_res[op]["comp_sim"] for op in ops_c], w,
               label="→compute", color="steelblue", alpha=0.85)
        ax.bar(xc+w/2, [cheat_res[op]["copy_sim"] for op in ops_c], w,
               label="→copy",    color="tomato",    alpha=0.85)
        ax.set_xticks(xc); ax.set_xticklabels(ops_c)
        ax.set(ylim=(0, 1), title="F  Cheat proximity"); ax.legend(fontsize=8)

    plt.suptitle(f"SAE Fingerprint — {MODEL_NAME}  best_layer={best_layer}  "
                 f"d_sae={D_SAE}  k={saes[best_layer].k}", fontsize=12, y=1.01)
    plt.savefig(OUT_DIR / "summary.png", dpi=150, bbox_inches="tight"); plt.close()

    # 13. Save results.json
    out = dict(
        model=MODEL_NAME, layers=layers, best_layer=best_layer,
        n_train=len(train_recs), d_sae=D_SAE,
        layer_sweep={str(l): dict(
            var_expl=layer_sweep[l]["var_expl"],
            n_live=layer_sweep[l]["n_live"],
            auc=layer_sweep[l]["auc"],
            fp_sizes=layer_sweep[l]["fp_sizes"],
            jaccards=layer_sweep[l]["jaccards"],
        ) for l in layers},
        fps_best={op: int(fps_all.get(op, np.zeros(D_SAE, bool)).sum()) for op in OPS_EVAL},
        jaccard={op: float(jaccards.get(op, float("nan"))) for op in OPS_EVAL},
        auc={sl: float(m) for sl, (m, _) in auc_results.items()},
        cheat={op: {k: float(v) for k, v in r.items()} for op, r in cheat_res.items()},
    )
    with open(ckpt_results(), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults → {ckpt_results()}")
    print("Plots   → " + str(OUT_DIR))
    print("=" * 60)
    print(f"  Best layer : {best_layer}")
    for op in OPS_EVAL:
        print(f"  {op}: {out['fps_best'][op]:4d} features  Jaccard={jaccards.get(op,0):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
