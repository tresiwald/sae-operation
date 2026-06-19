"""
SAE Operational Fingerprint — Math Operations on Gemma-3-1B

Pipeline:
  1. Load Gemma-3-1B float32 on MPS
  2. Generate data via data_gen.py  (add/sub/mul/div/copy, 3 formats, 5 bins)
  3. Collect residual-stream activations (batch_size=1, no NaN)
  4. Train shared TopK SAE on all training activations
  5. Run full fingerprint analysis 4×:
       - all formats combined
       - symbolic only
       - mixed only
       - verbal only
  6. Evaluate 4 holdout sets (H1–H4), each also split by format

Run:
  source .venv/bin/activate
  python code/math_fingerprint.py
"""

import os, sys, json, re as _re, random as _random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from data_gen import make_dataset, make_multi_op_holdout, split_dataset, make_ctrl_data, BINS

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Inference device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/gemma-3-1b-pt"

# Layers to sweep. One SAE is trained per layer; all hooks fire in a single
# inference pass. Set to None to sweep all layers.
LAYERS       = [4, 8, 13, 17, 22, 25]   # early / mid / late coverage

# n_per_cell=150 → ~10.7k training examples (80/20 split)
N_PER_CELL   = 150

SAE_K        = 32
SAE_RATIO    = 8           # d_sae = d_model * SAE_RATIO
SAE_LR       = 3e-4
SAE_EPOCHS   = 30
SAE_BATCH    = 512
WARMUP_STEPS = 200

MEASURE_ACCURACY = False   # set True to run greedy-decode accuracy (~44 min on MPS)
ACC_SAMPLE   = 200         # records per op to sample if MEASURE_ACCURACY=True

FP_THRESHOLD = 0.5         # Cohen's d threshold for fingerprint membership
AUX_W        = 1 / 32
DEAD_THR     = 1e-4

FORMATS      = ["symbolic", "mixed", "verbal"]
OPS_EVAL     = ["add", "sub", "mul", "div"]

OUT_DIR  = Path(os.getenv("SAE_OUT_DIR",  "results"))             # plots + JSON
CKPT_DIR = Path(os.getenv("SAE_CKPT_DIR", str(OUT_DIR / "checkpoints")))  # large binaries
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Model ──────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} in float32...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval().to(DEVICE)

D_MODEL  = model.config.hidden_size
D_SAE    = D_MODEL * SAE_RATIO
N_LAYERS = model.config.num_hidden_layers

if LAYERS is None:
    LAYERS = list(range(N_LAYERS))

print(f"d_model={D_MODEL}  d_sae={D_SAE}  n_layers={N_LAYERS}  sweeping={LAYERS}")

preds = tokenizer("3+5=", return_tensors="pt").to(DEVICE)
with torch.no_grad():
    lg = model(**preds).logits[0, -1].float().cpu().topk(3)
print(f"Sanity  3+5= → {[(tokenizer.decode([i]).strip(), round(v.item(),1)) for i,v in zip(lg.indices,lg.values)]}")

# ── 2. Multi-layer activation hooks ──────────────────────────────────────────
# All hooks fire in a single inference pass — one buffer per layer.
_layer_bufs: dict[int, list[torch.Tensor]] = {l: [] for l in LAYERS}

def _make_hook(layer_idx: int):
    def _hook(module, args):
        hidden = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
        _layer_bufs[layer_idx].append(hidden[:, -1, :].detach().float().cpu())
    return _hook

_hook_handles = [
    model.model.layers[l].register_forward_pre_hook(_make_hook(l))
    for l in LAYERS
]

@torch.no_grad()
def collect_acts_all_layers(records, desc="acts"):
    """Single inference pass; fills _layer_bufs for every layer in LAYERS."""
    for l in LAYERS:
        _layer_bufs[l].clear()
    for i, rec in enumerate(tqdm(records, desc=desc, leave=False)):
        inp = tokenizer(rec["prompt"], return_tensors="pt").to(DEVICE)
        model(**inp)
        if DEVICE.type == "mps" and i % 500 == 499:
            torch.mps.empty_cache()
    return {l: torch.cat(_layer_bufs[l], dim=0) for l in LAYERS}

# ── 3. Accuracy (multi-token greedy decode) ───────────────────────────────────
@torch.no_grad()
def measure_accuracy(records):
    for rec in tqdm(records, desc="acc", leave=False):
        if rec.get("expected") is None:
            rec["correct"] = None
            continue
        inp = tokenizer(rec["prompt"], return_tensors="pt").to(DEVICE)
        out = model.generate(
            inp["input_ids"], max_new_tokens=6, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(out[0, inp["input_ids"].shape[1]:]).strip()
        m = _re.match(r"\d+", decoded)
        rec["correct"] = (int(m.group()) == rec["expected"]) if m else False

# ── 4. Data generation ────────────────────────────────────────────────────────
print(f"\nGenerating dataset (n_per_cell={N_PER_CELL})...")
all_data = make_dataset(n_per_cell=N_PER_CELL, ops=OPS_EVAL)
train, hold_per_op, hold_cheat = split_dataset(all_data, holdout_frac=0.2)

multi_data         = make_multi_op_holdout(n_per_cell=N_PER_CELL // 3)
hold_multi_compute = [r for r in multi_data if r["variant"] == "compute"]
hold_multi_cheat   = [r for r in multi_data if r["variant"] == "cheat"]

# ctrl = neutral non-math text; used as Cohen's d baseline for fingerprinting.
# copy = number-repetition condition; used as a labelled comparison point.
# Neither enters the fingerprint definition — ctrl anchors it.
N_CTRL = max(1000, len(train) // 10)
ctrl_data = make_ctrl_data(N_CTRL)

train_corpus = [r for r in train if r["variant"] in ("compute", "copy")] + ctrl_data

print(f"  Train corpus:       {len(train_corpus):>6}  (incl. {N_CTRL} ctrl)")
print(f"  H1 per-op compute:  {len(hold_per_op):>6}")
print(f"  H2 per-op cheat:    {len(hold_cheat):>6}")
print(f"  H3 multi-op:        {len(hold_multi_compute):>6}")
print(f"  H4 multi-op cheat:  {len(hold_multi_cheat):>6}")

# ── 5. Accuracy on H1 (per-op × format × bin) ────────────────────────────────
acc_table = {}   # (op, fmt, bin) → float
if MEASURE_ACCURACY:
    import random as _rand
    _rng = _random.Random(0)
    _sample = []
    for op in OPS_EVAL:
        grp = [r for r in hold_per_op if r["op"] == op and r["variant"] == "compute"]
        _rng.shuffle(grp)
        _sample.extend(grp[:ACC_SAMPLE])
    print(f"\nMeasuring accuracy on {len(_sample)} sampled H1 records ...")
    measure_accuracy(_sample)
    print(f"\n  {'op':4s}  {'fmt':8s}  {'bin':3s}  {'n':>4s}  acc")
    for op in OPS_EVAL:
        for fmt in FORMATS:
            for bin_name in BINS:
                grp = [r for r in _sample
                       if r["op"] == op and r["fmt"] == fmt and r["bin"] == bin_name]
                if not grp:
                    continue
                acc = float(np.mean([r["correct"] for r in grp]))
                acc_table[(op, fmt, bin_name)] = acc
                print(f"  {op:4s}  {fmt:8s}  {bin_name:3s}  {len(grp):>4d}  {acc:.1%}")
else:
    print("\n(Accuracy measurement skipped — set MEASURE_ACCURACY=True to enable)")

# ── 6. Collect activations at all layers in one pass ─────────────────────────
train_recs: list[dict] = train_corpus
CKPT_ACTS = CKPT_DIR / "acts_checkpoint.pt"

norm_by_layer: dict[int, torch.Tensor] = {}
norm_stats:    dict[int, tuple]         = {}

if CKPT_ACTS.exists():
    print(f"\nLoading cached activations from {CKPT_ACTS} ...")
    ckpt = torch.load(CKPT_ACTS, weights_only=True)
    for l in LAYERS:
        norm_by_layer[l] = ckpt["norm"][l]
        norm_stats[l]    = (ckpt["mu"][l], ckpt["sig"][l])
        print(f"  Layer {l:2d}: {norm_by_layer[l].shape}")
    for h in _hook_handles:
        h.remove()
    del model
else:
    print("\nCollecting activations (all layers in one inference pass)...")
    print(f"  ctrl={sum(1 for r in train_recs if r['op']=='ctrl')}  "
          f"copy={sum(1 for r in train_recs if r['op']=='copy')}  "
          f"math={sum(1 for r in train_recs if r['op'] not in ('ctrl','copy'))}")

    raw_by_layer = collect_acts_all_layers(train_recs, desc="train (all layers)")
    for h in _hook_handles:
        h.remove()

    for l, raw in raw_by_layer.items():
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mu  = raw.mean(dim=0, keepdim=True)
        sig = raw.std(dim=0,  keepdim=True).clamp(min=1e-6)
        norm_by_layer[l] = (raw - mu) / sig
        norm_stats[l]    = (mu, sig)
        print(f"  Layer {l:2d}: {raw.shape}  mean={norm_by_layer[l].mean():.4f}  std={norm_by_layer[l].std():.4f}")

    torch.save(dict(
        norm={l: norm_by_layer[l] for l in LAYERS},
        mu={l: norm_stats[l][0] for l in LAYERS},
        sig={l: norm_stats[l][1] for l in LAYERS},
    ), CKPT_ACTS)
    print(f"  Activations saved → {CKPT_ACTS}")

    del model
    if str(DEVICE) == "mps":
        torch.mps.empty_cache()

# ── 7. TopK SAE (one per layer) ───────────────────────────────────────────────
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.k = k
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.W_enc    = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc    = nn.Parameter(torch.zeros(d_sae))
        self.W_dec    = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec    = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = nn.functional.normalize(self.W_dec.data, dim=1)

    def encode(self, x):
        pre = (x - self.pre_bias) @ self.W_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, vals.clamp(min=0))
        return acts

    def decode(self, acts): return acts @ self.W_dec + self.b_dec
    def forward(self, x):   acts = self.encode(x); return acts, self.decode(acts)

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.data = nn.functional.normalize(self.W_dec.data, dim=1)

def train_sae(train_acts: torch.Tensor, layer_idx: int) -> tuple[TopKSAE, list, float, int]:
    """Train one SAE; return (sae, history, var_expl, n_live)."""
    sae       = TopKSAE(D_MODEL, D_SAE, SAE_K)
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    loader    = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_acts), batch_size=SAE_BATCH, shuffle=True)
    feat_usage = torch.zeros(D_SAE)
    history    = []
    step       = 0

    for epoch in range(SAE_EPOCHS):
        e_mse = 0.0
        for (x,) in loader:
            lr = SAE_LR * min(1.0, step / max(1, WARMUP_STEPS))
            for pg in optimizer.param_groups: pg["lr"] = lr
            step += 1
            acts, x_hat = sae(x)
            with torch.no_grad():
                feat_usage = 0.99 * feat_usage + 0.01 * (acts > 0).float().mean(0)
            mse  = (x - x_hat).pow(2).mean()
            dead = (feat_usage < DEAD_THR).float()
            aux  = ((x - x_hat).detach() - (acts * dead) @ sae.W_dec).pow(2).mean() \
                   if dead.sum() > 0 else torch.tensor(0.0)
            optimizer.zero_grad()
            (mse + AUX_W * aux).backward()
            nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step(); sae.normalise_decoder()
            e_mse += mse.item()
        n_dead   = (feat_usage < DEAD_THR).sum().item()
        mean_mse = e_mse / len(loader)
        if not torch.isfinite(torch.tensor(mean_mse)):
            print(f"    L{layer_idx} epoch {epoch+1}: NaN — stopping"); break
        history.append(dict(epoch=epoch+1, mse=mean_mse, dead=n_dead))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    L{layer_idx}  ep {epoch+1:2d}/{SAE_EPOCHS}  mse={mean_mse:.5f}  dead={n_dead}/{D_SAE}")

    sae.eval()
    with torch.no_grad():
        s = train_acts[:1000]
        var_expl = float(1 - (s - sae(s)[1]).pow(2).sum() / s.pow(2).sum())
    n_live = int((feat_usage > DEAD_THR).sum())
    return sae, history, var_expl, n_live

# ── 8. Train SAE for each layer ───────────────────────────────────────────────
saes:       dict[int, TopKSAE] = {}
sae_stats:  dict[int, dict]    = {}
CKPT_SAE = CKPT_DIR / "sae_checkpoint.pt"

if CKPT_SAE.exists():
    print(f"\nLoading cached SAEs from {CKPT_SAE} ...")
    ckpt_sae = torch.load(CKPT_SAE, weights_only=False)
    for l in LAYERS:
        sae = TopKSAE(D_MODEL, D_SAE, SAE_K)
        sae.load_state_dict(ckpt_sae["state_dicts"][l])
        sae.eval()
        saes[l]      = sae
        sae_stats[l] = ckpt_sae["stats"][l]
        print(f"  Layer {l}: var_expl={sae_stats[l]['var_expl']:.2%}  live={sae_stats[l]['n_live']}")
else:
    print(f"\nTraining SAEs for layers {LAYERS}  (d_sae={D_SAE}  k={SAE_K}  epochs={SAE_EPOCHS})")
    for l in LAYERS:
        print(f"\n  ── Layer {l} ──")
        sae, history, var_expl, n_live = train_sae(norm_by_layer[l], l)
        saes[l]      = sae
        sae_stats[l] = dict(layer=l, history=history, var_expl=var_expl, n_live=n_live)
        print(f"    Layer {l}: var_expl={var_expl:.2%}  live={n_live}/{D_SAE}")
    torch.save(dict(
        state_dicts={l: saes[l].state_dict() for l in LAYERS},
        stats=sae_stats,
    ), CKPT_SAE)
    print(f"  SAEs saved → {CKPT_SAE}")

# Training curves (one row per layer)
fig, axes = plt.subplots(len(LAYERS), 2, figsize=(10, 3 * len(LAYERS)), squeeze=False)
for row, l in enumerate(LAYERS):
    hist = sae_stats[l]["history"]
    eps  = [h["epoch"] for h in hist]
    axes[row, 0].plot(eps, [h["mse"]  for h in hist], marker="o", ms=2)
    axes[row, 0].set(title=f"Layer {l} — MSE", xlabel="Epoch")
    axes[row, 1].plot(eps, [h["dead"] for h in hist], marker="o", ms=2, color="tomato")
    axes[row, 1].set(title=f"Layer {l} — Dead/{D_SAE}", xlabel="Epoch")
plt.tight_layout()
plt.savefig(OUT_DIR / "sae_training_all_layers.png", dpi=120)
plt.close()

# ── 9. Feature extraction helpers ─────────────────────────────────────────────
@torch.no_grad()
def to_features(sae: TopKSAE, acts_tensor: torch.Tensor) -> np.ndarray:
    return np.concatenate([
        sae.encode(acts_tensor[i:i+512]).numpy()
        for i in range(0, len(acts_tensor), 512)
    ], axis=0)

def get_feat_slice(feats_np, op=None, fmt=None) -> np.ndarray:
    mask = np.ones(len(train_recs), dtype=bool)
    if op  is not None: mask &= np.array([r["op"]  == op  for r in train_recs])
    if fmt is not None: mask &= np.array([r["fmt"] == fmt for r in train_recs])
    return feats_np[mask]

# ── 10. Fingerprint analysis helpers ─────────────────────────────────────────
def cosine(a, b):
    na, nb = norm(a), norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

def compute_fingerprints(feats_by_op: dict) -> tuple[dict, dict]:
    """Baseline = ctrl (neutral non-math text)."""
    if "ctrl" not in feats_by_op or len(feats_by_op["ctrl"]) == 0:
        return {}, {}
    base_mean = feats_by_op["ctrl"].mean(axis=0)
    base_std  = feats_by_op["ctrl"].std(axis=0) + 1e-8
    fps, effs = {}, {}
    for op in OPS_EVAL:
        if op not in feats_by_op or len(feats_by_op[op]) == 0: continue
        op_mean = feats_by_op[op].mean(axis=0)
        pooled  = np.sqrt((base_std**2 + feats_by_op[op].std(axis=0)**2 + 1e-8) / 2)
        d = (op_mean - base_mean) / pooled
        effs[op] = d; fps[op] = d > FP_THRESHOLD
    return fps, effs

def containment(fp_a, fp_b):
    return float((fp_a & fp_b).sum()) / (fp_a.sum() + 1e-8)

def jaccard(fps_by_fmt: list) -> float:
    if not fps_by_fmt: return 0.0
    union = np.array(fps_by_fmt).any(axis=0)
    inter = np.array(fps_by_fmt).all(axis=0)
    return float(inter.sum()) / (union.sum() + 1e-8)

# ── 11. Per-layer analysis ────────────────────────────────────────────────────
layer_results: dict[int, dict] = {}

print("\nRunning fingerprint analysis per layer …")
for l in LAYERS:
    sae     = saes[l]
    acts    = norm_by_layer[l]
    feats_np = np.nan_to_num(to_features(sae, acts), nan=0.0)

    # Build feature dicts for all formats and per-format
    feats_all = {op: get_feat_slice(feats_np, op=op) for op in OPS_EVAL}
    feats_all["copy"] = get_feat_slice(feats_np, op="copy")
    feats_all["ctrl"] = get_feat_slice(feats_np, op="ctrl")

    feats_by_fmt = {}
    for fmt in FORMATS:
        fd = {op: get_feat_slice(feats_np, op=op, fmt=fmt) for op in OPS_EVAL}
        fd["copy"] = get_feat_slice(feats_np, op="copy", fmt=fmt)
        fd["ctrl"] = feats_all["ctrl"]   # ctrl has no fmt tag
        feats_by_fmt[fmt] = fd

    fps_all, _   = compute_fingerprints(feats_all)
    fps_by_fmt   = {fmt: compute_fingerprints(feats_by_fmt[fmt])[0] for fmt in FORMATS}

    # Jaccard across formats per op
    jaccards = {}
    for op in OPS_EVAL:
        fps_list = [fps_by_fmt[f].get(op, np.zeros(D_SAE, bool)) for f in FORMATS]
        jaccards[op] = jaccard(fps_list)

    # AUC
    X = np.concatenate([feats_all[op] for op in OPS_EVAL if len(feats_all.get(op, [])) >= 10]
                       + [feats_all["ctrl"]])
    ops_p = [op for op in OPS_EVAL if len(feats_all.get(op, [])) >= 10]
    y = np.array([1]*sum(len(feats_all[op]) for op in ops_p) + [0]*len(feats_all["ctrl"]))
    auc_mean = auc_std = float("nan")
    if len(ops_p) and len(feats_all["ctrl"]) >= 10:
        aucs = cross_val_score(LogisticRegression(max_iter=500, C=0.1),
                               X, y, cv=min(5, int(y.sum())), scoring="roc_auc")
        auc_mean, auc_std = float(aucs.mean()), float(aucs.std())

    layer_results[l] = dict(
        var_expl   = sae_stats[l]["var_expl"],
        n_live     = sae_stats[l]["n_live"],
        fp_sizes   = {op: int(fps_all.get(op, np.zeros(D_SAE, bool)).sum()) for op in OPS_EVAL},
        jaccards   = jaccards,
        auc        = auc_mean,
        feats_all  = feats_all,
        fps_all    = fps_all,
        fps_by_fmt = fps_by_fmt,
        feats_by_fmt = feats_by_fmt,
    )
    fp_str = "  ".join(f"{op}={layer_results[l]['fp_sizes'][op]}" for op in OPS_EVAL)
    print(f"  Layer {l:2d}: var={sae_stats[l]['var_expl']:.1%}  live={sae_stats[l]['n_live']}  "
          f"AUC={auc_mean:.3f}  fps=[{fp_str}]")

# ── 12. Layer sweep visualisations ────────────────────────────────────────────
COLORS = {"add":"steelblue","sub":"seagreen","mul":"darkorange",
          "div":"mediumpurple","copy":"tomato","ctrl":"gray"}

# (a) Fingerprint size per layer × op — heatmap + line plot
fp_matrix = np.array([[layer_results[l]["fp_sizes"].get(op, 0)
                        for op in OPS_EVAL] for l in LAYERS])  # (n_layers, n_ops)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns_ax = axes[0]
import seaborn as sns
sns.heatmap(fp_matrix.T, annot=True, fmt="d", ax=sns_ax, cmap="YlOrRd",
            xticklabels=[f"L{l}" for l in LAYERS], yticklabels=OPS_EVAL, linewidths=0.5)
sns_ax.set_title("Fingerprint features per layer × op", fontweight="bold")
sns_ax.set_xlabel("Layer"); sns_ax.set_ylabel("Operation")

ax2 = axes[1]
for k, op in enumerate(OPS_EVAL):
    ax2.plot(LAYERS, fp_matrix[:, k], marker="o", label=op, color=list(COLORS.values())[k])
ax2.set_xlabel("Layer"); ax2.set_ylabel("# fingerprint features")
ax2.set_title("Fingerprint size across layers"); ax2.legend(); ax2.set_xticks(LAYERS)
plt.tight_layout()
plt.savefig(OUT_DIR / "layer_sweep_fingerprints.png", dpi=150)
plt.close()

# (b) AUC, variance explained, live features across layers
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
layer_labels = [f"L{l}" for l in LAYERS]
axes[0].plot(LAYERS, [layer_results[l]["auc"]      for l in LAYERS], marker="o", color="steelblue")
axes[0].set(title="Linear AUC (math vs ctrl)", xlabel="Layer", ylabel="AUC", ylim=(0,1)); axes[0].set_xticks(LAYERS)
axes[1].plot(LAYERS, [layer_results[l]["var_expl"] for l in LAYERS], marker="o", color="seagreen")
axes[1].set(title="Variance explained", xlabel="Layer", ylabel="var_expl"); axes[1].set_xticks(LAYERS)
axes[2].plot(LAYERS, [layer_results[l]["n_live"]   for l in LAYERS], marker="o", color="darkorange")
axes[2].axhline(D_SAE, ls="--", color="gray", alpha=0.4); axes[2].set_xticks(LAYERS)
axes[2].set(title=f"Live features (/{D_SAE})", xlabel="Layer", ylabel="n_live")
plt.suptitle(f"Layer sweep — {MODEL_NAME}", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "layer_sweep_metrics.png", dpi=150)
plt.close()

# (c) Jaccard (format invariance) across layers
fig, ax = plt.subplots(figsize=(9, 3.5))
for k, op in enumerate(OPS_EVAL):
    ax.plot(LAYERS, [layer_results[l]["jaccards"].get(op, 0) for l in LAYERS],
            marker="o", label=op, color=list(COLORS.values())[k])
ax.set_xlabel("Layer"); ax.set_ylabel("Jaccard (format invariance)"); ax.set_ylim(0, 1)
ax.set_title("Format invariance of fingerprints across layers"); ax.legend(); ax.set_xticks(LAYERS)
ax.axhline(0.5, ls="--", color="gray", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_DIR / "layer_sweep_jaccard.png", dpi=150)
plt.close()
print(f"\nLayer sweep plots → {OUT_DIR}/layer_sweep_*.png")

# Pick best layer for detailed analysis (highest mean fingerprint size)
best_layer = max(LAYERS, key=lambda l: np.mean(list(layer_results[l]["fp_sizes"].values())))
print(f"\nBest layer for fingerprints: {best_layer}  "
      f"(fp_sizes={layer_results[best_layer]['fp_sizes']})")

# ── 13. Detailed analysis on best layer ──────────────────────────────────────
print(f"\nDetailed analysis on layer {best_layer} …")
feats_all    = layer_results[best_layer]["feats_all"]
fps_combined = layer_results[best_layer]["fps_all"]
feats_by_fmt = layer_results[best_layer]["feats_by_fmt"]
fps_by_fmt   = layer_results[best_layer]["fps_by_fmt"]

# Format invariance summary
print(f"\n  FORMAT INVARIANCE (layer {best_layer}):")
print(f"  {'op':4s}  {'all':>6s}", "  ".join(f"{f:>9s}" for f in FORMATS))
for op in OPS_EVAL:
    n_all  = int(fps_combined.get(op, np.zeros(D_SAE, bool)).sum())
    by_fmt = [int(fps_by_fmt[f].get(op, np.zeros(D_SAE, bool)).sum()) for f in FORMATS]
    j      = layer_results[best_layer]["jaccards"].get(op, 0)
    print(f"  {op:4s}  {n_all:>6d}", "  ".join(f"{n:>9d}" for n in by_fmt),
          f"  Jaccard={j:.3f}")

# PCA — all formats + per format
import seaborn as sns
SUB = 200
fig, axes_pca = plt.subplots(1, len(FORMATS) + 1, figsize=(5*(len(FORMATS)+1), 4.5))

def pca_plot(ax, fd, title):
    ops_p = [op for op in OPS_EVAL + ["copy","ctrl"] if op in fd and len(fd[op]) >= 5]
    sub_f = {op: fd[op][np.random.choice(len(fd[op]), min(SUB,len(fd[op])), replace=False)]
             for op in ops_p}
    X = np.nan_to_num(np.concatenate(list(sub_f.values())), nan=0.0)
    lbls = sum([[op]*len(sub_f[op]) for op in ops_p], [])
    if X.shape[0] < 5: ax.set_title(title + " (no data)"); return
    pcs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    for op in ops_p:
        m = np.array(lbls) == op
        ax.scatter(pcs[m,0], pcs[m,1], c=COLORS.get(op,"k"), alpha=0.4, s=10, label=op)
    ax.legend(fontsize=7); ax.set_title(title, fontsize=9)

np.random.seed(0)
pca_plot(axes_pca[0], feats_all, f"All formats  (L{best_layer})")
for i, fmt in enumerate(FORMATS):
    pca_plot(axes_pca[i+1], feats_by_fmt[fmt], fmt.capitalize())
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_best_layer.png", dpi=150)
plt.close()

# ── 14. H2: Cheat holdout on best layer ──────────────────────────────────────
print(f"\nH2: Cheat holdout (layer {best_layer}) — reloading model …")
model2 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model2.eval().to(DEVICE)
cheat_layer_buf: list[torch.Tensor] = []

def _cheat_hook(module, args):
    hidden = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
    cheat_layer_buf.append(hidden[:,-1,:].detach().float().cpu())

cheat_hook = model2.model.layers[best_layer].register_forward_pre_hook(_cheat_hook)
mu, sig    = norm_stats[best_layer]
best_sae   = saes[best_layer]

cheat_feats: dict[str, np.ndarray] = {}
for op in OPS_EVAL:
    recs = [r for r in hold_cheat if r["op"] == op][:300]
    if not recs: continue
    cheat_layer_buf.clear()
    for rec in tqdm(recs, desc=f"cheat/{op}", leave=False):
        inp = tokenizer(rec["prompt"], return_tensors="pt").to(DEVICE)
        model2(**inp)
    acts = torch.cat(cheat_layer_buf, dim=0)
    acts = torch.nan_to_num(acts, nan=0.0)
    acts = (acts - mu) / sig
    cheat_feats[op] = np.nan_to_num(to_features(best_sae, acts), nan=0.0)

cheat_hook.remove(); del model2
if str(DEVICE) == "mps": torch.mps.empty_cache()

copy_centroid = feats_all["copy"].mean(axis=0)
print(f"\n  {'op':5s}  {'→compute':>10s}  {'→copy':>8s}  verdict")
for op in OPS_EVAL:
    if op not in cheat_feats: continue
    cc = cheat_feats[op].mean(axis=0); comp_c = feats_all[op].mean(axis=0)
    cs = cosine(cc, comp_c); ks = cosine(cc, copy_centroid)
    print(f"  {op:5s}  {cs:>10.4f}  {ks:>8.4f}  {'→copy' if ks>cs else '→compute'}")

# ── 15. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Model:       {MODEL_NAME}")
print(f"  Layers swept:{LAYERS}")
print(f"  Best layer:  {best_layer}")
print(f"  Train rows:  {len(train_recs)}")
print(f"  SAE:         d_sae={D_SAE}  k={SAE_K}  epochs={SAE_EPOCHS}")
print()
print(f"  {'Layer':>6}  {'var_expl':>9}  {'n_live':>7}  {'AUC':>6}  " +
      "  ".join(f"{op:>5}" for op in OPS_EVAL))
for l in LAYERS:
    r = layer_results[l]
    fp_str = "  ".join(f"{r['fp_sizes'].get(op,0):>5}" for op in OPS_EVAL)
    marker = "  ◀ best" if l == best_layer else ""
    print(f"  {l:>6}  {r['var_expl']:>9.1%}  {r['n_live']:>7}  {r['auc']:>6.3f}  {fp_str}{marker}")
print("=" * 65)

json.dump(dict(
    model=MODEL_NAME, layers=LAYERS, best_layer=best_layer,
    n_train=len(train_recs), d_sae=D_SAE, k=SAE_K,
    layer_results={str(l): dict(
        var_expl=layer_results[l]["var_expl"],
        n_live=layer_results[l]["n_live"],
        auc=layer_results[l]["auc"],
        fp_sizes=layer_results[l]["fp_sizes"],
        jaccards=layer_results[l]["jaccards"],
    ) for l in LAYERS},
    accuracy_table={str(k): v for k, v in acc_table.items()},
), open(OUT_DIR / "results.json", "w"), indent=2)
print(f"Results → {OUT_DIR}/results.json")
print(f"Plots   → {OUT_DIR}/layer_sweep_*.png  sae_training_all_layers.png  pca_best_layer.png")
