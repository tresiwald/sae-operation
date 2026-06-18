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

import sys, json, re as _re, random as _random
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
LAYER        = 17

# n_per_cell=150 → ~10.7k training examples (80/20 split)
N_PER_CELL   = 150

SAE_K        = 32
SAE_RATIO    = 8           # d_sae = d_model * SAE_RATIO
SAE_LR       = 3e-4
SAE_EPOCHS   = 30
SAE_BATCH    = 512
WARMUP_STEPS = 200

FP_THRESHOLD = 0.5         # Cohen's d threshold for fingerprint membership
AUX_W        = 1 / 32
DEAD_THR     = 1e-4

FORMATS      = ["symbolic", "mixed", "verbal"]
OPS_EVAL     = ["add", "sub", "mul", "div"]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Model ──────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} in float32...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval().to(DEVICE)

D_MODEL = model.config.hidden_size
D_SAE   = D_MODEL * SAE_RATIO
print(f"d_model={D_MODEL}  d_sae={D_SAE}  float32 on {DEVICE}")

preds = tokenizer("3+5=", return_tensors="pt").to(DEVICE)
with torch.no_grad():
    lg = model(**preds).logits[0, -1].float().cpu().topk(3)
print(f"Sanity  3+5= → {[(tokenizer.decode([i]).strip(), round(v.item(),1)) for i,v in zip(lg.indices,lg.values)]}")

# ── 2. Activation hook ────────────────────────────────────────────────────────
_buf: list[torch.Tensor] = []

def _pre_hook(module, args):
    hidden = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
    _buf.append(hidden[:, -1, :].detach().float().cpu())

hook_handle = model.model.layers[LAYER].register_forward_pre_hook(_pre_hook)

@torch.no_grad()
def collect_acts(records, desc="acts"):
    """Single-example inference; no padding NaN."""
    _buf.clear()
    for rec in tqdm(records, desc=desc, leave=False):
        inp = tokenizer(rec["prompt"], return_tensors="pt").to(DEVICE)
        model(**inp)
    return torch.cat(_buf, dim=0)

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
print("\nMeasuring accuracy (H1 per-op holdout)...")
measure_accuracy(hold_per_op)

acc_table = {}   # (op, fmt, bin) → float
print(f"\n  {'op':4s}  {'fmt':8s}  {'bin':3s}  {'n':>4s}  acc")
for op in OPS_EVAL:
    for fmt in FORMATS:
        for bin_name in BINS:
            grp = [r for r in hold_per_op
                   if r["op"] == op and r["fmt"] == fmt
                   and r["bin"] == bin_name and r["variant"] == "compute"]
            if not grp:
                continue
            acc = float(np.mean([r["correct"] for r in grp]))
            acc_table[(op, fmt, bin_name)] = acc
            print(f"  {op:4s}  {fmt:8s}  {bin_name:3s}  {len(grp):>4d}  {acc:.1%}")

# ── 6. Collect training activations ───────────────────────────────────────────
print("\nCollecting training activations...")

# Flat indexed list so we can slice by (op, fmt) later
train_recs: list[dict] = train_corpus
print(f"  ctrl: {sum(1 for r in train_recs if r['op']=='ctrl')}  "
      f"copy: {sum(1 for r in train_recs if r['op']=='copy')}  "
      f"math: {sum(1 for r in train_recs if r['op'] not in ('ctrl','copy'))}")
train_acts_raw = collect_acts(train_recs, desc="train")

n_nan = torch.isnan(train_acts_raw).sum().item()
n_inf = torch.isinf(train_acts_raw).sum().item()
if n_nan or n_inf:
    print(f"  WARNING: {n_nan} NaN + {n_inf} Inf — replacing with 0")
    train_acts_raw = torch.nan_to_num(train_acts_raw, nan=0.0, posinf=0.0, neginf=0.0)

act_mean = train_acts_raw.mean(dim=0, keepdim=True)
act_std  = train_acts_raw.std(dim=0,  keepdim=True).clamp(min=1e-6)
train_acts = (train_acts_raw - act_mean) / act_std
print(f"  {train_acts.shape}  mean={train_acts.mean():.4f}  std={train_acts.std():.4f}")

del model
if str(DEVICE) == "mps":
    torch.mps.empty_cache()

# ── 7. TopK SAE ───────────────────────────────────────────────────────────────
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

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        return acts, self.decode(acts)

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.data = nn.functional.normalize(self.W_dec.data, dim=1)

sae = TopKSAE(D_MODEL, D_SAE, SAE_K)
print(f"\nSAE: d_sae={D_SAE}  k={SAE_K}  params={sum(p.numel() for p in sae.parameters()):,}")

# ── 8. Train SAE ──────────────────────────────────────────────────────────────
optimizer   = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
loader      = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_acts), batch_size=SAE_BATCH, shuffle=True,
)
feat_usage  = torch.zeros(D_SAE)
history     = []
step        = 0

print("\nTraining SAE...")
for epoch in range(SAE_EPOCHS):
    e_mse = 0.0
    for (x,) in loader:
        lr = SAE_LR * min(1.0, step / max(1, WARMUP_STEPS))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
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
        optimizer.step()
        sae.normalise_decoder()
        e_mse += mse.item()

    n_dead   = (feat_usage < DEAD_THR).sum().item()
    mean_mse = e_mse / len(loader)
    if not torch.isfinite(torch.tensor(mean_mse)):
        print(f"  Epoch {epoch+1:2d}  NaN/Inf — stopping"); break
    history.append(dict(epoch=epoch+1, mse=mean_mse, dead=n_dead))
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{SAE_EPOCHS}  mse={mean_mse:.5f}  dead={n_dead}/{D_SAE}")

sae.eval()
with torch.no_grad():
    s = train_acts[:1000]
    var_expl = 1 - (s - sae(s)[1]).pow(2).sum() / s.pow(2).sum()
n_live = (feat_usage > DEAD_THR).sum().item()
print(f"  Variance explained: {var_expl.item():.2%}  Live: {n_live}/{D_SAE}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot([h["epoch"] for h in history], [h["mse"] for h in history], marker="o")
a1.set(title="SAE MSE", xlabel="Epoch")
a2.plot([h["epoch"] for h in history], [h["dead"] for h in history], marker="o", color="tomato")
a2.set(title=f"Dead features (/{D_SAE})", xlabel="Epoch")
plt.tight_layout(); plt.savefig(OUT_DIR / "sae_training.png", dpi=150); plt.close()

# ── 9. SAE feature extraction ─────────────────────────────────────────────────
@torch.no_grad()
def to_features(acts_tensor: torch.Tensor) -> np.ndarray:
    return np.concatenate([
        sae.encode(acts_tensor[i:i+512]).numpy()
        for i in range(0, len(acts_tensor), 512)
    ], axis=0)

# Encode entire training corpus once; slice later by (op, fmt)
all_feats_np = np.nan_to_num(to_features(train_acts), nan=0.0)  # (N, D_SAE)

def get_feat_slice(op=None, fmt=None) -> np.ndarray:
    """Return SAE features for records matching (op, fmt) filter."""
    mask = np.ones(len(train_recs), dtype=bool)
    if op  is not None: mask &= np.array([r["op"]  == op  for r in train_recs])
    if fmt is not None: mask &= np.array([r["fmt"] == fmt for r in train_recs])
    return all_feats_np[mask]

# ── 10. Fingerprint analysis (reusable) ───────────────────────────────────────
def cosine(a, b):
    na, nb = norm(a), norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

def compute_fingerprints(feats_by_op: dict) -> tuple[dict, dict]:
    """
    feats_by_op: {op: np.ndarray (N, D_SAE)}
    Returns (fingerprints, effect_sizes) dicts.
    Baseline = 'ctrl' (neutral non-math text).
    'copy' is a labelled condition, not the baseline — it sits in feature space
    alongside the ops and is used separately for cheat proximity scoring.
    """
    base_mean = feats_by_op["ctrl"].mean(axis=0)
    base_std  = feats_by_op["ctrl"].std(axis=0) + 1e-8
    fingerprints, effect_sizes = {}, {}
    for op in OPS_EVAL:
        if op not in feats_by_op or len(feats_by_op[op]) == 0:
            continue
        op_mean = feats_by_op[op].mean(axis=0)
        pooled  = np.sqrt((base_std**2 + feats_by_op[op].std(axis=0)**2 + 1e-8) / 2)
        d = (op_mean - base_mean) / pooled
        effect_sizes[op] = d
        fingerprints[op] = d > FP_THRESHOLD
    return fingerprints, effect_sizes

def containment(fp_a, fp_b):
    return float((fp_a & fp_b).sum()) / (fp_a.sum() + 1e-8)

def compute_rdm(f, n=200, seed=0):
    np.random.seed(seed)
    sub = f[np.random.choice(len(f), min(n, len(f)), replace=False)]
    nrm = norm(sub, axis=1, keepdims=True).clip(min=1e-8)
    sim = (sub / nrm) @ (sub / nrm).T
    return np.nan_to_num(1 - sim, nan=1.0)

def rdm_corr(a, b):
    idx = np.triu_indices(len(a), k=1)
    rho, _ = spearmanr(a[idx], b[idx])
    return rho

def run_analysis(label: str, feats_by_op: dict) -> dict:
    """Run fingerprint + containment + RSA for a given feature slice."""
    fps, effs = compute_fingerprints(feats_by_op)
    if not fps:
        return {}

    print(f"\n{'─'*55}")
    print(f"  Analysis: {label}")
    print(f"{'─'*55}")

    # Fingerprint sizes
    print(f"  Fingerprints (Cohen's d > {FP_THRESHOLD}):")
    for op in OPS_EVAL:
        n = fps.get(op, np.zeros(D_SAE, dtype=bool)).sum()
        print(f"    {op}: {n} features")

    # Containment matrix
    print(f"  Containment [row ⊆ col]:")
    print(f"    {'':5}", "  ".join(f"{o:>6}" for o in OPS_EVAL if o in fps))
    for a in OPS_EVAL:
        if a not in fps: continue
        row = "  ".join(f"{containment(fps[a], fps[b]):>6.3f}"
                        for b in OPS_EVAL if b in fps)
        print(f"    {a:5}  {row}")

    c_am = containment(fps.get("add", np.zeros(D_SAE, bool)),
                       fps.get("mul", np.zeros(D_SAE, bool)))
    c_ma = containment(fps.get("mul", np.zeros(D_SAE, bool)),
                       fps.get("add", np.zeros(D_SAE, bool)))
    verdict = "SUPPORTED" if c_am > c_ma else "NOT SUPPORTED"
    shared  = int(np.array([fps[o] for o in OPS_EVAL if o in fps]).all(axis=0).sum())
    print(f"  Hierarchy (add⊆mul={c_am:.3f} mul⊆add={c_ma:.3f}): {verdict}")
    print(f"  Shared math core (all ops): {shared} features")

    # RSA
    rdms = {op: compute_rdm(feats_by_op[op])
            for op in OPS_EVAL + ["copy", "ctrl"] if op in feats_by_op and len(feats_by_op[op]) >= 10}
    if len(rdms) >= 2:
        print(f"  RDM (Spearman ρ):")
        rdm_ops = list(rdms.keys())
        print(f"    {'':6}", "  ".join(f"{o:>6}" for o in rdm_ops))
        for a in rdm_ops:
            row = "  ".join(f"{rdm_corr(rdms[a], rdms[b]):>6.3f}" for b in rdm_ops)
            print(f"    {a:6}  {row}")

    return dict(
        label=label,
        fingerprint_sizes={op: int(fps.get(op, np.zeros(D_SAE, bool)).sum()) for op in OPS_EVAL},
        shared_core=shared,
        containment_add_mul=float(c_am), containment_mul_add=float(c_ma),
        hierarchy_verdict=verdict,
    )

# ── 11. Run analysis: all formats + per-format ───────────────────────────────
analysis_results = []

# ── All formats combined ──────────────────────────────────────────────────────
feats_all = {op: get_feat_slice(op=op) for op in OPS_EVAL}
feats_all["copy"] = get_feat_slice(op="copy")
feats_all["ctrl"] = get_feat_slice(op="ctrl")
r = run_analysis("ALL FORMATS", feats_all)
analysis_results.append(r)

# Save fingerprints from combined analysis for later holdout scoring
fps_combined, effs_combined = compute_fingerprints(feats_all)

# ── Per-format ────────────────────────────────────────────────────────────────
feats_by_fmt = {}  # fmt → {op → np.ndarray}
for fmt in FORMATS:
    fd = {op: get_feat_slice(op=op, fmt=fmt) for op in OPS_EVAL}
    fd["copy"] = get_feat_slice(op="copy", fmt=fmt)
    fd["ctrl"] = get_feat_slice(op="ctrl")   # ctrl has no fmt tag; share across slices
    feats_by_fmt[fmt] = fd
    if all(len(v) >= 10 for v in fd.values()):
        r = run_analysis(f"FORMAT: {fmt.upper()}", fd)
    else:
        print(f"\n  (skipping {fmt} — insufficient data)")
        r = {"label": fmt}
    analysis_results.append(r)

# ── 12. Format-invariance summary ─────────────────────────────────────────────
print(f"\n{'═'*55}")
print("  FORMAT INVARIANCE — fingerprint size comparison")
print(f"{'═'*55}")
print(f"  {'op':4s}  {'all':>6s}", "  ".join(f"{f:>8s}" for f in FORMATS))
for op in OPS_EVAL:
    n_all  = int(fps_combined.get(op, np.zeros(D_SAE, bool)).sum())
    by_fmt = []
    for fmt in FORMATS:
        fps_f, _ = compute_fingerprints(feats_by_fmt[fmt])
        by_fmt.append(int(fps_f.get(op, np.zeros(D_SAE, bool)).sum()))
    print(f"  {op:4s}  {n_all:>6d}", "  ".join(f"{n:>8d}" for n in by_fmt))

# Jaccard overlap of fingerprints across formats (per op)
print(f"\n  Fingerprint overlap (Jaccard) across formats:")
for op in OPS_EVAL:
    fps_fmt = []
    for fmt in FORMATS:
        fps_f, _ = compute_fingerprints(feats_by_fmt[fmt])
        fps_fmt.append(fps_f.get(op, np.zeros(D_SAE, bool)))
    if not any(f.sum() > 0 for f in fps_fmt):
        continue
    union = np.array(fps_fmt).any(axis=0)
    inter = np.array(fps_fmt).all(axis=0)
    jaccard = float(inter.sum()) / (union.sum() + 1e-8)
    print(f"  {op}: intersection={inter.sum()}  union={union.sum()}  Jaccard={jaccard:.3f}")

# ── 13. Linear classification per format ─────────────────────────────────────
print(f"\n  Linear AUC (math vs copy) per format slice:")
for label, fd in [("all", feats_all)] + [(f, feats_by_fmt[f]) for f in FORMATS]:
    ops_present = [op for op in OPS_EVAL if op in fd and len(fd[op]) >= 10]
    ctrl_present = "ctrl" in fd and len(fd["ctrl"]) >= 10
    if not ops_present or not ctrl_present:
        continue
    # AUC: math ops vs neutral ctrl (the natural "is math happening?" classifier)
    X = np.concatenate([fd[op] for op in ops_present] + [fd["ctrl"]])
    y = np.array([1] * sum(len(fd[op]) for op in ops_present) + [0] * len(fd["ctrl"]))
    if len(np.unique(y)) < 2 or X.shape[0] < 20:
        continue
    aucs = cross_val_score(LogisticRegression(max_iter=500, C=0.1),
                           X, y, cv=min(5, int(y.sum())), scoring="roc_auc")
    print(f"  {label:8s}  AUC={aucs.mean():.3f} ± {aucs.std():.3f}")

# ── 14. PCA: all ops, coloured by op; faceted by format ──────────────────────
fig, axes = plt.subplots(1, len(FORMATS) + 1, figsize=(5 * (len(FORMATS) + 1), 5))
COLORS = {"add": "steelblue", "sub": "green", "mul": "darkorange",
          "div": "purple", "copy": "tomato", "ctrl": "gray"}
SUB = 200

def pca_plot(ax, fd, title):
    ops_p = [op for op in OPS_EVAL + ["copy", "ctrl"] if op in fd and len(fd[op]) >= 5]
    sub_f = {op: fd[op][np.random.choice(len(fd[op]), min(SUB, len(fd[op])), replace=False)]
             for op in ops_p}
    X = np.nan_to_num(np.concatenate(list(sub_f.values())), nan=0.0)
    lbls = sum([[op] * len(sub_f[op]) for op in ops_p], [])
    if X.shape[0] < 5:
        ax.set_title(title + " (no data)"); return
    pcs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    for op in ops_p:
        m = np.array(lbls) == op
        ax.scatter(pcs[m, 0], pcs[m, 1], c=COLORS[op], alpha=0.4, s=10, label=op)
    ax.legend(fontsize=7); ax.set_title(title, fontsize=9)

np.random.seed(0)
pca_plot(axes[0], feats_all, "All formats")
for i, fmt in enumerate(FORMATS):
    pca_plot(axes[i + 1], feats_by_fmt[fmt], fmt.capitalize())
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_by_format.png", dpi=150)
plt.close()
print(f"\nPCA plot → {OUT_DIR}/pca_by_format.png")

# ── 15. H2: Cheat holdout (reload model) ────────────────────────────────────
print("\nH2: Cheat holdout — collecting activations (reloading model)...")
model2 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model2.eval().to(DEVICE)
_buf.clear()
hook2 = model2.model.layers[LAYER].register_forward_pre_hook(_pre_hook)

cheat_sample = {op: [r for r in hold_cheat if r["op"] == op][:300] for op in OPS_EVAL}
cheat_feats_by_op: dict[str, dict[str, np.ndarray]] = {"all": {}, **{f: {} for f in FORMATS}}

for op, recs in cheat_sample.items():
    if not recs:
        continue
    acts = collect_acts(recs, desc=f"cheat/{op}")
    acts = torch.nan_to_num(acts, nan=0.0)
    acts = (acts - act_mean) / act_std
    f_all = np.nan_to_num(to_features(acts), nan=0.0)
    cheat_feats_by_op["all"][op] = f_all
    for fmt in FORMATS:
        idx = [i for i, r in enumerate(recs) if r["fmt"] == fmt]
        cheat_feats_by_op[fmt][op] = f_all[idx] if idx else np.zeros((0, D_SAE))

hook2.remove()
del model2
if str(DEVICE) == "mps":
    torch.mps.empty_cache()

# Compare cheat vs compute centroid per (op, fmt-slice)
print("\n  Cheat fingerprint similarity (cosine) to compute / copy centroids:")
print(f"  {'slice':10s}  {'op':4s}  comp_sim  copy_sim  verdict")
copy_centroid = feats_all["copy"].mean(axis=0)

for slice_label, cheat_fd, ref_fd in (
    [("all",  cheat_feats_by_op["all"],  feats_all)] +
    [(f,      cheat_feats_by_op[f],      feats_by_fmt[f]) for f in FORMATS]
):
    for op in OPS_EVAL:
        c_feats = cheat_fd.get(op)
        if c_feats is None or len(c_feats) == 0:
            continue
        cheat_c  = c_feats.mean(axis=0)
        comp_c   = ref_fd[op].mean(axis=0) if op in ref_fd and len(ref_fd[op]) else cheat_c
        comp_sim = cosine(cheat_c, comp_c)
        copy_sim = cosine(cheat_c, copy_centroid)
        v = "→copy" if copy_sim > comp_sim else "→compute"
        print(f"  {slice_label:10s}  {op:4s}  {comp_sim:>8.4f}  {copy_sim:>8.4f}  {v}")

# ── 16. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Model:              {MODEL_NAME}  layer {LAYER}")
print(f"  Training examples:  {len(train_acts)}")
print(f"  SAE:                d_sae={D_SAE}  k={SAE_K}  epochs={SAE_EPOCHS}")
print(f"  Variance explained: {var_expl.item():.2%}  Live: {n_live}/{D_SAE}")
for op in OPS_EVAL:
    n = int(fps_combined.get(op, np.zeros(D_SAE, bool)).sum())
    print(f"  Fingerprint {op}:     {n} features")
print("=" * 65)

json.dump(dict(
    model=MODEL_NAME, layer=LAYER, n_train=len(train_acts),
    var_explained=float(var_expl.item()), n_live=n_live, d_sae=D_SAE,
    analysis=analysis_results,
    accuracy_table={str(k): v for k, v in acc_table.items()},
), open(OUT_DIR / "results.json", "w"), indent=2)
print(f"Results → {OUT_DIR}/results.json  |  Plots → {OUT_DIR}/pca_by_format.png")
