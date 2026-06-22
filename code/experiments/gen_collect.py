"""
Generation-time activation collection (teacher-forced, multi-position).

For each compute problem `347*8=` with answer `2776`, we run a single forward
pass on the full string `347*8=2776` and capture the residual at each position
that PRODUCES an answer token (the `=` predicts the first digit, the first digit
predicts the second, …). Each compute problem also gets a magnitude-matched COPY
twin — "Repeat the number 2776: 2776" — giving a digit-production baseline with
no arithmetic.

Captures one layer (the best layer) to keep the checkpoint small.

Outputs: results/checkpoints/gen_acts.pt
           keys: acts [N, d_model] (normalized), mu, sig, layer, meta (list)
"""

import os, sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import OPS_EVAL
import experiments._common as C
from experiments.gen_common import gen_acts_path

GEN_N     = int(os.getenv("SAE_GEN_N", "3000"))    # compute problems (each + 1 copy twin)
GEN_BATCH = int(os.getenv("SAE_GEN_BATCH", "16"))  # sequences per forward (CUDA)


def _meta_for(base, ans, n_p, n_ans, f_ids, j):
    """Build the per-position metadata record (answer token j)."""
    tid = f_ids[n_p + j]
    ts  = tok_decode(tid)
    digit_clean = (n_ans == len(ans))
    return {**base,
            "tok_index": j, "tok_str": ts,
            "place_from_left": j,
            "place_from_right": (len(ans) - 1 - j) if digit_clean else -1,
            "target_digit": int(ts) if (ts.isdigit() and len(ts) == 1) else -1,
            "is_leading": int(j == 0),
            "is_final": int(j == n_ans - 1),
            "answer_len": n_ans}


# the decode fn is bound once in main() so _meta_for stays tokenizer-agnostic
def tok_decode(tid):  # overwritten in main
    raise RuntimeError("tok_decode not bound")


def _prep(pairs, tok):
    """Tokenise once: keep (base, ans, f_ids, n_p, n_ans) for sequences with an answer."""
    items = []
    for base, prompt, ans in pairs:
        p_ids = tok(prompt)["input_ids"]
        f_ids = tok(prompt + ans)["input_ids"]
        n_p, n_ans = len(p_ids), len(f_ids) - len(p_ids)
        if n_ans >= 1:
            items.append((base, ans, f_ids, n_p, n_ans))
    return items


@torch.no_grad()
def capture_single(items, model, tok, layer, device):
    buf = {}
    def hook(mod, args):
        h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
        buf["h"] = h[0].detach().float().cpu()
    handle = model.model.layers[layer].register_forward_pre_hook(hook)
    acts, meta = [], []
    for base, ans, f_ids, n_p, n_ans in tqdm(items, desc="gen acts"):
        buf.clear()
        model(torch.tensor([f_ids], device=device))
        h = buf["h"]
        for j in range(n_ans):
            pos = n_p - 1 + j
            if 0 <= pos < h.shape[0]:
                acts.append(h[pos])
                meta.append(_meta_for(base, ans, n_p, n_ans, f_ids, j))
    handle.remove()
    return torch.stack(acts), meta


@torch.no_grad()
def capture_batched(items, model, tok, layer, device, batch_size):
    """Left-pad each batch so real tokens are right-aligned; index answer
    positions per sequence using its pad offset."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    buf = {}
    def hook(mod, args):
        h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
        buf["h"] = h.detach().float().cpu()          # [B, seq, d]
    handle = model.model.layers[layer].register_forward_pre_hook(hook)

    acts, meta = [], []
    for s in tqdm(range(0, len(items), batch_size), desc="gen acts (batched)"):
        chunk   = items[s:s + batch_size]
        max_len = max(len(it[2]) for it in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attn      = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for bi, (_, _, f_ids, _, _) in enumerate(chunk):
            off = max_len - len(f_ids)
            input_ids[bi, off:] = torch.tensor(f_ids)
            attn[bi, off:]      = 1
        buf.clear()
        model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
        h = buf["h"]
        for bi, (base, ans, f_ids, n_p, n_ans) in enumerate(chunk):
            off = max_len - len(f_ids)
            for j in range(n_ans):
                pos = off + (n_p - 1 + j)
                if 0 <= pos < h.shape[1]:
                    acts.append(h[bi, pos])
                    meta.append(_meta_for(base, ans, n_p, n_ans, f_ids, j))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    handle.remove()
    return torch.stack(acts), meta


def capture(pairs, model, tok, layer, device):
    items = _prep(pairs, tok)
    if device.type == "cuda" and GEN_BATCH > 1:
        tok.padding_side = "left"
        print(f"  batched capture (batch_size={GEN_BATCH})")
        return capture_batched(items, model, tok, layer, device, GEN_BATCH)
    print("  single-record capture (batch_size=1)")
    return capture_single(items, model, tok, layer, device)


def main():
    with open(C.ckpt_data(), "rb") as f:
        train = C.pickle_load(f)["train_corpus"]
    best = C.best_layer_from_results(default=13)
    print(f"Gen-collect — layer {best}")

    # compute problems: symbolic, has an answer
    comp = [r for r in train if r["variant"] == "compute"
            and r.get("fmt") == "symbolic" and r.get("expected") is not None]
    import random
    random.Random(0).shuffle(comp)
    comp = comp[:GEN_N]

    pairs = []
    for r in comp:
        N = str(r["expected"])
        meta = dict(op=r["op"], fmt=r["fmt"], bin=r["bin"],
                    a=r["a"], b=r["b"], expected=r["expected"])
        pairs.append(({**meta, "variant": "compute"}, r["prompt"], N))
        pairs.append(({**meta, "variant": "copy"}, f"Repeat the number {N}: ", N))
    print(f"  {len(comp)} compute problems + {len(comp)} copy twins "
          f"= {len(pairs)} sequences")

    model, tok, device = C.load_model()
    global tok_decode
    tok_decode = lambda tid: tok.decode([tid]).strip()
    acts, meta = capture(pairs, model, tok, best, device)
    del model
    print(f"  captured {acts.shape[0]} answer-token positions")

    mu  = acts.mean(0, keepdim=True)
    sig = acts.std(0, keepdim=True).clamp(min=1e-6)
    norm = (acts - mu) / sig

    out = gen_acts_path()
    torch.save(dict(acts=norm, mu=mu, sig=sig, layer=best, meta=meta), out)
    print(f"  Saved → {out}  ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
