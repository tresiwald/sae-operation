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

GEN_N = int(os.getenv("SAE_GEN_N", "3000"))   # compute problems (each + 1 copy twin)


@torch.no_grad()
def capture(pairs, model, tok, layer, device):
    """pairs: list of (meta_base, prompt, answer_str). Returns (acts[N,d], meta)."""
    buf = {}

    def hook(mod, args):
        h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
        buf["h"] = h[0].detach().float().cpu()      # [seq, d_model]

    handle = model.model.layers[layer].register_forward_pre_hook(hook)
    acts, meta = [], []
    for base, prompt, ans in tqdm(pairs, desc="gen acts"):
        p_ids = tok(prompt, return_tensors="pt")["input_ids"]
        f_ids = tok(prompt + ans, return_tensors="pt")["input_ids"].to(device)
        n_p, n_f = p_ids.shape[1], f_ids.shape[1]
        n_ans = n_f - n_p
        if n_ans < 1:
            continue
        buf.clear(); model(f_ids)
        h = buf["h"]
        digit_clean = (n_ans == len(ans))
        for j in range(n_ans):
            pos = n_p - 1 + j
            if pos < 0 or pos >= h.shape[0]:
                continue
            tid = f_ids[0, n_p + j].item()
            ts  = tok.decode([tid]).strip()
            acts.append(h[pos])
            meta.append({**base,
                         "tok_index": j, "tok_str": ts,
                         "place_from_left": j,
                         "place_from_right": (len(ans) - 1 - j) if digit_clean else -1,
                         "target_digit": int(ts) if (ts.isdigit() and len(ts) == 1) else -1,
                         "is_leading": int(j == 0),
                         "is_final": int(j == n_ans - 1),
                         "answer_len": n_ans})
    handle.remove()
    return torch.stack(acts), meta


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
