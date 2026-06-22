"""
Data generation for SAE operational fingerprint experiment.

Conditions:
  - Operations:  add, sub, mul, div (exact), copy
  - Variants:    compute, cheat (answer hinted), copy (no-math baseline)
  - Formats:     symbolic, mixed, verbal  (verbal only for n <= 999)
  - Ranges:      5 magnitude bins up to 99999

Four holdout splits (not used for SAE training):
  1. per_op       — standard compute, each op individually
  2. per_op_cheat — answer hinted, each op individually
  3. multi_op     — compositional: (a+b)*c=, a+(b*c)=, etc.
  4. multi_op_cheat — compositional with answer hinted

Each record is a dict with keys:
  prompt, op, variant, fmt, bin, a, b, expected
  (multi-op records additionally have: expr, ops_used)
"""

import random, itertools, re

# ── Number → words (up to 99999) ─────────────────────────────────────────────
_ONES  = ["", "one","two","three","four","five","six","seven","eight","nine",
          "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
          "seventeen","eighteen","nineteen"]
_TENS  = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def _num_to_words(n: int) -> str:
    if n == 0:   return "zero"
    if n < 20:   return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + _num_to_words(r) if r else "")
    if n < 10000:
        th, r = divmod(n, 1000)
        return _ONES[th] + " thousand" + (" " + _num_to_words(r) if r else "")
    if n < 100000:
        th, r = divmod(n, 1000)
        return _num_to_words(th) + " thousand" + (" " + _num_to_words(r) if r else "")
    raise ValueError(f"n={n} out of range")

# ── Operand ranges (bins) ─────────────────────────────────────────────────────
BINS = {
    "1d":  (1,       9),
    "2d":  (10,      99),
    "3d":  (100,     999),
    "4d":  (1000,    9999),
    "5d":  (10000,   99999),
}

# Verbal format only for numbers that can be spelled cleanly
VERBAL_BINS = {"1d", "2d", "3d"}

# ── Prompt templates ──────────────────────────────────────────────────────────
# Each template is (format, compute_tmpl, cheat_tmpl, copy_tmpl)
# Placeholders: {a}, {b}, {c} (answer), {wa}, {wb}, {wc} (word forms)

TEMPLATES = {
    "add": [
        ("symbolic",
         "{a}+{b}=",
         "{a}+{b}={c}. So {a}+{b}=",
         None),
        ("mixed",
         "What is {a} plus {b}?",
         "What is {a} plus {b}? It equals {c}. So what is {a} plus {b}?",
         "The sum is {c}. The sum is"),
        ("verbal",
         "What is {wa} plus {wb}?",
         "{wa} plus {wb} is {wc}. So {wa} plus {wb} is",
         "The answer is {wc}. The answer is"),
    ],
    "sub": [
        ("symbolic",
         "{a}-{b}=",
         "{a}-{b}={c}. So {a}-{b}=",
         None),
        ("mixed",
         "What is {a} minus {b}?",
         "What is {a} minus {b}? It equals {c}. So what is {a} minus {b}?",
         "The result is {c}. The result is"),
        ("verbal",
         "What is {wa} minus {wb}?",
         "{wa} minus {wb} is {wc}. So {wa} minus {wb} is",
         "The answer is {wc}. The answer is"),
    ],
    "mul": [
        ("symbolic",
         "{a}*{b}=",
         "{a}*{b}={c}. So {a}*{b}=",
         None),
        ("mixed",
         "What is {a} times {b}?",
         "What is {a} times {b}? It equals {c}. So what is {a} times {b}?",
         "The product is {c}. The product is"),
        ("verbal",
         "What is {wa} times {wb}?",
         "{wa} times {wb} is {wc}. So {wa} times {wb} is",
         "The answer is {wc}. The answer is"),
    ],
    "div": [
        ("symbolic",
         "{a}/{b}=",
         "{a}/{b}={c}. So {a}/{b}=",
         None),
        ("mixed",
         "What is {a} divided by {b}?",
         "What is {a} divided by {b}? It equals {c}. So what is {a} divided by {b}?",
         "The quotient is {c}. The quotient is"),
        ("verbal",
         "What is {wa} divided by {wb}?",
         "{wa} divided by {wb} is {wc}. So {wa} divided by {wb} is",
         "The answer is {wc}. The answer is"),
    ],
}

# Standalone copy templates (no math — pure number repetition)
COPY_TEMPLATES = [
    ("symbolic", "Repeat: {c}."),
    ("mixed",    "The number is {c}. Write the number:"),
    ("verbal",   "The answer is {wc}. The answer is"),
]

# Non-math control templates — neutral language, no numbers, no arithmetic.
# Used as the Cohen's d baseline for fingerprint extraction (not a condition to analyse).
_CTRL_TEMPLATES = [
    ("The capital of {country} is",           {"country": ["France","Germany","Japan","Brazil","Canada","Italy","Spain","Egypt"]}),
    ("The color of the sky is",               {}),
    ("She walked into the {room} and",        {"room": ["kitchen","library","office","garden","basement","hallway"]}),
    ("The largest {animal} in the world is",  {"animal": ["mammal","reptile","bird","fish","insect"]}),
    ("He picked up the {obj} and",            {"obj": ["book","pen","phone","key","bag","bottle","cup"]}),
    ("Scientists recently discovered that",   {}),
    ("The best way to learn a language is",   {}),
    ("After the storm the {place} was",       {"place": ["forest","street","beach","rooftop","market"]}),
    ("In the morning she always",             {}),
    ("The old {thing} had been there for years", {"thing": ["building","bridge","tree","clock","statue"]}),
]

def make_ctrl_data(n: int, seed: int = 7) -> list[dict]:
    """Generate n non-math control records (op='ctrl', variant='ctrl')."""
    rng  = random.Random(seed)
    data = []
    for _ in range(n):
        tmpl, fills = rng.choice(_CTRL_TEMPLATES)
        prompt = tmpl
        for k, opts in fills.items():
            prompt = prompt.replace("{" + k + "}", rng.choice(opts))
        data.append(dict(op="ctrl", variant="ctrl", fmt="none",
                         bin="none", a=None, b=None, expected=None, prompt=prompt))
    return data

# ── Sampler ───────────────────────────────────────────────────────────────────
def _safe_words(n):
    try:
        return _num_to_words(n) if n is not None else ""
    except ValueError:
        return str(n)

def _fill(tmpl: str, a, b, c) -> str:
    wa = _safe_words(a)
    wb = _safe_words(b)
    wc = _safe_words(c)
    return (tmpl
            .replace("{a}", str(a) if a is not None else "")
            .replace("{b}", str(b) if b is not None else "")
            .replace("{c}", str(c) if c is not None else "")
            .replace("{wa}", wa)
            .replace("{wb}", wb)
            .replace("{wc}", wc))

def _sample_pair(op, bin_name, rng):
    lo, hi = BINS[bin_name]
    for _ in range(100):   # retry to find valid pair
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if op == "add":
            return a, b, a + b
        if op == "sub":
            a, b = max(a, b), min(a, b)
            return a, b, a - b
        if op == "mul":
            # cap answer at 10^8 to keep it finite
            if a * b < 100_000_000:
                return a, b, a * b
        if op == "div":
            # only exact divisors where quotient > 0
            if b != 0 and a % b == 0 and a // b > 0:
                return a, b, a // b
    return None  # no valid pair found

def _sample_pair_loguniform(op, rng, mag_lo, mag_hi):
    """Sample one (a, b, result) with operands drawn LOG-uniformly over
    10^mag_lo .. 10^mag_hi, so log10(operand) is ~uniform. This flattens the
    magnitude distribution and (for add/sub/mul) matches operand-magnitude
    coverage across operations, so magnitude can be regressed out cleanly.

    div is handled as quotient×divisor (its dividend therefore differs — an
    unavoidable asymmetry, recorded via the operands for later regression)."""
    def lu():                       # one log-uniform integer ≥ 1
        return max(1, int(10 ** rng.uniform(mag_lo, mag_hi)))
    for _ in range(200):
        if op == "add":
            a, b = lu(), lu()
            return a, b, a + b
        if op == "sub":
            a, b = lu(), lu()
            a, b = max(a, b), min(a, b)
            return a, b, a - b
        if op == "mul":
            a, b = lu(), lu()
            if a * b < 100_000_000:
                return a, b, a * b
        if op == "div":
            b = lu()
            q = max(1, int(10 ** rng.uniform(0, mag_hi)))   # quotient = result
            a = q * b
            if a < 100_000_000:
                return a, b, q
    return None


def make_loguniform_dataset(
    n_per_op: int = 1500,
    ops: list = None,
    mag_lo: float = 0.0,
    mag_hi: float = 4.0,
    seed: int = 42,
    include_copy: bool = True,
) -> list:
    """Magnitude-stratified dataset: operands sampled LOG-uniformly so the result
    magnitude is spread evenly across scales and has common support across
    operations (the fix for the magnitude-vs-operation confound).

    Same record schema as make_dataset (op, variant, fmt, bin, a, b, expected,
    prompt); `bin` is the result's digit-count (1d..5d, capped) so downstream
    per-bin grouping still works and now reflects magnitude.
    """
    if ops is None:
        ops = ["add", "sub", "mul", "div"]
    rng  = random.Random(seed)
    data = []

    def bin_of(r):
        return f"{min(max(len(str(abs(r))), 1), 5)}d"

    for op in ops:
        templates = TEMPLATES[op]
        made = 0
        while made < n_per_op:
            pair = _sample_pair_loguniform(op, rng, mag_lo, mag_hi)
            if pair is None:
                continue
            a, b, c = pair
            bin_name = bin_of(c)
            spellable = (a < 100000 and b < 100000 and c < 100000)
            for fmt, compute_t, cheat_t, copy_t in templates:
                if fmt == "verbal" and not spellable:
                    continue
                base = dict(op=op, bin=bin_name, fmt=fmt, a=a, b=b, expected=c)
                data.append({**base, "variant": "compute",
                              "prompt": _fill(compute_t, a, b, c)})
                data.append({**base, "variant": "cheat",
                              "prompt": _fill(cheat_t, a, b, c)})
                if copy_t:
                    data.append({**base, "variant": "copy_op",
                                  "prompt": _fill(copy_t, a, b, c)})
            made += 1

    # standalone copy examples (no math) — log-uniform numbers
    if include_copy:
        for _ in range(n_per_op):
            c = max(1, int(10 ** rng.uniform(mag_lo, mag_hi)))
            for fmt, copy_t in COPY_TEMPLATES:
                if fmt == "verbal" and c >= 100000:
                    continue
                wc = _num_to_words(c) if c < 100000 else ""
                data.append(dict(op="copy", variant="copy", fmt=fmt,
                                 bin=f"{min(len(str(c)),5)}d", a=None, b=None,
                                 expected=c,
                                 prompt=copy_t.replace("{c}", str(c)).replace("{wc}", wc)))

    rng.shuffle(data)
    return data


def make_dataset(
    n_per_cell: int = 200,
    ops: list = None,
    seed: int = 42,
    include_copy: bool = True,
) -> list[dict]:
    """
    Generate all records.

    n_per_cell: examples per (op, variant, fmt, bin) cell.
    Returns list of dicts with keys:
      prompt, op, variant, fmt, bin, a, b, expected
    """
    if ops is None:
        ops = ["add", "sub", "mul", "div"]

    rng   = random.Random(seed)
    data  = []

    for op in ops:
        templates = TEMPLATES[op]
        for bin_name in BINS:
            for fmt, compute_t, cheat_t, copy_t in templates:
                # Skip verbal for large bins
                if fmt == "verbal" and bin_name not in VERBAL_BINS:
                    continue

                generated = 0
                attempts  = 0
                while generated < n_per_cell and attempts < n_per_cell * 10:
                    attempts += 1
                    pair = _sample_pair(op, bin_name, rng)
                    if pair is None:
                        continue
                    a, b, c = pair

                    base = dict(op=op, bin=bin_name, fmt=fmt, a=a, b=b, expected=c)

                    # compute
                    data.append({**base, "variant": "compute",
                                  "prompt": _fill(compute_t, a, b, c)})
                    # cheat
                    data.append({**base, "variant": "cheat",
                                  "prompt": _fill(cheat_t, a, b, c)})
                    # op-specific copy (uses same answer token)
                    if copy_t:
                        data.append({**base, "variant": "copy_op",
                                      "prompt": _fill(copy_t, a, b, c)})
                    generated += 1

    # Standalone copy examples (no math)
    if include_copy:
        # Sample random numbers from each bin to copy
        for bin_name in BINS:
            lo, hi = BINS[bin_name]
            for fmt, copy_t in COPY_TEMPLATES:
                if fmt == "verbal" and bin_name not in VERBAL_BINS:
                    continue
                for _ in range(n_per_cell):
                    c = rng.randint(lo, hi)
                    wa = wb = ""
                    wc = _num_to_words(c) if bin_name in VERBAL_BINS else ""
                    prompt = (copy_t
                              .replace("{c}", str(c))
                              .replace("{wc}", wc))
                    data.append(dict(op="copy", variant="copy", fmt=fmt,
                                     bin=bin_name, a=None, b=None, expected=c,
                                     prompt=prompt))

    rng.shuffle(data)
    return data


# ── Multi-operation holdout generation ───────────────────────────────────────
# Compositional expressions: (a OP1 b) OP2 c  and  a OP1 (b OP2 c)

_MULTI_SYM = {
    "add": "+",
    "mul": "*",
    "sub": "-",
    "div": "/",
}

_MULTI_WORD = {
    "add": "plus",
    "mul": "times",
    "sub": "minus",
    "div": "divided by",
}


def _eval_expr(a, op1, b, op2, c, structure):
    """Return (result, None) or (None, reason_str) if invalid."""
    if structure == "left":   # (a OP1 b) OP2 c
        mid_a, mid_b, mid_op = a, b, op1
        outer_b, outer_op    = c, op2
    else:                     # a OP1 (b OP2 c)
        mid_a, mid_b, mid_op = b, c, op2
        outer_b, outer_op    = a, op1

    # compute inner
    if mid_op == "add":   mid = mid_a + mid_b
    elif mid_op == "sub":
        if mid_a < mid_b: return None, "sub negative"
        mid = mid_a - mid_b
    elif mid_op == "mul": mid = mid_a * mid_b
    elif mid_op == "div":
        if mid_b == 0 or mid_a % mid_b != 0: return None, "div invalid"
        mid = mid_a // mid_b

    # compute outer
    if structure == "left":
        lhs, rhs, outer_op2 = mid, outer_b, outer_op
    else:
        lhs, rhs, outer_op2 = outer_b, mid, outer_op

    if outer_op2 == "add":   result = lhs + rhs
    elif outer_op2 == "sub":
        if lhs < rhs: return None, "outer sub negative"
        result = lhs - rhs
    elif outer_op2 == "mul": result = lhs * rhs
    elif outer_op2 == "div":
        if rhs == 0 or lhs % rhs != 0: return None, "outer div invalid"
        result = lhs // rhs
    else:
        return None, "unknown op"

    if result < 0 or result > 10_000_000: return None, "out of range"
    return result, None


def make_multi_op_holdout(
    n_per_cell: int = 100,
    ops: list = None,
    seed: int = 99,
) -> list[dict]:
    """
    Generate multi-operation holdout records.
    Returns records with variant in {"compute", "cheat"}.
    Each record also has: expr (str), ops_used (tuple), structure (str).
    """
    if ops is None:
        ops = ["add", "sub", "mul", "div"]

    rng  = random.Random(seed)
    data = []

    op_pairs = [(o1, o2) for o1 in ops for o2 in ops]  # includes same-op pairs

    for op1, op2 in op_pairs:
        for bin_name in BINS:
            lo, hi = BINS[bin_name]
            generated = 0
            attempts  = 0
            while generated < n_per_cell and attempts < n_per_cell * 20:
                attempts += 1
                a = rng.randint(lo, hi)
                b = rng.randint(lo, hi)
                c = rng.randint(lo, hi)
                structure = rng.choice(["left", "right"])

                result, err = _eval_expr(a, op1, b, op2, c, structure)
                if err:
                    continue

                s1, s2 = _MULTI_SYM[op1], _MULTI_SYM[op2]
                w1, w2 = _MULTI_WORD[op1], _MULTI_WORD[op2]

                if structure == "left":
                    sym_expr  = f"({a}{s1}{b}){s2}{c}"
                    word_expr = f"({a} {w1} {b}) {w2} {c}"
                else:
                    sym_expr  = f"{a}{s1}({b}{s2}{c})"
                    word_expr = f"{a} {w1} ({b} {w2} {c})"

                base = dict(
                    op="multi", bin=bin_name, fmt="symbolic",
                    a=a, b=b, expected=result,
                    expr=sym_expr, ops_used=(op1, op2), structure=structure,
                )

                # compute
                data.append({**base,
                              "variant": "compute",
                              "prompt":  f"{sym_expr}="})
                # cheat
                data.append({**base,
                              "variant": "cheat",
                              "prompt":  f"{sym_expr}={result}. So {sym_expr}="})
                # mixed format
                data.append({**base,
                              "fmt": "mixed",
                              "variant": "compute",
                              "prompt":  f"What is {word_expr}?"})
                data.append({**base,
                              "fmt": "mixed",
                              "variant": "cheat",
                              "prompt":  f"What is {word_expr}? It equals {result}. So what is {word_expr}?"})

                generated += 1

    rng.shuffle(data)
    return data


# ── Matched-operand probes (Experiment 1: hard negatives) ─────────────────────
_PROBE_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "nonsense": "#"}


def make_matched_probes(n_per_bin: int = 100, seed: int = 123,
                        formats: list = None, include_nonsense: bool = True,
                        bins: list = None) -> list:
    """
    Matched-operand hard negatives: the SAME operand pair (a, b) is presented
    under every operator, so the only thing that differs between an `add` probe
    and a `mul` probe with the same `pair_id` is the operator token.

    This controls for digit identity / length / position — if operation
    fingerprints stay separable here, they encode the operation, not the digits.

    With `formats=["symbolic","mixed","verbal"]` the SAME operand pairs are also
    rendered in each surface format (sharing `pair_id` across formats), which
    lets a cross-format test ask whether the separability is the *operation* or
    just the operator *token* (`+` vs `plus`).

    Operand pairs are drawn from the given BINS (default: all), constrained so
    that every real operation is valid on the pair (sub non-negative, div exact).
    When verbal format is requested, bins are restricted to VERBAL_BINS so the
    same pairs render cleanly in words.

    Returns records with keys:
      op (add/sub/mul/div[/nonsense]), variant='compute', fmt, bin, a, b,
      expected, pair_id, prompt
    """
    if formats is None:
        formats = ["symbolic"]
    if bins is None:
        bins = list(VERBAL_BINS) if "verbal" in formats else list(BINS)

    rng  = random.Random(seed)
    data = []
    pid  = 0
    real_ops = ["add", "sub", "mul", "div"]
    for bin_name in bins:
        lo, hi = BINS[bin_name]
        made, attempts = 0, 0
        while made < n_per_bin and attempts < n_per_bin * 100:
            attempts += 1
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            if b == 0:
                continue
            if a < b:                          # keep subtraction non-negative
                a, b = b, a
            if not (a % b == 0 and a // b > 0):  # require exact division
                continue
            expected = {"add": a + b, "sub": a - b, "mul": a * b, "div": a // b}
            for fmt in formats:
                for op in real_ops:
                    if fmt == "symbolic":
                        prompt = f"{a}{_PROBE_SYM[op]}{b}="
                    else:
                        compute_t = dict((f, ct) for f, ct, _, _ in TEMPLATES[op])[fmt]
                        prompt = _fill(compute_t, a, b, expected[op])
                    data.append(dict(op=op, variant="compute", fmt=fmt,
                                     bin=bin_name, a=a, b=b, expected=expected[op],
                                     pair_id=pid, prompt=prompt))
                # nonsense operator only makes sense symbolically
                if include_nonsense and fmt == "symbolic":
                    data.append(dict(op="nonsense", variant="compute", fmt="symbolic",
                                     bin=bin_name, a=a, b=b, expected=None,
                                     pair_id=pid, prompt=f"{a}#{b}="))
            pid  += 1
            made += 1
    rng.shuffle(data)
    return data


# ── Hard problems (Experiment 2: error mechanism) ─────────────────────────────
def make_hard_problems(ops: list = None, bins: list = None,
                       n_per_cell: int = 150, seed: int = 321) -> list:
    """
    Symbolic compute problems in larger operand bins where the model starts to
    make mistakes — needed to get a mix of correct and wrong answers for the
    error-mechanism analysis.

    Defaults to mul/div at 3d-5d, which break earliest.
    """
    if ops  is None: ops  = ["mul", "div"]
    if bins is None: bins = ["3d", "4d", "5d"]
    rng  = random.Random(seed)
    data = []
    for op in ops:
        for bin_name in bins:
            made, attempts = 0, 0
            while made < n_per_cell and attempts < n_per_cell * 50:
                attempts += 1
                pair = _sample_pair(op, bin_name, rng)
                if pair is None:
                    continue
                a, b, c = pair
                data.append(dict(op=op, variant="compute", fmt="symbolic",
                                 bin=bin_name, a=a, b=b, expected=c,
                                 prompt=f"{a}{_PROBE_SYM[op]}{b}="))
                made += 1
    rng.shuffle(data)
    return data


# ── Train / holdout split ─────────────────────────────────────────────────────
def split_dataset(data: list[dict], holdout_frac: float = 0.2, seed: int = 0):
    """
    Stratified split by (op, variant, fmt, bin).

    Returns:
        train          — for SAE training  (compute + copy only; no cheat)
        holdout_per_op — per-op compute holdout
        holdout_cheat  — per-op cheat holdout
        (multi-op holdouts are built separately via make_multi_op_holdout)
    """
    from collections import defaultdict
    rng = random.Random(seed)

    # Separate cheat records from training data
    cheat_records   = [r for r in data if r["variant"] == "cheat"]
    non_cheat       = [r for r in data if r["variant"] != "cheat"]

    # Stratified split for non-cheat
    groups = defaultdict(list)
    for rec in non_cheat:
        key = (rec["op"], rec["variant"], rec["fmt"], rec["bin"])
        groups[key].append(rec)

    train, holdout_per_op = [], []
    for group in groups.values():
        rng.shuffle(group)
        n_hold = max(1, int(len(group) * holdout_frac))
        holdout_per_op.extend(group[:n_hold])
        train.extend(group[n_hold:])

    rng.shuffle(train)
    rng.shuffle(holdout_per_op)
    rng.shuffle(cheat_records)

    return train, holdout_per_op, cheat_records


# ── Quick summary ─────────────────────────────────────────────────────────────
def summarise(data: list[dict], label: str = ""):
    from collections import Counter
    c = Counter((r["op"], r["variant"], r["fmt"], r["bin"]) for r in data)
    print(f"\n{'─'*60}")
    print(f"  {label or 'Dataset'}  ({len(data)} records)")
    print(f"{'─'*60}")
    print(f"  {'op':8} {'variant':12} {'fmt':10} {'bin':5}  count")
    for (op, variant, fmt, bin_), n in sorted(c.items()):
        print(f"  {op:8} {variant:12} {fmt:10} {bin_:5}  {n}")


if __name__ == "__main__":
    N = 30   # small for quick preview

    data = make_dataset(n_per_cell=N)
    train, hold_per_op, hold_cheat = split_dataset(data)

    multi = make_multi_op_holdout(n_per_cell=N // 2)
    hold_multi_compute = [r for r in multi if r["variant"] == "compute"]
    hold_multi_cheat   = [r for r in multi if r["variant"] == "cheat"]

    summarise(train,            "TRAIN  (SAE input)")
    summarise(hold_per_op,      "HOLDOUT 1 — per-op compute")
    summarise(hold_cheat,       "HOLDOUT 2 — per-op cheat")
    summarise(hold_multi_compute, "HOLDOUT 3 — multi-op compute")
    summarise(hold_multi_cheat,   "HOLDOUT 4 — multi-op cheat")

    print("\nSample prompts from each holdout:")
    for label, dataset in [
        ("per-op",       hold_per_op),
        ("per-op cheat", hold_cheat),
        ("multi-op",     hold_multi_compute),
        ("multi cheat",  hold_multi_cheat),
    ]:
        recs = random.sample(dataset, min(3, len(dataset)))
        print(f"\n  [{label}]")
        for r in recs:
            print(f"    {r['prompt']!r:70s} → {r['expected']}")
