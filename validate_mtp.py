"""Validate native MTP self-speculation across draft depths.

Loads the model ONCE, runs a non-speculative greedy baseline, then MTP at
several depths. For each depth asserts token-identical output (losslessness)
and reports accept-length (speedup predictor) and tok/s.

Usage: python validate_mtp.py [MODEL_PATH] [MAX_TOKENS] [--kv-bits N]
"""

import sys
import time

import mlx.core as mx

from mlx_lm import load
from mlx_lm.generate import generate_step, mtp_speculative_generate_step

MODEL = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else "/Users/AWillet/optiq_output/Qwen3.6-27B-Thinking-MLX-mixed-7.6bit"
)
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 200
KV_BITS = None
if "--kv-bits" in sys.argv:
    KV_BITS = int(sys.argv[sys.argv.index("--kv-bits") + 1])
DEPTHS = [1, 2, 3, 4, 6]

PROMPT = (
    "Write a complete Python implementation of a red-black tree with insert, "
    "delete, search, and full docstrings."
)
kv = dict(kv_bits=KV_BITS, kv_group_size=64, quantized_kv_start=0)
greedy = lambda x: mx.argmax(x, axis=-1)


def as_int(t):
    return int(t.item()) if hasattr(t, "item") else int(t)


def main():
    print(f"Loading {MODEL} ...", flush=True)
    model, tok = load(MODEL)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        **{"enable_thinking": False},
    )
    prompt = mx.array(ids)
    print(f"prompt {prompt.size} tok | max_new {MAX_TOKENS} | kv_bits={KV_BITS}\n")

    base = []
    t0 = time.perf_counter()
    for token, _ in generate_step(prompt, model, max_tokens=MAX_TOKENS, sampler=greedy, **kv):
        base.append(as_int(token))
        if len(base) >= MAX_TOKENS:
            break
    bdt = time.perf_counter() - t0
    base_tps = len(base) / bdt
    print(f"baseline: {len(base)} tok @ {base_tps:.1f} tok/s\n")
    print(f"{'depth':>5} {'tok/s':>7} {'speedup':>8} {'accept-len':>11} {'accepted':>10} {'lossless':>9}")

    for D in DEPTHS:
        toks, ndraft = [], 0
        t0 = time.perf_counter()
        for token, _, fd in mtp_speculative_generate_step(
            prompt, model, max_tokens=MAX_TOKENS, num_draft_tokens=D, sampler=greedy, **kv
        ):
            toks.append(as_int(token))
            ndraft += int(bool(fd))
            if len(toks) >= MAX_TOKENS:
                break
        dt = time.perf_counter() - t0
        emitted = len(toks)
        tau = emitted / max(emitted - ndraft, 1)
        tps = emitted / dt
        n = min(len(base), emitted)
        div = next((i for i in range(n) if base[i] != toks[i]), -1)
        lossless = "yes" if div == -1 else f"@{div}"
        print(f"{D:>5} {tps:>7.1f} {tps/base_tps:>7.2f}x {tau:>11.2f} {ndraft:>5}/{emitted:<4} {lossless:>9}")


if __name__ == "__main__":
    main()
