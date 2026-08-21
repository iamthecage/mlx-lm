"""Compare greedy vanilla and native-MTP generation on a local model.

The script deliberately does not download a model itself.  ``mlx_lm.load``
accepts a local model directory (or a Hugging Face id when the caller chooses
to allow that), and the same model instance is used for both generation paths.

Example::

    python -m mlx_lm.examples.validate_mtp_penalties \
        --model /models/qwen3.5-0.8b-mtp \
        --prompt "A short proof:" --max-tokens 64 --mtp-depth 4 \
        --presence-penalty 1.5 --repetition-penalty 1.1 --context-size 64 \
        --compare-logprobs
"""

import argparse
import time

import mlx.core as mx

from ..generate import generate_step, mtp_speculative_generate_step
from ..sample_utils import make_logits_processors, make_sampler
from ..utils import load


def _parser():
    parser = argparse.ArgumentParser(
        description="Validate greedy native MTP against greedy vanilla generation."
    )
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--mtp-depth",
        "--num-draft-tokens",
        dest="mtp_depth",
        type=int,
        default=4,
        help="Number of MTP draft positions per verification round.",
    )
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=0.0)
    parser.add_argument(
        "--context-size",
        type=int,
        default=20,
        help="Context size used by both penalty processors.",
    )
    parser.add_argument(
        "--compare-logprobs",
        action="store_true",
        help="Also compare per-token target log-probability vectors.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _processors(args):
    return make_logits_processors(
        presence_penalty=args.presence_penalty,
        presence_context_size=args.context_size,
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.context_size,
    )


def _first_divergence(lhs, rhs):
    for i, (a, b) in enumerate(zip(lhs, rhs)):
        if a != b:
            return i
    return min(len(lhs), len(rhs)) if len(lhs) != len(rhs) else None


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.mtp_depth < 1:
        raise ValueError("--mtp-depth must be positive")
    if args.context_size < 1:
        raise ValueError("--context-size must be positive")

    model, tokenizer = load(args.model, trust_remote_code=args.trust_remote_code)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not prompt:
        raise ValueError("The prompt must contain at least one token")
    prompt = mx.array(prompt, dtype=mx.uint32)
    sampler = make_sampler(temp=0.0)

    start = time.perf_counter()
    vanilla = list(
        generate_step(
            prompt,
            model,
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=_processors(args),
        )
    )
    vanilla_time = time.perf_counter() - start

    start = time.perf_counter()
    mtp = list(
        mtp_speculative_generate_step(
            prompt,
            model,
            max_tokens=args.max_tokens,
            num_draft_tokens=args.mtp_depth,
            sampler=sampler,
            logits_processors=_processors(args),
        )
    )
    mtp_time = time.perf_counter() - start

    vanilla_tokens = [int(token) for token, _ in vanilla]
    mtp_tokens = [int(token) for token, _, _ in mtp]
    divergence = _first_divergence(vanilla_tokens, mtp_tokens)

    if args.compare_logprobs:
        if len(vanilla) != len(mtp):
            raise AssertionError("cannot compare logprobs for unequal token lengths")
        max_error = 0.0
        for (_, vanilla_lp), (_, mtp_lp, _) in zip(vanilla, mtp):
            max_error = max(max_error, float(mx.max(mx.abs(vanilla_lp - mtp_lp)).item()))
        print(f"max target logprob absolute error: {max_error:.6g}")

    accepted = sum(1 for _, _, from_draft in mtp if from_draft)
    target_tokens = len(mtp) - accepted
    accept_length = len(mtp) / max(target_tokens, 1)
    print(
        f"tokens: {len(mtp)}; first divergence: "
        f"{'none' if divergence is None else divergence}"
    )
    print(f"accepted drafts: {accepted}; accept length: {accept_length:.3f}")
    print(f"vanilla tokens/sec: {len(vanilla) / max(vanilla_time, 1e-9):.3f}")
    print(f"native MTP tokens/sec: {len(mtp) / max(mtp_time, 1e-9):.3f}")
    if divergence is not None:
        raise AssertionError(
            f"greedy divergence at token {divergence}: "
            f"vanilla={vanilla_tokens[divergence:divergence + 1]} "
            f"mtp={mtp_tokens[divergence:divergence + 1]}"
        )
    return 0


if __name__ == "__main__":
    main()
