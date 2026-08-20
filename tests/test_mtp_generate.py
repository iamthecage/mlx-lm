# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step, mtp_speculative_generate_step
from mlx_lm.sample_utils import make_logits_processors


class _TinyCache:
    """Small exact-trimmable cache for MTP generation tests."""

    def __init__(self):
        self.offset = 0
        self.trim_calls = []

    @property
    def state(self):
        return mx.array([self.offset], dtype=mx.int32)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(int(n), self.offset)
        self.trim_calls.append(n)
        self.offset -= n
        return n

    def start_speculation(self):
        pass

    def stop_speculation(self):
        pass


class _TinyBackbone:
    def __call__(self, tokens, cache=None):
        if cache:
            cache[0].offset += tokens.shape[1]
        # Hidden values are the input token ids.  The fake target and MTP head
        # below turn those values into deterministic logits.
        return tokens.astype(mx.float32)[..., None]

    def norm(self, hidden):
        return hidden


class _TinyMTPModel:
    """A tiny MTP-capable model with independently-scaled draft logits."""

    layers = [object()]
    vocab_size = 32

    def __init__(self, draft_scale=2.0, target_scale=2.0, draft_offset=0):
        self.mtp = object()
        self.model = _TinyBackbone()
        self.draft_scale = draft_scale
        self.target_scale = target_scale
        self.draft_offset = draft_offset

    def make_cache(self):
        return [_TinyCache()]

    def make_mtp_cache(self):
        return [_TinyCache()]

    def _logits(self, tokens, scale, offset=0):
        tokens = tokens.astype(mx.int32)
        primary = (tokens + offset) % self.vocab_size
        secondary = (primary + 1) % self.vocab_size
        vocab = mx.arange(self.vocab_size)
        return mx.where(
            vocab[None, None, :] == primary[..., None],
            mx.array(scale),
            mx.where(
                vocab[None, None, :] == secondary[..., None],
                mx.array(1.5),
                mx.array(0.0),
            ),
        )

    def logits(self, hidden):
        return self._logits(hidden[..., 0], self.target_scale)

    def __call__(self, tokens, cache=None, input_embeddings=None):
        del input_embeddings
        hidden = self.model(tokens, cache=cache)
        return self.logits(hidden)

    def mtp_step(self, hidden, tokens, mtp_cache):
        del hidden
        if mtp_cache:
            mtp_cache[0].offset += tokens.shape[1]
        return (
            self._logits(tokens, self.draft_scale, self.draft_offset),
            tokens.astype(mx.float32)[..., None],
        )


class _RejectingTargetModel(_TinyMTPModel):
    """Target outputs stay off the deliberately bad draft sequence."""

    def logits(self, hidden):
        tokens = hidden[..., 0].astype(mx.int32)
        primary = mx.where(
            tokens == 3,
            mx.array(4),
            mx.where(
                tokens == 9,
                mx.array(5),
                mx.where(tokens == 10, mx.array(6), (tokens + 1) % self.vocab_size),
            ),
        )
        secondary = (primary + 1) % self.vocab_size
        vocab = mx.arange(self.vocab_size)
        return mx.where(
            vocab[None, None, :] == primary[..., None],
            mx.array(self.target_scale),
            mx.where(
                vocab[None, None, :] == secondary[..., None],
                mx.array(1.5),
                mx.array(0.0),
            ),
        )


def _tokens(step, **kwargs):
    return [token for token, _, _ in mtp_speculative_generate_step(
        mx.array([1, 2, 3], dtype=mx.uint32),
        _TinyMTPModel(),
        max_tokens=step,
        num_draft_tokens=2,
        **kwargs,
    )]


class TestMTPGeneration(unittest.TestCase):
    def test_processors_see_one_dimensional_history_and_two_dimensional_logits(self):
        seen = []

        def processor(tokens, logits):
            self.assertEqual(tokens.ndim, 1)
            self.assertEqual(logits.shape, (1, _TinyMTPModel.vocab_size))
            seen.append(tokens.tolist())
            return logits

        out = _tokens(6, logits_processors=[processor])
        self.assertEqual(out, [3, 3, 3, 3, 3, 3])
        self.assertGreaterEqual(len(seen), 4)

    def test_draft_history_is_temporary_and_positionally_chained(self):
        seen = []

        def processor(tokens, logits):
            seen.append(tokens.tolist())
            return logits

        list(_tokens(4, logits_processors=[processor]))
        # Bootstrap, two draft positions, then three target verification
        # positions.  Draft position two includes the first sampled draft.
        self.assertEqual(seen[0], [1, 2, 3])
        self.assertEqual(seen[1], [1, 2, 3, 3])
        self.assertEqual(seen[2], [1, 2, 3, 3, 3])
        self.assertEqual(seen[3], [1, 2, 3, 3])
        self.assertEqual(seen[4], [1, 2, 3, 3, 3])

    def test_target_history_uses_target_samples_and_rejected_drafts_do_not_leak(self):
        seen = []

        def processor(tokens, logits):
            seen.append(tokens.tolist())
            return logits

        # The draft head proposes 9/10 while the target emits 4/5, forcing a
        # rejection at every round.
        list(
            mtp_speculative_generate_step(
                mx.array([1, 2, 3], dtype=mx.uint32),
                _RejectingTargetModel(draft_offset=6),
                max_tokens=7,
                num_draft_tokens=2,
                logits_processors=[processor],
            )
        )
        self.assertEqual(seen[3:6], [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
        for history in seen[3:6]:
            self.assertNotIn(9, history)
            self.assertNotIn(10, history)

    def test_presence_and_repetition_penalties_are_non_neutral_and_match_vanilla(self):
        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        for processors in (
            make_logits_processors(presence_penalty=2.0),
            make_logits_processors(repetition_penalty=2.0),
        ):
            with self.subTest(processors=processors):
                vanilla = [
                    token
                    for token, _ in generate_step(
                        prompt,
                        _TinyMTPModel(),
                        max_tokens=7,
                        logits_processors=processors,
                    )
                ]
                mtp = _tokens(7, logits_processors=processors)
                self.assertNotEqual(vanilla[0], 3)
                self.assertEqual(mtp, vanilla)

    def test_accepted_tokens_expose_target_logprobs(self):
        model = _TinyMTPModel(draft_scale=9.0, target_scale=2.0)
        out = list(
            mtp_speculative_generate_step(
                mx.array([1, 2, 3], dtype=mx.uint32),
                model,
                max_tokens=5,
                num_draft_tokens=2,
            )
        )
        self.assertTrue(out[1][2])
        target_logits = model._logits(
            mx.array([3]), model.target_scale
        ).squeeze(0).squeeze(0)
        target_lp = target_logits - mx.logsumexp(target_logits)
        self.assertTrue(mx.allclose(out[1][1], target_lp).item())

    def test_prompt_cache_tokens_prefix_processor_history(self):
        seen = []

        def processor(tokens, logits):
            seen.append(tokens.tolist())
            return logits

        list(
            mtp_speculative_generate_step(
                mx.array([3, 4], dtype=mx.uint32),
                _TinyMTPModel(),
                prompt_cache=[],
                prompt_cache_tokens=mx.array([1, 2], dtype=mx.uint32),
                max_tokens=1,
                logits_processors=[processor],
            )
        )
        self.assertEqual(seen[0], [1, 2, 3, 4])

    def test_generator_close_rolls_back_interrupted_round_once(self):
        model_cache = _TinyCache()
        mtp_cache = _TinyCache()
        generator = mtp_speculative_generate_step(
            mx.array([1, 2, 3], dtype=mx.uint32),
            _RejectingTargetModel(draft_offset=6),
            prompt_cache=[model_cache, mtp_cache],
            max_tokens=99,
            num_draft_tokens=2,
            logits_processors=[lambda _, logits: logits],
        )
        next(generator)  # bootstrap
        next(generator)  # correction token; the round is still pending
        generator.close()
        self.assertEqual(model_cache.trim_calls, [2])
        self.assertEqual(mtp_cache.trim_calls, [1])
        generator.close()
        self.assertEqual(model_cache.trim_calls, [2])
        self.assertEqual(mtp_cache.trim_calls, [1])

    def test_no_processor_path_matches_vanilla(self):
        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        vanilla = [
            token
            for token, _ in generate_step(prompt, _TinyMTPModel(), max_tokens=8)
        ]
        mtp = _tokens(8)
        self.assertEqual(mtp, vanilla)


if __name__ == "__main__":
    unittest.main()
