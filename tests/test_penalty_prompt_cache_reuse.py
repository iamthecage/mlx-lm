# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.sample_utils import make_logits_processors
from mlx_lm.server import _prepare_cached_prompt


class _StatefulCache:
    """Tiny cache whose logical state is the complete token history."""

    def __init__(self, tokens=()):
        self._tokens = [int(token) for token in mx.array(tokens).tolist()]

    @property
    def state(self):
        return mx.array(self._tokens, dtype=mx.uint32)

    @state.setter
    def state(self, value):
        self._tokens = [int(token) for token in mx.array(value).tolist()]

    @property
    def tokens(self):
        return list(self._tokens)

    def append(self, token):
        self._tokens.append(int(token))

    @property
    def nbytes(self):
        return len(self._tokens) * 4

    def is_trimmable(self):
        return True

    def trim(self, n):
        removed = min(n, len(self._tokens))
        if removed:
            del self._tokens[-removed:]
        return removed

    def __deepcopy__(self, memo):
        clone = type(self)(self._tokens)
        memo[id(self)] = clone
        return clone


class _AlignmentSensitiveModel:
    """A deterministic model where cache position affects every logits row."""

    layers = [object()]
    vocab_size = 16

    def make_cache(self):
        return [_StatefulCache()]

    def __call__(self, tokens, cache=None, input_embeddings=None):
        del input_embeddings
        state = cache[0]
        rows = []
        for token in tokens[0].tolist():
            history = state.tokens
            state.append(token)
            position = len(history) + 1

            # Include both the sequence checksum and the cache length.  A
            # duplicated bootstrap token therefore changes the logits.
            checksum = sum((index + 1) * value for index, value in enumerate(history))
            anchor = (checksum + 7 * position + 3 * token) % self.vocab_size
            ids = mx.arange(self.vocab_size)
            logits = 0.05 * (ids + 1).astype(mx.float32)
            logits = logits + mx.where(
                ids == anchor, mx.array(4.0), mx.array(0.0)
            )

            # Keep every prompt token at a positive, visible score so both
            # penalties alter cached-prefix entries in the first logprob row.
            for index, previous in enumerate(state.tokens):
                logits = logits + mx.where(
                    ids == previous,
                    mx.array(1.5 + 0.01 * index),
                    mx.array(0.0),
                )
            rows.append(logits)
        return mx.stack(rows, axis=0)[None]


class _RecordingProcessor:
    def __init__(self, processors, history):
        self._processors = processors
        self.history = history

    def __call__(self, tokens, logits):
        self.history.append(tokens.tolist())
        for processor in self._processors:
            logits = processor(tokens, logits)
        return logits


class TestRealLRUPromptCachePenaltyReuse(unittest.TestCase):
    model_key = "alignment-sensitive-test-model"
    prompt = mx.array([1, 4, 2, 6, 3], dtype=mx.uint32)
    cached_prefix = prompt[:3]
    max_tokens = 4

    @staticmethod
    def _uncached(processors=None):
        return list(
            generate_step(
                TestRealLRUPromptCachePenaltyReuse.prompt,
                _AlignmentSensitiveModel(),
                max_tokens=TestRealLRUPromptCachePenaltyReuse.max_tokens,
                logits_processors=processors,
            )
        )

    @staticmethod
    def _prefilled_cache(model, tokens):
        prompt_cache = model.make_cache()
        # max_tokens=0 still performs the prompt prefill and bootstrap model
        # call, but yields no generated tokens.
        list(generate_step(tokens, model, max_tokens=0, prompt_cache=prompt_cache))
        return prompt_cache

    @staticmethod
    def _assert_results_equal(test, expected, actual):
        test.assertEqual([token for token, _ in expected], [token for token, _ in actual])
        test.assertEqual(len(expected), len(actual))
        for (_, expected_logprobs), (_, actual_logprobs) in zip(expected, actual):
            test.assertTrue(mx.array_equal(expected_logprobs, actual_logprobs).item())
            test.assertEqual(expected_logprobs.shape, (_AlignmentSensitiveModel.vocab_size,))

    @staticmethod
    def _processors(presence_penalty, repetition_penalty):
        return make_logits_processors(
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            presence_context_size=32,
            repetition_context_size=32,
        )

    def _run_shorter_prefix_case(self, presence_penalty, repetition_penalty):
        processors = self._processors(presence_penalty, repetition_penalty)
        uncached = self._uncached(processors)

        source_model = _AlignmentSensitiveModel()
        source_cache = self._prefilled_cache(source_model, self.cached_prefix)
        prompt_cache = LRUPromptCache(max_size=8)
        prompt_cache.insert_cache(
            self.model_key,
            self.cached_prefix.tolist(),
            source_cache,
        )
        fetched, rest = prompt_cache.fetch_nearest_cache(
            self.model_key, self.prompt.tolist()
        )
        self.assertEqual(rest, self.prompt[3:].tolist())
        prepared, rest, prefix_count = _prepare_cached_prompt(
            self.prompt.tolist(), fetched, rest
        )
        self.assertEqual(prefix_count, len(self.cached_prefix))
        self.assertEqual(rest, self.prompt[3:].tolist())

        history = []
        recording_processor = _RecordingProcessor(processors, history)
        cached = list(
            generate_step(
                mx.array(rest, dtype=mx.uint32),
                _AlignmentSensitiveModel(),
                max_tokens=self.max_tokens,
                prompt_cache=prepared,
                prompt_cache_tokens=self.cached_prefix,
                logits_processors=[recording_processor],
            )
        )
        self._assert_results_equal(self, uncached, cached)
        self.assertEqual(history[0], self.prompt.tolist())

        # Generation mutates the fetched copy, never the entry held by LRU.
        stored, stored_rest = prompt_cache.fetch_nearest_cache(
            self.model_key, self.cached_prefix.tolist()
        )
        self.assertEqual(stored_rest, [])
        self.assertEqual(stored[0].state.tolist(), self.cached_prefix.tolist())

    def _run_exact_hit_case(self, presence_penalty, repetition_penalty):
        processors = self._processors(presence_penalty, repetition_penalty)
        uncached = self._uncached(processors)

        source_model = _AlignmentSensitiveModel()
        source_cache = self._prefilled_cache(source_model, self.prompt)
        prompt_cache = LRUPromptCache(max_size=8)
        prompt_cache.insert_cache(self.model_key, self.prompt.tolist(), source_cache)
        fetched, rest = prompt_cache.fetch_nearest_cache(
            self.model_key, self.prompt.tolist()
        )
        self.assertEqual(rest, [])
        prepared, rest, prefix_count = _prepare_cached_prompt(
            self.prompt.tolist(), fetched, rest
        )
        self.assertEqual(prefix_count, len(self.prompt) - 1)
        self.assertEqual(rest, [self.prompt[-1].item()])
        self.assertEqual(prepared[0].state.tolist(), self.prompt[:-1].tolist())

        history = []
        recording_processor = _RecordingProcessor(processors, history)
        cached = list(
            generate_step(
                mx.array(rest, dtype=mx.uint32),
                _AlignmentSensitiveModel(),
                max_tokens=self.max_tokens,
                prompt_cache=prepared,
                prompt_cache_tokens=self.prompt[:-1],
                logits_processors=[recording_processor],
            )
        )
        self._assert_results_equal(self, uncached, cached)
        self.assertEqual(history[0], self.prompt.tolist())

        # The trim and subsequent generation happened only on the deep copy.
        stored, stored_rest = prompt_cache.fetch_nearest_cache(
            self.model_key, self.prompt.tolist()
        )
        self.assertEqual(stored_rest, [])
        self.assertEqual(stored[0].state.tolist(), self.prompt.tolist())

    def test_shorter_prefix_reuse_presence_penalty(self):
        self._run_shorter_prefix_case(1.5, None)

    def test_shorter_prefix_reuse_repetition_penalty(self):
        self._run_shorter_prefix_case(None, 1.1)

    def test_shorter_prefix_reuse_combined_penalties(self):
        self._run_shorter_prefix_case(1.5, 1.1)

    def test_exact_hit_reuse_presence_penalty(self):
        self._run_exact_hit_case(1.5, None)

    def test_exact_hit_reuse_repetition_penalty(self):
        self._run_exact_hit_case(None, 1.1)

    def test_exact_hit_reuse_combined_penalties(self):
        self._run_exact_hit_case(1.5, 1.1)

    def test_penalties_change_cached_prefix_logits(self):
        baseline = self._uncached()
        for presence_penalty, repetition_penalty in (
            (1.5, None),
            (None, 1.1),
            (1.5, 1.1),
        ):
            with self.subTest(
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
            ):
                penalized = self._uncached(
                    self._processors(presence_penalty, repetition_penalty)
                )
                for token in self.prompt.tolist():
                    self.assertNotEqual(
                        float(
                            mx.abs(
                                baseline[0][1][token] - penalized[0][1][token]
                            ).item()
                        ),
                        0.0,
                    )


if __name__ == "__main__":
    unittest.main()
