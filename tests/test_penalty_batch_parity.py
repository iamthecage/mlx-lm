import unittest

import mlx.core as mx

from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models.cache import TokenBuffer
from mlx_lm.sample_utils import make_logits_processors


class _PenaltyModel:
    """Small stateless model whose logits make token-history penalties visible."""

    vocab_size = 7

    def make_cache(self):
        # The model deliberately has no cache state.  This keeps these tests
        # focused on batching and logical processor history.
        return []

    def __call__(self, tokens, cache=None, input_embeddings=None):
        del cache, input_embeddings
        rows = []
        for row in tokens.tolist():
            row_logits = []
            for token in row:
                preferred = (int(token) + 2) % self.vocab_size
                alternate = (int(token) + 3) % self.vocab_size
                third = (int(token) + 1) % self.vocab_size
                scores = [-2.0] * self.vocab_size
                scores[preferred] = 3.8
                scores[alternate] = 3.6
                scores[third] = 1.2
                row_logits.append(scores)
            rows.append(row_logits)
        logits = mx.array(rows, dtype=mx.float32)
        mx.eval(logits)
        return logits


def _processors(*, repetition_penalty=None, presence_penalty=None, seen=None):
    processors = make_logits_processors(
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
    )
    if seen is None:
        return processors

    def record(tokens, logits):
        seen.append(tokens.tolist())
        return logits

    return [record, *processors]


def _single_result(model, prompt, max_tokens, processors):
    result = {"tokens": [], "logprobs": []}
    for token, logprobs in generate_step(
        mx.array(prompt, dtype=mx.uint32),
        model,
        max_tokens=max_tokens,
        logits_processors=processors,
    ):
        mx.eval(logprobs)
        result["tokens"].append(token)
        result["logprobs"].append(mx.array(logprobs))
    mx.synchronize()
    return result


def _batch_results(model, prompts, max_tokens, processors):
    generator = BatchGenerator(
        model,
        max_tokens=max(max_tokens),
        prefill_batch_size=len(prompts),
        prefill_step_size=2,
        completion_batch_size=len(prompts),
    )
    uids = generator.insert(
        prompts,
        max_tokens=max_tokens,
        logits_processors=processors,
    )
    results = {uid: {"tokens": [], "logprobs": []} for uid in uids}
    try:
        while True:
            responses = generator.next_generated()
            if not responses:
                break
            for response in responses:
                results[response.uid]["tokens"].append(response.token)
                results[response.uid]["logprobs"].append(
                    mx.array(response.logprobs)
                )
    finally:
        mx.synchronize()
        generator.close()
    return [results[uid] for uid in uids]


class TestPenaltyBatchParity(unittest.TestCase):
    def assert_result_equal(self, expected, actual):
        self.assertEqual(expected["tokens"], actual["tokens"])
        self.assertEqual(len(expected["logprobs"]), len(actual["logprobs"]))
        for expected_logprobs, actual_logprobs in zip(
            expected["logprobs"], actual["logprobs"]
        ):
            self.assertTrue(
                mx.allclose(
                    expected_logprobs, actual_logprobs, rtol=1e-6, atol=1e-6
                )
            )

    def test_batch_matches_single_for_penalty_configurations(self):
        prompts = [
            [1, 6, 2, 4],
            [3, 5, 1],
            [0, 6, 4, 2],
        ]
        max_tokens = [4, 3, 5]
        configurations = [
            ("none", {}),
            ("repetition_zero", {"repetition_penalty": 0.0}),
            ("repetition_one", {"repetition_penalty": 1.0}),
            ("repetition", {"repetition_penalty": 1.1}),
            ("presence", {"presence_penalty": 1.5}),
            (
                "presence_and_repetition",
                {"presence_penalty": 1.5, "repetition_penalty": 1.1},
            ),
        ]

        baseline = None
        for name, kwargs in configurations:
            with self.subTest(configuration=name):
                single = [
                    _single_result(
                        _PenaltyModel(),
                        prompt,
                        limit,
                        _processors(**kwargs),
                    )
                    for prompt, limit in zip(prompts, max_tokens)
                ]
                batch = _batch_results(
                    _PenaltyModel(),
                    prompts,
                    max_tokens,
                    [_processors(**kwargs) for _ in prompts],
                )
                for expected, actual in zip(single, batch):
                    self.assert_result_equal(expected, actual)

                if baseline is None:
                    baseline = single
                elif (
                    kwargs.get("repetition_penalty", 0.0) not in (0.0, 1.0)
                    or kwargs.get("presence_penalty", 0.0)
                ):
                    self.assertTrue(
                        any(
                            expected["tokens"] != base["tokens"]
                            for expected, base in zip(single, baseline)
                        )
                    )

    def test_cached_prefix_history_matches_uncached_prompt(self):
        full_prompt = [1, 6, 2, 4]
        cached_prefix = full_prompt[:2]
        suffix = full_prompt[2:]
        kwargs = {"presence_penalty": 1.5, "repetition_penalty": 1.1}
        uncached_history = []
        uncached = _single_result(
            _PenaltyModel(),
            full_prompt,
            4,
            _processors(**kwargs, seen=uncached_history),
        )

        cached_history = []
        generator = BatchGenerator(
            _PenaltyModel(),
            max_tokens=4,
            prefill_batch_size=1,
            prefill_step_size=2,
            completion_batch_size=1,
        )
        (uid,) = generator.insert(
            [suffix],
            max_tokens=[4],
            caches=[[]],
            all_tokens=[cached_prefix],
            logits_processors=[_processors(**kwargs, seen=cached_history)],
        )
        cached = {"tokens": [], "logprobs": []}
        try:
            while True:
                responses = generator.next_generated()
                if not responses:
                    break
                for response in responses:
                    self.assertEqual(response.uid, uid)
                    cached["tokens"].append(response.token)
                    cached["logprobs"].append(mx.array(response.logprobs))
        finally:
            mx.synchronize()
            generator.close()

        self.assert_result_equal(uncached, cached)
        self.assertGreaterEqual(len(uncached_history), 1)
        self.assertGreaterEqual(len(cached_history), 1)
        self.assertEqual(uncached_history[0], full_prompt)
        self.assertEqual(cached_history[0], full_prompt)

    def test_lifecycle_filter_and_late_extension_preserve_context(self):
        prompts = [
            [1, 6, 2, 4],
            [3, 5, 1],
        ]
        max_tokens = [6, 2]
        kwargs = {"presence_penalty": 1.5, "repetition_penalty": 1.1}
        baseline = [
            _single_result(
                _PenaltyModel(),
                prompt,
                limit,
                _processors(**kwargs) if i == 0 else _processors(),
            )
            for i, (prompt, limit) in enumerate(zip(prompts, max_tokens))
        ]

        late_prompt = [2, 4, 6]
        late_max_tokens = 3
        late_baseline = _single_result(
            _PenaltyModel(), late_prompt, late_max_tokens, _processors(**kwargs)
        )

        generator = BatchGenerator(
            _PenaltyModel(),
            max_tokens=max(max_tokens),
            prefill_batch_size=2,
            prefill_step_size=2,
            completion_batch_size=3,
        )
        uids = generator.insert(
            prompts,
            max_tokens=max_tokens,
            logits_processors=[_processors(**kwargs), []],
        )
        results = {uid: {"tokens": [], "logprobs": []} for uid in uids}
        try:
            first_responses = generator.next_generated()
            for response in first_responses:
                results[response.uid]["tokens"].append(response.token)
                results[response.uid]["logprobs"].append(
                    mx.array(response.logprobs)
                )

            for uid, processors, context in zip(
                generator._generation_batch.uids,
                generator._generation_batch.logits_processors,
                generator._generation_batch._token_context,
            ):
                if processors:
                    self.assertIsInstance(context, TokenBuffer, msg=f"uid={uid}")
                else:
                    self.assertIsNone(context, msg=f"uid={uid}")

            second_responses = generator.next_generated()
            for response in second_responses:
                results[response.uid]["tokens"].append(response.token)
                results[response.uid]["logprobs"].append(
                    mx.array(response.logprobs)
                )
            self.assertNotIn(uids[1], generator._generation_batch.uids)

            (late_uid,) = generator.insert(
                [late_prompt],
                max_tokens=[late_max_tokens],
                logits_processors=[_processors(**kwargs)],
            )
            saw_extension = False
            while True:
                responses = generator.next_generated()
                if not responses:
                    break
                for response in responses:
                    results.setdefault(
                        response.uid, {"tokens": [], "logprobs": []}
                    )
                    results[response.uid]["tokens"].append(response.token)
                    results[response.uid]["logprobs"].append(
                        mx.array(response.logprobs)
                    )
                if late_uid in generator._generation_batch.uids:
                    saw_extension = len(generator._generation_batch) > 1
                    if saw_extension:
                        contexts = dict(
                            zip(
                                generator._generation_batch.uids,
                                generator._generation_batch._token_context,
                            )
                        )
                        self.assertIsInstance(contexts[uids[0]], TokenBuffer)
                        self.assertIsInstance(contexts[late_uid], TokenBuffer)
            self.assertTrue(saw_extension)
            self.assert_result_equal(baseline[0], results[uids[0]])
            self.assert_result_equal(baseline[1], results[uids[1]])
            self.assert_result_equal(late_baseline, results[late_uid])
        finally:
            mx.synchronize()
            generator.close()
