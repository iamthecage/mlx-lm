# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import types
import unittest
from queue import Queue
from unittest.mock import patch

import mlx.core as mx
import requests

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_logits_processors
from mlx_lm.server import (
    APIHandler,
    GenerationArguments,
    LogitsProcessorArguments,
    LRUPromptCache,
    ModelDescription,
    Response,
    ResponseGenerator,
    SamplingArguments,
    _make_logits_processors,
    _prepare_cached_prompt,
    _process_control_tokens,
)
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prefill_step_size": 2048,
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
                "allowed_origins": ["*"],
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer

    def load_default(self):
        return self.load("default_model", None, "default_model")


class MockCache:
    def __init__(self, value, is_trimmable: bool = True):
        self.value = value
        self._is_trimmable = is_trimmable

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value

    def is_trimmable(self):
        return self._is_trimmable

    def trim(self, n):
        assert self._is_trimmable
        return n


class TestProcessControlTokens(unittest.TestCase):
    @staticmethod
    def _r(text, state, match=None):
        return Response(text, 0, state, match, 0.0, None, ())

    def test_single_tool_call_passes_body_with_open_and_close_crossings(self):
        r = self._r
        stream = [
            r("hi ", "normal"),
            r("<tool_call>", "tool", match=(0,)),
            r("body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r(" bye", "normal"),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "hi body bye")
        states = [t.state for t in out]
        self.assertEqual(sum(1 for a, b in zip(states, states[1:]) if a != b), 2)

    def test_back_to_back_tool_calls_emit_state_crossings(self):
        r = self._r
        stream = [
            r("<tool_call>", "tool", match=(0,)),
            r("call1_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r("<tool_call>", "tool", match=(0,)),
            r("call2_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "call1_bodycall2_body")
        states = [t.state for t in out]
        crossings = sum(
            1 for a, b in zip(states, states[1:]) if a == "tool" and b == "normal"
        )
        self.assertEqual(crossings, 2)

    def test_multi_token_match_preserves_order(self):
        r = self._r
        match = (10, 11, 12)
        stream = [
            r("body", "tool"),
            r("</", "tool"),
            r("tool", "tool"),
            r("_call>", "normal", match=match),
            r(" ok", "normal"),
        ]
        ctx = types.SimpleNamespace(sequences={match: "</tool_call>"})
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual([t.text for t in out], ["body", "", "", "", " ok"])
        self.assertEqual(
            [t.state for t in out],
            ["tool", "tool", "tool", "normal", "normal"],
        )


class _PenaltyHistoryModel:
    """Small deterministic model for server cache/penalty parity tests."""

    vocab_size = 32
    layers = [object()]

    def __call__(self, tokens, cache=None, input_embeddings=None):
        del cache, input_embeddings
        rows = []
        for token in tokens[0].tolist():
            next_token = (token + 1) % self.vocab_size
            rows.append(
                mx.where(
                    mx.arange(self.vocab_size) == next_token,
                    mx.array(10.0),
                    mx.array(0.0),
                )
            )
        return mx.stack(rows, axis=0)[None]


class TestServerPenaltyAndCachePlumbing(unittest.TestCase):
    @staticmethod
    def _generation_args(repetition_penalty=0.0, presence_penalty=0.0):
        return GenerationArguments(
            model=ModelDescription("model", "draft", None),
            sampling=SamplingArguments(0.0, 1.0, 0, 0.0, 0.0, 0.0),
            logits=LogitsProcessorArguments(
                logit_bias=None,
                repetition_penalty=repetition_penalty,
                repetition_context_size=20,
                presence_penalty=presence_penalty,
                presence_context_size=20,
                frequency_penalty=0.0,
                frequency_context_size=20,
            ),
            stop_words=[],
            max_tokens=4,
            num_draft_tokens=2,
            logprobs=False,
            top_logprobs=-1,
            seed=None,
            chat_template_kwargs=None,
        )

    def test_repetition_factor_one_is_preserved_by_request_parser(self):
        """The API parser must not rewrite an explicitly requested 1.0."""
        body = {
            "prompt": "hello",
            "model": "default_model",
            "max_tokens": 1,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 4,
            "min_p": 0.2,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
        }
        raw_body = json.dumps(body).encode()
        captured = {}

        cli_args = types.SimpleNamespace(
            num_draft_tokens=3,
            temp=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            max_tokens=512,
            chat_template_args={},
        )

        def generate(request, args, progress_callback=None):
            del request, progress_callback
            captured["args"] = args
            raise RuntimeError("stop after argument construction")

        handler = object.__new__(APIHandler)
        handler.response_generator = types.SimpleNamespace(
            cli_args=cli_args, generate=generate
        )
        handler.path = "/v1/completions"
        handler.headers = {"Content-Length": str(len(raw_body))}
        handler.rfile = io.BytesIO(raw_body)
        handler.wfile = io.BytesIO()
        handler._dump_system_prompt = lambda: None
        handler._set_completion_headers = lambda status: None
        handler.end_headers = lambda: None
        handler.handle_text_completions = lambda: types.SimpleNamespace(
            request_type="text", prompt=body["prompt"], messages=[], tools=None
        )

        handler.do_POST()

        args = captured["args"]
        self.assertEqual(handler.repetition_penalty, 1.0)
        self.assertEqual(args.sampling.temperature, 0.3)
        self.assertEqual(args.sampling.top_p, 0.8)
        self.assertEqual(args.sampling.top_k, 4)
        self.assertEqual(args.sampling.min_p, 0.2)
        self.assertEqual(args.logits.repetition_penalty, 1.0)
        self.assertEqual(args.logits.presence_penalty, 1.5)

    def test_one_repetition_factor_has_no_processor(self):
        args = self._generation_args(repetition_penalty=1.0)
        self.assertEqual(_make_logits_processors(args), [])

    def test_one_repetition_factor_plus_presence_has_only_presence(self):
        args = self._generation_args(repetition_penalty=1.0, presence_penalty=1.5)
        processors = _make_logits_processors(args)
        self.assertEqual(len(processors), 1)

        tokens = mx.array([2, 2, 3])
        logits = mx.zeros((1, _PenaltyHistoryModel.vocab_size))
        result = processors[0](tokens, logits)
        self.assertEqual(result[0, 2].item(), -1.5)
        self.assertEqual(result[0, 3].item(), -1.5)

    def test_cached_single_request_passes_prefix_to_generation(self):
        prompt = [1, 2, 3, 4]
        captured = {}

        class PromptCache:
            def fetch_nearest_cache(self, model_key, tokens):
                self.model_key = model_key
                self.tokens = tokens
                return ["cached"], tokens[2:]

            def insert_cache(self, *args, **kwargs):
                pass

        provider = types.SimpleNamespace(
            model=object(),
            tokenizer=types.SimpleNamespace(
                has_thinking=False,
                has_tool_calling=False,
                tool_parser=lambda text, tools: {},
            ),
            draft_model=None,
            model_key=("model", None, None),
            cli_args=types.SimpleNamespace(
                mtp_draft=False,
                prefill_step_size=8,
                kv_bits=None,
                kv_group_size=64,
                quantized_kv_start=0,
            ),
        )
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = provider
        generator.prompt_cache = PromptCache()
        generator._is_distributed = False
        generator._tokenize = lambda tokenizer, request, args: (
            prompt,
            [prompt],
            ["assistant"],
            "normal",
        )
        generator._make_state_machine = lambda *args, **kwargs: (
            types.SimpleNamespace(
                make_state=lambda: None,
                match=lambda state, token: (state, None, "normal"),
            ),
            {},
        )
        generator._log_cache_stats = lambda: None

        def fake_stream_generate(**kwargs):
            captured.update(kwargs)
            return iter(())

        request = types.SimpleNamespace(request_type="text", prompt="hello")
        rqueue = Queue()
        with patch("mlx_lm.server.stream_generate", fake_stream_generate), patch(
            "mlx_lm.server._make_sampler", return_value=lambda logits: logits
        ), patch("mlx_lm.server._make_logits_processors", return_value=[]):
            generator._serve_single((rqueue, request, self._generation_args()))

        context = rqueue.get_nowait()
        self.assertEqual(context.prompt_cache_count, 2)
        self.assertIsNone(rqueue.get_nowait())
        self.assertEqual(captured["prompt"], [3, 4])
        self.assertEqual(captured["prompt_cache_tokens"], [1, 2])

    def test_speculative_single_requests_bypass_prompt_cache(self):
        prompt = [1, 2, 3, 4]

        class PromptCache:
            def fetch_nearest_cache(self, *args, **kwargs):
                raise AssertionError("speculative generation must not fetch cache")

            def insert_cache(self, *args, **kwargs):
                raise AssertionError("speculative generation must not store cache")

        for draft_model, mtp_draft in ((object(), False), (None, True)):
            with self.subTest(external_draft=draft_model is not None, mtp=mtp_draft):
                captured = {}
                provider = types.SimpleNamespace(
                    model=object(),
                    tokenizer=types.SimpleNamespace(
                        has_thinking=False,
                        has_tool_calling=False,
                        tool_parser=lambda text, tools: {},
                    ),
                    draft_model=draft_model,
                    model_key=("model", None, None),
                    cli_args=types.SimpleNamespace(
                        mtp_draft=mtp_draft,
                        prefill_step_size=8,
                        kv_bits=None,
                        kv_group_size=64,
                        quantized_kv_start=0,
                    ),
                )
                generator = ResponseGenerator.__new__(ResponseGenerator)
                generator.model_provider = provider
                generator.prompt_cache = PromptCache()
                generator._is_distributed = False
                generator._tokenize = lambda tokenizer, request, args: (
                    prompt,
                    [prompt],
                    ["assistant"],
                    "normal",
                )
                generator._make_state_machine = lambda *args, **kwargs: (
                    types.SimpleNamespace(
                        make_state=lambda: None,
                        match=lambda state, token: (state, None, "normal"),
                    ),
                    {},
                )
                generator._log_cache_stats = lambda: None

                def fake_stream_generate(**kwargs):
                    captured.update(kwargs)
                    return iter(())

                request = types.SimpleNamespace(request_type="text", prompt="hello")
                rqueue = Queue()
                with patch(
                    "mlx_lm.server.stream_generate", fake_stream_generate
                ), patch(
                    "mlx_lm.server.make_prompt_cache", return_value=["fresh"]
                ), patch(
                    "mlx_lm.server._make_sampler", return_value=lambda logits: logits
                ), patch("mlx_lm.server._make_logits_processors", return_value=[]):
                    generator._serve_single(
                        (rqueue, request, self._generation_args())
                    )

                context = rqueue.get_nowait()
                self.assertEqual(context.prompt_cache_count, 0)
                self.assertIsNone(rqueue.get_nowait())
                self.assertEqual(captured["prompt"], prompt)
                self.assertEqual(captured["prompt_cache_tokens"], [])
                self.assertEqual(
                    len(captured["prompt_cache"]), 2 if draft_model else 1
                )

    def test_cached_and_uncached_greedy_active_penalty_match(self):
        prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
        processors = make_logits_processors(presence_penalty=1.5)

        uncached = list(
            generate_step(
                prompt,
                _PenaltyHistoryModel(),
                max_tokens=4,
                prompt_cache=[],
                logits_processors=processors,
            )
        )
        cached = list(
            generate_step(
                prompt[2:],
                _PenaltyHistoryModel(),
                max_tokens=4,
                prompt_cache=[],
                prompt_cache_tokens=prompt[:2],
                logits_processors=make_logits_processors(presence_penalty=1.5),
            )
        )

        self.assertEqual(
            [token for token, _ in uncached], [token for token, _ in cached]
        )
        for (_, uncached_logits), (_, cached_logits) in zip(uncached, cached):
            self.assertTrue(mx.allclose(uncached_logits, cached_logits).item())

    def test_exact_cached_prompt_keeps_bootstrap_token(self):
        """An exact cache hit must still leave one prompt token to consume."""

        class TrimmableCache:
            def __init__(self):
                self.trimmed = 0

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.trimmed += n
                return n

        prompt = [1, 2, 3, 4]
        cached = TrimmableCache()
        prepared, rest, count = _prepare_cached_prompt(prompt, [cached], [])

        self.assertIs(prepared[0], cached)
        self.assertEqual(rest, [4])
        self.assertEqual(count, 3)
        self.assertEqual(cached.trimmed, 1)

    def test_exact_cached_nontrimmable_prompt_recomputes(self):
        class NonTrimmableCache:
            def is_trimmable(self):
                return False

        prompt = [1, 2, 3]
        prepared, rest, count = _prepare_cached_prompt(
            prompt, [NonTrimmableCache()], []
        )

        self.assertIsNone(prepared)
        self.assertEqual(rest, prompt)
        self.assertEqual(count, 0)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

        sm, _ = self.response_generator._make_state_machine(
            ("fake-empty-end", None, None),
            FakeTokenizer(),
            stop_words=[],
        )
        state = sm.make_state()
        state, _, s = sm.match(state, 100)
        self.assertEqual(s, "tool")
        for tok in [42, 43, 44]:
            state, _, s = sm.match(state, tok)
            self.assertEqual(s, "tool")
        state, _, s = sm.match(state, 2)
        self.assertIsNone(s)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        # Fetching a cache that is strictly a prefix doesn't remove it from the
        # lru cache
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 1)

        # Inserting a trimmable cache with shared prefix removes the prefixes
        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 1)

        # Fetching a cache with a shared prefix doesn't remove it either
        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache), 1)

        # Inserting a diverged cache actually creates another entry
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 2)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [1])
        c, t = cache.fetch_nearest_cache(model, [1, 3, 4])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [3, 4])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [2, 4, 5])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4, 5])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

    def test_insert_trimmable_cache_removes_immediate_prefix(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ab")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)

        cache.insert_cache(model, [1, 2, 3], [MockCache("abc")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 3)

    def test_insert_empty_tokens_does_not_self_destruct(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNotNone(c)
        self.assertEqual(t, [])

    def test_fetch_empty_tokens_after_root_eviction(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        cache.insert_cache(model, [1], [MockCache("a")])

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNone(c)
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])


if __name__ == "__main__":
    unittest.main()
