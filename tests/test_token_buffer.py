# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.cache import TokenBuffer


class TestTokenBuffer(unittest.TestCase):
    def assert_live_tokens(self, buffer, expected):
        self.assertEqual(len(buffer), len(expected))
        self.assertTrue(
            mx.array_equal(
                buffer.tokens,
                mx.array(expected, dtype=mx.int32),
            )
        )

    def test_construct_and_append_within_capacity(self):
        buffer = TokenBuffer([1, 2, 3])
        self.assert_live_tokens(buffer, [1, 2, 3])

        # The first update allocates a chunk, so the second update stays within
        # that allocation.
        buffer.update_and_fetch([4, 5])
        capacity = buffer._buffer.size
        self.assert_live_tokens(buffer, [1, 2, 3, 4, 5])
        buffer.update_and_fetch([6])
        self.assertEqual(buffer._buffer.size, capacity)
        self.assert_live_tokens(buffer, [1, 2, 3, 4, 5, 6])

    def test_append_across_growth_boundary(self):
        buffer = TokenBuffer()
        buffer.update_and_fetch(mx.arange(256))
        self.assert_live_tokens(buffer, list(range(256)))
        self.assertEqual(buffer._buffer.size, 256)

        buffer.update_and_fetch([256])
        self.assertEqual(buffer._buffer.size, 512)
        self.assert_live_tokens(buffer, list(range(257)))

    def test_partial_trim_retains_allocation(self):
        buffer = TokenBuffer([1, 2, 3, 4, 5])
        allocation = buffer._buffer

        self.assertEqual(buffer.trim(2), 2)
        self.assertIs(buffer._buffer, allocation)
        self.assert_live_tokens(buffer, [1, 2, 3])

    def test_trim_empty_beyond_size_and_negative(self):
        buffer = TokenBuffer()
        allocation = buffer._buffer
        self.assertEqual(buffer.trim(0), 0)
        self.assertIs(buffer._buffer, allocation)
        self.assert_live_tokens(buffer, [])

        buffer.update_and_fetch([1, 2, 3])
        allocation = buffer._buffer
        self.assertEqual(buffer.trim(10), 3)
        self.assertIs(buffer._buffer, allocation)
        self.assert_live_tokens(buffer, [])

        with self.assertRaises(ValueError):
            buffer.trim(-1)
        self.assert_live_tokens(buffer, [])

    def test_append_after_trim_replaces_discarded_values(self):
        buffer = TokenBuffer()
        buffer.update_and_fetch([1, 2, 3, 4, 5])
        self.assertEqual(buffer.trim(2), 2)
        self.assert_live_tokens(buffer, [1, 2, 3])

        fetched = buffer.update_and_fetch([9, 10])
        self.assertTrue(mx.array_equal(fetched, mx.array([1, 2, 3, 9, 10])))
        self.assert_live_tokens(buffer, [1, 2, 3, 9, 10])

        self.assertEqual(buffer.trim(100), 5)
        buffer.update_and_fetch([7, 8])
        self.assert_live_tokens(buffer, [7, 8])


if __name__ == "__main__":
    unittest.main()
