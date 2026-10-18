"""Known-value tests for the Wilson interval.

These exist because a reporting helper returned the centre as the lower bound
and the mistake went out in a table (docs/debugging.md). If the numbers below
change, the interval is wrong, not the test.
"""
import unittest

from src.naming.stats import wilson, wilson_pair


class TestWilson(unittest.TestCase):
    def test_known_value_10_of_19(self):
        """10/19 was the first interval this project ever quoted: [31.7, 72.7]."""
        lo, hi = wilson(10, 19)
        self.assertAlmostEqual(lo, 31.7, delta=0.2)
        self.assertAlmostEqual(hi, 72.7, delta=0.2)

    def test_known_value_58_of_204(self):
        lo, hi = wilson(58, 204)
        self.assertAlmostEqual(lo, 22.7, delta=0.3)
        self.assertAlmostEqual(hi, 35.0, delta=0.3)

    def test_known_value_90_of_204(self):
        lo, hi = wilson(90, 204)
        self.assertAlmostEqual(lo, 37.3, delta=0.4)
        self.assertAlmostEqual(hi, 51.0, delta=0.4)

    def test_zero_of_n(self):
        lo, hi = wilson(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 16.1, delta=0.3)   # one-sided upper edge

    def test_n_of_n(self):
        lo, hi = wilson(20, 20)
        self.assertAlmostEqual(lo, 83.9, delta=0.3)
        self.assertEqual(hi, 100.0)

    def test_no_data(self):
        self.assertEqual(wilson(0, 0), (0.0, 0.0))

    def test_interval_always_contains_the_point_estimate(self):
        for k, n in ((1, 3), (10, 19), (58, 204), (90, 204), (0, 20), (20, 20),
                     (1, 1000), (999, 1000), (5, 5)):
            point, lo, hi = wilson_pair(k, n)
            self.assertLessEqual(lo, point, f"{k}/{n}: lower bound above the point")
            self.assertGreaterEqual(hi, point, f"{k}/{n}: upper bound below the point")
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 100.0)

    def test_narrower_with_more_data(self):
        """Same proportion, more samples: the interval must shrink."""
        _, lo_small, hi_small = wilson_pair(29, 100)
        _, lo_big, hi_big = wilson_pair(290, 1000)
        self.assertLess(hi_big - lo_big, hi_small - lo_small)


if __name__ == "__main__":
    unittest.main()
