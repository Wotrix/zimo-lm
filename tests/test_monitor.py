# SPDX-License-Identifier: MIT
"""Rendering helpers of the terminal monitor (pure functions, no GPU needed)."""

import unittest

from monitor import (downsample, format_seconds, ppl_series, progress_bar,
                     smooth, sparkline)


class MonitorRenderTest(unittest.TestCase):
    def test_progress_bar_bounds(self):
        self.assertEqual(progress_bar(0, 10), "[..........]")
        self.assertEqual(progress_bar(100, 10), "[##########]")
        self.assertEqual(progress_bar(-5, 4), "[....]")
        self.assertEqual(progress_bar(150, 4), "[####]")
        self.assertEqual(len(progress_bar(42.7)), 30)

    def test_smooth_keeps_length_and_reduces_spikes(self):
        values = [10, 10, 10, 40, 10, 10, 10]
        out = smooth(values, 3)
        self.assertEqual(len(out), len(values))
        self.assertLess(max(out), 40)
        self.assertGreater(min(out), 9)

    def test_smooth_noop_when_too_short(self):
        self.assertEqual(smooth([1, 2], 7), [1, 2])

    def test_downsample_caps_width(self):
        self.assertEqual(len(downsample(list(range(100)), 20)), 20)
        self.assertEqual(downsample([1, 2, 3], 20), [1, 2, 3])

    def test_sparkline_flat_and_rising(self):
        flat = sparkline([5.0] * 10, 10)
        self.assertEqual(set(flat), {"▄"})
        rising = sparkline([1.0, 2.0, 3.0, 4.0], 4)
        self.assertEqual(len(rising), 4)
        self.assertLess(rising.index(rising[0]), rising.index(rising[-1]))
        self.assertEqual(rising[0], "▁")
        self.assertEqual(rising[-1], "█")

    def test_ppl_series_restarts_after_step_reset(self):
        rows = [
            {"step": 100, "ppl": 90.0},
            {"step": 200, "ppl": 80.0},
            {"step": 1, "ppl": 500.0},
            {"step": 2, "ppl": 400.0},
        ]
        self.assertEqual(ppl_series(rows), [500.0, 400.0])

    def test_ppl_series_ignores_rows_without_ppl(self):
        rows = [{"step": 1, "loss": 5.0}, {"step": 2, "ppl": 12.5}]
        self.assertEqual(ppl_series(rows), [12.5])

    def test_format_seconds(self):
        self.assertEqual(format_seconds(0), "0m")
        self.assertEqual(format_seconds(59), "0m")
        self.assertEqual(format_seconds(90), "1m")
        self.assertEqual(format_seconds(3661), "1h01m")


if __name__ == "__main__":
    unittest.main()
