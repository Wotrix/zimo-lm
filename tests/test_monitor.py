# SPDX-License-Identifier: MIT
"""Rendering helpers of the terminal monitor (pure functions, no GPU needed)."""

import unittest

from monitor import (bar, disp_width, downsample, fit, format_count,
                     format_seconds, pad, ppl_series, short_gpu_name, smooth,
                     sparkline)


class WidthTest(unittest.TestCase):
    """CJK is two columns wide; padding must use display width, not len."""

    def test_ascii_and_cjk_widths(self):
        self.assertEqual(disp_width("abc"), 3)
        self.assertEqual(disp_width("训练"), 4)
        self.assertEqual(disp_width("训练abc"), 7)
        self.assertEqual(disp_width("ー"), 2)   # Katakana prolonged sound mark

    def test_pad_uses_display_width(self):
        self.assertEqual(disp_width(pad("训练", 10)), 10)
        self.assertEqual(disp_width(pad("abc", 10)), 10)
        # already wider than the target: no negative padding
        self.assertEqual(pad("训练训练", 4), "训练训练")

    def test_fit_truncates_by_display_width(self):
        self.assertLessEqual(disp_width(fit("训练训练训练", 7)), 7)
        self.assertTrue(fit("训练训练训练", 7).endswith("…"))
        self.assertEqual(fit("abc", 10), "abc")


class RenderTest(unittest.TestCase):
    def test_bar_bounds(self):
        self.assertEqual(bar(0, 10), "░" * 10)
        self.assertEqual(bar(100, 10), "█" * 10)
        self.assertEqual(bar(-5, 4), "░░░░")
        self.assertEqual(bar(150, 4), "████")
        self.assertEqual(len(bar(42.7)), 34)

    def test_bar_custom_glyphs(self):
        self.assertEqual(bar(50, 4, "▮", "▯"), "▮▮▯▯")

    def test_smooth_reduces_spike(self):
        values = [10, 10, 10, 40, 10, 10, 10]
        out = smooth(values, 3)
        self.assertEqual(len(out), len(values))
        self.assertLess(max(out), 40)
        self.assertGreater(min(out), 9)

    def test_smooth_noop_when_short(self):
        self.assertEqual(smooth([1, 2], 7), [1, 2])

    def test_downsample_caps_width(self):
        self.assertEqual(len(downsample(list(range(100)), 20)), 20)
        self.assertEqual(downsample([1, 2, 3], 20), [1, 2, 3])

    def test_sparkline_flat_and_rising(self):
        flat = sparkline([5.0] * 10, 10)
        self.assertEqual(set(flat), {"▄"})
        rising = sparkline([1.0, 2.0, 3.0, 4.0], 4)
        self.assertEqual(rising[0], "▁")
        self.assertEqual(rising[-1], "█")

    def test_ppl_series_restarts_after_reset(self):
        rows = [
            {"step": 100, "ppl": 90.0},
            {"step": 200, "ppl": 80.0},
            {"step": 1, "ppl": 500.0},
            {"step": 2, "ppl": 400.0},
        ]
        self.assertEqual(ppl_series(rows), [500.0, 400.0])

    def test_ppl_series_skips_rows_without_ppl(self):
        rows = [{"step": 1, "loss": 5.0}, {"step": 2, "ppl": 12.5}]
        self.assertEqual(ppl_series(rows), [12.5])

    def test_format_seconds(self):
        self.assertEqual(format_seconds(0), "0m")
        self.assertEqual(format_seconds(90), "1m")
        self.assertEqual(format_seconds(3661), "1h01m")

    def test_format_count(self):
        self.assertEqual(format_count(999), "999")
        self.assertEqual(format_count(12500), "12.5k")
        self.assertEqual(format_count(4_314_240), "4.31M")
        self.assertEqual(format_count(2_500_000_000), "2.50B")

    def test_short_gpu_name(self):
        self.assertEqual(short_gpu_name("NVIDIA GeForce RTX 4090 D"), "RTX 4090 D")
        self.assertEqual(short_gpu_name("GeForce GTX 1650"), "GTX 1650")
        self.assertEqual(short_gpu_name("MTT S80"), "MTT S80")


if __name__ == "__main__":
    unittest.main()
