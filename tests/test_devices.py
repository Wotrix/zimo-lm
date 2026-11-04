# SPDX-License-Identifier: MIT
"""Device resolution must never claim a backend that is not there."""

import unittest

from device_utils import (backend_label, cuda_available, detect_backend,
                          musa_available, resolve_device)


class DeviceResolutionTest(unittest.TestCase):
    def test_detect_backend_is_known_value(self):
        self.assertIn(detect_backend(), {"cpu", "cuda", "musa"})

    def test_auto_matches_detection(self):
        self.assertEqual(resolve_device("auto"), detect_backend())
        self.assertEqual(resolve_device(None), detect_backend())

    def test_cpu_always_allowed(self):
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_missing_backend_degrades_to_cpu(self):
        if not musa_available():
            self.assertEqual(resolve_device("musa"), "cpu")
        if not cuda_available():
            # A machine with only MUSA must not be told "cuda".
            expected = "musa" if musa_available() else "cpu"
            self.assertEqual(resolve_device("cuda"), expected)

    def test_label_mentions_backend(self):
        label = backend_label()
        self.assertTrue(label.startswith(("CPU", "GPU")), label)


if __name__ == "__main__":
    unittest.main()
