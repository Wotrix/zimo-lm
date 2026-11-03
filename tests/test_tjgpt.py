# SPDX-License-Identifier: MIT
"""Round-trip test for the TJGPT container (export, header, size, vocab)."""

import struct
import tempfile
import unittest
from pathlib import Path

import torch

from tjgpt_export import (MAGIC, expected_size, read_tjgpt_header,
                          write_tjgpt)
from train_gpt import GPTLM


class TjgptFormatTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab = ["你", "好", "世", "界", "！", "我", "爱"]
        self.dim, self.layers, self.heads, self.block = 32, 2, 4, 16
        self.model = GPTLM(len(self.vocab) + 1, self.block, self.dim,
                           self.layers, self.heads)
        self.model.eval()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.tjgpt"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_header_and_size(self):
        write_tjgpt(self.model, self.vocab, self.path, block=self.block,
                    layers=self.layers, dim=self.dim, heads=self.heads)
        meta = read_tjgpt_header(self.path)
        self.assertEqual(meta["vocab"], len(self.vocab) + 1)
        self.assertEqual(meta["block"], self.block)
        self.assertEqual(meta["dim"], self.dim)
        self.assertEqual(meta["layers"], self.layers)
        self.assertEqual(meta["heads"], self.heads)
        self.assertFalse(meta["tied"])
        self.assertEqual(meta["words"], self.vocab)
        self.assertEqual(meta["data_offset"], 40 + len(
            ("\n".join(self.vocab) + "\n").encode("utf-8")))
        self.assertEqual(self.path.stat().st_size, expected_size(meta))

    def test_magic_and_version(self):
        write_tjgpt(self.model, self.vocab, self.path, block=self.block,
                    layers=self.layers, dim=self.dim, heads=self.heads)
        raw = self.path.read_bytes()
        self.assertEqual(raw[:8], MAGIC)
        version = struct.unpack("<I", raw[8:12])[0]
        self.assertEqual(version, 1)

    def test_weights_are_float32(self):
        write_tjgpt(self.model, self.vocab, self.path, block=self.block,
                    layers=self.layers, dim=self.dim, heads=self.heads)
        meta = read_tjgpt_header(self.path)
        raw = self.path.read_bytes()
        payload = len(raw) - meta["data_offset"]
        self.assertEqual(payload % 4, 0)
        # first token embedding row should match the checkpoint
        first = struct.unpack("<f", raw[meta["data_offset"]:
                                       meta["data_offset"] + 4])[0]
        expected = float(self.model.state_dict()["tok.weight"][0][0])
        self.assertAlmostEqual(first, expected, places=6)

    def test_tied_head_omits_head_block(self):
        write_tjgpt(self.model, self.vocab, self.path, block=self.block,
                    layers=self.layers, dim=self.dim, heads=self.heads,
                    tied=True)
        meta = read_tjgpt_header(self.path)
        self.assertTrue(meta["tied"])
        self.assertEqual(self.path.stat().st_size, expected_size(meta))

    def test_rejects_bad_magic(self):
        self.path.write_bytes(b"NOTATJG!" + b"\0" * 64)
        with self.assertRaises(ValueError):
            read_tjgpt_header(self.path)


if __name__ == "__main__":
    unittest.main()
