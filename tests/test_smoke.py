# SPDX-License-Identifier: MIT
"""End-to-end smoke tests for the standalone package.

Run from the package root:
  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(*args: str, cwd: str = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, *args], cwd=cwd, text=True, capture_output=True, check=True)


def read_text(path: str) -> str:
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


class PackageSmokeTests(unittest.TestCase):
    def test_prepare_corpus_jsonl_and_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "data.jsonl")
            out = os.path.join(tmp, "corpus.txt")
            with io.open(src, "w", encoding="utf-8") as fh:
                fh.write('["你好", "今天 天气 不错"]\n')
            run("prepare_corpus.py", "--input", src, "--output", out)
            lines = read_text(out).splitlines()
            self.assertEqual(lines, ["你 好", "今 天 天 气 不 错"])

            src2 = os.path.join(tmp, "data2.jsonl")
            with io.open(src2, "w", encoding="utf-8") as fh:
                fh.write('["再见"]\n')
            run("prepare_corpus.py", "--input", src2, "--output", out,
                "--append")
            lines = read_text(out).splitlines()
            self.assertEqual(lines[-1], "再 见")
            self.assertEqual(len(lines), 3)

    def test_train_gpt_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            run(
                "train_gpt.py",
                "--corpus", os.path.join(ROOT, "examples", "sample.txt"),
                "--tokens", "2000", "--epochs", "1", "--max-steps", "5",
                "--block", "16", "--batch", "4", "--dim", "32", "--layers", "1",
                "--heads", "2", "--lr", "1e-3", "--warmup", "10",
                "--min-count", "1", "--max-vocab", "2000", "--val-ratio", "0",
                "--device", "cpu", "--save-every", "100",
                "--out", out,
                "--metrics", os.path.join(out, "metrics.jsonl"),
                "--status", os.path.join(out, "status.json"),
            )
            self.assertTrue(os.path.exists(os.path.join(out, "model.pt")))
            status = json.loads(read_text(os.path.join(out, "status.json")))
            self.assertEqual(status["state"], "done")
            self.assertGreater(status["step"], 0)
            self.assertTrue(os.path.exists(os.path.join(out, "vocab.txt")))
            self.assertTrue(os.path.exists(os.path.join(out, "metrics.jsonl")))

    def test_static_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            status = {
                "state": "done", "vocab": 50, "tokens": 1000, "params": 1000,
                "total_steps": 200, "steps_per_epoch": 100, "epoch": 1,
                "step": 200, "elapsed_s": 10, "eta_s": 0,
                "tokens_per_s": 500, "last_loss": 3.0, "last_ppl": 20.0,
                "arch": "gpt2 1x32",
            }
            with io.open(os.path.join(run_dir, "status.json"), "w",
                         encoding="utf-8") as fh:
                json.dump(status, fh)
            with io.open(os.path.join(run_dir, "metrics.jsonl"), "w",
                         encoding="utf-8") as fh:
                for step, loss, ppl, t in [(50, 4.0, 55.0, 2),
                                           (100, 3.5, 33.0, 5),
                                           (200, 3.0, 20.0, 10)]:
                    fh.write(json.dumps({"step": step, "epoch": 1,
                                         "loss": loss, "ppl": ppl,
                                         "t": t}) + "\n")
            with io.open(os.path.join(run_dir, "train.log"), "w",
                         encoding="utf-8") as fh:
                fh.write("[lm] device=cpu\n[lm] done\n")
            output = os.path.join(tmp, "report.html")
            run("train_report.py", "--dir", run_dir, "--output", output,
                "--title", "Test LM")
            html = read_text(output)
            self.assertIn("Test LM", html)
            self.assertIn("lossChart", html)
            self.assertIn('"last_loss": 3.0', html)
            self.assertNotIn("__TITLE__", html)
            self.assertNotIn("__DATA__", html)

    def test_dashboard_template_is_neutral(self):
        sys.path.insert(0, ROOT)
        import train_dashboard  # noqa: E402
        page = train_dashboard.HTML
        self.assertIn("__TITLE__", page)
        self.assertIn("__SUBTITLE__", page)
        self.assertNotIn("LCCC", page)
        self.assertNotIn("\u6850\u5bb6", page)
        self.assertNotIn("\u5fae\u4fe1", page)

    def test_no_competitor_references(self):
        banned = ["\u5fae\u4fe1", "We" + "Chat", "nlr" + "gpt",
                  "\u9006\u5411", "\u53cd\u7f16\u8bd1",
                  "\u6850\u5bb6", "\u7ade\u54c1"]
        for name in os.listdir(ROOT):
            if not (name.endswith(".py") or name.endswith(".md")):
                continue
            path = os.path.join(ROOT, name)
            text = read_text(path)
            for term in banned:
                self.assertNotIn(term, text, f"{term} in {name}")


if __name__ == "__main__":
    unittest.main()
