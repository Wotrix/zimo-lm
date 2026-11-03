#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert a dialogue/text corpus into character-token training text.

Supported input formats:

  jsonl  one JSON value per line; a list of utterances (LCCC-style) or one string
         ["你好", "今天怎么样"]  ->  你 好 / 今 天 怎 么 样
  text   one utterance per line; the line is re-split into characters

The output is one utterance per line with characters separated by spaces, which
is what train_gpt.py's load_tokens()/build_vocab() expect.  Punctuation is kept
as its own token; the trainer decides whether to keep it in the vocabulary.

Examples:
  python prepare_corpus.py --input data.jsonl.gz --output corpus.txt
  python prepare_corpus.py --input data.jsonl --output corpus.txt --max-chars 30000000
  python prepare_corpus.py --input plain.txt --format text --output corpus.txt
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys

WS_RE = re.compile(r"\s+")


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return io.open(path, encoding="utf-8", errors="replace")


def utterances_from_line(line: str, fmt: str) -> list[str]:
    if fmt == "text":
        return [line]
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input .jsonl/.jsonl.gz/.txt")
    parser.add_argument("--output", required=True, help="output char-token text")
    parser.add_argument("--format", choices=("jsonl", "text"), default="jsonl")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="stop after roughly this many non-space characters "
                             "(0 = all)")
    parser.add_argument("--append", action="store_true",
                        help="append to an existing output file")
    parser.add_argument("--report", default="",
                        help="optional JSON report path")
    args = parser.parse_args()

    if args.max_chars < 0:
        parser.error("--max-chars must be >= 0")

    lines_out = 0
    chars = 0
    skipped = 0
    mode = "a" if args.append else "w"
    with open_text(args.input) as src, \
            io.open(args.output, mode, encoding="utf-8", newline="\n") as out:
        for raw in src:
            raw = raw.rstrip("\r\n")
            if not raw.strip():
                continue
            utterances = utterances_from_line(raw.strip(), args.format)
            if not utterances:
                skipped += 1
                continue
            for utterance in utterances:
                text = WS_RE.sub("", utterance)
                if not text:
                    continue
                out.write(" ".join(text) + "\n")
                lines_out += 1
                chars += len(text)
                if args.max_chars and chars >= args.max_chars:
                    break
            if args.max_chars and chars >= args.max_chars:
                break

    report = {
        "input": args.input,
        "output": args.output,
        "format": args.format,
        "lines_out": lines_out,
        "chars": chars,
        "skipped_lines": skipped,
        "max_chars": args.max_chars,
        "append": args.append,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        with io.open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return 0 if lines_out else 1


if __name__ == "__main__":
    sys.exit(main())
