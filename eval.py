# SPDX-License-Identifier: MIT
"""Hold-out evaluation for a trained character LM: PPL / NLL / bits-per-char.

    python eval.py --model runs/base_8x256 --corpus data/valid.txt
    python eval.py --model runs/a --model runs/b --corpus data/valid.txt
    python eval.py --model runs/a --corpus data/valid.txt --limit 2000000

``--model`` accepts a run directory (uses ``model_best.pt`` when present) or a
checkpoint file in any format the loader understands.  The vocabulary comes
from the checkpoint itself, or a ``vocab.txt`` next to it.

Windows are taken from a *comma-free* split of the corpus: the file is read as
one long token stream and sliced into ``block``-sized windows, so sentences are
not padded and every token after the first of a window is scored.  Text files
that are already character-spaced (as ``prepare_corpus.py`` produces) work
best; plain text is split into characters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from checkpoint_utils import (extract_vocab_words, inspect_checkpoint_file,
                              load_state_any)
from device_utils import resolve_device


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def build_model(state: dict, config: dict, vocab_size: int, arch: str):
    from train_gpt import GPTLM

    if arch == "gpt2":
        return GPTLM(vocab_size, int(config["block"]), int(config["dim"]),
                     int(config["layers"]), int(config.get("heads", 4)))
    raise SystemExit(f"[x] 暂不支持架构 {arch}（当前支持 gpt2）")


def load_model(model_arg: str, device: torch.device):
    path = Path(model_arg).expanduser()
    if path.is_dir():
        for name in ("model_best.pt", "model.pt"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            raise SystemExit(f"[x] {path} 里没有 model_best.pt / model.pt")

    info = load_state_any(path)
    if not info["loadable"]:
        raise SystemExit(f"[x] 无法读取 {path}：{info['error']}")

    config = dict(info["config"])
    from checkpoint_utils import infer_config_from_state
    for key, value in infer_config_from_state(info["state"]).items():
        config.setdefault(key, value)

    words = info["vocab"]
    if not words:
        sidecar = path.parent / "vocab.txt"
        if sidecar.is_file():
            words = [w for w in sidecar.read_text(encoding="utf-8").splitlines() if w]
    if not words:
        raise SystemExit(f"[x] {path} 没有词表，也没有同名 vocab.txt")
    vocab = {word: index + 1 for index, word in enumerate(words)}

    arch = "gpt2"
    model = build_model(info["state"], config, len(vocab) + 1, arch)
    state = info["state"]
    if any(key.startswith("module.") for key in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    return {
        "path": path,
        "model": model,
        "vocab": vocab,
        "block": int(config.get("block", 64)),
        "layers": int(config.get("layers", 0)),
        "dim": int(config.get("dim", 0)),
        "params": sum(p.numel() for p in model.parameters()),
        "words": words,
        "state": info["state"],
    }


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def read_tokens(path: Path, limit: int) -> list[int]:
    """Character tokens of a corpus file (space-separated tokens accepted)."""
    tokens: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens.extend(line.split())
            if limit and len(tokens) >= limit:
                break
    return tokens[:limit] if limit else tokens


def encode(tokens: list[str], vocab: dict[str, int]) -> torch.Tensor:
    ids = [vocab.get(tok, 0) for tok in tokens]
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data: torch.Tensor, block: int, device: torch.device,
             batch: int = 32, amp: bool = False) -> dict:
    """Mean NLL over every predicted token in block-sized windows."""
    usable = (data.numel() // (block + 1)) * (block + 1)
    if usable < block + 1:
        raise SystemExit("[x] 语料太短，连一个窗口都凑不出来")
    windows = data[:usable].view(-1, block + 1)

    total_nll = 0.0
    total_tokens = 0
    started = time.time()
    dtype = torch.bfloat16
    for start in range(0, windows.shape[0], batch):
        chunk = windows[start:start + batch].to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), reduction="sum")
        total_nll += float(loss)
        total_tokens += y.numel()

    mean_nll = total_nll / total_tokens
    return {
        "tokens": total_tokens,
        "windows": int(windows.shape[0]),
        "nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 20.0)),
        "bits_per_char": mean_nll / math.log(2),
        "seconds": time.time() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="用留出语料评测字符级 LM：PPL / 平均 NLL / bits-per-char",
    )
    parser.add_argument("--model", action="append", required=True,
                        help="运行目录或 checkpoint，可重复传多个做对比")
    parser.add_argument("--corpus", required=True, help="留出语料（不是训练语料）")
    parser.add_argument("--limit", type=int, default=2_000_000,
                        help="最多读多少 token（默认 200 万，0 = 全部）")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | musa")
    parser.add_argument("--amp", action="store_true", help="用 bf16 评测（更快）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    corpus = Path(args.corpus).expanduser()
    if not corpus.is_file():
        raise SystemExit(f"[x] 语料不存在：{corpus}")

    backend = resolve_device(args.device)
    device = torch.device(backend)
    raw = read_tokens(corpus, args.limit)
    if not raw:
        raise SystemExit("[x] 语料为空")

    results = []
    for model_arg in args.model:
        loaded = load_model(model_arg, device)
        data = encode(raw, loaded["vocab"])
        metrics = evaluate(loaded["model"], data, loaded["block"], device,
                           batch=args.batch, amp=args.amp)
        metrics.update({
            "model": str(loaded["path"]),
            "arch": f"gpt2 {loaded['layers']}x{loaded['dim']}",
            "params": loaded["params"],
            "block": loaded["block"],
        })
        results.append(metrics)
        if not args.json:
            print(f"[ok] {loaded['path'].parent.name}: "
                  f"ppl={metrics['ppl']:.2f} nll={metrics['nll']:.4f} "
                  f"({metrics['tokens']:,} tokens)")

    if args.json:
        print(json.dumps({"corpus": str(corpus), "results": results},
                         ensure_ascii=False, indent=2))
        return 0

    # comparison table
    print()
    print(f"{'model':<34}{'arch':<14}{'params':>12}{'ppl':>9}{'nll':>9}{'bits/char':>11}")
    print("-" * 89)
    best = min(results, key=lambda r: r["ppl"])
    for row in results:
        mark = " ←" if row is best and len(results) > 1 else ""
        print(f"{Path(row['model']).parent.name:<34}{row['arch']:<14}"
              f"{row['params']:>12,}{row['ppl']:>9.2f}{row['nll']:>9.4f}"
              f"{row['bits_per_char']:>11.3f}{mark}")
    print()
    print(f"语料：{corpus}（{results[0]['tokens']:,} tokens，"
          f"设备 {backend}）")
    if len(results) > 1:
        delta = results[0]["ppl"] - best["ppl"]
        print(f"最优：{Path(best['model']).parent.name}（PPL {best['ppl']:.2f}，"
              f"比第一个低 {delta:.2f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
