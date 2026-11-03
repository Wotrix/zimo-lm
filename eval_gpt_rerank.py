# SPDX-License-Identifier: MIT
"""Offline candidate reranking evaluation for a trained character LM.

Given a candidate CSV emitted by an IME harness (columns include ``input``,
``word``, ``spanwords``, ``final``) and a cases TSV (``input<TAB>gold``), this
script scores each candidate's word sequence with the LM, blends the LM score
with the engine's final score at several weights, and reports top-1 before and
after.  It is an offline helper and does not modify the engine.

Usage:
  python eval_gpt_rerank.py --model out/model_best.pt --vocab out/vocab.txt       --cases cases.tsv --csv candidates.tsv
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import io
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 4 * dim)
        self.proj = nn.Linear(4 * dim, dim)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        h = self.proj(F.gelu(self.fc(self.ln2(x))))
        return x + h


class GPTLM(nn.Module):
    def __init__(self, vocab, block, dim, layers, heads):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = nn.Embedding(block, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok(x) + self.pos(pos)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), 1)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_f(h))


def load_model(model_path: str, vocab_path: str):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    words = io.open(vocab_path, encoding="utf-8").read().split("\n")
    model = GPTLM(len(words) + 1, ck["block"], ck["dim"], ck["layers"],
                  ck.get("heads", 4))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, {w: i + 1 for i, w in enumerate(words)}, words


@torch.no_grad()
def seq_mean_logp(model, vocab, tokens, block):
    ids = [vocab.get(t, 0) for t in tokens][:block]
    if len(ids) < 2:
        return -20.0
    x = torch.tensor([ids[:-1]])
    logits = model(x)
    lp = torch.log_softmax(logits, dim=-1)
    return float(sum(lp[0, i, t] for i, t in enumerate(ids[1:])) / len(ids))


def load_cases(path: str):
    exp, cat = {}, {}
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) > 2 and "garbage_probe" in p[2]:
            continue
        exp[p[0]] = p[1]
        cat[p[0]] = (p[2].split("|")[-1] if len(p) > 2 else "all")
    return exp, cat


def load_csv(path: str):
    hdr = None
    rows = collections.defaultdict(list)
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("#"):
            hdr = line.lstrip("# ").rstrip("\n").split("\t")
            continue
        if not line.strip():
            continue
        f = dict(zip(hdr, line.rstrip("\n").split("\t")))
        rows[f["input"]].append(f)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--weights", default="0,0.5,1,2,4")
    args = ap.parse_args()

    model, vocab, words = load_model(args.model, args.vocab)
    print(f"[eval] vocab={len(words)+1} 参数量="
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    exp, cat = load_cases(args.cases)
    rows = load_csv(args.csv)
    usable = [k for k in rows if k in exp]
    print(f"[eval] 可比对案例 {len(usable)}", flush=True)

    # 预算每个候选的语言模型分（缓存）
    cache: dict[tuple, float] = {}

    def lm_score(f):
        toks = tuple(t for t in f["spanwords"].split("|") if t)
        if toks not in cache:
            cache[toks] = seq_mean_logp(model, vocab, list(toks), args.block)
        return cache[toks]

    def top1(pick):
        ok = 0
        per = collections.defaultdict(lambda: [0, 0])
        for inp in usable:
            g = rows[inp]
            best = pick(g)
            good = best["word"] == exp[inp]
            ok += 1 if good else 0
            per[cat[inp]][0] += 1
            per[cat[inp]][1] += 1 if good else 0
        return ok, per

    # 基线：CSV 顺序的第一名（生产排序）
    base_ok, base_per = top1(lambda g: g[0])
    print(f"\n[eval] 基线（生产 rank1）: {base_ok}/{len(usable)} = "
          f"{base_ok/len(usable):.1%}")
    for k, (n, o) in sorted(base_per.items()):
        print(f"        {k:10s} {o}/{n} = {o/n:.1%}")

    # 案例内标准化：同一输入下比较候选，消掉"词少天然概率高"的跨案例偏置。
    case_norm = {}
    for inp in usable:
        vals = [lm_score(f) for f in rows[inp]]
        if len(vals) > 1:
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
            sd = sd if sd > 1e-6 else 1.0
        else:
            mu, sd = 0.0, 1.0
        case_norm[inp] = (mu, sd)

    def znorm(inp, f):
        mu, sd = case_norm[inp]
        return (lm_score(f) - mu) / sd

    print("\n[eval] 语言分按案例内 z-score 标准化后混合")
    for w in [float(x) for x in args.weights.split(",")]:
        if w == 0:
            continue

        def pick(g, w=w):
            inp = g[0]["input"]
            band = min(int(f["band"]) for f in g)
            sub = [f for f in g if int(f["band"]) == band]
            return max(sub, key=lambda f: float(f["final"]) + w * znorm(inp, f))

        ok, per = top1(pick)
        print(f"[eval] w={w:<5} 混合重排: {ok}/{len(usable)} = {ok/len(usable):.1%}"
              f"  ({ok-base_ok:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
