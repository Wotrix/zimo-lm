# SPDX-License-Identifier: MIT
"""Train a small character-level GPT-2-style language model from scratch.

Architecture:
    token_embed(V,d) + pos_embed(L,d)
    N x [ LN -> causal self-attention -> residual -> LN -> GELU MLP -> residual ]
    LN_f
    lm_head (optionally tied to the token embedding via --tied)

This is a from-scratch reference implementation, not a wrapper around OpenAI
GPT-2 weights.  The model is trained with next-token cross-entropy and can be
used as a language-model scorer for reranking (see eval_gpt_rerank.py) or as a
small reproducible baseline for Chinese character-level LM research.

Outputs (per run directory):
    model.pt / model_best.pt   weights (fp32)
    vocab.txt                  token vocabulary
    metrics.jsonl              loss/PPL per 10 steps
    status.json                live state for train_dashboard.py
    train.log                  stdout/stderr log

Usage:
  python train_gpt.py --corpus corpus.txt --tokens 15000000 --epochs 3       --dim 128 --layers 3 --max-vocab 13000 --block 64 --out out       --metrics out/metrics.jsonl --status out/status.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from device_utils import resolve_device
import torch.nn.functional as F


def load_tokens(path: str, limit: int) -> list[str]:
    tokens: list[str] = []
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens.extend(line.split())
            if len(tokens) >= limit:
                break
    return tokens[:limit]


def build_vocab(tokens, min_count: int, max_vocab: int):
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    words = [w for w, c in freq.items() if c >= min_count]
    words.sort(key=lambda w: (-freq[w], w))
    words = words[: max(1, max_vocab - 1)]
    vocab = {w: i + 1 for i, w in enumerate(words)}  # 0 = <unk>
    return vocab, words


class Block(nn.Module):
    """GPT-2 block：pre-LN + causal self-attention + GELU MLP。"""

    def __init__(self, dim: int, heads: int):
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
        h = self.ln2(x)
        h = self.proj(F.gelu(self.fc(h)))
        return x + h


class GPTLM(nn.Module):
    def __init__(self, vocab: int, block: int, dim: int, layers: int, heads: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = nn.Embedding(block, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        # The reference 3-layer/128-d configuration uses an untied lm_head;
        # keep that default.  Use --tied for weight sharing.
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok(x) + self.pos(pos)
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.ln_f(h)
        return self.head(h)


def write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    # Windows: the dashboard reads status.json every second; os.replace needs
    # delete access and fails with PermissionError while a reader holds it.
    # Retry briefly instead of killing a multi-hour training run.
    last_error = None
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.02)
    try:
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError:
        if last_error is not None:
            raise last_error
        raise
    try:
        os.remove(tmp)
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="train_gpt.py",
        description="Zimo Studio 字符级语言模型训练器（从零训练，无预训练权重）",
        epilog='示例：\n  最小冒烟：\n    python train_gpt.py --corpus examples/sample.txt --tokens 200000 --epochs 1 --dim 64 --layers 2 --heads 2 --out runs/smoke\n  常规训练：\n    python train_gpt.py --corpus corpus.txt --tokens 30000000 --epochs 2 --dim 192 --layers 6 --heads 4 --out runs/base\n  GPU 续训：\n    python train_gpt.py --corpus corpus.txt --device cuda --out runs/base --resume\n完整说明：docs/cli.zh.md',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", default="corpus.txt",
                    help="训练语料路径（每行一句；字符之间用空格分隔更佳）")
    ap.add_argument("--tokens", type=int, default=15_000_000,
                    help="最多读取多少 token（默认 1500 万；越大越慢）")
    ap.add_argument("--epochs", type=int, default=3,
                    help="训练轮数（默认 3；小语料 5-10，大语料 1-3）")
    ap.add_argument("--block", type=int, default=64,
                    help="上下文长度，一次看多少个 token（默认 64）")
    ap.add_argument("--batch", type=int, default=32,
                    help="批大小（CPU 默认 32；GPU 可 64-256，显存不足就调小）")
    ap.add_argument("--dim", type=int, default=128,
                    help="隐藏维度，需能被 --heads 整除（默认 128）")
    ap.add_argument("--layers", type=int, default=3,
                    help="Transformer 层数（默认 3；4M 模型用 6）")
    ap.add_argument("--heads", type=int, default=4,
                    help="注意力头数（默认 4；dim 必须能被它整除）")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="学习率（默认 3e-4；大模型可降到 1e-4）")
    ap.add_argument("--min-count", type=int, default=20,
                    help="字符最低出现次数，低于它的丢弃（小语料请调成 1-3，否则词表会塌）")
    ap.add_argument("--max-vocab", type=int, default=13_000,
                    help="词表上限（默认 13000；字符级通常 4000-13000）")
    ap.add_argument("--out", default="out",
                    help="输出目录（存 model.pt / model_best.pt / vocab.txt）")
    ap.add_argument("--metrics", default="out/metrics.jsonl",
                    help="逐步指标文件（每行一条 JSON）")
    ap.add_argument("--status", default="out/status.json",
                    help="状态文件（state/step/loss/val_ppl 等）")
    ap.add_argument("--device", default="auto",
                    help="运行设备：auto | cpu | cuda | musa（默认自动检测；摩尔线程用 musa）")
    ap.add_argument("--tied", action="store_true",
                    help="共享输入词向量与输出头（默认不共享）")
    ap.add_argument("--resume", action="store_true",
                    help="从 --out 目录里的 model.pt 继续训练（断点续训）")
    ap.add_argument("--warmup", type=int, default=300,
                    help="线性 warmup 步数（默认 300；小语料可设 0）")
    ap.add_argument("--val-ratio", type=float, default=0.02,
                    help="验证集比例，从语料尾部切出（默认 0.02；0 = 不验证）")
    ap.add_argument("--save-every", type=int, default=500,
                    help="每 N 步保存一次检查点（默认 500；断线重跑依赖它）")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="最多训练多少步（0 = 不限制；调试时可设 50）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
    t0 = time.time()

    tokens = load_tokens(args.corpus, args.tokens)
    vocab, words = build_vocab(tokens, args.min_count, args.max_vocab)
    ids = np.array([vocab.get(t, 0) for t in tokens], dtype=np.int64)
    seq = args.block
    usable = (len(ids) // (seq + 1)) * (seq + 1)
    data = torch.from_numpy(ids[:usable].reshape(-1, seq + 1))
    print(f"[lm] tokens={len(tokens)} vocab={len(vocab)+1} windows={data.shape[0]}",
          flush=True)

    # MUSA (Moore Threads) exposes torch.musa, not torch.cuda; resolve centrally.
    backend = resolve_device(args.device)
    device = torch.device(backend)
    print(f"[lm] device={device} backend={backend}", flush=True)
    if backend == "musa":
        try:
            print(f"[lm] musa={torch.musa.get_device_name(0)}", flush=True)
        except Exception:  # noqa: BLE001
            pass

    torch.manual_seed(7)
    model = GPTLM(len(vocab) + 1, seq, args.dim, args.layers, args.heads).to(device)
    if args.tied:
        model.head.weight = model.tok.weight
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    params = sum(p.numel() for p in model.parameters())
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(os.path.join(args.out, "model.pt")):
        ck = torch.load(os.path.join(args.out, "model.pt"), map_location="cpu",
                        weights_only=False)
        try:
            model.load_state_dict(ck["state"])
            start_epoch = int(ck.get("epoch", 0))
            global_step = int(ck.get("step", 0))
            print(f"[lm] resumed epoch={start_epoch} step={global_step}", flush=True)
        except RuntimeError as exc:
            print("[lm] WARNING: 检查点不兼容，从头训练:", flush=True)
            print(f"[lm]   {exc}", flush=True)

    # Speed/ETA must use steps from THIS process, not the resumed global step:
    # elapsed starts at zero on every launch.
    session_start_step = global_step

    # 验证集：取尾部若干窗口，训练只用前面的
    n_val = max(1, int(data.shape[0] * args.val_ratio)) if args.val_ratio > 0 else 0
    train_data = data[: data.shape[0] - n_val] if n_val else data
    val_data = data[data.shape[0] - n_val :] if n_val else None
    steps_per_epoch = max(1, train_data.shape[0] // args.batch)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    write_json(args.status, {
        "state": "running", "vocab": len(vocab) + 1, "tokens": len(tokens),
        "params": params, "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch, "epoch": start_epoch,
        "step": global_step, "elapsed_s": 0, "eta_s": None,
        "tokens_per_s": 0, "last_loss": None, "last_ppl": None,
        "arch": f"gpt2 {args.layers}x{args.dim}",
    })

    best_val = float("inf")
    metrics = io.open(args.metrics, "a", encoding="utf-8")
    running, run_count = 0.0, 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        order = torch.randperm(train_data.shape[0])
        for b in range(steps_per_epoch):
            idx = order[b * args.batch : (b + 1) * args.batch]
            if idx.numel() == 0:
                continue
            chunk = train_data[idx].to(device, non_blocking=True)
            x, y = chunk[:, :-1], chunk[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            # warmup + 余弦退火
            if global_step < args.warmup:
                lr = args.lr * (global_step + 1) / max(1, args.warmup)
            else:
                prog = (global_step - args.warmup) / max(1, total_steps - args.warmup)
                lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            lv = float(loss.detach())
            running += lv
            run_count += 1
            global_step += 1
            if global_step % 10 == 0:
                elapsed = time.time() - t0
                avg = running / max(run_count, 1)
                session_steps = max(1, global_step - session_start_step)
                tok_s = session_steps * args.batch * seq / max(elapsed, 1e-6)
                eta = (total_steps - global_step) * (elapsed / session_steps)
                ppl = math.exp(min(avg, 20.0))
                metrics.write(json.dumps({
                    "step": global_step, "epoch": epoch,
                    "loss": round(avg, 4), "ppl": round(min(1e6, ppl), 1),
                    "t": round(elapsed, 1),
                }, ensure_ascii=False) + "\n")
                metrics.flush()
                write_json(args.status, {
                    "state": "running", "vocab": len(vocab) + 1,
                    "tokens": len(tokens), "params": params,
                    "total_steps": total_steps,
                    "steps_per_epoch": steps_per_epoch,
                    "epoch": epoch, "step": global_step,
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(eta, 1),
                    "tokens_per_s": round(tok_s, 0),
                    "last_loss": round(avg, 4),
                    "last_ppl": round(min(1e6, ppl), 1),
                    "arch": f"gpt2 {args.layers}x{args.dim}",
                })
                if global_step % 200 == 0:
                    print(f"[lm] epoch={epoch} step={global_step} loss={avg:.4f} "
                          f"ppl={ppl:.1f} eta={eta/60:.1f}min", flush=True)
                running, run_count = 0.0, 0
            if args.save_every and global_step % args.save_every == 0:
                torch.save({"state": model.state_dict(), "dim": args.dim,
                            "layers": args.layers, "heads": args.heads,
                            "block": seq, "epoch": epoch, "step": global_step,
                            "vocab": words},
                           os.path.join(args.out, "model.pt"))
                with io.open(os.path.join(args.out, "vocab.txt"), "w",
                             encoding="utf-8") as f:
                    f.write("\n".join(words))
            if args.max_steps and global_step >= args.max_steps:
                break
        if val_data is not None and val_data.shape[0] > 0:
            model.eval()
            vloss = 0.0; vn = 0
            with torch.no_grad():
                for b in range(min(200, val_data.shape[0] // args.batch or 1)):
                    vc = val_data[b * args.batch:(b + 1) * args.batch].to(device)
                    if vc.shape[0] < 2:
                        continue
                    xv, yv = vc[:, :-1], vc[:, 1:]
                    lg = model(xv)
                    vloss += float(F.cross_entropy(
                        lg.reshape(-1, lg.size(-1)), yv.reshape(-1)))
                    vn += 1
            model.train()
            v = vloss / max(vn, 1)
            vppl = math.exp(min(v, 20.0))
            print(f"[lm] epoch={epoch} val_loss={v:.4f} val_ppl={vppl:.1f}", flush=True)
            metrics.write(json.dumps({"step": global_step, "epoch": epoch,
                                      "loss": round(v, 4), "ppl": round(min(1e6, vppl), 1),
                                      "val": 1, "t": round(time.time() - t0, 1)},
                                     ensure_ascii=False) + "\n")
            metrics.flush()
            if v < best_val:
                best_val = v
                torch.save({"state": model.state_dict(), "dim": args.dim,
                            "layers": args.layers, "heads": args.heads,
                            "block": seq, "epoch": epoch + 1, "step": global_step,
                            "vocab": words},
                           os.path.join(args.out, "model_best.pt"))
        torch.save({"state": model.state_dict(), "dim": args.dim,
                    "layers": args.layers, "heads": args.heads,
                    "block": seq, "epoch": epoch + 1, "step": global_step,
                    "vocab": words},
                   os.path.join(args.out, "model.pt"))
        with io.open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(words))
        if args.max_steps and global_step >= args.max_steps:
            break

    metrics.close()
    write_json(args.status, {
        "state": "done", "vocab": len(vocab) + 1, "tokens": len(tokens),
        "params": params, "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch, "epoch": epoch,
        "step": global_step, "elapsed_s": round(time.time() - t0, 1),
        "eta_s": 0, "tokens_per_s": 0,
        "last_loss": round(float(loss.detach()), 4), "last_ppl": None,
        "arch": f"gpt2 {args.layers}x{args.dim}",
    })
    print("[lm] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
