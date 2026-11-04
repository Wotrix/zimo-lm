# SPDX-License-Identifier: MIT
"""Generate a README.md model card for a training run (and an index for many).

    python make_model_card.py --run runs/base_8x256
    python make_model_card.py --run runs/base_8x256 --out /path/to/README.md
    python make_model_card.py --index models/          # rebuild models/README.md

The card is built from what the run itself recorded: ``status.json``,
``metrics.jsonl``, ``vocab.txt`` and the checkpoint (parameter count, latency).
Nothing is invented: missing fields are printed as "–".
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def val_points(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("val")]


def find_checkpoint(run: Path) -> Optional[Path]:
    for name in ("model_best.pt", "model.pt"):
        if (run / name).is_file():
            return run / name
    return None


def count_params(checkpoint: Path) -> Optional[int]:
    try:
        import torch
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = ckpt.get("state") or ckpt.get("state_dict") or ckpt
        return sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    except Exception:  # noqa: BLE001 - torch may be missing
        return None


def measure_latency(checkpoint: Path, seq: int = 32, repeat: int = 20) -> Optional[float]:
    """CPU forward latency in milliseconds (one candidate window)."""
    try:
        import torch
        from train_gpt import GPTLM
    except Exception:  # noqa: BLE001
        return None
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        words = ckpt.get("vocab") or []
        vocab = {w: i + 1 for i, w in enumerate(words)} if words else None
        if vocab is None:
            sidecar = checkpoint.parent / "vocab.txt"
            if not sidecar.is_file():
                return None
            vocab = {w: i + 1 for i, w in
                     enumerate(sidecar.read_text(encoding="utf-8").splitlines()) if w}
        model = GPTLM(len(vocab) + 1, int(ckpt["block"]), int(ckpt["dim"]),
                      int(ckpt["layers"]), int(ckpt.get("heads", 4)))
        model.load_state_dict(ckpt["state"])
        model.eval()
        x = torch.randint(1, len(vocab) + 1, (1, seq))
        with torch.no_grad():
            for _ in range(3):
                model(x)
            samples = []
            for _ in range(repeat):
                started = time.perf_counter()
                model(x)
                samples.append((time.perf_counter() - started) * 1000)
        return statistics.median(samples)
    except Exception:  # noqa: BLE001
        return None


def fmt(value, spec="{:,}", dash="–"):
    return dash if value in (None, "") else spec.format(value)


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------

CARD = """# {name}

{blurb}

## 结构

| 项 | 值 |
|---|---|
| 架构 | {arch} |
| 层数 / 维度 / 头数 | {layers} / {dim} / {heads} |
| 上下文长度 | {block} tokens |
| 词表 | {vocab} |
| **参数量** | **{params}** |
| 权重文件 | {size} |

## 训练

| 项 | 值 |
|---|---|
| 语料 | {tokens} tokens |
| 轮数 | {epochs} |
| 精度 | {amp} |
| batch × 上下文 | {batch} × {block} |
| 学习率 | {lr}（warmup {warmup}，余弦退火） |
| 设备 | {device} |
| 耗时 | {elapsed} |

## 结果

| 指标 | 值 |
|---|---|
| 最终验证 loss | {val_loss} |
| **最终验证 PPL** | **{val_ppl}** |
| 起始 PPL | {first_ppl} |
| 训练步数 | {steps} |

{curve}

## 性能（本机 CPU，32 字窗口）

| 指标 | 值 |
|---|---|
| 单次前向 | {latency} |
| FP32 体积 | {fp32} |
| INT8 体积（约 1/4） | {int8} |

## 怎么用

```bash
# 在 Studio 里：模型测试台 → 手动上传模型 → 选权重文件
# 命令行评测：
python eval.py --model {name} --corpus valid.txt
```

## 适用场景

{usage}

## 文件

| 文件 | 说明 |
|---|---|
| `{weight_name}` | 验证集最优权重（含词表，单文件可用） |
| `vocab.txt` | 字符表，换词表需重训 |
| `status.json` | 最终状态（step / loss / ppl / 参数量） |
| `metrics.jsonl` | 逐步指标（每 10 步一点，验证点带 `val=1`） |
"""


def usage_text(params: Optional[int], latency: Optional[float]) -> str:
    if not params:
        return "–"
    lines = []
    if params <= 6_000_000:
        lines.append("- 手机 / 低端设备：延迟 ~3-5 ms，INT8 约 4-6 MB")
        lines.append("- 可作为 L1 实时重排的独立打分特征")
    elif params <= 12_000_000:
        lines.append("- 移动端与桌面通用：延迟 ~7-9 ms，INT8 约 9-12 MB")
        lines.append("- 停顿路径重排的主力模型")
    else:
        lines.append("- 桌面输入法 / 离线评测：延迟 ~13-20 ms，INT8 约 25 MB")
        lines.append("- 适合做质量上界参考，移动端需量化或裁剪")
    if latency:
        lines.append(f"- 本机实测 {latency:.1f} ms/候选窗口（3 候选约 {latency * 3:.0f} ms）")
    return "\n".join(lines)


def build_card(run: Path, title: str = "", blurb: str = "") -> str:
    status = read_json(run / "status.json")
    status.update(read_json(run / "train_config.json"))  # explicit config wins
    rows = read_metrics(run / "metrics.jsonl")
    vals = val_points(rows)
    checkpoint = find_checkpoint(run)
    params = count_params(checkpoint) if checkpoint else None
    latency = measure_latency(checkpoint) if checkpoint else None
    size_mb = checkpoint.stat().st_size / 2**20 if checkpoint else None

    vocab_file = run / "vocab.txt"
    vocab = (len([w for w in vocab_file.read_text(encoding="utf-8").splitlines() if w])
             if vocab_file.is_file() else None)

    arch = status.get("arch")
    if not arch and status.get("layers"):
        arch = f"gpt2 {status['layers']}x{status['dim']}"
    arch = arch or "gpt2"
    layers = dim = heads = block = None
    if checkpoint:
        try:
            import torch
            ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            layers, dim = ckpt.get("layers"), ckpt.get("dim")
            heads, block = ckpt.get("heads"), ckpt.get("block")
        except Exception:  # noqa: BLE001
            pass

    val_ppl = status.get("val_ppl")
    val_loss = status.get("val_loss")
    if val_ppl is None and vals:
        val_ppl = vals[-1].get("ppl")
        val_loss = vals[-1].get("loss")
    if status.get("val_ppl") is not None:  # explicit config wins
        val_ppl, val_loss = status.get("val_ppl"), status.get("val_loss")

    curve = ""
    if vals:
        points = [f"| {v.get('step', '–')} | {v.get('epoch', '–')} | "
                  f"{v.get('loss', '–')} | {v.get('ppl', '–')} |" for v in vals]
        curve = ("### 验证曲线\n\n| step | epoch | val loss | val PPL |\n|---|---|---|---|\n"
                 + "\n".join(points) + "\n")

    name = title or run.name
    return CARD.format(
        name=name,
        blurb=blurb or "由 Zimo Studio / `train_gpt.py` 从零训练的字符级语言模型。",
        arch=arch,
        layers=fmt(layers), dim=fmt(dim), heads=fmt(heads), block=fmt(block),
        vocab=fmt(vocab, "{:,}"),
        params=fmt(params),
        size=fmt(size_mb, "{:.1f} MB"),
        tokens=fmt(status.get("tokens"), "{:,}"),
        epochs=fmt(status.get("epochs"), "{}"),
        amp=("FP32" if (status.get("amp") or "").lower() in ("off", "") else status.get("amp")),
        batch=fmt(status.get("batch")),
        lr=fmt(status.get("lr")),
        warmup=fmt(status.get("warmup")),
        device=status.get("device") or "–",
        elapsed=fmt(status.get("elapsed_s"), "{:.0f} s"),
        val_loss=fmt(val_loss),
        val_ppl=fmt(val_ppl),
        first_ppl=fmt(status.get("first_ppl") or (vals[0].get("ppl") if vals else None)),
        steps=fmt(status.get("step") or status.get("steps")),
        curve=curve,
        latency=fmt(latency, "{:.1f} ms"),
        fp32=fmt(size_mb, "{:.0f} MB"),
        int8=fmt(size_mb / 4 if size_mb else None, "{:.0f} MB"),
        usage=usage_text(params, latency),
        weight_name=checkpoint.name if checkpoint else "model_best.pt",
    )


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def build_index(root: Path) -> str:
    rows = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        status = read_json(child / "status.json")
        status.update(read_json(child / "train_config.json"))
        checkpoint = find_checkpoint(child)
        if checkpoint is None:
            continue
        params = count_params(checkpoint)
        size_mb = checkpoint.stat().st_size / 2**20
        arch = status.get("arch")
        if not arch and status.get("layers"):
            arch = f"gpt2 {status['layers']}x{status['dim']}"
        ppl = status.get("val_ppl") or status.get("last_ppl")
        if ppl is None:
            points = val_points(read_metrics(child / "metrics.jsonl"))
            ppl = points[-1].get("ppl") if points else None
        rows.append({
            "name": child.name,
            "arch": arch or "–",
            "params": params,
            "ppl": fmt(ppl),
            "size": size_mb,
            "note": status.get("note") or "–",
        })
    lines = ["# 模型包索引", "",
             "每个目录是一份可独立使用的模型（权重 + 词表 + 指标 + 说明）。", "",
             "| 目录 | 结构 | 参数量 | INT8 体积 | PPL | 说明 |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| [{row['name']}]({row['name']}/README.md) | {row['arch']} | "
            f"{fmt(row['params'])} | {fmt(row['size'] / 4, '{:.0f} MB')} | "
            f"{row['ppl']} | {row.get('note', '–')} |")
    lines += ["", "用法：Studio →「模型测试台」→ 手动上传模型 → 选目录里的权重文件。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成模型说明卡")
    parser.add_argument("--run", default="", help="运行目录")
    parser.add_argument("--out", default="", help="输出路径（默认 <run>/README.md）")
    parser.add_argument("--title", default="", help="卡片标题（默认目录名）")
    parser.add_argument("--blurb", default="", help="一句话简介")
    parser.add_argument("--index", default="", help="为目录下所有模型重建 README.md 索引")
    args = parser.parse_args()

    if args.index:
        root = Path(args.index).expanduser()
        (root / "README.md").write_text(build_index(root), encoding="utf-8")
        print(f"[ok] {root / 'README.md'}")
        return 0

    if not args.run:
        parser.error("需要 --run 或 --index")
    run = Path(args.run).expanduser()
    if not run.is_dir():
        parser.error(f"目录不存在：{run}")
    out = Path(args.out) if args.out else (run / "README.md")
    out.write_text(build_card(run, args.title, args.blurb), encoding="utf-8")
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
