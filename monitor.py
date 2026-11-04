# SPDX-License-Identifier: MIT
"""Terminal progress panel for a Zimo training run (stdlib only).

    python monitor.py                      # newest run under runs/
    python monitor.py --run runs/20261104_120000_gpt2
    python monitor.py --run runs/xxx -w    # refresh every 10s (Ctrl+C to stop)
    python monitor.py --run runs/xxx -w -i 5

Reads ``status.json`` and ``metrics.jsonl`` from the run directory, so it works
for a run started by the Studio, by ``train_gpt.py`` on a rented GPU box, or
after the page was closed.  GPU facts come from ``nvidia-smi`` when present;
everything else is plain stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

BLOCKS = "▁▂▃▄▅▆▇█"
BAR_WIDTH = 28


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def find_newest_run(root: Path) -> Optional[Path]:
    """Newest directory under ``root`` that contains a status.json."""
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir()
                  if p.is_dir() and (p / "status.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "status.json").stat().st_mtime)


def read_status(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    rows = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def ppl_series(rows: list[dict]) -> list[float]:
    """PPL values of the *current* run (restarts after a step reset)."""
    if not rows:
        return []
    start = 0
    for index in range(1, len(rows)):
        if (rows[index].get("step") or 0) < (rows[index - 1].get("step") or 0):
            start = index
    return [float(r["ppl"]) for r in rows[start:] if r.get("ppl")]


# ---------------------------------------------------------------------------
# rendering helpers (unit-tested, no state)
# ---------------------------------------------------------------------------

def smooth(values: list[float], window: int = 7) -> list[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    for index in range(len(values)):
        low = max(0, index - window + 1)
        chunk = values[low:index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def downsample(values: list[float], width: int) -> list[float]:
    if len(values) <= width:
        return list(values)
    step = max(1, len(values) // width)
    sampled = values[::step]
    return sampled[-width:]


def sparkline(values: list[float], width: int = 44) -> str:
    if not values:
        return ""
    sample = downsample(values, width)
    low, high = min(sample), max(sample)
    if high - low < 1e-9:
        return BLOCKS[3] * len(sample)
    return "".join(BLOCKS[min(7, int((v - low) / (high - low) * 7))]
                   for v in sample)


def progress_bar(percent: float, width: int = BAR_WIDTH) -> str:
    percent = max(0.0, min(100.0, percent))
    filled = int(round(percent / 100 * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    hours, rest = divmod(seconds, 3600)
    minutes, _ = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

def gpu_info() -> Optional[list[str]]:
    """nvidia-smi fields: name, util, mem_used, mem_total, temp, power."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu="
             "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return None
    return [part.strip() for part in line[0].split(",")]


def cpu_info() -> str:
    try:
        cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        cores = os.cpu_count() or 1
    if hasattr(os, "getloadavg"):
        load = os.getloadavg()[0]
        return f"{cores} 核   负载 {load:.1f}"
    return f"{cores} 核"


def memory_info() -> str:
    try:
        import psutil  # optional on Windows
        vm = psutil.virtual_memory()
        return f"{vm.used / 2**30:.0f} / {vm.total / 2**30:.0f} GB"
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            info = dict(line.split(":", 1) for line in handle.read().splitlines())
        total = int(info["MemTotal"].split()[0]) / 2**20
        avail = int(info["MemAvailable"].split()[0]) / 2**20
        return f"{total - avail:.0f} / {total:.0f} GB"
    except (OSError, KeyError, ValueError):
        return "–"


def disk_info(path: Path) -> str:
    try:
        usage = shutil.disk_usage(str(path))
        return f"{usage.used / 2**30:.1f} / {usage.total / 2**30:.0f} GB"
    except OSError:
        return "–"


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

def render(run_dir: Path) -> str:
    status = read_status(run_dir)
    rows = read_metrics(run_dir)
    series = ppl_series(rows)
    gpu = gpu_info()

    title = gpu[0] if gpu else "CPU"
    lines = [
        "=" * 64,
        f" ZIMO TRAIN MONITOR            {time.strftime('%H:%M:%S')}   {title}",
        "=" * 64,
        f" 目录   {run_dir.name}",
    ]

    if not status:
        lines.append(" (status.json 尚未生成，等待第一步)")
        return "\n".join(lines)

    params = status.get("params") or 0
    lines.append(f" 模型   {status.get('arch', '?'):<16} 参数 {params:,}")
    lines.append(f" 语料   {status.get('tokens', 0):,} tokens   词表 {status.get('vocab', 0):,}")
    lines.append("")

    step = status.get("step") or 0
    total = status.get("total_steps") or 0
    percent = step * 100 / total if total else 0
    lines.append(f" 进度   {progress_bar(percent)} {percent:5.1f}%")
    lines.append(f"        step {step:,} / {total:,}   epoch {status.get('epoch')}")
    lines.append("")

    last_ppl = status.get("last_ppl")
    if last_ppl is None and series:
        last_ppl = round(series[-1], 1)
    lines.append(f" 指标   loss {status.get('last_loss')}    ppl {last_ppl}")
    speed = status.get("tokens_per_s") or 0
    lines.append(
        f" 速率   {speed:,.0f} tok/s   已跑 {format_seconds(status.get('elapsed_s') or 0)}"
        f"   剩余 {format_seconds(status.get('eta_s') or 0)}")
    spark = sparkline(smooth(series))
    if spark:
        lines.append(f" 趋势   {spark}  (PPL)")
        first, last = series[0], series[-1]
        drop = (1 - last / first) * 100 if first else 0
        lines.append(f" 变化   PPL {first:.0f} -> {last:.1f}   (-{drop:.0f}%)")
    lines.append("")

    if gpu:
        _, util, used, mem_total, temp, power = gpu
        lines.append(f" GPU    {gpu[0]}")
        lines.append(f"        利用率 {util}   显存 {used} / {mem_total}   温度 {temp}   功耗 {power}")
    memory = memory_info()
    cpu_line = f" CPU    {cpu_info()}"
    if memory != "–":
        cpu_line += f"   内存 {memory}"
    lines.append(cpu_line)
    lines.append(f" 磁盘   {disk_info(run_dir)}")
    lines.append("-" * 64)
    state = status.get("state", "?")
    if state == "done":
        lines.append(" 状态   训练已完成 ✓")
    elif state == "running":
        lines.append(" 状态   训练中（Ctrl+C 只停止查看）")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description="Zimo 训练进度面板（读取 status.json / metrics.jsonl）",
    )
    parser.add_argument("--run", default="", help="运行目录；默认取 runs/ 下最新的")
    parser.add_argument("--runs-root", default="runs", help="runs 根目录（默认 runs）")
    parser.add_argument("-w", "--watch", action="store_true", help="持续刷新")
    parser.add_argument("-i", "--interval", type=float, default=10.0, help="刷新间隔秒（默认 10）")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run).expanduser()
    else:
        run_dir = find_newest_run(Path(args.runs_root).expanduser())
        if run_dir is None:
            print(f"[x] {args.runs_root}/ 下没有找到带 status.json 的运行目录", file=sys.stderr)
            return 1

    if not run_dir.is_dir():
        print(f"[x] 目录不存在：{run_dir}", file=sys.stderr)
        return 1
    if not (run_dir / "status.json").is_file():
        print(f"[!] {run_dir} 里没有 status.json（还没开始写入？）", file=sys.stderr)

    try:
        while True:
            if args.watch:
                sys.stdout.write("\033[H\033[J")  # clear (no Windows flicker)
            print(render(run_dir))
            sys.stdout.flush()
            if not args.watch:
                return 0
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n(已停止查看，训练不受影响)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
