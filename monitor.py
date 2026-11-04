# SPDX-License-Identifier: MIT
"""Terminal progress panel for a Zimo training run (stdlib only).

    python monitor.py                      # newest run under runs/
    python monitor.py --run runs/xxx
    python monitor.py --run runs/xxx -w    # refresh every 10s (Ctrl+C to stop)
    python monitor.py --run runs/xxx -w -i 5
    python monitor.py --run runs/xxx --no-color

Reads ``status.json`` and ``metrics.jsonl`` from the run directory, so it works
for a run started by the Studio, by ``train_gpt.py`` on a rented GPU box, or
after the page was closed.  GPU facts come from ``nvidia-smi`` when present.

Layout: a fixed-width card.  CJK glyphs are two columns wide, so every padded
line goes through ``disp_width`` instead of ``len`` — otherwise the right
border drifts (that was the "scattered" look).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

CARD_WIDTH = 62          # inner width in display columns
BAR_WIDTH = 34
BLOCKS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# width-aware text helpers
# ---------------------------------------------------------------------------

def char_width(char: str) -> int:
    """2 for East Asian wide/fullwidth glyphs, 0 for combining, else 1."""
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def disp_width(text: str) -> int:
    return sum(char_width(ch) for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - disp_width(text))


def fit(text: str, width: int) -> str:
    """Truncate to a display width, appending an ellipsis when cut."""
    if disp_width(text) <= width:
        return text
    out, used = "", 0
    for ch in text:
        w = char_width(ch)
        if used + w > width - 1:
            break
        out += ch
        used += w
    return out + "…"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def find_newest_run(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir()
                  if p.is_dir() and (p / "status.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "status.json").stat().st_mtime)


def read_status(run_dir: Path, path: Optional[Path] = None) -> dict:
    target = Path(path) if path else (run_dir / "status.json")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def read_metrics(run_dir: Path, override: Optional[Path] = None) -> list[dict]:
    path = Path(override) if override else (run_dir / "metrics.jsonl")
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
    """PPL values of the *current* run (steps restart between runs)."""
    if not rows:
        return []
    start = 0
    for index in range(1, len(rows)):
        if (rows[index].get("step") or 0) < (rows[index - 1].get("step") or 0):
            start = index
    return [float(r["ppl"]) for r in rows[start:] if r.get("ppl")]


# ---------------------------------------------------------------------------
# small renderers (unit-tested)
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
    return values[::step][-width:]


def sparkline(values: list[float], width: int = 34) -> str:
    if not values:
        return ""
    sample = downsample(values, width)
    low, high = min(sample), max(sample)
    if high - low < 1e-9:
        return BLOCKS[3] * len(sample)
    return "".join(BLOCKS[min(7, int((v - low) / (high - low) * 7))]
                   for v in sample)


def bar(percent: float, width: int = BAR_WIDTH, full: str = "█",
        empty: str = "░") -> str:
    percent = max(0.0, min(100.0, percent))
    filled = int(round(percent / 100 * width))
    return full * filled + empty * (width - filled)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    hours, rest = divmod(seconds, 3600)
    minutes, _ = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def format_count(value: float) -> str:
    value = float(value or 0)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return str(int(value))


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

def gpu_info() -> Optional[list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu="
             "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (out.stdout or "").strip().splitlines()
    if not lines:
        return None
    return [part.strip() for part in lines[0].split(",")]


def short_gpu_name(name: str) -> str:
    for prefix in ("NVIDIA ", "GeForce ", "NVIDIA Corporation "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def to_float(text: str) -> Optional[float]:
    digits = "".join(ch for ch in (text or "") if ch.isdigit() or ch == ".")
    try:
        return float(digits)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

class Colors:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)


def render(run_dir: Path, status_path: Optional[Path] = None,
           metrics_path: Optional[Path] = None,
           color: bool = False) -> str:
    status = read_status(run_dir, status_path)
    series = ppl_series(read_metrics(run_dir, metrics_path))
    gpu = gpu_info()
    c = Colors(color)

    # Colours must not affect padding: build plain text, colour last.
    def row(content: str) -> str:
        return "│ " + pad(content, CARD_WIDTH) + " │"

    def coloured_row(plain_parts: list[tuple[str, Optional[str]]]) -> str:
        """plain_parts: [(text, colour_code|None)] laid out by display width."""
        plain = "".join(text for text, _ in plain_parts)
        tail = " " * max(0, CARD_WIDTH - disp_width(plain))
        if not color:
            return "│ " + plain + tail + " │"
        painted = "".join(c._wrap(code, text) if code else text
                          for text, code in plain_parts)
        return "│ " + painted + tail + " │"

    top = "┌" + "─" * (CARD_WIDTH + 2) + "┐"
    mid = "├" + "─" * (CARD_WIDTH + 2) + "┤"
    bottom = "└" + "─" * (CARD_WIDTH + 2) + "┘"

    title = "ZIMO · 训练监控"
    gpu_short = short_gpu_name(gpu[0]) if gpu else "CPU"
    right = fit(f"{gpu_short}  {time.strftime('%H:%M:%S')}", 34)
    gap = max(1, CARD_WIDTH - disp_width(title) - disp_width(right))
    lines = [top, coloured_row([(title, "1"), (" " * gap, None), (right, "2")])]

    if not status:
        lines.append(mid)
        lines.append(row("(status.json 尚未生成，等待第一步)"))
        lines.append(bottom)
        return "\n".join(lines)

    step = status.get("step") or 0
    total = status.get("total_steps") or 0
    percent = step * 100 / total if total else 0
    last_ppl = status.get("last_ppl")
    if last_ppl is None and series:
        last_ppl = round(series[-1], 1)

    # ---- progress ----
    lines.append(mid)
    bar_text = bar(percent)
    pct_text = f"  {percent:5.1f}%"
    lines.append(coloured_row([(" " + bar_text, "32"), (pct_text, None)]))

    # ---- metrics grid: ASCII labels keep the columns stable ----
    grid = [
        ("step", f"{step:,} / {total:,}", "epoch", f"{status.get('epoch', 0)}"),
        ("loss", f"{status.get('last_loss')}", "ppl", f"{last_ppl}"),
        ("speed", f"{format_count(status.get('tokens_per_s'))} tok/s", "eta",
         format_seconds(status.get("eta_s") or 0)),
    ]
    for left_label, left_value, right_label, right_value in grid:
        lines.append(coloured_row([
            (" " + pad(left_label, 6), "2"), (pad(left_value, 23), None),
            (pad(right_label, 6), "2"), (right_value, None),
        ]))

    # ---- trend ----
    spark = sparkline(smooth(series))
    if spark:
        first, last = series[0], series[-1]
        drop = (1 - last / first) * 100 if first else 0
        lines.append(row(""))
        lines.append(coloured_row([(" trend ", "2"), (spark, "36")]))
        lines.append(coloured_row(
            [(f"       {first:.0f} → {last:.1f}", None),
             (f"  (-{drop:.0f}%)", "2")]))

    # ---- hardware ----
    lines.append(mid)
    if gpu:
        _, util, used, mem_total, temp, power = gpu
        used_gb = (to_float(used) or 0) / 1024
        total_gb = (to_float(mem_total) or 1) / 1024
        mem_pct = used_gb / total_gb * 100 if total_gb else 0
        lines.append(coloured_row([
            (" GPU  ", "2"), (bar(mem_pct, 18, "▮", "▯"), "32"),
            (f"  {used_gb:4.1f}/{total_gb:.0f} GB", None),
        ]))
        lines.append(coloured_row([
            ("      ", None),
            (f"util {(to_float(util) or 0):3.0f}%  ·  {temp}°C  ·  {power}", None),
        ]))
    else:
        lines.append(coloured_row([
            (" GPU  ", "2"), ("未检测到 nvidia-smi（CPU 训练）", None)]))

    try:
        cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        cores = os.cpu_count() or 1
    load = f"   load {os.getloadavg()[0]:.1f}" if hasattr(os, "getloadavg") else ""
    try:
        disk = shutil.disk_usage(str(run_dir))
        disk_text = f"   disk {disk.used / 2**30:.1f}/{disk.total / 2**30:.0f} GB"
    except OSError:
        disk_text = ""
    lines.append(coloured_row([
        (" CPU  ", "2"), (f"{cores} cores{load}", None), (disk_text, "2")]))

    # ---- model + state ----
    lines.append(mid)
    lines.append(coloured_row([
        (" model ", "2"),
        (fit(f"{status.get('arch', '?')} · "
             f"{format_count(status.get('params'))} params · "
             f"vocab {status.get('vocab', 0):,}", CARD_WIDTH - 7), None),
    ]))
    lines.append(coloured_row([
        (" data  ", "2"),
        (fit(f"{format_count(status.get('tokens'))} tokens · {run_dir.name}",
             CARD_WIDTH - 7), None),
    ]))

    state = status.get("state", "?")
    if state == "done":
        lines.append(coloured_row([(" ● done", "32"), ("   训练已完成", None)]))
    elif state == "running":
        lines.append(coloured_row([(" ● running", "33"),
                                   ("   Ctrl+C 只停止查看", "2")]))
    else:
        lines.append(coloured_row([(f" ● {state}", "2")]))

    lines.append(bottom)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description="Zimo 训练进度面板（读取 status.json / metrics.jsonl）",
    )
    parser.add_argument("--run", default="", help="运行目录；默认取 runs/ 下最新的")
    parser.add_argument("--runs-root", default="runs", help="runs 根目录（默认 runs）")
    parser.add_argument("-w", "--watch", action="store_true", help="持续刷新")
    parser.add_argument("-i", "--interval", type=float, default=10.0,
                        help="刷新间隔秒（默认 10）")
    parser.add_argument("--status", default="", help="指定 status.json")
    parser.add_argument("--metrics", default="", help="指定 metrics.jsonl")
    parser.add_argument("--no-color", action="store_true", help="关闭颜色")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run).expanduser()
    else:
        run_dir = find_newest_run(Path(args.runs_root).expanduser())
        if run_dir is None:
            print(f"[x] {args.runs_root}/ 下没有找到带 status.json 的运行目录",
                  file=sys.stderr)
            return 1

    if not run_dir.is_dir():
        print(f"[x] 目录不存在：{run_dir}", file=sys.stderr)
        return 1

    color = (not args.no_color and sys.stdout.isatty()
             and not os.environ.get("NO_COLOR"))
    try:
        while True:
            if args.watch:
                sys.stdout.write("\033[H\033[J")
            print(render(run_dir,
                         Path(args.status) if args.status else None,
                         Path(args.metrics) if args.metrics else None,
                         color=color))
            sys.stdout.flush()
            if not args.watch:
                return 0
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n(已停止查看，训练不受影响)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
