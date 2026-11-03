#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Zimo Studio — a minimal one-click Streamlit trainer.

What it does
------------
- Pick a trainer (GPT-2 char LM / LSTM LM / MLP listwise ranker).
- Upload a `.txt` corpus or a selection-events `.csv` / `.jsonl`.
- Start the trainer as a subprocess and stream its stdout into the UI.
- Show a progress bar, live loss/PPL cards, a loss chart, and the log tail.
- After training, score candidate sentences with the trained LM (GPT-2/LSTM)
  and rank them by PPL.  MLP listwise rankers need 32-dim feature vectors and
  therefore do not support raw-text PPL scoring.

Run
---
  python -m pip install streamlit torch numpy pandas
  streamlit run app.py

The app locates the trainers automatically:
  tools/lm/train_gpt.py      or  train_gpt.py
  tools/lm/train_nnlm.py     or  train_nnlm.py
  tools/ranker/train_mlp.py  or  train_mlp.py

Set ZIMO_TRAINER_ROOT if the trainers live in a different checkout:
  ZIMO_TRAINER_ROOT=/path/to/project streamlit run app.py

This is a local development tool.  Do not expose it to the public internet: it
spawns local processes and reads local files.

Note on Windows: the studio deliberately does not open status.json while the
training process is alive.  It reads stdout + metrics.jsonl during training and
status.json only after exit, so an unpatched trainer's os.replace() cannot be
blocked by a UI reader.
"""

from __future__ import annotations

import csv
import hashlib
import html
import gzip
import io
import json
import math
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from tjgpt_export import write_tjgpt
from onnx_export import (cpp_snippet, export_onnx, export_safetensors,
                         export_tflite, export_torchscript,
                         onnx_file_size_mb, onnx_ppl, quantize_onnx_int8)
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint_utils import (CHECKPOINT_SUFFIXES, extract_vocab_words,
                              find_latest_run, infer_config_from_state,
                              inspect_checkpoint_file, load_state_any)

# ---------------------------------------------------------------------------
# Paths and trainer discovery
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    env = (os.environ.get("ZIMO_TRAINER_ROOT")
           or os.environ.get("TINY_CHAR_LM_TRAINER_ROOT")
           or os.environ.get("TONGJIA_TRAINER_ROOT")
           or os.environ.get("TONGJIA_ROOT"))
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        APP_DIR,
        APP_DIR.parent,
        APP_DIR.parent.parent,
        APP_DIR.parent.parent.parent,
    ]
    for candidate in candidates:
        if ((candidate / "tools" / "lm" / "train_gpt.py").exists()
                or (candidate / "train_gpt.py").exists()):
            return candidate
    return APP_DIR


ROOT = _find_project_root()
TRAINER_ROOT = Path(
    os.environ.get("ZIMO_TRAINER_ROOT")
    or os.environ.get("TINY_CHAR_LM_TRAINER_ROOT")
    or os.environ.get("TONGJIA_TRAINER_ROOT")
    or str(ROOT)).expanduser().resolve()
RUNS_DIR = Path(
    os.environ.get("ZIMO_RUNS_DIR")
    or os.environ.get("TINY_CHAR_LM_RUNS_DIR")
    or os.environ.get("TONGJIA_RUNS_DIR")
    or str(ROOT / "runs"))
PLAYGROUND_DIR = RUNS_DIR / "playground"
DEMO_MODEL_DIR = APP_DIR / "examples" / "demo"
APP_VERSION = "0.4.0"

# Relative by default so the UI never shows an absolute personal path;
# set ZIMO_DATA_DIR to point somewhere else.
LOCAL_DATA_HINT = os.environ.get("ZIMO_DATA_DIR", "examples")

CLASSIC_DASHBOARD_URL = os.environ.get(
    "ZIMO_CLASSIC_DASHBOARD_URL", "http://127.0.0.1:8765")

TRAINER_CANDIDATES = {
    "gpt2": ("tools/lm/train_gpt.py", "train_gpt.py"),
    "lstm": ("tools/lm/train_nnlm.py", "train_nnlm.py"),
    "mlp": ("tools/ranker/train_mlp.py", "train_mlp.py"),
}

MODEL_OPTIONS = {
    "gpt2": {"kind": "text", "zh": "GPT-2 字符级语言模型",
             "en": "GPT-2 character LM"},
    "lstm": {"kind": "text", "zh": "LSTM 语言模型",
             "en": "LSTM language model"},
    "mlp": {"kind": "events", "zh": "MLP Listwise Ranker",
            "en": "MLP listwise ranker"},
    "tjtr1": {"kind": "events", "zh": "TJTR1 交叉编码器（CLI）",
              "en": "TJTR1 cross-encoder (CLI)"},
}

# Six trainers, three camps.  "studio" means this Studio has a one-click
# training command; CLI-only models are still documented in the guide.
MODEL_GUIDE = [
    {
        "camp_zh": "1. 无监督文本打分（只需纯文本）",
        "camp_en": "1. Unsupervised text scoring (plain text only)",
        "models": [
            {
                "key": "gpt2",
                "zh": "GPT-2 字符级语言模型",
                "en": "GPT-2 character LM",
                "summary_zh": "字符级自回归语言模型，输出候选文本的平均对数似然 / PPL，适用于候选重排与纠错。",
                "summary_en": "A character-level autoregressive LM scoring candidates by mean log-likelihood / PPL; used for reranking and correction.",
                "data_zh": "纯文本，每行一句",
                "data_en": "plain text, one sentence per line",
                "use_zh": "输出候选文本的 PPL，用于候选重排与纠错",
                "use_en": "score text with PPL for naturalness; reranking/correction",
                "studio": True,
            },
            {
                "key": "lstm",
                "zh": "LSTM 语言模型",
                "en": "LSTM language model",
                "summary_zh": "轻量循环语言模型；任务与 GPT-2 相同，体积更小、推理更快，长上下文能力较弱。",
                "summary_en": "A lightweight recurrent LM with the same scoring task; smaller and faster, with weaker long-context capacity.",
                "data_zh": "纯文本，每行一句",
                "data_en": "plain text, one sentence per line",
                "use_zh": "同类文本打分任务，体积更小、推理更快，适合资源受限设备",
                "use_en": "like GPT-2 but smaller/faster for very low-end devices",
                "studio": True,
            },
        ],
    },
    {
        "camp_zh": "2. 有监督用户日志重排（需要点击日志）",
        "camp_en": "2. Supervised user-log reranking (click logs required)",
        "models": [
            {
                "key": "tjtr1",
                "zh": "TJTR1 交叉编码器",
                "en": "TJTR1 cross-encoder",
                "summary_zh": "Transformer 编码器交叉重排模型；输入上下文与候选列表，直接学习用户选择。",
                "summary_en": "A Transformer encoder cross-ranker over context and candidates; learns the user's choices directly.",
                "data_zh": "用户选择日志：上下文 + 候选列表 + 选中项（JSONL/CSV）",
                "data_en": "selection events: context + candidate list + chosen item (JSONL/CSV)",
                "use_zh": "直接学习用户选择，适用于候选重排",
                "use_en": "predict the user's choice directly; better for reranking than a plain LM",
                "studio": False,
            },
            {
                "key": "mlp",
                "zh": "TJMLP1 Listwise Ranker",
                "en": "TJMLP1 listwise ranker",
                "summary_zh": "基于 32 维特征的低延迟列表重排器；适用于实时按键/停顿路径。",
                "summary_en": "A low-latency listwise reranker over 32-dim features; suited to real-time keystroke/pause paths.",
                "data_zh": "带 32 维特征的点击日志（JSONL/CSV scores JSON）",
                "data_en": "click logs with 32-dim features (JSONL/CSV scores JSON)",
                "use_zh": "极低延迟的实时重排",
                "use_en": "very low-latency real-time reranking",
                "studio": True,
            },
        ],
    },
    {
        "camp_zh": "3. 传统低延迟兜底（需要分词语料）",
        "camp_en": "3. Traditional low-latency fallback (tokenized corpus required)",
        "models": [
            {
                "key": "ngram3",
                "zh": "3-gram 量化语言模型",
                "en": "3-gram quantized LM",
                "summary_zh": "统计语言模型；基于相邻词频，延迟低，用作停顿/兜底打分。",
                "summary_en": "A statistical LM based on adjacent-word counts; low latency for pause-path/fallback scoring.",
                "data_zh": "已分词语料 + 词表 TSV",
                "data_en": "tokenized corpus + vocabulary TSV",
                "use_zh": "为停顿路径与传统兜底提供低延迟打分",
                "use_en": "fast scores for pause-path/traditional fallback",
                "studio": False,
            },
            {
                "key": "word_ngram",
                "zh": "词级 n-gram",
                "en": "word-level n-gram",
                "summary_zh": "词级统计语言模型；用于旧版本兼容与迁移。",
                "summary_en": "A word-level statistical LM for legacy compatibility and migration.",
                "data_zh": "已分词语料 + 项目词典",
                "data_en": "tokenized corpus + project dictionary",
                "use_zh": "旧版兼容",
                "use_en": "legacy compatibility",
                "studio": False,
            },
        ],
    },
]


def guide_for(key: str) -> tuple[Optional[dict], Optional[dict]]:
    for camp in MODEL_GUIDE:
        for model in camp["models"]:
            if model["key"] == key:
                return model, camp
    return None, None


def tr(zh: str, en: str) -> str:
    """Tiny inline i18n: Chinese by default, English when the toggle is on."""
    return en if st.session_state.get("lang", "zh") == "en" else zh


_EXPLICIT_TRAINER_ROOT = any(
    os.environ.get(name) for name in (
        "ZIMO_TRAINER_ROOT", "TINY_CHAR_LM_TRAINER_ROOT",
        "TONGJIA_TRAINER_ROOT"))


def trainer_roots() -> list[Path]:
    """Package root, plus nearby project checkouts that contain trainers."""
    roots = [TRAINER_ROOT]
    if not _EXPLICIT_TRAINER_ROOT:
        bases = [ROOT, ROOT.parent, ROOT.parent.parent]
        for base in bases:
            roots.append(base)
            try:
                for child in base.iterdir():
                    if (child.is_dir()
                            and (child / "tools" / "lm").is_dir()):
                        roots.append(child)
            except OSError:
                continue
    unique = []
    seen = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def trainer_path(key: str) -> Optional[Path]:
    for root in trainer_roots():
        for relative in TRAINER_CANDIDATES.get(key, ()):
            path = root / relative
            if path.is_file():
                return path
    return None


DEFAULT_STATE: dict[str, Any] = {
    "proc": None,
    "queue": None,
    "reader": None,
    "log": [],
    "metrics": [],
    "val_points": [],
    "status": {},
    "run_dir": "",
    "arch_key": "",
    "command": "",
    "finished": False,
    "error": "",
    "started_at": 0.0,
    "pid": 0,
    "log_offset": 0,
    "train_config": {},
    "epochs": 2,
    "batch": 32,
    "dataset_tokens": 0,
    "tokens_budget": 0,
    "last_status_read": 0.0,
    "returncode": None,
    "lang": "zh",
    "theme": "dark",
    "play_a": None,
    "play_b": None,
}


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def pid_alive(pid: int) -> bool:
    """True when the process still exists (works while detached)."""
    if not pid:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                capture_output=True, timeout=5)
            return str(int(pid)).encode() in out.stdout
        os.kill(int(pid), 0)
        return True
    except Exception:  # noqa: BLE001 - a gone process is not an error
        return False


def is_running() -> bool:
    return pid_alive(int(st.session_state.get("pid") or 0))


def launch_detached(command: list[str], cwd: Path, run_dir: Path) -> int:
    """Start the trainer detached from Streamlit.

    stdout/stderr go to ``run_dir/train.log`` so the UI can tail it; the PID is
    written to ``run_dir/train.pid``.  Closing the browser or dropping SSH no
    longer kills the run.
    """
    log_path = run_dir / "train.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    kwargs: dict = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": io.open(log_path, "a", encoding="utf-8", buffering=1),
        "stderr": subprocess.STDOUT,
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
    else:
        kwargs["start_new_session"] = True
    handle = kwargs["stdout"]
    proc = subprocess.Popen(command, **kwargs)
    handle.close()
    (run_dir / "train.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def terminate_pid(pid: int) -> None:
    if not pid:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                           capture_output=True, timeout=10)
        else:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
    except Exception:  # noqa: BLE001 - already gone
        pass


def read_log_from(path: Path, offset: int) -> tuple[list[str], int]:
    """Incrementally read appended lines from a log file."""
    if not path.is_file():
        return [], offset
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size < offset:
        offset = 0
    if size == offset:
        return [], offset
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        chunk = fh.read()
        offset = fh.tell()
    return chunk.splitlines(), offset


def write_run_config(run_dir: Path, config: dict) -> None:
    try:
        (run_dir / "train_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def read_run_config(run_dir: Path) -> dict:
    path = run_dir / "train_config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def auto_attach_run() -> None:
    """Adopt a run that is still alive in the background (e.g. after reload)."""
    if st.session_state.get("pid"):
        return
    if not RUNS_DIR.is_dir():
        return
    candidates = []
    for path in RUNS_DIR.iterdir():
        pid_file = path / "train.pid"
        if path.is_dir() and pid_file.is_file():
            try:
                candidates.append((pid_file.stat().st_mtime, path, pid_file))
            except OSError:
                continue
    for _mtime, path, pid_file in sorted(candidates, reverse=True)[:5]:
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            continue
        if pid_alive(pid):
            status = read_json_safe(path / "status.json")
            st.session_state["pid"] = pid
            st.session_state["run_dir"] = str(path)
            st.session_state["arch_key"] = str(
                status.get("arch") or path.name.rsplit("_", 1)[-1])
            st.session_state["log_offset"] = 0
            st.session_state["finished"] = False
            st.session_state["error"] = ""
            st.session_state["command"] = f"attached: {path.name}"
            return


def unfinished_runs(limit: int = 20) -> list[dict]:
    """Runs that own a checkpoint and a corpus, newest first."""
    items = []
    if not RUNS_DIR.is_dir():
        return items
    for path in RUNS_DIR.iterdir():
        if not path.is_dir() or path.name == "playground":
            continue
        model = next((path / name for name in ("model.pt", "model_best.pt")
                      if (path / name).is_file()), None)
        corpus = next((path / name for name in ("corpus.txt", "events.jsonl")
                       if (path / name).is_file()), None)
        if model is None or corpus is None:
            continue
        status = read_json_safe(path / "status.json")
        try:
            mtime = model.stat().st_mtime
        except OSError:
            continue
        items.append({
            "dir": path,
            "arch": str(status.get("arch") or path.name.rsplit("_", 1)[-1]),
            "state": status.get("state", "unknown"),
            "step": status.get("step", "–"),
            "mtime": mtime,
        })
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items[:limit]


def reset_run_state() -> None:
    for key in ("log", "metrics", "val_points", "status", "run_dir",
                "arch_key", "command", "finished", "error", "started_at",
                "dataset_tokens", "tokens_budget", "last_status_read",
                "returncode", "pid", "log_offset", "train_config"):
        st.session_state[key] = DEFAULT_STATE[key]
    st.session_state["proc"] = None
    st.session_state["queue"] = None
    st.session_state["reader"] = None


# ---------------------------------------------------------------------------
# Subprocess stdout pump and parsers
# ---------------------------------------------------------------------------

RE_INIT = re.compile(r"tokens=(\d+)\s+vocab=(\d+)\s+windows=(\d+)")
RE_STEP = re.compile(
    r"epoch=(\d+)\s+step=(\d+)\s+loss=([0-9.]+)\s+ppl=([0-9.]+)")
RE_STEP_NOPPL = re.compile(
    r"epoch=(\d+)\s+step=(\d+)\s+loss=([0-9.]+)")
RE_VAL = re.compile(r"val_loss=([0-9.]+)\s+val_ppl=([0-9.]+)")
RE_ETA = re.compile(r"eta=([0-9.]+)min")
RE_DEVICE = re.compile(r"device=([A-Za-z0-9_]+)")
RE_MLP_EVAL = re.compile(r"top1 train=([0-9.]+) val=([0-9.]+)")


def pump_stdout(stream, output_queue: queue.Queue) -> None:
    """Read the child's stdout line by line and put it on a thread-safe queue.

    The training subprocess is started with `-u` (unbuffered) and
    `PYTHONUNBUFFERED=1`; without that, Python buffers stdout when it is a pipe
    and the UI would only see output at exit.
    """
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line.rstrip("\r\n"))
    finally:
        output_queue.put(None)


def merge_metrics(points: list[dict]) -> None:
    by_step = {m.get("step"): m
               for m in st.session_state.get("metrics", [])
               if "step" in m}
    val_by_step = {m.get("step"): m
                   for m in st.session_state.get("val_points", [])
                   if "step" in m}
    for point in points:
        step = point.get("step")
        if step is None:
            continue
        if point.get("val"):
            # Keep validation points separate: they must not overwrite the
            # train loss at the same step on the loss chart.
            val_by_step[step] = point
            if point.get("ppl") is not None:
                st.session_state.status["val_ppl"] = point["ppl"]
            continue
        by_step[step] = point
    st.session_state.metrics = sorted(
        by_step.values(), key=lambda m: m.get("step", 0))[-5000:]
    st.session_state.val_points = sorted(
        val_by_step.values(), key=lambda m: m.get("step", 0))[-200:]


def add_metric(point: dict) -> None:
    merge_metrics([point])


def parse_line(line: str) -> None:
    """Parse one stdout line into log/status/metrics state."""
    st.session_state.log.append(line)
    if len(st.session_state.log) > 500:
        st.session_state.log = st.session_state.log[-500:]

    status = st.session_state.status

    m = RE_INIT.search(line)
    if m:
        tokens, vocab, windows = (int(x) for x in m.groups())
        status["tokens"] = tokens
        status["vocab"] = vocab
        status["windows"] = windows
        batch = max(1, int(st.session_state.get("batch", 32)))
        status.setdefault("steps_per_epoch", max(1, windows // batch))
        status.setdefault(
            "total_steps",
            status["steps_per_epoch"] * st.session_state.get("epochs", 1))

    m = RE_DEVICE.search(line)
    if m:
        status["device"] = m.group(1)

    m = RE_STEP.search(line)
    if m:
        epoch, step, loss, ppl = m.groups()
        status.update({"epoch": int(epoch), "step": int(step),
                       "last_loss": float(loss), "last_ppl": float(ppl)})
        add_metric({"step": int(step), "epoch": int(epoch),
                    "loss": float(loss), "ppl": float(ppl)})
    else:
        m = RE_STEP_NOPPL.search(line)
        if m:
            epoch, step, loss = m.groups()
            ppl = math.exp(min(float(loss), 20.0))
            status.update({"epoch": int(epoch), "step": int(step),
                           "last_loss": float(loss), "last_ppl": ppl})
            add_metric({"step": int(step), "epoch": int(epoch),
                        "loss": float(loss), "ppl": ppl})

    m = RE_VAL.search(line)
    if m:
        loss, ppl = m.groups()
        status["val_loss"] = float(loss)
        status["val_ppl"] = float(ppl)
        if "step" in status:
            add_metric({"step": int(status["step"]),
                        "loss": float(status.get("last_loss", 0.0)),
                        "ppl": float(ppl), "val": 1})

    m = RE_ETA.search(line)
    if m:
        status["eta_s"] = float(m.group(1)) * 60.0

    m = RE_MLP_EVAL.search(line)
    if m:
        status["train_top1"] = float(m.group(1))
        status["val_top1"] = float(m.group(2))

    lower = line.lower()
    if "traceback" in lower or "permissionerror" in lower:
        st.session_state.error = line.strip()
    elif "error:" in lower and "error=" not in lower:
        st.session_state.error = line.strip()


def drain_queue() -> None:
    q = st.session_state.get("queue")
    if q is None:
        return
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            st.session_state.finished = True
            continue
        parse_line(line)


def read_json_safe(path: Path, retries: int = 5) -> dict:
    for _ in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except PermissionError:
            time.sleep(0.02)
    return {}


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    points = []
    try:
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except PermissionError:
        return []
    return points


def refresh_from_files(allow_status: bool = False) -> None:
    """Merge metrics.jsonl, and (only after exit) status.json, into state.

    While the process is alive we deliberately do **not** open status.json:
    on Windows a reader handle can make an unpatched trainer's os.replace()
    fail.  stdout + metrics.jsonl provide everything the live UI needs;
    status.json is read once the process has exited.
    """
    run_dir = st.session_state.get("run_dir")
    if not run_dir:
        return
    run_path = Path(run_dir)

    if allow_status:
        status = read_json_safe(run_path / "status.json")
        if status:
            st.session_state.status.update(status)
            if status.get("state") == "done":
                st.session_state.finished = True

    points = read_metrics(run_path / "metrics.jsonl")
    if points:
        merge_metrics(points)


def sync_process() -> None:
    """Tail the detached run's log and metrics; detect exit."""
    pid = int(st.session_state.get("pid") or 0)
    run_dir = st.session_state.get("run_dir")
    if not pid or not run_dir:
        return
    alive = pid_alive(pid)
    lines, offset = read_log_from(Path(run_dir) / "train.log",
                                  int(st.session_state.get("log_offset", 0)))
    if lines:
        st.session_state["log_offset"] = offset
        for line in lines:
            parse_line(line)
    refresh_from_files(allow_status=not alive)
    if not alive:
        st.session_state["pid"] = 0
        st.session_state["finished"] = True
        state = st.session_state.get("status", {}).get("state")
        if state != "done" and not st.session_state.get("error"):
            st.session_state["error"] = (
                "训练进程已退出（被终止或异常），请查看日志。")


# ---------------------------------------------------------------------------
# Uploaded data preparation
# ---------------------------------------------------------------------------

def _clean_lines(raw_lines, clean_opts: Optional[dict] = None,
                 limit_chars: Optional[int] = None):
    """Apply the cleaning pipeline to an iterable of raw text lines.

    Rules run in a fixed order: custom regex -> strip punctuation -> strip
    digits/Latin -> split long lines -> dedupe.  ``limit_chars`` stops early
    so huge corpora are not converted in full when only a token budget is
    needed.
    """
    opts = clean_opts or {}
    pattern = None
    regex_src = str(opts.get("regex") or "").strip()
    if regex_src:
        try:
            pattern = re.compile(regex_src)
        except re.error as exc:
            raise ValueError(f"自定义正则无效：{exc}") from exc

    seen = set()
    written = 0
    out = []
    for line in raw_lines:
        compact = re.sub(r"\s+", "", line)
        if not compact:
            continue
        if pattern is not None:
            compact = re.sub(r"\s+", "", pattern.sub("", compact))
        if not compact:
            continue
        if opts.get("strip_punct"):
            compact = "".join(
                ch for ch in compact
                if "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff")
        if opts.get("strip_alnum"):
            compact = re.sub(r"[0-9A-Za-z]", "", compact)
        if not compact:
            continue
        pieces = [compact]
        if opts.get("split_long") and len(compact) > 64:
            pieces = [compact[i:i + 64] for i in range(0, len(compact), 64)]
        for piece in pieces:
            if opts.get("dedup"):
                if piece in seen:
                    continue
                seen.add(piece)
            out.append(" ".join(piece))
            written += len(piece)
            if limit_chars and written >= limit_chars:
                return out
    return out


def _iter_file_lines(path: Path):
    """Yield raw text lines from .txt, .jsonl or .jsonl.gz.

    LCCC-style JSONL records are lists of utterances; each utterance becomes
    its own line so the character LM sees sentence boundaries.
    """
    name = path.name.lower()
    if name.endswith(".gz"):
        opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")  # noqa: E731
    else:
        opener = lambda: io.open(path, encoding="utf-8", errors="replace")  # noqa: E731
    with opener() as fh:
        for line in fh:
            if not name.endswith((".jsonl", ".jsonl.gz", ".json")):
                yield line
                continue
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(record, list):
                for utterance in record:
                    if isinstance(utterance, str):
                        yield utterance
            elif isinstance(record, dict):
                for key in ("content", "sentence", "text", "dialog"):
                    value = record.get(key)
                    if isinstance(value, str):
                        yield value
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                yield item
                conversations = record.get("conversations")
                if isinstance(conversations, list):
                    for turn in conversations:
                        if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                            yield turn["content"]


def resolve_local_paths(raw: str, kind: str) -> list[Path]:
    """Split a user-entered path list; expand directories to matching files."""
    parts = [piece.strip().strip('"') for piece in re.split(r"[;\n]", raw or "")
             if piece.strip()]
    if not parts:
        raise ValueError("请填写本地语料路径（文件或目录）。")
    suffixes = ((".txt", ".text") if kind == "text"
                else (".csv", ".jsonl", ".jsonl.gz"))
    if kind == "text":
        suffixes = (".txt", ".text", ".jsonl", ".jsonl.gz")
    files: list[Path] = []
    for part in parts:
        path = Path(part).expanduser()
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.name.lower().endswith(suffixes):
                    files.append(child)
        elif path.is_file():
            if not path.name.lower().endswith(suffixes):
                raise ValueError(f"不支持的文件类型：{path.name}")
            files.append(path)
        else:
            raise ValueError(f"路径不存在：{path}")
    if not files:
        raise ValueError("该路径下没有匹配的数据文件。")
    return files


def save_local_corpus(paths: list[Path], run_dir: Path,
                      clean_opts: Optional[dict] = None,
                      limit_chars: Optional[int] = None) -> Path:
    """Build corpus.txt straight from local files (no browser upload)."""
    def raw_lines():
        for path in paths:
            yield from _iter_file_lines(path)

    lines = _clean_lines(raw_lines(), clean_opts, limit_chars)
    if not lines:
        raise ValueError("清洗后语料为空：请检查清洗选项或路径。")
    path = run_dir / "corpus.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prepare_local_data(paths: list[Path], kind: str, run_dir: Path,
                       clean_opts: Optional[dict] = None,
                       limit_chars: Optional[int] = None) -> Path:
    if kind == "text":
        return save_local_corpus(paths, run_dir, clean_opts, limit_chars)
    out = run_dir / "events.jsonl"
    count = 0
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        for path in paths:
            if path.name.lower().endswith(".jsonl"):
                opener = (gzip.open if path.name.lower().endswith(".gz")
                          else io.open)
                with opener(path, "rt", encoding="utf-8", errors="replace") as src:
                    for line in src:
                        text = line.strip()
                        if text:
                            fh.write(text + "\n")
                            count += 1
            else:
                frame = pd.read_csv(path, dtype=str)
                for _, row in frame.iterrows():
                    try:
                        scores = json.loads(row.get("scores", ""))
                    except (TypeError, ValueError):
                        continue
                    fh.write(json.dumps({
                        "input": row.get("input", ""),
                        "selected_text": row.get("selected_text", ""),
                        "selected_index": int(row.get("selected_index", -1) or -1),
                        "prev_committed": row.get("prev_committed", ""),
                        "scores": scores,
                    }, ensure_ascii=False) + "\n")
                    count += 1
    if not count:
        raise ValueError("路径里没有可用的选择事件。")
    return out


def _as_uploads(upload) -> list:
    """Normalise a single uploader value or a multi-file list into a list."""
    if upload is None:
        return []
    if isinstance(upload, (list, tuple)):
        return [item for item in upload if item is not None]
    return [upload]


def save_text_corpus(upload, run_dir: Path,
                     clean_opts: Optional[dict] = None) -> Path:
    """Normalise uploaded .txt files to one character-tokenised line each."""
    parts = []
    for item in _as_uploads(upload):
        parts.append(item.getvalue().decode("utf-8-sig", errors="replace"))
    raw_lines = (line for chunk in parts for line in chunk.splitlines())
    lines = _clean_lines(raw_lines, clean_opts)
    if not lines:
        raise ValueError("清洗后语料为空：请检查清洗选项或更换文件。")
    path = run_dir / "corpus.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_event_data(upload, run_dir: Path) -> Path:
    """Accept selection_events .jsonl directly, or .csv with a JSON scores column.

    Multiple uploads are merged: every .jsonl is concatenated, every .csv is
    parsed and appended to the same output file.
    """
    uploads = _as_uploads(upload)
    if all(item.name.lower().endswith(".jsonl") for item in uploads) and uploads:
        path = run_dir / "events.jsonl"
        with io.open(path, "wb") as fh:
            for item in uploads:
                data = item.getvalue()
                fh.write(data if data.endswith(b"\n") else data + b"\n")
        return path
    upload = uploads[0]
    raw = upload.getvalue()
    name = upload.name.lower()
    if name.endswith(".jsonl"):
        path = run_dir / "events.jsonl"
        path.write_bytes(raw)
        return path

    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        raw_scores = row.get("scores") or row.get("candidates") or ""
        if not raw_scores.strip():
            raise ValueError(
                "CSV 缺少 `scores` 列（JSON 数组）。MLP 需要每个候选的 "
                "word/features；仅 top 列表不够。")
        try:
            scores = json.loads(raw_scores)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"scores 不是合法 JSON: {exc}") from exc
        if not isinstance(scores, list) or len(scores) < 2:
            raise ValueError("每个事件的 scores 至少需要 2 个候选。")
        if not all(isinstance(item, dict) for item in scores):
            raise ValueError(
                "scores 必须是候选字典数组（每项含 word/features）。")
        selected_text = row.get("selected_text", "")
        selected_index = row.get("selected_index", "")
        try:
            selected_index = int(selected_index)
        except (TypeError, ValueError):
            selected_index = -1
        event = {
            "input": row.get("input", ""),
            "selected_text": selected_text,
            "selected_index": selected_index,
            "prev_committed": row.get("prev_committed", ""),
            "scores": scores,
        }
        rows.append(event)
    # extra csv/jsonl files selected together are appended to the same file
    for item in uploads[1:]:
        if item.name.lower().endswith(".csv"):
            extra = pd.read_csv(io.BytesIO(item.getvalue()))
            for _, row in extra.iterrows():
                try:
                    scores = json.loads(row.get("scores", ""))
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "input": row.get("input", ""),
                    "selected_text": row.get("selected_text", ""),
                    "selected_index": int(row.get("selected_index", -1) or -1),
                    "prev_committed": row.get("prev_committed", ""),
                    "scores": scores,
                })
    if not rows:
        raise ValueError("CSV 没有可用事件。")
    path = run_dir / "events.jsonl"
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        for event in rows:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def prepare_upload(upload, kind: str, run_dir: Path,
                   clean_opts: Optional[dict] = None) -> Path:
    uploads = _as_uploads(upload)
    if not uploads:
        raise ValueError("请先上传训练数据。")
    if kind == "text":
        if not all(item.name.lower().endswith((".txt", ".text"))
                   for item in uploads):
            raise ValueError("GPT-2/LSTM 训练需要 .txt 纯文本。")
        return save_text_corpus(uploads, run_dir, clean_opts)
    if not all(item.name.lower().endswith((".csv", ".jsonl"))
               for item in uploads):
        raise ValueError("MLP 训练需要 selection_events 的 .csv 或 .jsonl。")
    return save_event_data(uploads, run_dir)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_command(arch_key: str, data_path: Path, run_dir: Path,
                  epochs: int, batch: int, tokens: int,
                  use_gpu: bool,
                  arch_params: Optional[dict] = None,
                  resume: bool = False,
                  max_vocab: int = 13000, val_ratio: float = 0.02,
                  learning_rate: float = 3e-4,
                  warmup: int = 300) -> list[str]:
    trainer = trainer_path(arch_key)
    if trainer is None:
        raise FileNotFoundError(
            f"找不到 {arch_key} 训练器；请检查 ZIMO_TRAINER_ROOT。")
    base = [sys.executable, "-u", str(trainer)]
    ap = arch_params or {}
    layers_v = int(ap.get("layers", 6))
    dim_v = int(ap.get("dim", 192))
    heads_v = int(ap.get("heads", 4))
    block_v = int(ap.get("block", 64))

    if arch_key == "gpt2":
        resume_args = ["--resume"] if resume else []
        return base + resume_args + [
            "--corpus", str(data_path),
            "--tokens", str(tokens),
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--block", str(block_v),
            "--dim", str(dim_v),
            "--layers", str(layers_v),
            "--heads", str(heads_v),
            "--lr", f"{learning_rate:g}",
            "--warmup", str(int(warmup)),
            "--min-count", "20",
            "--max-vocab", str(int(max_vocab)),
            "--val-ratio", f"{val_ratio:g}",
            "--save-every", "500",
            "--device", "cuda" if use_gpu else "cpu",
            "--out", str(run_dir),
            "--metrics", str(run_dir / "metrics.jsonl"),
            "--status", str(run_dir / "status.json"),
        ]

    if arch_key == "lstm":
        resume_args = ["--resume"] if resume else []
        return base + resume_args + [
            "--corpus", str(data_path),
            "--tokens", str(tokens),
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--bptt", "64",
            "--dim", str(dim_v),
            "--hidden", "512",
            "--layers", str(layers_v),
            "--lr", f"{learning_rate:g}",
            "--min-count", "3",
            "--max-vocab", str(int(max_vocab)),
            "--out", str(run_dir),
            "--metrics", str(run_dir / "metrics.jsonl"),
            "--status", str(run_dir / "status.json"),
        ]

    # MLP listwise ranker: feature events, not raw text.
    return base + [
        "--events", str(data_path),
        "--output", str(run_dir / "l1_ranker.bin"),
        "--hidden", "16",
        "--epochs", str(epochs),
        "--min-events", "10",  # demo-friendly; production should use >=1000
    ]


def start_training(upload, arch_key: str, arch_kind: str, epochs: int,
                   batch: int, tokens: int, use_gpu: bool,
                   arch_params: Optional[dict] = None,
                   clean_opts: Optional[dict] = None,
                   local_paths: Optional[list] = None,
                   max_vocab: int = 13000, val_ratio: float = 0.02,
                   learning_rate: float = 3e-4, warmup: int = 300) -> None:
    if is_running():
        st.error("已有训练进程在运行，请先停止或等待结束。")
        return
    if not local_paths and not _as_uploads(upload):
        st.error("请填写本地语料路径，或展开「或上传小样本」选择文件。")
        return
    if trainer_path(arch_key) is None:
        st.error("当前项目里找不到对应的 trainer，请设置 ZIMO_TRAINER_ROOT。")
        return
    if use_gpu and arch_key == "gpt2" and not torch.cuda.is_available():
        st.error("选择了 GPU，但当前 PyTorch 检测不到 CUDA。")
        return
    if (arch_key == "gpt2" and arch_params
            and int(arch_params.get("dim", 192)) % int(arch_params.get("heads", 4)) != 0):
        st.error("维度必须能被注意力头数整除（dim % heads == 0），请调整后重试。")
        return

    reset_run_state()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{arch_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if local_paths:
            data_path = prepare_local_data(
                local_paths, arch_kind, run_dir, clean_opts,
                limit_chars=int(tokens * 1.1) if arch_kind == "text" else None)
        else:
            data_path = prepare_upload(upload, arch_kind, run_dir, clean_opts)
        dataset_tokens = 0
        if arch_kind == "text":
            dataset_tokens = sum(
                len(line.split())
                for line in data_path.read_text(encoding="utf-8").splitlines())
            dataset_tokens = min(dataset_tokens, int(tokens))
            distinct = len({
                ch for line in data_path.read_text(encoding="utf-8").splitlines()
                for ch in line.replace(" ", "")})
            if dataset_tokens < 5000 or distinct < 100:
                st.error(tr(
                    f"语料太小（{dataset_tokens:,} tokens / {distinct} 种字符）。"
                    "训练器的 min_count=20 会把低频字全部过滤，模型学不到东西；"
                    "建议至少 10 万 token、数百种字符。",
                    f"Corpus too small ({dataset_tokens:,} tokens / {distinct} "
                    "distinct chars). min_count=20 filters rare characters, so "
                    "the model learns nothing; use 100k+ tokens."))
                return
        st.session_state.dataset_tokens = dataset_tokens
        st.session_state.tokens_budget = int(tokens)
        command = build_command(arch_key, data_path, run_dir, epochs, batch,
                                tokens, use_gpu, arch_params,
                                max_vocab=max_vocab, val_ratio=val_ratio,
                                learning_rate=learning_rate, warmup=warmup)
    except Exception as exc:  # noqa: BLE001 - surface input errors in the UI
        st.error(str(exc))
        return

    run_config = {
        "arch": arch_key, "arch_kind": arch_kind, "epochs": int(epochs),
        "batch": int(batch), "tokens": int(tokens), "use_gpu": bool(use_gpu),
        "arch_params": arch_params or {}, "clean_opts": clean_opts or {},
        "corpus": str(data_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_run_config(run_dir, run_config)
    try:
        pid = launch_detached(command, TRAINER_ROOT, run_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"启动训练失败：{exc}")
        return

    st.session_state.pid = pid
    st.session_state.log_offset = 0
    st.session_state.train_config = run_config
    st.session_state.proc = None
    st.session_state.queue = None
    st.session_state.reader = None
    st.session_state.run_dir = str(run_dir)
    st.session_state.arch_key = arch_key
    st.session_state.command = " ".join(shlex.quote(x) for x in command)
    st.session_state.started_at = time.time()
    st.session_state.epochs = epochs
    st.session_state.batch = batch
    st.rerun()


def stop_training() -> None:
    pid = int(st.session_state.get("pid") or 0)
    if pid and pid_alive(pid):
        terminate_pid(pid)
        st.session_state.error = "已手动停止训练。"
    st.session_state.pid = 0
    st.session_state.finished = True
    st.rerun()


def resume_training(run_dir: Path, arch_key: str, epochs: int, batch: int,
                    tokens: int, use_gpu: bool,
                    arch_params: Optional[dict] = None) -> None:
    """Continue an existing run from its checkpoint (``--resume``)."""
    if is_running():
        st.error("已有训练在运行，请先停止。")
        return
    corpus = run_dir / "corpus.txt"
    events = run_dir / "events.jsonl"
    if corpus.is_file():
        data_path, arch_kind = corpus, "text"
    elif events.is_file():
        data_path, arch_kind = events, "events"
    else:
        st.error(f"这个运行目录里没有 corpus.txt / events.jsonl，无法续训：{run_dir}")
        return
    config = read_run_config(run_dir)
    if config.get("arch_params") and not arch_params:
        arch_params = config["arch_params"]
    command = build_command(arch_key, data_path, run_dir, epochs, batch, tokens,
                            use_gpu, arch_params, resume=True)
    try:
        pid = launch_detached(command, TRAINER_ROOT, run_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"续训启动失败：{exc}")
        return
    st.session_state.pid = pid
    st.session_state.log_offset = 0
    st.session_state.run_dir = str(run_dir)
    st.session_state.arch_key = arch_key
    st.session_state.command = " ".join(shlex.quote(x) for x in command)
    st.session_state.started_at = time.time()
    st.session_state.finished = False
    st.session_state.error = ""
    st.session_state.epochs = int(epochs)
    st.session_state.batch = int(batch)
    st.session_state.status = read_json_safe(run_dir / "status.json")
    st.session_state.log = []
    st.session_state.metrics = []
    st.session_state.val_points = []
    st.rerun()


# ---------------------------------------------------------------------------
# Model scoring (GPT-2 / LSTM)
# ---------------------------------------------------------------------------

class Block(nn.Module):
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
        return x + self.proj(F.gelu(self.fc(h)))


class GPTLM(nn.Module):
    def __init__(self, vocab: int, block: int, dim: int, layers: int,
                 heads: int):
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
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_f(h))


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, hidden: int = 512,
                 layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.drop = nn.Dropout(0.1)
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        e = self.drop(self.embed(x))
        h, hidden = self.lstm(e, hidden)
        return self.out(self.drop(h)), hidden


def _read_vocab(path: Path) -> dict[str, int]:
    words = [w for w in path.read_text(encoding="utf-8").splitlines() if w]
    return {word: index + 1 for index, word in enumerate(words)}


@st.cache_resource(show_spinner=False)
def load_scorer(arch_key: str, model_path: str, vocab_path: str = "",
                mtime: float = 0.0, vocab_words_json: str = ""):
    """Load a checkpoint in any supported format.

    Accepts ``.pt`` / ``.pth`` / ``.ckpt`` / ``.bin`` / ``.safetensors``.
    Vocabulary: explicit JSON list > sidecar vocab file > embedded list.
    Hyper-parameters: checkpoint keys > sidecar config.json > inferred from
    tensor shapes.  ``mtime`` / ``vocab_words_json`` act as cache keys.
    """
    del mtime
    info = load_state_any(Path(model_path))
    if not info["loadable"]:
        raise ValueError(info["error"] or "无法读取该模型文件")

    config = dict(info["config"])
    for key, value in infer_config_from_state(info["state"]).items():
        config.setdefault(key, value)

    embedded = info["vocab"]
    if vocab_words_json:
        words = json.loads(vocab_words_json)
        vocab = {word: index + 1 for index, word in enumerate(words)}
    elif vocab_path and Path(vocab_path).is_file():
        vocab_path_obj = Path(vocab_path)
        if vocab_path_obj.suffix.lower() == ".json":
            data = json.loads(vocab_path_obj.read_text(encoding="utf-8"))
            words = extract_vocab_words(data) or list(data.keys())
            vocab = {word: index + 1 for index, word in enumerate(words)}
        else:
            vocab = _read_vocab(vocab_path_obj)
    elif embedded:
        vocab = {word: index + 1 for index, word in enumerate(embedded)}
    else:
        raise ValueError("checkpoint 中没有词表，也未找到 vocab.txt / vocab.json。")

    if arch_key == "gpt2":
        block = int(config.get("block", 64))
        model = GPTLM(len(vocab) + 1, block, int(config["dim"]),
                      int(config["layers"]), int(config.get("heads", 4)))
    elif arch_key == "lstm":
        model = LSTMLM(len(vocab) + 1, int(config.get("dim", 256)),
                       int(config.get("hidden", 512)),
                       int(config.get("layers", 1)))
        block = int(config.get("block", 64))
    else:
        raise ValueError("unsupported scorer")

    state = info["state"]
    if any(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(
            "权重与结构不匹配：请确认模型类型选择正确，"
            "或该文件不是本工具训练出的模型。") from exc
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    try:
        file_mb = Path(model_path).stat().st_size / 1048576
    except OSError:
        file_mb = 0.0
    meta = {
        "params": params,
        "layers": int(config.get("layers", 0)),
        "dim": int(config.get("dim", 0)),
        "heads": int(config.get("heads", 0)) if arch_key == "gpt2" else 0,
        "hidden": int(config.get("hidden", 0)) if arch_key == "lstm" else 0,
        "block": block,
        "file_mb": file_mb,
        "format": info["format"],
    }
    return {"kind": arch_key, "model": model, "vocab": vocab,
            "block": block, "meta": meta}


def _char_ids(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(ch, 0) for ch in text if not ch.isspace()]


@torch.no_grad()
def score_candidate(scorer: dict, context: str, candidate: str) -> Optional[dict]:
    """Mean NLL / PPL of the candidate characters, conditioned on context.

    We use a sliding window of `block` tokens so long candidates are still
    scored correctly.  The first candidate character is skipped when there is
    no preceding token at all (the model has no BOS token).
    """
    ctx = _char_ids(context, scorer["vocab"])
    cand = _char_ids(candidate, scorer["vocab"])
    if not cand:
        return None
    ids = ctx + cand
    ctx_len = len(ctx)
    model = scorer["model"]
    kind = scorer["kind"]
    block = max(2, int(scorer["block"]))
    nlls: list[float] = []

    for pos in range(ctx_len, len(ids)):
        start = max(0, pos - block + 1)
        window = ids[start:pos + 1]
        if len(window) < 2:
            continue
        x = torch.tensor([window[:-1]], dtype=torch.long)
        y = torch.tensor([window[-1]], dtype=torch.long)
        out = model(x)
        logits = out[0] if kind == "lstm" else out
        last = logits[0, -1]
        nll = F.cross_entropy(last.unsqueeze(0), y, reduction="sum").item()
        nlls.append(nll)

    if not nlls:
        return None
    mean_nll = sum(nlls) / len(nlls)
    return {
        "ppl": math.exp(min(mean_nll, 20.0)),
        "mean_nll": mean_nll,
        "scored_chars": len(nlls),
    }


@torch.no_grad()
def char_nlls(scorer: dict, context: str, text: str) -> list[dict]:
    """Per-character NLL for `text`, conditioned on `context`."""
    ctx = _char_ids(context, scorer["vocab"])
    cand = _char_ids(text, scorer["vocab"])
    if not cand:
        return []
    ids = ctx + cand
    chars = [ch for ch in text if not ch.isspace()]
    model = scorer["model"]
    kind = scorer["kind"]
    block = max(2, int(scorer["block"]))
    rows = []
    for pos in range(len(ctx), len(ids)):
        offset = pos - len(ctx)
        start = max(0, pos - block + 1)
        window = ids[start:pos + 1]
        if len(window) < 2:
            continue
        x = torch.tensor([window[:-1]], dtype=torch.long)
        y = torch.tensor([window[-1]], dtype=torch.long)
        out = model(x)
        logits = out[0] if kind == "lstm" else out
        nll = float(F.cross_entropy(logits[0, -1].unsqueeze(0), y,
                                    reduction="sum"))
        rows.append({"position": offset + 1,
                     "char": chars[offset] if offset < len(chars) else "?",
                     "nll": round(nll, 4),
                     "prob": round(math.exp(-min(nll, 20.0)), 5)})
    return rows


@torch.no_grad()
def _top_chars(scorer: dict, prefix_ids: list[int], k: int = 3) -> list[tuple[str, float]]:
    """Most likely next characters after a prefix."""
    block = max(2, int(scorer["block"]))
    window = prefix_ids[-block:]
    if not window:
        return []
    x = torch.tensor([window], dtype=torch.long)
    out = scorer["model"](x)
    logits = out[0] if scorer["kind"] == "lstm" else out
    probs = torch.softmax(logits[0, -1], dim=-1)
    values, indices = torch.topk(probs, min(k, probs.numel()))
    inverse = {index: word for word, index in scorer["vocab"].items()}
    return [(inverse.get(int(i), "?"), float(v))
            for v, i in zip(values, indices)]


def suggest_corrections(scorer: dict, text: str, context: str = "",
                        max_edits: int = 3, candidate_k: int = 3,
                        min_gain: float = 0.05) -> list[dict]:
    """Greedy single-character corrections, accepted only when they lower PPL.

    For each position we take the model's top-k next characters, re-score the
    whole sentence with that one replacement, and keep the best edit only if it
    improves PPL by at least ``min_gain`` (5%).  Repeats for a few passes, so a
    genuine typo is fixed while correct-but-uncommon characters survive.
    """
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) < 2:
        return []
    ctx = _char_ids(context, scorer["vocab"])

    def score(sentence: str) -> Optional[float]:
        result = score_candidate(scorer, context, sentence)
        return result["ppl"] if result else None

    current = "".join(chars)
    best_ppl = score(current)
    if best_ppl is None:
        return []
    edits: list[dict] = []
    for _ in range(max(1, int(max_edits))):
        best_edit = None
        for pos in range(len(chars)):
            prefix = ctx + _char_ids(current[:pos], scorer["vocab"])
            if not prefix:
                continue
            for cand, cand_prob in _top_chars(scorer, prefix, candidate_k):
                if not cand or cand == chars[pos]:
                    continue
                trial_chars = list(chars)
                trial_chars[pos] = cand
                trial = "".join(trial_chars)
                trial_ppl = score(trial)
                if trial_ppl is None:
                    continue
                if trial_ppl > best_ppl * (1.0 - min_gain):
                    continue
                if best_edit is None or trial_ppl < best_edit["ppl"]:
                    best_edit = {"pos": pos + 1, "char": chars[pos],
                                 "suggest": cand, "ppl": trial_ppl,
                                 "p_suggest": round(cand_prob, 6)}
        if best_edit is None:
            break
        chars[best_edit["pos"] - 1] = best_edit["suggest"]
        current = "".join(chars)
        before = best_ppl
        best_ppl = best_edit["ppl"]
        edits.append({
            "pos": best_edit["pos"],
            "char": best_edit["char"],
            "suggest": best_edit["suggest"],
            "ppl_before": round(before, 2),
            "ppl_after": round(best_ppl, 2),
            "p_suggest": best_edit["p_suggest"],
        })
    return edits


def apply_corrections(text: str, hints: list[dict]) -> str:
    """Apply hints (1-based positions over non-space characters)."""
    chars = [ch for ch in text if not ch.isspace()]
    for hint in hints:
        index = int(hint["pos"]) - 1
        if 0 <= index < len(chars):
            chars[index] = hint["suggest"]
    return "".join(chars)


@torch.no_grad()
def generate_text(scorer: dict, prompt: str, max_new: int = 40,
                  temperature: float = 0.8, top_k: int = 0) -> str:
    """Autoregressive continuation; temperature <= 0 means greedy."""
    model = scorer["model"]
    kind = scorer["kind"]
    vocab = scorer["vocab"]
    block = max(2, int(scorer["block"]))
    inverse = {index: word for word, index in vocab.items()}
    ids = _char_ids(prompt, vocab)
    if not ids:
        ids = [0]
    produced = []
    for _ in range(max(1, int(max_new))):
        window = ids[-block:]
        x = torch.tensor([window], dtype=torch.long)
        out = model(x)
        logits = out[0] if kind == "lstm" else out
        last = logits[0, -1]
        if temperature <= 0:
            token = int(torch.argmax(last))
        else:
            scaled = last / max(1e-3, float(temperature))
            if top_k and 0 < top_k < scaled.numel():
                values, indices = torch.topk(scaled, int(top_k))
                probs = torch.softmax(values, dim=-1)
                token = int(indices[int(torch.multinomial(probs, 1))])
            else:
                probs = torch.softmax(scaled, dim=-1)
                token = int(torch.multinomial(probs, 1))
        ids.append(token)
        produced.append(inverse.get(token, "？"))
    return "".join(produced)


def find_model(run_dir: Path, arch_key: str) -> tuple[Optional[Path], Optional[Path]]:
    if arch_key in ("gpt2", "lstm"):
        model = None
        for name in ("model_best.pt", "model.pt"):
            candidate = run_dir / name
            if candidate.is_file():
                model = candidate
                break
        vocab = run_dir / "vocab.txt"
        return model, (vocab if vocab.is_file() else None)
    model = run_dir / "l1_ranker.bin"
    return (model if model.is_file() else None), None


def read_log_tail(path: Path, max_lines: int = 80) -> list[str]:
    if not path.is_file():
        return []
    try:
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\r\n") for line in fh.readlines()[-max_lines:]]
    except OSError:
        return []


def open_directory(path: str | Path) -> None:
    """Open a folder in the local file manager (studio is a local tool)."""
    target = str(Path(path).resolve())
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", target])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def load_existing_run(run_dir: str, arch_key: str) -> None:
    """Adopt an existing run directory so its model can be inspected/scored."""
    path = Path(run_dir).expanduser()
    if not path.is_dir():
        st.error(f"目录不存在：{path}")
        return
    model, _ = find_model(path, arch_key)
    if model is None:
        st.error("这个目录里没有 model_best.pt / model.pt / l1_ranker.bin。")
        return
    reset_run_state()
    st.session_state.proc = None
    st.session_state.queue = None
    st.session_state.reader = None
    st.session_state.run_dir = str(path.resolve())
    st.session_state.arch_key = arch_key
    st.session_state.status = read_json_safe(path / "status.json")
    merge_metrics(read_metrics(path / "metrics.jsonl"))
    st.session_state.log = read_log_tail(path / "train.log")
    st.session_state.finished = True
    st.session_state.error = ""
    st.session_state.command = f"loaded existing run: {path}"
    st.rerun()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "–"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def fmt_compact(n: Optional[float]) -> str:
    if n is None or not math.isfinite(n):
        return "–"
    a = abs(n)
    if a >= 1e9:
        return f"{n / 1e9:.2f}B"
    if a >= 1e6:
        return f"{n / 1e6:.2f}M"
    if a >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(int(round(n * 10) / 10))


_THEMES = {
    "dark": {
        "bg": "#0b0f16",
        "panel": "#131a26",
        "panel2": "#18202e",
        "line": "rgba(148,163,184,.10)",
        "fg": "#e8eefc",
        "dim": "#93a4c3",
        "muted": "#64748b",
        "ring_track": "rgba(148,163,184,.14)",
        "ring_inner": "#131a26",
        "box_bg": "#0f1622",
        "box_text": "#dbe4f2",
        "sidebar_bg": "#0e131c",
        "input_bg": "#0f1622",
        "input_text": "#e8eefc",
        "input_border": "rgba(148,163,184,.18)",
        "code_bg": "#0f1622",
        "code_text": "#dbe4f2",
        "log_val": "#34d399",
        "log_epoch": "#8fb8ff",
        "log_done": "#67e8f9",
        "log_warn": "#fbbf24",
        "sec_btn_bg": "#18202e",
        "sec_btn_border": "rgba(148,163,184,.16)",
        "sec_btn_text": "#e8eefc",
        "tag_color": "#9fc0ff",
        "note_text": "#93a4c3",
        "dialog_bg": "#131a26",
        "alert_bg": "#131a26",
        "accent": "#5b8def",
        "accent_hover": "#6f9bf2",
        "accent_text": "#0b0f16",
    },
    "light": {
        "bg": "#f4f6fa",
        "panel": "#ffffff",
        "panel2": "#eef1f7",
        "line": "rgba(15,23,42,.08)",
        "fg": "#0f172a",
        "dim": "#475569",
        "muted": "#64748b",
        "ring_track": "rgba(15,23,42,.08)",
        "ring_inner": "#ffffff",
        "box_bg": "#f2f5f9",
        "box_text": "#1e293b",
        "sidebar_bg": "#ffffff",
        "input_bg": "#ffffff",
        "input_text": "#0f172a",
        "input_border": "rgba(15,23,42,.16)",
        "code_bg": "#f2f5f9",
        "code_text": "#1e293b",
        "log_val": "#047857",
        "log_epoch": "#2f6feb",
        "log_done": "#0e7490",
        "log_warn": "#b45309",
        "sec_btn_bg": "#ffffff",
        "sec_btn_border": "rgba(15,23,42,.14)",
        "sec_btn_text": "#0f172a",
        "tag_color": "#2f6feb",
        "note_text": "#475569",
        "dialog_bg": "#ffffff",
        "alert_bg": "#f2f5f9",
        "accent": "#2f6feb",
        "accent_hover": "#275fd2",
        "accent_text": "#ffffff",
    },
}


def inject_css() -> None:
    """Theme-aware CSS following the house rules: flat colour fields, tone
    steps instead of borders/shadows/gradients, no eyebrows, text face for
    numbers, square-ish buttons.  Light theme also overrides Streamlit's
    native dark chrome (dialog, dataframe canvas, code tokens, alerts)."""
    theme = st.session_state.get("theme", "dark")
    t = _THEMES["light" if theme == "light" else "dark"]
    light = theme == "light"
    css = """
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  [data-testid="stAppViewContainer"] {
    background: __BG__;
    color: __FG__;
  }
  [data-testid="stHeader"] { background: __BG__; }
  [data-testid="stSidebar"] {
    background: __SIDEBAR__;
  }
  [data-testid="stSidebar"] * { color: __FG__; }
  [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color: __DIM__; }
  [data-testid="stMarkdownContainer"] { color: __FG__; }
  [data-testid="stMetricValue"] {
    color: __FG__; font-variant-numeric: tabular-nums; font-weight: 650;
  }
  [data-testid="stMetricLabel"] p { color: __MUTED__; }
  div[data-baseweb="input"], div[data-baseweb="textarea"] {
    background: __INPUT_BG__;
    border-color: __INPUT_BORDER__;
  }
  div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
    color: __INPUT_TEXT__ !important;
  }
  [data-baseweb="select"] > div { background: __INPUT_BG__; color: __INPUT_TEXT__; }
  [data-baseweb="popover"] [role="listbox"] { background: __INPUT_BG__; }
  [data-baseweb="popover"] [role="option"] { color: __INPUT_TEXT__; }
  [data-baseweb="tab-list"] { border-bottom: none; gap: 4px; }
  [data-baseweb="tab"] {
    color: __DIM__; border-radius: 8px 8px 0 0; padding: 8px 14px;
  }
  [data-baseweb="tab"][aria-selected="true"] {
    color: __FG__ !important;
    background: __PANEL2__;
  }
  [data-baseweb="tab-highlight"] { background-color: __ACCENT__; }
  [data-testid="stExpander"] { border: none; background: __PANEL__; }
  [data-testid="stExpander"] summary { color: __FG__; }
  div[data-testid="stCode"] pre, div[data-testid="stCode"] code {
    background: __CODE_BG__ !important; color: __CODE_TEXT__ !important;
  }
  div[data-testid="stAlert"] {
    background: __ALERT_BG__; color: __FG__;
    border: none; border-radius: 12px;
  }
  div[data-testid="stAlert"] p { color: __FG__; }
  [data-testid="stDialog"] div[data-testid="stVerticalBlock"], [role="dialog"] {
    background: __DIALOG_BG__;
    color: __FG__;
  }
  [role="dialog"] [data-testid="stMarkdownContainer"] { color: __FG__; }
  [role="dialog"] h1, [role="dialog"] h2, [role="dialog"] h3 { color: __FG__; }
  .stApp { color: __FG__; }
  h1, h2, h3 { color: __FG__; letter-spacing: -.02em; }
  .tj-title {
    font-size: 38px; line-height: 1.06; letter-spacing: -.035em;
    font-weight: 700; margin: 0 0 10px;
  }
  .tj-sub { color: __DIM__; font-size: 15px; max-width: 720px; }
  .tj-card {
    background: __PANEL__;
    border: none; border-radius: 14px;
    padding: 18px 20px; margin-top: 14px;
  }
  .tj-card b { color: __FG__; }
  div.stButton > button[kind="primary"],
  div[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: __ACCENT__; border: 1px solid __ACCENT__; color: __ACCENT_TEXT__;
    font-weight: 600; border-radius: 4px;
    box-shadow: none; transition: background .15s ease;
  }
  div.stButton > button[kind="primary"]:hover {
    background: __ACCENT_HOVER__; border-color: __ACCENT_HOVER__;
  }
  div.stButton > button[kind="secondary"],
  [data-testid="stBaseButton-secondary"] {
    background: __SEC_BTN_BG__ !important; border: 1px solid __SEC_BTN_BORDER__ !important;
    color: __SEC_BTN_TEXT__ !important; font-weight: 500; border-radius: 4px !important;
    box-shadow: none;
  }
  div.stButton > button[kind="secondary"]:hover,
  [data-testid="stBaseButton-secondary"]:hover {
    background: __PANEL2__ !important;
    border-color: __SEC_BTN_BORDER__ !important;
  }
  .tj-code {
    font-size: 12px; color: __DIM__; word-break: break-all;
  }
  .tj-monitor-head { display:flex; align-items:center; gap:20px; margin:2px 0 10px; }
  .tj-ring {
    width:96px; height:96px; border-radius:50%; flex:none;
    background: conic-gradient(__ACCENT__ calc(var(--pct) * 1%),
                              __RING_TRACK__ 0);
    display:grid; place-items:center; box-shadow: none;
  }
  .tj-ring span {
    width:74px; height:74px; border-radius:50%; background: __RING_INNER__;
    display:grid; place-items:center;
    font-size: 18px; font-weight: 700; color: __FG__;
    font-variant-numeric: tabular-nums;
  }
  .tj-monitor-meta .big { font-size:19px; font-weight:700; letter-spacing:-.02em; }
  .tj-monitor-meta .small { font-size:12px; color:__DIM__; margin-top:3px;
    font-variant-numeric: tabular-nums; }
  .tj-logbox {
    margin:0; max-height:300px; overflow-y:auto; padding:12px 14px;
    background: __BOX_BG__; border-radius:12px;
    font:12px/1.55 "Cascadia Code",ui-monospace,monospace;
    color: __BOX_TEXT__; white-space:pre-wrap; word-break:break-word;
  }
  .tj-logline { display:block; }
  .tj-logline.val { color:__LOG_VAL__; }
  .tj-logline.epoch { color:__LOG_EPOCH__; }
  .tj-logline.done { color:__LOG_DONE__; }
  .tj-logline.warn { color:__LOG_WARN__; }
  .tj-guidebox {
    max-height:300px; overflow-y:auto; padding:12px 14px;
    border-radius:12px;
    background: __BOX_BG__; font-size:13px; line-height:1.55;
    color: __BOX_TEXT__;
  }
  .tj-guidebox ul { margin:4px 0 10px 16px; padding:0; }
  .tj-guidebox li { margin:5px 0; }
  .tj-guidebox b { color: __FG__; }
  .tj-guide-camp { margin-top:8px; color: __FG__; font-weight:650; }
  .tj-guide-camp:first-child { margin-top:0; }
  .tj-tag {
    display:inline-block; margin-left:4px; padding:1px 6px;
    border-radius:999px;
    color: __TAG_COLOR__; font-size:11px;
  }
  .tj-guide-note {
    margin-top:10px; padding-top:8px;
    color: __NOTE_TEXT__;
  }
  .tj-version {
    font-size: 11px; color: __MUTED__; text-align: center; margin-top: 6px;
    font-variant-numeric: tabular-nums;
  }
"""
    if light:
        css += """
  [data-testid="stDataFrame"] {
    filter: invert(0.92) hue-rotate(180deg);
    border-radius: 12px; overflow: hidden;
  }
  div[data-testid="stCode"] span {
    color: __CODE_TEXT__ !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label,
  [data-testid="stSidebar"] [data-testid="stCheckbox"] label { color: __FG__; }
"""
    css = (css
           .replace("__BG__", t["bg"])
           .replace("__PANEL2__", t["panel2"])
           .replace("__PANEL__", t["panel"])
           .replace("__FG__", t["fg"])
           .replace("__SIDEBAR__", t["sidebar_bg"])
           .replace("__LINE__", t["line"])
           .replace("__DIM__", t["dim"])
           .replace("__MUTED__", t["muted"])
           .replace("__ACCENT_HOVER__", t["accent_hover"])
           .replace("__ACCENT__", t["accent"])
           .replace("__ACCENT_TEXT__", t["accent_text"])
           .replace("__INPUT_BG__", t["input_bg"])
           .replace("__INPUT_TEXT__", t["input_text"])
           .replace("__INPUT_BORDER__", t["input_border"])
           .replace("__CODE_BG__", t["code_bg"])
           .replace("__CODE_TEXT__", t["code_text"])
           .replace("__RING_TRACK__", t["ring_track"])
           .replace("__RING_INNER__", t["ring_inner"])
           .replace("__BOX_BG__", t["box_bg"])
           .replace("__BOX_TEXT__", t["box_text"])
           .replace("__LOG_VAL__", t["log_val"])
           .replace("__LOG_EPOCH__", t["log_epoch"])
           .replace("__LOG_DONE__", t["log_done"])
           .replace("__LOG_WARN__", t["log_warn"])
           .replace("__SEC_BTN_BG__", t["sec_btn_bg"])
           .replace("__SEC_BTN_BORDER__", t["sec_btn_border"])
           .replace("__SEC_BTN_TEXT__", t["sec_btn_text"])
           .replace("__TAG_COLOR__", t["tag_color"])
           .replace("__NOTE_TEXT__", t["note_text"])
           .replace("__DIALOG_BG__", t["dialog_bg"])
           .replace("__ALERT_BG__", t["alert_bg"]))
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)



def render_documentation() -> None:
    """Full in-app manual: a spec-style document for all six trainers."""
    lang = st.session_state.get("lang", "zh")

    zh = """## 一、Zimo Studio 是什么

Zimo Studio 是一个**本地运行的端侧微型语言模型训练与测试工作台**，覆盖完整闭环：

```
语料准备 → 字符化/分词 → 模型训练 → 实时监控 → 候选测试 → ONNX 导出 → INT8 量化
```

所有计算在本地 CPU/GPU 完成，不联网、数据不出本机。目标场景是输入法等端侧应用：
训练 1M–25M 参数的小模型，为候选文本打分或直接学习用户选择行为，用于候选重排与纠错。

排序链路中的位置（输入法视角）：

```
按键 → 拼音 DAG → 候选生成 → L1 TJMLP1（实时重排）
                              → L2 TJTR1（停顿语义重排）
                              → L3 用户个性化
主语言模型 tons.gram 负责 beam 解码；本工作台训练的是上述神经组件
```

---

## 二、模型类型：神经网络与统计模型

六个训练器分属两大类。**神经网络**从数据里学参数（需要梯度下降），泛化好但需要
数据和算力；**统计模型**直接计数（无梯度），极小极快但只看固定窗口、无泛化。

| 模型类别 | 代表训练器 | 学习方式 | 神经网络 | 推理计算 |
|---|---|---|---|---|
| 自回归语言模型（Decoder-only Transformer） | train_gpt.py | 无监督 next-token | 是（深度） | 矩阵乘法 |
| 循环语言模型（LSTM） | train_nnlm.py | 无监督 next-token | 是（浅层递归） | 逐步矩阵乘 |
| 交叉编码重排器（Encoder Transformer） | train_tiny_transformer.py | 有监督 listwise | 是（深度） | 矩阵乘法 |
| 特征列表重排器（MLP） | train_mlp.py | 有监督 listwise | 是（浅层） | 极少量矩阵乘 |
| 统计语言模型（n-gram） | train_quantized_lm.py / train_v3.py | 计数 + Kneser-Ney 平滑 | 否 | 查表 + 二分查找 |

取舍直觉：**要泛化和上下文理解 → 神经网络；要极小体积、极低延迟、零训练数据 →
统计模型**。输入法里两者并存：n-gram 兜底打分，神经模型提供语义证据。

按输出形态又分两种：**生成式**（GPT-2/LSTM 输出整个词表上的分布，可用于采样生成）
与**判别式**（TJTR1/TJMLP1 只输出每个候选的分数，不能生成文本）。

---

## 三、六个训练器完整规格

### 1. GPT-2 字符级语言模型（train_gpt.py）｜默认推荐

| 项 | 规格 |
|---|---|
| 类型 | 神经网络 · Decoder-only Transformer · 自回归生成式 |
| 架构 | token embedding + 可学习位置 embedding，N 层 pre-LN block（LayerNorm → 因果多头自注意力 → 残差 → LayerNorm → GELU FFN(4x) → 残差），最终 LayerNorm，独立线性输出头 |
| 注意力 | **单向（因果）**：位置 i 只能 attending ≤ i 的 token，保证自回归性质 |
| 参考规模 | 4,314,240 参数 = 6 层 / 192 维 / 4 头 / 上下文 64 / 词表 4,251（可配置） |
| 输入 | int64 字符 id 序列 `[batch, seq_len≤64]`；汉字 = 1 token，英文字母 = 1 token，标点 = 1 token |
| 输出 | float32 logits `[batch, seq_len, V]`；log_softmax 后得到每个位置的条件概率分布（可采样生成） |
| 训练信号 | **自回归 next-token 交叉熵**（无监督）：位置 i 的标签是 i+1 处的真实字符，因果 mask 防止偷看未来 |
| 推理 | 候选评分 = 滑窗计算候选字符的平均 NLL，`PPL = exp(mean NLL)`；前文作为条件拼在窗口头部 |
| 用途 | 候选自然度打分（重排间接证据）、纠错、域内文本基线 |
| 边界 | 单向注意力导致它只能"逐句独立打分"，不能在同一次前向里对比多个候选；无拼音概念 |

### 2. LSTM 语言模型（train_nnlm.py）

| 项 | 规格 |
|---|---|
| 类型 | 神经网络 · 循环网络（RNN 族）· 自回归生成式 |
| 架构 | Embedding → Dropout(0.1) → 多层 LSTM（默认 1 层，hidden 512，dim 256）→ Linear 输出 |
| 注意力 | 无（递归结构按时间步串行计算，天然只看过去） |
| 输入 | int64 id `[batch, bptt=65]` |
| 输出 | logits `[batch, seq, V]` + 隐状态 |
| 训练信号 | 与 GPT-2 相同的 next-token 交叉熵（无监督） |
| 用途 | 与 GPT-2 同任务的轻量替代；体积更小、推理更快 |
| 边界 | 长距离依赖弱于 Transformer；递归计算无法并行，GPU 收益低 |

### 3. TJTR1 交叉编码器（train_tiny_transformer.py）｜CLI

| 项 | 规格 |
|---|---|
| 类型 | 神经网络 · Encoder-only Transformer（BERT 式）· 判别式重排器 |
| 架构 | token embedding + 可学习位置 embedding，N 层 pre-LN block（LayerNorm → 合并 WQKV 投影的多头自注意力 → wo 残差 → LayerNorm → GELU FFN → 残差），有效 token masked mean pooling，线性标量打分头 |
| 注意力 | **双向（无因果 mask）**：序列内每个字符互相可见，因此上下文与各候选可以在同一次前向中直接对比 |
| 参考规模 | 约 0.4M 参数 = 2 层 / 64 维 / 4 头 / FFN 128 / max_seq 48 / 词表 ≤4000 |
| 输入 | **一条编码序列**：前文字符 + 候选边界 + 全部候选字符，int64 `[1, seq_len≤48]`（上下文与候选列表拼在一起） |
| 输出 | 每个候选一个标量 logit → listwise softmax 得到选择概率；C++ 运行时返回 `log(sigmoid(logit)) ∈ (-inf, 0]`，分数地板 -8 |
| 训练信号 | **listwise 交叉熵（有监督）**：训练数据是 selection_events（上下文 + 候选列表 + 用户实际选中的下标），以选中项为正类对整张候选分数向量做 CE |
| 推理 | 挂在停顿/长句路径（`Session::RescoreCandidates`，80ms 预算）；与主模型 tons.gram 异构，构成"第二意见" |
| 用途 | 候选精准重排；同架构可直接迁移到 ASR n-best 重排、整句纠错对打分、意图配对等任何"列表选择"问题 |
| 边界 | 必须有用户点击日志；上下文+全部候选需装入 48 字符窗口；不能生成文本 |

### 4. TJMLP1 Listwise Ranker（train_mlp.py）

| 项 | 规格 |
|---|---|
| 类型 | 神经网络 · 浅层前馈网络（MLP）· 判别式重排器 |
| 架构 | 共享权重的两层 MLP：32 维特征 → hidden(默认 16) → 每候选 1 个分数；同一候选列表内的分数做 listwise softmax |
| 注意力 | 无（纯前馈网络） |
| 输入 | 每候选 32 维 float 特征向量：词典分、用户分、语言模型分、搭配分、上下文分、path_log、音节数、tag one-hot 等（定义在 `model_feature.h`，append-only 契约） |
| 输出 | 每候选分数 → 选择概率；导出 TJMLP1 二进制（版本头 + 权重），由 C++ `MlpRanker` 加载 |
| 训练信号 | listwise 交叉熵（有监督）：数据为 selection_events v2，优先使用事件里的精确 32 维 `features` 数组，缺失字段按映射规则补 0 |
| 推理 | 微秒级（两层全连接），放在按键路径做实时重排（L1） |
| 用途 | 实时候选重排、特征消融实验 |
| 边界 | 不看文本本体——特征质量决定精度上限；只有纯文本无法训练 |

### 5. 3-gram 量化语言模型（train_quantized_lm.py）｜CLI

| 项 | 规格 |
|---|---|
| 类型 | **统计模型（非神经网络）** · 词级计数 + 平滑 |
| 架构 | unigram/bigram/trigram 频次 → 插值 Kneser-Ney 平滑（D=0.75）→ 每阶 log 概率 8-bit 仿射量化 → TJNGRMQ 容器（144 字节头 + 有序表），mmap 零拷贝加载 |
| 输入 | 分词语料（词表约束 FMM 切分）+ 词表 TSV；运行时输入为词 id 序列 |
| 输出 | `P(w|前1词)` / `P(w|前2词)` 的 log 概率（uint8 量化值经 scale/bias 还原） |
| 训练信号 | 极大似然计数（无梯度、无反向传播） |
| 推理 | trigram → bigram → unigram 直接落级（无 backoff 权重），有序表二分查找 |
| 用途 | 停顿路径兜底打分、低资源设备（含 MCU）、与神经模型构成异构第二意见 |
| 边界 | 只看前 2 个词；对未见组合无泛化；需要分词语料而非原始文本 |

### 6. 词级 n-gram（train_v3.py）｜CLI

| 项 | 规格 |
|---|---|
| 类型 | **统计模型（非神经网络）** · float32 版本（NIANGRM 格式） |
| 输入/输出/训练信号 | 与 3-gram 相同，词表来自项目词典 + 技术词表 |
| 用途 | 旧版本兼容与迁移 |
| 边界 | 新项目不建议从此起步 |

---

## 四、注意力与训练信号：两种范式

**范式 A · 自回归语言模型（GPT-2 / LSTM）**

- 单向注意力/递归：位置 i 只见前文
- 训练信号：无监督的 next-token 交叉熵，语料越多越好
- 打分方式：对每个候选**独立**计算 `P(候选|前文)`，再互相比较
- 比喻：一位只根据前文判断"接下来写什么最顺"的校对员

**范式 B · 交叉编码重排器（TJTR1 / TJMLP1）**

- 双向注意力/前馈：上下文与全部候选进入同一序列/同一批特征
- 训练信号：有监督的 listwise 交叉熵，学习目标 = 复现用户的选择
- 打分方式：候选之间在注意力层直接交互，输出"用户会选它"的相对概率
- 比喻：把所有选项摆在桌上、知道标准答案的阅卷老师

两者的分数含义不同（自然度 vs 选择概率），因此引擎把它们放在不同链路：
范式 A 供 beam/兜底/独立证据，范式 B 供停顿期精准重排。

---

## 五、你能用它训练什么：四种产物与十个场景

架构是通用的，绑死输入法的只是**训练语料和任务包装**。这个工具箱的本质是一个
**通用微型语言模型训练器**：输入任意纯文本（或用户点击日志），产出一个可在
PC / 树莓派 / ESP32 上推理的小模型。

### 按手里的数据分：四种可训练的产物

| 你手里的数据 | 训练产物 | 典型用途 |
|---|---|---|
| 任意领域纯文本：客服对话、法律、医疗、诗词、方言、代码注释、英文语料…… | 领域字符级打分器（GPT-2 / LSTM） | 文本质量分级、语料清洗、异常检测、域内补全 |
| 用户点击日志（上下文 + 候选列表 + 选中项） | 交叉重排器（TJTR1）或特征重排器（TJMLP1） | 搜索 / ASR / MT 的 n-best 重排、意图匹配 |
| 带 32 维特征的日志 | TJMLP1 Listwise Ranker | 低延迟实时重排 |
| 已分词语料 + 词表 | 统计 n-gram（3-gram / 词级） | MCU 与极低资源设备打分 |

### 十个真实场景

1. **输入法候选重排**——源场景：同音消歧（他/她/它）、长句排序
2. **编辑器 ghost typing**——任意应用的联想补全打分
3. **ASR 语音识别重排**——语音 n-best 候选与拼音候选同构，架构直接复用
4. **机器翻译 n-best 重排**——交叉编码器重排是 MT 领域的经典用法
5. **离线文本清洗**——用 PPL 过滤低质量语料与垃圾评论，数据不出本机
6. **异常检测**——日志 / 弹幕流里 PPL 突增即异常片段
7. **边缘智能硬件**——树莓派 / ESP32 上的本地打分（智能家居面板、电子键盘、玩具）
8. **教学工具**——几分钟训完一个完整、可解释、每一步可见的小 LM
9. **研究基线**——tiny-LM 消融、tokenizer 对比、蒸馏实验的标准参照
10. **低资源语言 / 方言 / 古文保存**——字符级免分词，任何文字体系都可直接训练

### 三个典型组合

- **客服质检**：客服对话语料 → GPT-2 打分器（回答自然度分级）+ 客服操作日志 →
  TJMLP1（话术排序）
- **离线智能键盘**：训练 → 测试 → INT8 量化 → 树莓派 / 手机部署的完整链路
- **课堂教学**：`examples/sample.txt` + 5 分钟冒烟训练，学生能看到从 loss 曲线
  到 ONNX 产物的每一步

### 按架构的通用能力

**自回归 LM（GPT-2 / LSTM）**

- 文本打分与过滤：PPL 高 = 不像该领域文本 → 语料清洗、垃圾评论过滤、数据质量分级
- 异常检测：日志/弹幕里 PPL 突然飙高的片段即异常内容
- 文本生成：架构支持采样续写（当前 Studio 只做了打分，加一个采样循环即可）
- 前缀补全：任何应用的联想输入
- 教学与研究：参数量小到可以完全理解，几分钟可复现，是 tiny-LM 实验的标准基线

**交叉编码器（TJTR1）**——任何"n-best 列表重排"问题：

- ASR 语音识别候选重排（与拼音候选同构）
- 机器翻译 n-best 重排（交叉编码器重排是 MT 领域的经典用法）
- 搜索/匹配排序（query 与候选文档配对打分）
- 问答匹配、相似句判断、意图配对
- 整句纠错（错句/正句候选对打分）
- 数据清洗：反查"模型极高置信但用户没选"的记录，定位标注或引擎问题

**统计 n-gram**：拼写字检查、MCU 打分器、极低资源自动补全。

重要提醒：**架构通用 ≠ 权重通用**。当前 4M 权重学的是 LCCC 对话分布，
拿去给法律文书打分会失真——换领域请换语料重训。


---

## 六、部署平台：PC · 树莓派 · ESP32

| 平台 | 运行时 | 推荐模型 | 状态 |
|---|---|---|---|
| PC / 服务器 | onnxruntime / PyTorch | FP32 或 INT8 ONNX | 即用（导出台直出） |
| 树莓派（ARM） | onnxruntime aarch64 | INT8 ONNX（4.3MB） | 即用，单次前向估计个位数~几十毫秒 |
| ESP32 | TJNGRMQ 3-gram（C + mmap） | 统计模型（几百 KB） | 即用形态，纯查表无矩阵乘 |
| ESP32 神经模型 | TFLite Micro（全整数量化） | 需重训 ≤0.5M 参数 | 需补工作：全整数量化 + ONNX→TFLite 转换 |

ESP32 的内存账：SRAM 约 520KB、PSRAM 4–8MB、Flash 4–16MB。
fp32 4M 模型（16MB）放不进 RAM；int8（4.3MB）在 8MB PSRAM 上很紧；
**0.2–0.5M 参数的小模型 int8 后约 0.2–0.5MB**，放 flash 流式读取可行，
单序列推理估计 10–100ms（ESP32-S3）。

注意：当前导出台的 INT8 是**仅权重量化**；TFLite Micro 需要**全整数量化**
（激活值也 int8，需代表性数据集校准）。这是从 ONNX 到 MCU 的主要补齐项。

---

## 七、三个工作区操作指南

**训练控制台**
1. 左侧选择任务（A 纯文本打分 / B 点击日志重排）与模型
2. 选择数据来源：**本地路径**（默认，直接读磁盘，支持目录与 .jsonl.gz，不限大小）
   或 **上传文件**（可多选合并，适合小样本）；再选硬件、调整 Epoch / Batch / Token 预算
3. 开始训练：后台子进程运行，页面展示进度环、Loss/PPL/吞吐曲线、最佳验证 PPL、彩色日志
4. 结束后在"训练产物"区查看路径，可直接打开目录或下载 .pt
   （Windows 防坑：训练期间不读 status.json，仅监听 stdout 与 metrics.jsonl）

**模型测试台**
1. 模型来源：自动加载 runs/ 最新产物，或手动上传模型。支持
   **.pt / .pth / .ckpt / .bin / .safetensors**；词表取自 checkpoint 内置
   （``vocab`` 键或 safetensors metadata）、``vocab.txt`` / ``vocab.json``；
   结构参数取自 checkpoint、``config.json``，缺失时按权重形状反推
2. **模型参数对比**：并排显示文件、架构、参数量、层数/维度/头数、上下文、词表、体积、词表来源
3. **测试方式**：模型 A / 模型 B / A/B 对比（单模型测试不必两个都加载）
4. 输入前文上下文与若干候选句（每行一条），按 PPL 升序展示，Rank 1 绿色、偏离值红色
5. 对比模式额外给出 PPL 差、更优列，以及双方平均 PPL 与胜负统计
6. 通用工具：逐字 NLL（哪个字最意外）、批量打分（每行一句 + CSV 导出）、
   文本生成（贪心或温度采样，用于观察模型学到了什么）
7. **手动纠错**：自己打字比对两句（正确句 vs 错别字句，直接给判定），
   或输入一句让模型逐字给建议——每处修改都会整句重算 PPL，
   只有确实降低 PPL 的改动才保留，改前/改后可直接对比

**导出与量化**
1. 同样支持自动加载 / 手动上传
2. 选择精度：FP32（原始）或 INT8（动态量化，体积约 1/4）
3. 点击"开始转换"：导出 ONNX（动态轴，支持任意 batch 与序列长度）；
   选 INT8 时自动调用 onnxruntime 动态量化压缩权重
4. 产物：`model_fp32.onnx`、`model_int8.onnx`（可选）、`model.safetensors`、
   `model.tjgpt`、`vocab.txt`、README，自动打包为 .zip 供下载
5. 格式怎么选：**safetensors** 用于交换/归档；**ONNX** 给第三方或移动端（需 onnxruntime）；
   **TJGPT** 给本项目的 C++ 引擎直读（零依赖，配套 `TjgptModel` 运行时）
5. 一致性检查：输入一段文本，对比 PyTorch / ONNX FP32 / ONNX INT8 的 PPL

---

## 八、高级自定义：五个维度

**维度一 · 自定义模型结构（捏人）**
侧边栏「高级参数」里所有训练参数都是**自由输入**，不限定档位：
层数 1–64、维度 8–2048、头数 1–64、上下文 8–2048、词表上限 100–200000、
验证集比例、学习率、Warmup、Token 预算（最高 50 亿）、Batch（最高 4096）。
顶部有四个预设（端侧 4M / 均衡 9M / 大号 25M / 极限 44M）一键填入，填完仍可手改。
下方实时估算参数量（≈ 12 × Layers × Dim² + Vocab × Dim）与 INT8 体积，
并用你填的词表上限计算。参数原样拼进训练命令
（`--layers` / `--dim` / `--heads` / `--block` / `--max-vocab` / `--lr`
/ `--warmup` / `--val-ratio`）。约束：dim 必须能被 heads 整除，否则拒绝启动。

**维度二 · 数据清洗（洗菜）**
上传框下方「数据清洗」多选：去除所有标点（只留汉字）、去除数字与英文字母、
64 字强制切分（不丢弃）、自动去重，以及自定义正则（匹配内容删除）。
规则在字符切分前依次应用；正则语法错误会在开始训练时报错。

**维度三 · 自定义评测（定制考题）**
测试台「定制考题」：上传两列 CSV（正确句, 错别字句），一键评测输出纠错准确率
Top-1（正确句 PPL 低于错句即胜出）与逐对明细。另有「自定义高亮阈值」：
开启后可设 PPL ≤ 绿、PPL ≥ 红两个阈值，替代默认的 Rank 1 绿 / 5× 最优红。

**维度四 · 自定义导出（终极外挂）**
导出台可选 ONNX（FP32/INT8）、TorchScript（libtorch C++）、TFLite（实验，需
tensorflow + onnx-tf），自定义模型包名（决定 zip 文件名，仅允许字母数字点横线下划线），
并即时生成适配本次参数的 onnxruntime C++ 推理代码供复制。

**维度五 · 自定义词典（专有名词保护）**
测试台「自定义词典保护」：上传 custom_dict.txt（每行一个专有名词），
包含保护词的候选不会被标红，可选择优先排序；结果表新增「词库保护」列。

## 九、指标定义

| 指标 | 定义 | 方向 |
|---|---|---|
| PPL（困惑度） | exp(平均 NLL)，模型对句子的不确定度 | 越低越符合训练分布 |
| NLL（平均负对数似然） | PPL 的底层统计量 | 越低越好 |
| Rank | 按分数排序后的名次 | 1 为最高 |
| Loss | 训练目标函数值 | 下降 = 在学习 |
| val PPL | 验证集 PPL | 衡量泛化，不作为最终验收 |
| 吞吐 | 每秒处理的 token 数 | 越高越快 |
| INT8 | 8-bit 权重量化 | 体积约 1/4，分数轻微偏差 |

> 注意：PPL 只是训练信号，**不能作为上线验收**。最终验收必须看
> 端到端候选重排指标（Top1 救回/弄坏、Correct@1）。

---

## 十、数据准备

| 模型 | 最小可运行 | 推荐规模 | 格式 |
|---|---|---|---|
| GPT-2 / LSTM | ~10M token | 40–80M token | .txt，每行一句 |
| TJTR1 | 数千条点击事件 | 1 万条以上，覆盖真实候选分布 | JSONL/CSV：上下文+候选+选中项 |
| TJMLP1 | 1000 条以上 | 1 万条以上，含 32 维特征 | JSONL/CSV，scores 为 JSON 数组 |
| 3-gram / 词级 n-gram | 数千万 token | 数亿 token | 分词语料 + 词表 TSV |

侧边栏提供示例语料与示例点击日志下载，可直接作为格式模板；
本地路径模式支持目录批量读取与 LCCC 风格的 .jsonl / .jsonl.gz。

---

## 十一、常见问题

- **找不到 trainer**：独立包只自带 GPT-2；LSTM/MLP 需要把 `ZIMO_TRAINER_ROOT`
  指向包含 `tools/lm/train_nnlm.py`、`tools/ranker/train_mlp.py` 的项目目录，
  或将本包放在同级目录让其自动发现。
- **页面提示 File change / Rerun**：代码更新后点击 Rerun 才会加载新脚本，
  旧页面会一直跑旧代码。
- **词表从哪来**：新训练的 .pt 内置词表（`"vocab"` 字段）；旧模型需要同目录的
  vocab.txt。测试台与导出台都会自动选择可用的来源。
- **INT8 分数会变吗**：会有轻微偏差（参考值：FP32 6.318 → INT8 6.308），
  体积降至约 1/4，适合端侧部署。
- **GPU 选项无效**：LSTM/MLP 训练器当前为 CPU 实现；仅 GPT-2 使用 CUDA。
- **ONNX 是什么**：跨框架推理格式；导出后不依赖 PyTorch，可被
  onnxruntime / 移动端推理器加载。

---

## 十二、能力边界

- 不做拼音解码：按键 → 候选由 DAG/Viterbi/词典完成，本工作台只负责其后的打分与重排
- 不生成文本：TJTR1/TJMLP1 是判别式模型；GPT-2/LSTM 可生成但 Studio 当前只做打分
- 不是通用大模型：4M 级参数 + 对话域语料，能力边界明确
- 完全本地：无网络请求；请勿将 Streamlit 服务暴露到公网
"""

    en = """## 1. What is Zimo Studio

Zimo Studio is a **local workbench for training and evaluating edge-scale
micro language models**, covering the full loop:

```
corpus prep -> char/token split -> training -> live monitoring
-> candidate testing -> ONNX export -> INT8 quantization
```

Everything runs locally (CPU/GPU), no network.  Target scenario: IME-like
apps where a 1M-25M parameter model scores candidates or learns user
selection behavior for reranking.

Ranking pipeline position (IME view):

```
keystroke -> pinyin DAG -> candidate generation -> L1 TJMLP1 (real-time)
                                                -> L2 TJTR1 (pause rerank)
                                                -> L3 personalization
Primary LM tons.gram drives beam decoding; this workbench trains the
neural components above.
```

---

## 2. Model types: neural vs statistical

| Model class | Trainer | Learning | Neural | Inference cost |
|---|---|---|---|---|
| Autoregressive LM (decoder-only Transformer) | train_gpt.py | unsupervised next-token | yes (deep) | matrix multiplies |
| Recurrent LM (LSTM) | train_nnlm.py | unsupervised next-token | yes (shallow recurrent) | stepwise matmuls |
| Cross-encoder reranker (encoder Transformer) | train_tiny_transformer.py | supervised listwise | yes (deep) | matrix multiplies |
| Feature listwise ranker (MLP) | train_mlp.py | supervised listwise | yes (shallow) | tiny matmuls |
| Statistical n-gram | train_quantized_lm.py / train_v3.py | counting + Kneser-Ney | no | table lookup |

Rule of thumb: **generalization and context -> neural; tiny size, lowest
latency, zero training data -> statistical**.  IMEs use both: n-gram as
fallback scoring, neural models as semantic evidence.

By output shape: GPT-2/LSTM are **generative** (a distribution over the
whole vocab, sampling possible); TJTR1/TJMLP1 are **discriminative**
(a score per candidate, no generation).

---

## 3. Six trainers, full specs

### 1. GPT-2 character LM (train_gpt.py) - default

| Item | Spec |
|---|---|
| Type | Neural, decoder-only Transformer, autoregressive & generative |
| Architecture | token + learned position embedding, N pre-LN blocks (LayerNorm -> causal multi-head attention -> residual -> LayerNorm -> GELU FFN(4x) -> residual), final LayerNorm, untied linear head |
| Attention | **Causal (unidirectional)**: position i attends only to <= i |
| Reference size | 4,314,240 params = 6 layers / 192 dims / 4 heads / context 64 / vocab 4,251 |
| Input | int64 char-id sequence `[batch, seq_len<=64]`; one Han char = 1 token, one Latin letter = 1 token |
| Output | float32 logits `[batch, seq_len, V]`; log_softmax gives per-char conditionals (sampling possible) |
| Training signal | **Autoregressive next-token cross-entropy** (unsupervised) |
| Inference | candidate score = sliding-window mean NLL over candidate chars, `PPL = exp(mean NLL)` |
| Uses | naturalness scoring, correction, in-domain baselines |
| Boundaries | scores candidates independently (no cross-candidate comparison in one pass); no pinyin concept |

### 2. LSTM language model (train_nnlm.py)

| Item | Spec |
|---|---|
| Type | Neural, recurrent (RNN family), autoregressive & generative |
| Architecture | Embedding -> Dropout(0.1) -> stacked LSTM (1 layer, hidden 512, dim 256) -> Linear |
| Attention | none (recurrent, sequential over time steps) |
| Input | int64 ids `[batch, bptt=65]` |
| Output | logits `[batch, seq, V]` + hidden state |
| Training signal | same next-token cross-entropy (unsupervised) |
| Uses | lighter/faster alternative to GPT-2 |
| Boundaries | weaker long-range dependency; recurrence limits parallelism |

### 3. TJTR1 cross-encoder (train_tiny_transformer.py) - CLI

| Item | Spec |
|---|---|
| Type | Neural, encoder-only Transformer (BERT-style), discriminative reranker |
| Architecture | token + learned position embedding, N pre-LN blocks (LayerNorm -> fused-WQKV multi-head attention -> wo residual -> LayerNorm -> GELU FFN -> residual), masked mean pooling over valid tokens, linear scalar head |
| Attention | **Bidirectional (no causal mask)**: every char attends to every other, so context and all candidates interact within one forward pass |
| Reference size | ~0.4M params = 2 layers / 64 dims / 4 heads / FFN 128 / max_seq 48 / vocab <=4000 |
| Input | **one encoded sequence**: context chars + candidate boundaries + all candidates, int64 `[1, seq_len<=48]` |
| Output | one scalar logit per candidate -> listwise softmax = selection probability; C++ runtime returns `log(sigmoid(logit)) in (-inf, 0]`, floor -8 |
| Training signal | **listwise cross-entropy (supervised)**: selection events (context + candidate list + chosen index); the chosen candidate is the positive class |
| Inference | pause/long-sentence path (`Session::RescoreCandidates`, 80ms budget); heterogeneous second opinion vs tons.gram |
| Uses | precise candidate reranking; the same architecture transfers to ASR n-best reranking, correction-pair scoring, intent matching - any list-selection problem |
| Boundaries | requires click logs; context + all candidates must fit 48 chars; cannot generate text |

### 4. TJMLP1 listwise ranker (train_mlp.py)

| Item | Spec |
|---|---|
| Type | Neural, shallow feed-forward (MLP), discriminative reranker |
| Architecture | shared-weight 2-layer MLP: 32-dim feature -> hidden(16) -> one score per candidate; listwise softmax over the list |
| Attention | none (pure feed-forward) |
| Input | per-candidate 32-dim float feature vector: dictionary/user/LM/collocation/context scores, path_log, syllables, tag one-hots (`model_feature.h`, append-only contract) |
| Output | per-candidate scores -> selection probability; exported as TJMLP1 binary loaded by C++ MlpRanker |
| Training signal | listwise cross-entropy (supervised) on selection events v2; exact `features` arrays preferred |
| Inference | microsecond-level (two dense layers); L1 real-time reranking on the keystroke path |
| Uses | real-time reranking, feature ablations |
| Boundaries | never sees raw text; feature quality caps accuracy |

### 5. 3-gram quantized LM (train_quantized_lm.py) - CLI

| Item | Spec |
|---|---|
| Type | **Statistical (no neural net)**, word-level counting + smoothing |
| Architecture | counts -> interpolated Kneser-Ney (D=0.75) -> per-order 8-bit affine quantized log-probs -> TJNGRMQ container, mmap zero-copy |
| Input | tokenized corpus (vocab-constrained FMM) + vocab TSV |
| Output | log P(w|w-1) / P(w|w-2,w-1) (uint8 quantized, dequantized via scale/bias) |
| Training signal | maximum-likelihood counting (no gradients) |
| Inference | trigram -> bigram -> unigram fall-through (no backoff weights), binary search |
| Uses | pause-path fallback scoring, low-resource devices including MCUs |
| Boundaries | sees only the previous 2 words; no generalization to unseen combinations |

### 6. Word-level n-gram (train_v3.py) - CLI

Same family in float32 (NIANGRM); vocab from project dictionaries.
Purpose: legacy compatibility.

---

## 3b. Two paradigms: attention and training signal

**Paradigm A - autoregressive LM (GPT-2 / LSTM)**

- Causal attention / recurrence: position i sees only the past
- Signal: unsupervised next-token cross-entropy; more text helps
- Scoring: each candidate is scored **independently** as P(candidate|context)
- Mental model: a proofreader judging "what fits next"

**Paradigm B - cross-encoder reranker (TJTR1 / TJMLP1)**

- Bidirectional attention / feed-forward over context + all candidates together
- Signal: supervised listwise cross-entropy reproducing the user's choice
- Scoring: candidates interact inside the attention layers; the output is a
  relative "user would pick this" probability
- Mental model: a grader who sees all options and knows the answer key

Their scores mean different things (naturalness vs selection probability),
so the engine places them on different paths: A for beam/fallback/independent
evidence, B for precise pause-path reranking.

---

## 5. What can you train: four products, ten scenarios

The architectures are generic; only the **training data and task framing**
are IME-specific.  At its core this toolbox is a **generic micro language
model trainer**: feed it any plain text (or click logs) and it produces a
small model that runs on a PC, a Raspberry Pi, or an ESP32.

### Four trainable products, by the data you have

| Data you hold | Trainable product | Typical use |
|---|---|---|
| Any domain plain text: support chats, legal, medical, poetry, dialects, code comments, English corpora... | Domain character scorer (GPT-2 / LSTM) | text grading, corpus cleaning, anomaly detection, in-domain completion |
| Click logs (context + candidate list + chosen item) | Cross-encoder (TJTR1) or feature ranker (TJMLP1) | n-best reranking for search / ASR / MT, intent matching |
| Logs with 32-dim features | TJMLP1 Listwise Ranker | low-latency real-time reranking |
| Tokenized corpus + vocab | statistical n-gram (3-gram / word-level) | scoring on MCUs and tiny devices |

### Ten real scenarios

1. **IME candidate reranking** - the origin scenario: homophone disambiguation,
   long-sentence ordering
2. **Editor ghost typing** - completion scoring for any app
3. **ASR reranking** - speech n-best candidates are isomorphic to pinyin
   candidates; the architecture transfers directly
4. **Machine translation n-best reranking** - a classic cross-encoder use
5. **Offline text cleaning** - filter low-quality corpora and spam by PPL,
   nothing leaves the machine
6. **Anomaly detection** - PPL spikes in logs / chat streams mark outliers
7. **Edge hardware** - local scoring on Raspberry Pi / ESP32 (smart panels,
   keyboards, toys)
8. **Teaching** - a complete, explainable mini LM trained in minutes, every
   step visible
9. **Research baseline** - the standard reference for tiny-LM ablations,
   tokenizer comparisons, distillation experiments
10. **Low-resource languages / dialects / classical text** - char-level
    means no segmenter; any writing system trains directly

### Three typical combos

- **Support-quality audit**: support-chat corpus -> GPT-2 scorer (response
  naturalness grading) + agent logs -> TJMLP1 (script ordering)
- **Offline smart keyboard**: the full README loop (train -> test -> INT8 ->
  Raspberry Pi / phone)
- **Classroom**: `examples/sample.txt` + a 5-minute smoke training; students
  see every step from the loss curve to the ONNX artifact

### General capability by architecture

**Autoregressive LM (GPT-2 / LSTM)**: text scoring and filtering (high PPL =
off-distribution -> corpus cleaning, spam filtering, data grading); anomaly
detection (PPL spikes in logs/chat); text generation (sampling is supported
by the architecture, the Studio currently only scores); prefix completion for
any app; teaching/research baseline for tiny-LM experiments.

**Cross-encoder (TJTR1)** - any n-best list reranking problem: ASR candidate
reranking, machine-translation n-best reranking (a classic cross-encoder
use), search/matching ranking, QA matching, similar-sentence judgment,
correction-pair scoring, data cleaning (find high-confidence disagreements
with user choices).

**Statistical n-gram**: spell checking, MCU scoring, ultra-low-resource
autocomplete.

Reminder: **generic architecture != generic weights**.  The current 4M
weights model the LCCC chat distribution; retrain on domain text for other
domains.


---

## 6. Deployment: PC, Raspberry Pi, ESP32

| Platform | Runtime | Recommended model | Status |
|---|---|---|---|
| PC / server | onnxruntime / PyTorch | FP32 or INT8 ONNX | works today (export tab) |
| Raspberry Pi (ARM) | onnxruntime aarch64 | INT8 ONNX (4.3MB) | works today; single forward estimated single-digit to tens of ms |
| ESP32 | TJNGRMQ 3-gram (C + mmap) | statistical model (few hundred KB) | natural fit: pure lookup, no matmuls |
| ESP32 neural | TFLite Micro (full-integer) | retrain <=0.5M params | needs work: full-integer quant + ONNX->TFLite conversion |

ESP32 memory math: ~520KB SRAM, 4-8MB PSRAM, 4-16MB flash.  The fp32 4M
model (16MB) does not fit in RAM; int8 (4.3MB) is tight on 8MB PSRAM;
**a 0.2-0.5M parameter model at int8 (~0.2-0.5MB)** streamed from flash is
feasible, with estimated 10-100ms per sequence on ESP32-S3.

Note: the export tab's INT8 is **weight-only quantization**; TFLite Micro
requires **full-integer quantization** (activations too, calibrated with a
representative dataset).  That conversion is the main gap on the MCU path.

---

## 7. Workspaces

**Training console**: pick task and model, then fill a **local path**
(default; reads straight from disk, accepts directories and .jsonl.gz, no
size limit) or use the optional upload panel for small samples.  Training
launches as a **detached background process** (`train.pid` + `train.log`), so
closing the browser or dropping SSH does not stop it, and reopening the page
re-attaches automatically.  Progress is read by tailing the log
(loss/PPL/throughput charts, best validation PPL).  Interrupted runs can be
continued from the "Resume from checkpoint" panel (`--resume`), and the full
command can be copied to a rented GPU shell.  On Windows status.json is only
read after the process exits (file-lock safety).

**Model playground**: auto-load the newest run under `runs/` or upload a
model.  Supported containers: **.pt / .pth / .ckpt / .bin / .safetensors**.
Vocabulary comes from the checkpoint (`vocab` key or safetensors metadata) or
a `vocab.txt` / `vocab.json` sidecar; architecture hyper-parameters come from
the checkpoint, a `config.json`, or are inferred from tensor shapes.  Score candidates (one per line) with preceding
context; results are sorted by PPL with Rank 1 highlighted green and
outliers red.  Load a second model for A/B comparison.

**Export & quantization**: export ONNX with dynamic axes
(input_ids [batch, seq_len] -> logits), optionally INT8 dynamic quantization
(about 1/4 size).  Artifacts are bundled into a .zip
(ONNX + vocab.txt + README) with per-file download buttons and a
PyTorch-vs-ONNX PPL parity check.

---

## 8. Advanced customization: five dimensions

**Dimension 1 - custom architecture ("build your own")**
Every training parameter is **free form**, no fixed tiers: layers 1-64, hidden
dim 8-2048, heads 1-64, context 8-2048, vocab cap 100-200000, validation ratio,
learning rate, warmup, token budget (up to 5B) and batch (up to 4096).  Four
presets (Edge 4M / Balanced 9M / Large 25M / Max 44M) fill the fields once and
stay editable.  A live estimate shows parameters (~12 x Layers x Dim^2 + Vocab
x Dim, using your vocab cap) and INT8 size, and the values are passed straight
to the trainer (`--layers` / `--dim` / `--heads` / `--block` / `--max-vocab`
/ `--lr` / `--warmup` / `--val-ratio`).  dim must be divisible by heads.

**Dimension 2 - data cleaning**
A multiselect under the uploader: strip all punctuation (Han only), strip digits
and Latin letters, split lines longer than 64 chars (without dropping), dedupe,
plus a custom regex whose matches are deleted.  Rules apply before character
splitting; an invalid regex fails loudly when training starts.

**Dimension 3 - custom evaluation**
Playground > Custom eval: upload a two-column CSV (correct sentence, typo
sentence) and get Top-1 correction accuracy (the correct sentence wins when its
PPL is lower) with a per-pair table.  Custom highlight thresholds let you set
PPL <= green and PPL >= red instead of the default Rank-1 green / 5x red.

**Dimension 4 - custom export**
The export tab offers ONNX (FP32/INT8), TorchScript (libtorch C++), TFLite
(experimental; needs tensorflow + onnx-tf), a bundle name that becomes the zip
file name (letters/digits/dot/dash/underscore only), and generated onnxruntime
C++ inference code adapted to this export's parameters.

**Dimension 5 - custom dictionary (proper-noun protection)**
Playground > Custom dictionary: upload custom_dict.txt (one term per line).
Candidates containing a protected term are never flagged red, can be sorted
first, and gain a "Protected" column in the results table.

## 9. Metrics

| Metric | Definition | Direction |
|---|---|---|
| PPL (perplexity) | model uncertainty over the sentence | lower = closer to training distribution |
| NLL (mean negative log-likelihood) | underlying statistic for PPL | lower is better |
| Rank | position after sorting by score | 1 is best |
| Loss | training objective | decreasing = learning |
| val PPL | held-out PPL | generalization only |
| Throughput | tokens per second | higher is faster |
| INT8 | 8-bit weight quantization | about 1/4 size, slight drift |

PPL is a training signal, **not an acceptance metric**; final acceptance
requires end-to-end reranking metrics (Top-1 rescued/broken, Correct@1).

---

## 10. Data preparation

| Model | Minimum | Recommended | Format |
|---|---|---|---|
| GPT-2 / LSTM | ~10M tokens | 40-80M tokens | .txt, one sentence per line |
| TJTR1 | thousands of events | 10k+, realistic candidate sets | JSONL/CSV: context+candidates+chosen |
| TJMLP1 | 1000+ events | 10k+ with 32-dim features | JSONL/CSV, scores as JSON |
| n-gram family | tens of millions of tokens | hundreds of millions | tokenized corpus + vocab TSV |

Sample corpus and sample click-log downloads are in the sidebar; the local
path mode also reads directories and LCCC-style .jsonl / .jsonl.gz.

---

## 11. FAQ

- **Trainer not found**: the package ships GPT-2 only.  Point
  `ZIMO_TRAINER_ROOT` to a checkout containing `tools/lm/train_nnlm.py` /
  `tools/ranker/train_mlp.py`, or place this package next to such a project.
- **File change / Rerun prompt**: click Rerun to load edited code; the old
  page keeps executing the previous script until you do.
- **Where does the vocabulary come from**: new .pt files embed it;
  legacy models use the sidecar vocab.txt.  Both tabs pick automatically.
- **INT8 drift**: slight score changes are expected (reference 6.318 ->
  6.308) at about 1/4 the size.
- **GPU option ignored**: LSTM/MLP trainers are CPU-only; only GPT-2 uses CUDA.
- **What is ONNX**: a cross-framework inference format; no PyTorch needed
  at inference time.

---

## 12. Boundaries

- No pinyin decoding: keystrokes -> candidates is done by DAG/Viterbi +
  dictionaries; this workbench only scores/reranks afterwards.
- No text generation in this Studio: TJTR1/TJMLP1 are discriminative;
  GPT-2/LSTM can generate but the app currently only scores.
- Not a general LLM: 4M-scale parameters, dialogue-domain data.
- Fully local: no network requests; do not expose the Streamlit server publicly.
"""

    st.markdown(zh if lang == "zh" else en)

@st.dialog(tr("Zimo Studio 使用说明", "Zimo Studio Documentation"), width="large")
def open_documentation() -> None:
    render_documentation()


def hero() -> None:
    st.markdown(f"""
<div class="tj-title">{tr("端侧超轻量语言模型工作台", "Edge-scale language model workbench")}</div>
<div class="tj-sub">
  {tr("一个用于训练、监控与评测超轻量级端侧语言模型的可视化工作台。上传语料、选择架构，实时查看 loss / PPL / 进度。",
      "A visual workbench for training, monitoring, and evaluating ultra-lightweight language models for edge deployment. Upload a corpus, pick an architecture, and watch loss / PPL / progress live.")}
</div>
""", unsafe_allow_html=True)


def model_guide() -> None:
    """Sidebar reference: three camps and six trainers in a scrollable box."""
    lang = st.session_state.get("lang", "zh")
    with st.expander(tr("模型选型", "Model selection"), expanded=False):
        parts = ['<div class="tj-guidebox">']
        for camp in MODEL_GUIDE:
            parts.append(
                f'<div class="tj-guide-camp">'
                f'{html.escape(camp["camp_" + lang])}</div><ul>')
            for model in camp["models"]:
                if model["studio"]:
                    tag = (tr("Studio 一键", "Studio one-click")
                           if trainer_path(model["key"]) is not None
                           else tr("Studio（需 trainer）", "Studio (trainer needed)"))
                else:
                    tag = tr("CLI", "CLI")
                parts.append(
                    f'<li><b>{html.escape(model[lang])}</b> — '
                    f'{html.escape(model["summary_" + lang])} '
                    f'{html.escape(tr("输入：", "Input: "))}'
                    f'{html.escape(model["data_" + lang])} '
                    f'<span class="tj-tag">{html.escape(tag)}</span></li>')
            parts.append('</ul>')
        note = tr(
            "没有用户选择日志时，默认使用 GPT-2。仅有纯文本时选第一阵营；"
            "有用户点击日志时选第二阵营；需要低延迟兜底或旧版兼容时选第三阵营。",
            "Without user-selection logs, GPT-2 is the default. Plain text -> "
            "camp 1; click logs -> camp 2; low-latency fallback/legacy -> camp 3.")
        parts.append(f'<div class="tj-guide-note">{html.escape(note)}</div></div>')
        st.markdown("".join(parts), unsafe_allow_html=True)


def sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### ZIMO <span style='color:#22d3ee'>LM</span>",
                    unsafe_allow_html=True)
        cuda = torch.cuda.is_available()
        st.caption(tr(
            f"硬件：GPU · {torch.cuda.get_device_name(0)}" if cuda
            else "硬件：CPU（未检测到 CUDA）",
            f"Hardware: GPU · {torch.cuda.get_device_name(0)}" if cuda
            else "Hardware: CPU (no CUDA)"))
        st.caption(tr("端侧小模型工作台", "Edge-scale LM workbench"))

        task = st.radio(
            tr("你想做什么？", "What do you want to do?"),
            [tr("A · 自然语言打分（无监督，纯文本 .txt）",
                "A · Text scoring (unsupervised, plain .txt)"),
             tr("B · 智能重排（有监督，点击日志）",
                "B · Smart rerank (supervised, click logs)")],
            key="sidebar_task",
            help=tr("A 只需要纯文本，训练 GPT-2/LSTM；"
                    "B 需要用户点击日志，训练 MLP/TJTR1。",
                    "A needs plain text only (GPT-2/LSTM); B needs click "
                    "logs (MLP/TJTR1)."))
        task_b = task.startswith("B")
        st.session_state["sidebar_task_b"] = task_b
        model_keys = ["mlp", "tjtr1"] if task_b else ["gpt2", "lstm"]
        lang = st.session_state.get("lang", "zh")
        if st.session_state.pop("_force_gpt2", False):
            st.session_state["model_arch"] = "gpt2"
        arch_key = st.selectbox(
            tr("模型架构", "Model architecture"), model_keys,
            key="model_arch",
            format_func=lambda key: MODEL_OPTIONS[key][lang],
            help=tr(
                "不知道选哪个就选 GPT-2。纯文本 → GPT-2/LSTM；点击日志 → TJTR1/TJMLP1；极速兜底 → 3-gram/词级 n-gram。Studio 一键支持 GPT-2/LSTM/MLP。",
                "Pick GPT-2 if unsure. Plain text -> GPT-2/LSTM; click logs -> TJTR1/TJMLP1; low-latency fallback -> 3-gram/word n-gram. Studio one-click supports GPT-2/LSTM/MLP."))
        arch_kind = MODEL_OPTIONS[arch_key]["kind"]
        entry, _camp = guide_for(arch_key)
        if entry:
            st.caption(f"{tr('输入要求', 'Input')}：{entry['data_' + lang]}")
        model_guide()

        path = trainer_path(arch_key)
        if path is None:
            model_name = MODEL_OPTIONS[arch_key][lang]
            st.warning(tr(
                f"找不到 {model_name} 的 trainer。独立包只自带 GPT-2；"
                "改选 GPT-2，或把 ZIMO_TRAINER_ROOT 指到含 "
                "tools/lm/train_nnlm.py 的项目 checkout。",
                f"No trainer for {model_name}. This package ships GPT-2 only; "
                "pick GPT-2, or set ZIMO_TRAINER_ROOT to a checkout containing "
                "tools/lm/train_nnlm.py."))
            if arch_key != "gpt2" and st.button(
                    tr("改用 GPT-2", "Use GPT-2"), use_container_width=True,
                    key="fallback_gpt2"):
                st.session_state["_force_gpt2"] = True
                st.rerun()
        else:
            try:
                shown = path.relative_to(TRAINER_ROOT)
            except ValueError:
                shown = path
            st.caption(f"trainer: `{shown}`")

        if arch_kind == "text":
            data_help = tr(
                "纯文本：每行一句；也接受 LCCC 的 .jsonl / .jsonl.gz。",
                "Plain text: one sentence per line; .jsonl / .jsonl.gz also accepted.")
        else:
            data_help = tr(
                "点击日志：上下文 + 候选列表 + 选中项；CSV 的 scores 必须是 JSON 数组。",
                "Click logs: context + candidate list + chosen item; the CSV scores column must be a JSON array.")

        # Data lives on this machine, so the path field IS the input; the
        # uploader is only a fallback for small samples.
        default_path = str(LOCAL_DATA_HINT)
        raw_path = st.text_input(
            tr("语料路径（文件或目录，可用 ; 分隔多个）",
               "Corpus path (file or directory; separate with ;)"),
            value=st.session_state.get("local_data_path", default_path),
            key="local_data_path",
            help=tr("训练在本机进行，直接读磁盘，无大小限制。目录会收集匹配类型的文件；"
                    "支持 .txt、.jsonl、.jsonl.gz。",
                    "Training runs locally, so this reads from disk with no size "
                    "limit. A directory collects matching files; .txt, .jsonl and "
                    ".jsonl.gz are supported."))

        upload = None
        local_paths: list[Path] = []
        if raw_path.strip():
            try:
                local_paths = resolve_local_paths(raw_path, arch_kind)
                total_mb = sum(f.stat().st_size for f in local_paths) / 1048576
                st.caption(tr(
                    f"找到 {len(local_paths)} 个文件，合计 {total_mb:,.1f} MB",
                    f"{len(local_paths)} files, {total_mb:,.1f} MB total"))
            except ValueError as exc:
                st.warning(str(exc))

        with st.expander(tr("或上传小样本（可选）", "Or upload a small sample (optional)"),
                         expanded=False):
            st.caption(tr(
                "仅用于几百 MB 以内的小文件；已填本地路径时以路径为准。",
                "For small files only; when a local path is set, the path wins."))
            if arch_kind == "text":
                upload = st.file_uploader(
                    tr("训练文本（.txt，可多选）", "Training text (.txt, multi-select)"),
                    type=["txt", "text"], accept_multiple_files=True,
                    key="train_upload",
                    help=data_help + tr(
                        " 支持一次选择多个文件，按选择顺序合并。",
                        " Multiple files are merged in selection order."))
            else:
                upload = st.file_uploader(
                    "selection_events（.csv / .jsonl，可多选）",
                    type=["csv", "jsonl"], accept_multiple_files=True,
                    key="train_upload",
                    help=data_help + tr(" 支持一次选择多个文件，按顺序合并。",
                                        " Multiple files are merged in order."))

        sample_path = APP_DIR / "examples" / "sample.txt"
        sample_events = json.dumps({
            "input": "nihao", "selected_text": "你好", "selected_index": 0,
            "prev_committed": "",
            "scores": [
                {"word": "你好", "dict": 0.8, "lm": 2.1, "syll": 2},
                {"word": "泥嚎", "dict": 0.3, "lm": 0.4, "syll": 2},
            ]}, ensure_ascii=False) + chr(10)
        dl1, dl2 = st.columns(2)
        with dl1:
            if sample_path.is_file():
                st.download_button(
                    tr("示例语料", "Sample corpus"),
                    data=sample_path.read_bytes(),
                    file_name="zimo_sample_corpus.txt", mime="text/plain",
                    use_container_width=True, key="download_sample_corpus",
                    help=tr("一个可直接训练的最小 .txt 示例。",
                            "A minimal ready-to-train .txt sample."))
        with dl2:
            st.download_button(
                tr("示例点击日志", "Sample click log"),
                data=sample_events.encode("utf-8"),
                file_name="zimo_sample_events.jsonl",
                mime="application/x-ndjson",
                use_container_width=True, key="download_sample_events",
                help=tr("selection_events 的最小 JSONL 示例。",
                        "A minimal selection_events JSONL sample."))

        clean_opts = {}
        if arch_kind == "text":
            st.markdown(f"**{tr('数据清洗（可选）', 'Data cleaning (optional)')}**")
            clean_choice = st.multiselect(
                tr("清洗规则", "Cleaning rules"),
                [tr("去除所有标点符号（只留汉字）", "Strip punctuation (Han only)"),
                 tr("去除数字和英文字母", "Strip digits and Latin letters"),
                 tr("强制切分长句（64 字切分，不丢弃）", "Split long lines at 64 chars"),
                 tr("自动去重", "Deduplicate lines")],
                key="clean_rules",
                help=tr("按字符切分前先应用这些规则；全部不选 = 保持原样。",
                        "Applied before character splitting; none = keep as-is."))
            clean_regex = st.text_input(
                tr("自定义正则（过滤匹配内容）", "Custom regex (strip matches)"),
                key="clean_regex",
                placeholder=tr("例如：哈哈哈+|转发微博", "e.g.哈哈哈+|转发微博"),
                help=tr("匹配到的内容会从语料中删除；语法错误会在开始训练时报错。",
                        "Matches are removed from the corpus; invalid syntax fails at training start."))
            clean_opts = {
                "strip_punct": any("标点" in c for c in clean_choice),
                "strip_alnum": any("数字" in c for c in clean_choice),
                "split_long": any("切分" in c for c in clean_choice),
                "dedup": any("去重" in c for c in clean_choice),
                "regex": clean_regex,
            }

        uploads = [] if local_paths else _as_uploads(upload)
        if uploads:
            bad = [u.name for u in uploads
                   if arch_kind == "text" and not u.name.lower().endswith((".txt", ".text"))
                   or arch_kind == "events" and not u.name.lower().endswith((".csv", ".jsonl"))]
            if bad:
                st.warning(tr(f"这些文件类型不匹配：{'、'.join(bad[:3])}",
                              f"These files do not match: {', '.join(bad[:3])}"))
            elif len(uploads) > 1:
                total_mb = sum(len(u.getvalue()) for u in uploads) / 1048576
                st.caption(tr(f"已选择 {len(uploads)} 个文件，合计 {total_mb:.1f} MB，将按顺序合并。",
                              f"{len(uploads)} files selected, {total_mb:.1f} MB total, merged in order."))

        if arch_key == "mlp":
            st.info(tr(
                "MLP 需要点击日志的 32 维特征；只有纯文本请选 GPT-2。",
                "MLP needs 32-dim features from click logs; for plain text pick GPT-2."))
            st.caption(tr("CSV 的 scores 列必须是 JSON。",
                          "The CSV scores column must be JSON."))
        elif arch_key == "lstm":
            st.caption(tr("LSTM 体积更小、推理更快，但长上下文能力弱于 GPT-2，适合资源受限设备。",
                          "LSTM is smaller/faster but has weaker context than GPT-2; good for very low-end devices."))
        else:
            st.caption(tr("GPT-2 是默认推荐：只要有纯文本就能训。",
                          "GPT-2 is the default recommendation: plain text is enough."))

        cuda = torch.cuda.is_available()
        hardware = st.radio(
            tr("硬件", "Hardware"), ["CPU", "GPU (CUDA)"],
            index=1 if cuda else 0, horizontal=True,
            help=tr("GPT-2 可用 CUDA；LSTM/MLP 目前 CPU-only。",
                    "GPT-2 can use CUDA; LSTM/MLP are CPU-only."))
        if hardware.startswith("GPU"):
            if cuda:
                st.caption(f"CUDA: {torch.cuda.get_device_name(0)}")
            else:
                st.warning(tr("当前环境没有 CUDA；GPT-2 会选择 CPU。",
                              "CUDA is unavailable; GPT-2 will use CPU."))
        if arch_key != "gpt2":
            st.caption(tr("该 trainer 目前 CPU-only，硬件选择会被忽略。",
                          "This trainer is CPU-only; the hardware choice is ignored."))

        with st.expander(tr("高级参数", "Advanced"), expanded=False):
            epochs = st.number_input(
                tr("Epoch 数量", "Epochs"),
                min_value=1, max_value=100, value=2, step=1,
                help=tr("2 是默认起点；语料大可以 4–10，观察验证 PPL 是否继续下降。",
                        "2 is a good start; use 4–10 for larger corpora and watch val PPL."))
            batch = st.number_input(
                "Batch Size", min_value=1, max_value=4096, value=32, step=1,
                help=tr("CPU 默认 32；GPU 可试 64–256；显存不足就调小。",
                        "32 on CPU; try 64–256 on GPU; lower it if you run out of memory."))
            tokens = st.number_input(
                tr("Token 预算", "Token budget"),
                min_value=100_000, max_value=5_000_000_000,
                value=30_000_000, step=10_000_000, format="%d",
                help=tr("最多读多少 token；4M 模型参考 30M；越多越慢。",
                        "How many tokens to read; 30M is the 4M-model reference; more is slower."))
            st.markdown(f"**{tr('网络结构（可自由填写）', 'Architecture (free form)')}**")
            preset = st.selectbox(
                tr("预设（可再手改）", "Preset (still editable)"),
                [tr("自定义", "Custom"),
                 tr("端侧 4M（4层/192维/64）", "Edge 4M (4L/192D/64)"),
                 tr("均衡 9M（8层/256维/128）", "Balanced 9M (8L/256D/128)"),
                 tr("大号 25M（12层/384维/128）", "Large 25M (12L/384D/128)"),
                 tr("极限 44M（12层/512维/128）", "Max 44M (12L/512D/128)")],
                key="arch_preset",
                help=tr("选一个起点后下面的数字仍可随意修改。",
                        "Pick a starting point; the numbers stay editable."))
            presets = {
                1: {"layers": 4, "dim": 192, "heads": 4, "block": 64},
                2: {"layers": 8, "dim": 256, "heads": 8, "block": 128},
                3: {"layers": 12, "dim": 384, "heads": 12, "block": 128},
                4: {"layers": 12, "dim": 512, "heads": 8, "block": 128},
            }
            key_index = [
                tr("自定义", "Custom"),
                tr("端侧 4M（4层/192维/64）", "Edge 4M (4L/192D/64)"),
                tr("均衡 9M（8层/256维/128）", "Balanced 9M (8L/256D/128)"),
                tr("大号 25M（12层/384维/128）", "Large 25M (12L/384D/128)"),
                tr("极限 44M（12层/512维/128）", "Max 44M (12L/512D/128)")].index(preset)
            if key_index > 0 and st.session_state.get("applied_preset") != preset:
                chosen = presets[key_index]
                st.session_state["arch_layers"] = chosen["layers"]
                st.session_state["arch_dim"] = chosen["dim"]
                st.session_state["arch_heads"] = chosen["heads"]
                st.session_state["arch_block"] = chosen["block"]
                st.session_state["applied_preset"] = preset
                st.rerun()

            c1, c2 = st.columns(2)
            layers = c1.number_input(
                tr("层数 Layers", "Layers"), min_value=1, max_value=64,
                value=6, step=1, key="arch_layers",
                help=tr("1–64 任意值。越大越强，也越慢越占内存。",
                        "Any value 1-64; larger is stronger but slower."))
            dim = c2.number_input(
                tr("维度 Hidden Dim", "Hidden dim"), min_value=8, max_value=2048,
                value=192, step=8, key="arch_dim",
                help=tr("8–2048 任意值，需能被头数整除。",
                        "Any value 8-2048; must be divisible by heads."))
            c3, c4 = st.columns(2)
            heads = c3.number_input(
                tr("注意力头数 Heads", "Attention heads"), min_value=1, max_value=64,
                value=4, step=1, key="arch_heads",
                help=tr("1–64；dim 必须能被它整除。",
                        "1-64; dim must be divisible by this."))
            block = c4.number_input(
                tr("上下文 Context", "Context window"), min_value=8, max_value=2048,
                value=64, step=8, key="arch_block",
                help=tr("8–2048 个 token；越大越能看长句，也越费显存。",
                        "8-2048 tokens; longer context costs more memory."))
            arch_params = {"layers": int(layers), "dim": int(dim),
                           "heads": int(heads), "block": int(block)}

            st.markdown(f"**{tr('训练细节（可自由填写）', 'Training details (free form)')}**")
            d1, d2 = st.columns(2)
            max_vocab = d1.number_input(
                tr("词表上限", "Max vocab"), min_value=100, max_value=200_000,
                value=13000, step=500, key="train_max_vocab",
                help=tr("字符级时通常 4000–13000；调小可减体积。",
                        "For char-level models 4000-13000 is typical."))
            val_ratio = d2.number_input(
                tr("验证集比例", "Validation ratio"), min_value=0.0, max_value=0.5,
                value=0.02, step=0.01, format="%.2f", key="train_val_ratio",
                help=tr("从语料尾部留出多少做验证；0 = 不验证。",
                        "Fraction held out for validation; 0 disables it."))
            d3, d4 = st.columns(2)
            learning_rate = d3.number_input(
                tr("学习率 LR", "Learning rate"), min_value=1e-6, max_value=1e-1,
                value=3e-4, step=1e-5, format="%.6f", key="train_lr",
                help=tr("默认 3e-4；大模型可降到 1e-4。",
                        "3e-4 by default; lower it for bigger models."))
            warmup = d4.number_input(
                tr("Warmup 步数", "Warmup steps"), min_value=0, max_value=20_000,
                value=300, step=50, key="train_warmup",
                help=tr("前多少步线性升温；小语料可设 0。",
                        "Linear warmup steps; 0 disables it."))
            dim_invalid = dim % heads != 0
            if dim_invalid:
                st.warning(tr("当前 dim 不能被 heads 整除，无法开始训练；请调整。",
                              "dim is not divisible by heads; adjust before training."))
            vocab_est = int(max_vocab)
            est_params = 12 * layers * dim * dim + vocab_est * dim
            est_int8_mb = est_params / 1e6 * 1.07
            st.caption(tr(
                f"预计 {est_params / 1e6:.2f}M 参数，INT8 约 {est_int8_mb:.1f} MB"
                f"（按 {vocab_est:,} 词表估算）",
                f"Estimated {est_params / 1e6:.2f}M params, ~{est_int8_mb:.1f} MB "
                f"INT8 (vocab {vocab_est:,})"))

        with st.expander(tr("打开已有训练", "Open existing run"), expanded=False):
            existing = st.text_input(
                tr("运行目录", "Run directory"), key="existing_run",
                placeholder=r"E:\...\runs\20260101_120000_gpt2")
            if st.button(tr("加载已有训练", "Load existing run"),
                         use_container_width=True, key="load_existing"):
                load_existing_run(existing, arch_key)

        running = is_running()
        start = st.button(tr("开始训练", "Start training"), type="primary",
                          use_container_width=True, disabled=running,
                          help=tr("默认选 GPT-2 即可开始；MLP 需要点击日志。",
                                  "GPT-2 is the safe default; MLP needs click logs."))
        stop = st.button(tr("停止", "Stop"), use_container_width=True,
                         disabled=not running)
        if running:
            st.caption(tr("训练中 · 页面自动刷新", "Training · auto-refreshing"))
        elif st.session_state.get("finished"):
            st.caption(tr("本次训练已结束", "This run has finished"))
        else:
            st.caption(tr("就绪", "Ready"))

        with st.expander(tr("续训（从 checkpoint 继续）",
                            "Resume from checkpoint"), expanded=False):
            st.caption(tr(
                "训练在后台独立运行，关掉浏览器或断开 SSH 都不会中断；"
                "重新打开页面会自动接管。",
                "Training runs detached: closing the browser or dropping SSH "
                "does not stop it, and reopening this page re-attaches."))
            candidates = unfinished_runs()
            if not candidates:
                st.caption(tr(
                    "runs/ 里还没有可续训的目录（需要 corpus.txt + model.pt）。",
                    "No resumable run yet (needs corpus.txt + model.pt)."))
            else:
                labels = [f"{item['dir'].name} · {item['state']} · step {item['step']}"
                          for item in candidates]
                choice = st.selectbox(tr("运行目录", "Run directory"), labels,
                                      key="resume_pick")
                picked = candidates[labels.index(choice)]
                if st.button(tr("继续训练", "Resume training"),
                             use_container_width=True, key="resume_run",
                             disabled=running):
                    resume_training(picked["dir"], arch_key, epochs, batch,
                                    tokens, hardware.startswith("GPU"),
                                    arch_params)
        st.divider()
        with st.expander(tr("术语词典（新手必看）", "Glossary (for beginners)")):
            zh_gloss = (
                "- **单向注意力**：只能看前文，像顺着打字一样。" + chr(10)
                + "- **6 层 / 192 维**：模型的思考深度和脑容量。" + chr(10)
                + "- **PPL（困惑度）**：越低越好，代表像人话的程度。" + chr(10)
                + "- **NLL**：底层打分依据。" + chr(10)
                + "- **INT8 量化**：给模型瘦身，从 17MB 变 4.3MB。")
            en_gloss = (
                "- **Causal attention**: can only see preceding text, like typing." + chr(10)
                + "- **6 layers / 192 dims**: the model's depth and capacity." + chr(10)
                + "- **PPL (perplexity)**: lower is better; how human-like the text is." + chr(10)
                + "- **NLL**: the underlying scoring statistic." + chr(10)
                + "- **INT8 quantization**: shrinks the model, 17MB to 4.3MB.")
            st.markdown(tr(zh_gloss, en_gloss))

        st.divider()
        lang_label = st.radio(
            "语言 / Language", ["中文", "English"], horizontal=True,
            index=0 if st.session_state.get("lang", "zh") == "zh" else 1,
            key="language_choice")
        new_lang = "zh" if lang_label == "中文" else "en"
        if st.session_state.get("lang") != new_lang:
            st.session_state.lang = new_lang
            st.rerun()

        st.divider()
        theme_label = st.radio(
            "主题 / Theme",
            [tr("深色 Dark", "深色 Dark"), tr("浅色 Light", "浅色 Light")],
            horizontal=True, key="theme_choice",
            index=0 if st.session_state.get("theme", "dark") == "dark" else 1,
            help=tr("切换深色 / 浅色主题，立即生效。",
                    "Switch between dark and light themes; applies immediately."))
        new_theme = "light" if "浅色" in theme_label or "Light" in theme_label.split()[-1] else "dark"
        if st.session_state.get("theme") != new_theme:
            st.session_state.theme = new_theme
            st.rerun()

        if st.button(tr("使用说明 / Docs", "使用说明 / Docs"),
                     use_container_width=True, key="open_docs_sidebar",
                     help=tr("打开完整说明文档：模型架构、训练方式、工作区操作、指标与常见问题。",
                             "Open the full manual: architectures, training, workspaces, metrics, FAQ.")):
            open_documentation()
        st.markdown(f'<div class="tj-version">Zimo Studio v{APP_VERSION} · MIT</div>',
                    unsafe_allow_html=True)

    arch_params = None
    if arch_key == "gpt2" and not dim_invalid:
        arch_params = {"layers": int(layers), "dim": int(dim),
                       "heads": int(heads), "block": int(block)}

    return {
        "upload": upload, "arch_key": arch_key, "arch_kind": arch_kind,
        "epochs": int(epochs), "batch": int(batch), "tokens": int(tokens),
        "use_gpu": hardware.startswith("GPU"), "start": start, "stop": stop,
        "arch_params": arch_params, "clean_opts": clean_opts,
        "local_paths": local_paths,
        "max_vocab": int(max_vocab), "val_ratio": float(val_ratio),
        "learning_rate": float(learning_rate), "warmup": int(warmup),
    }


def metric_cards(status: dict, proc_running: bool) -> None:
    step = int(status.get("step", 0) or 0)
    total = int(status.get("total_steps", 0) or 0)
    epochs = max(1, int(st.session_state.get("epochs", 1)))
    spe = int(status.get("steps_per_epoch", 0) or 0)
    tokens = int(status.get("tokens", 0) or 0)
    if total <= 0 and spe > 0:
        total = spe * epochs
    if total <= 0:
        actual = int(status.get("tokens") or st.session_state.get("dataset_tokens") or 0)
        budget = int(st.session_state.get("tokens_budget") or 0)
        if budget > 0:
            actual = min(actual, budget) if actual else budget
        if actual > 0:
            spe = max(1, (actual // 65) // max(1, int(st.session_state.get("batch", 32))))
            total = spe * epochs
    pct = min(100.0, 100.0 * step / total) if total else 0.0
    epoch = int(status.get("epoch", 0) or 0)

    metrics = st.session_state.get("metrics", [])
    val_points = st.session_state.get("val_points", [])
    val_ppls = [p["ppl"] for p in val_points
                if isinstance(p.get("ppl"), (int, float))]
    best_val = min(val_ppls) if val_ppls else status.get("val_ppl")

    loss = status.get("last_loss")
    ppl = status.get("last_ppl")
    if loss is None and metrics:
        loss = metrics[-1].get("loss")
    if ppl is None and metrics:
        ppl = metrics[-1].get("ppl")
    speed = status.get("tokens_per_s")
    if (not isinstance(speed, (int, float)) or speed <= 0) and len(metrics) >= 2:
        a, b = metrics[-2], metrics[-1]
        ds = int(b.get("step", 0)) - int(a.get("step", 0))
        dt = float(b.get("t", 0)) - float(a.get("t", 0))
        if ds > 0 and dt > 0:
            speed = ds * int(st.session_state.get("batch", 32)) * 64 / dt
    eta = status.get("eta_s")
    if not isinstance(eta, (int, float)) and total > step > 0:
        elapsed = time.time() - float(st.session_state.get("started_at") or time.time())
        eta = (total - step) * elapsed / max(step, 1)

    per_step = (tokens / spe) if spe and tokens else int(
        st.session_state.get("batch", 32)) * 64
    processed = step * per_step
    total_passes = tokens * epochs if tokens else 0

    st.markdown(f"""
<div class="tj-monitor-head">
  <div class="tj-ring" style="--pct:{pct:.1f}"><span>{pct:.0f}%</span></div>
  <div class="tj-monitor-meta">
    <div class="big">{tr("Step", "Step")} {step:,} / {total:,}</div>
    <div class="small">{tr("Epoch", "Epoch")} {min(epoch + 1, epochs)} / {epochs}
      · {tr("ETA", "ETA")} {fmt_time(eta)}
      {('· ' + tr('运行中', 'running')) if proc_running else ''}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.progress(min(1.0, pct / 100.0))

    row1 = st.columns(4)
    row1[0].metric(tr("Loss", "Loss"),
                   f"{loss:.4f}" if isinstance(loss, (int, float)) else "–")
    row1[1].metric(tr("Perplexity", "Perplexity"),
                   f"{ppl:.1f}" if isinstance(ppl, (int, float)) else "–")
    row1[2].metric(tr("最佳验证 PPL", "Best val PPL"),
                   f"{best_val:.1f}" if isinstance(best_val, (int, float)) else "–")
    row1[3].metric(tr("吞吐 (tok/s)", "Throughput (tok/s)"),
                   fmt_compact(speed) if isinstance(speed, (int, float)) else "–")

    row2 = st.columns(4)
    row2[0].metric(tr("已处理 token", "Processed tokens"),
                   f"{fmt_compact(processed)} / {fmt_compact(total_passes)}")
    row2[1].metric(tr("Epoch", "Epoch"), f"{min(epoch + 1, epochs)} / {epochs}")
    params = status.get("params")
    row2[2].metric(tr("参数量", "Parameters"),
                   fmt_compact(params) if isinstance(params, (int, float)) else "–")
    vocab = status.get("vocab")
    row2[3].metric(tr("字符表", "Vocabulary"),
                   f"{int(vocab):,}" if isinstance(vocab, (int, float)) else "–")


def _line_chart(df: pd.DataFrame, key: str, color: str,
                log_scale: bool = False) -> Optional[alt.Chart]:
    if df.empty or key not in df:
        return None
    y_scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
    number_format = ".4f" if key == "loss" else (".1f" if key == "ppl" else ".0f")
    return (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="step"),
            y=alt.Y(f"{key}:Q", title=key, scale=y_scale),
            tooltip=["step:Q", alt.Tooltip(f"{key}:Q", format=number_format)],
        )
        .properties(height=240)
    )


def monitor_charts() -> None:
    metrics = st.session_state.get("metrics", [])
    val_points = st.session_state.get("val_points", [])
    if not metrics:
        st.info(tr("等待第一条 loss 记录……", "Waiting for the first loss point…"))
        return
    frame = pd.DataFrame(metrics)
    if "step" not in frame:
        st.info(tr("指标格式不完整。", "Metric format is incomplete."))
        return
    frame = frame.dropna(subset=["step"]).sort_values("step")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{tr('Loss', 'Loss')}**")
        loss_df = frame.dropna(subset=["loss"])
        chart = _line_chart(loss_df, "loss", "#6ea8ff")
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption(tr("等待 loss 数据……", "Waiting for loss data…"))

    with right:
        st.markdown(f"**{tr('Perplexity', 'Perplexity')}**")
        ppl_df = frame.dropna(subset=["ppl"])
        ppl_df = ppl_df[ppl_df["ppl"] > 0]
        chart = _line_chart(ppl_df, "ppl", "#34d399", log_scale=True)
        val_df = pd.DataFrame(val_points)
        if chart is not None and not val_df.empty and "ppl" in val_df:
            val_df = val_df.dropna(subset=["ppl"])
            val_df = val_df[val_df["ppl"] > 0]
            if not val_df.empty:
                points = (
                    alt.Chart(val_df)
                    .mark_point(size=55, filled=True, color="#e8eefc")
                    .encode(
                        x=alt.X("step:Q"),
                        y=alt.Y("ppl:Q", scale=alt.Scale(type="log")),
                        tooltip=["step:Q", alt.Tooltip("ppl:Q", format=".1f")],
                    )
                )
                chart = (chart + points).resolve_scale(y="shared")
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption(tr("等待 PPL 数据……", "Waiting for PPL data…"))

    status = st.session_state.get("status", {})
    spe = int(status.get("steps_per_epoch", 0) or 0)
    tokens = int(status.get("tokens", 0) or 0)
    per_step = (tokens / spe) if spe and tokens else int(
        st.session_state.get("batch", 32)) * 64
    speed_rows = []
    for i in range(1, len(metrics)):
        a, b = metrics[i - 1], metrics[i]
        ds = int(b.get("step", 0)) - int(a.get("step", 0))
        dt = float(b.get("t", 0)) - float(a.get("t", 0))
        if ds > 0 and dt > 0:
            speed_rows.append({"step": int(b["step"]), "tps": ds * per_step / dt})
    st.markdown(f"**{tr('吞吐 (tok/s)', 'Throughput (tok/s)')}**")
    speed_df = pd.DataFrame(speed_rows)
    speed_chart = _line_chart(speed_df, "tps", "#22d3ee") if not speed_df.empty else None
    if speed_chart is not None:
        st.altair_chart(speed_chart, use_container_width=True)
    else:
        st.caption(tr("等待吞吐数据……", "Waiting for throughput data…"))


def model_info(status: dict, run_dir: Optional[Path]) -> None:
    with st.expander(tr("模型与数据", "Model & data"), expanded=False):
        rows = [
            (tr("架构", "Architecture"), status.get("arch", "–")),
            (tr("参数量", "Parameters"),
             f"{int(status['params']):,}" if isinstance(status.get("params"), (int, float)) else "–"),
            (tr("字符表", "Vocabulary"),
             f"{int(status['vocab']):,}" if isinstance(status.get("vocab"), (int, float)) else "–"),
            (tr("训练 token", "Training tokens"),
             f"{int(status['tokens']):,}" if isinstance(status.get("tokens"), (int, float)) else "–"),
            (tr("每 epoch 步数", "Steps/epoch"),
             f"{int(status['steps_per_epoch']):,}" if isinstance(status.get("steps_per_epoch"), (int, float)) else "–"),
            (tr("总步数", "Total steps"),
             f"{int(status['total_steps']):,}" if isinstance(status.get("total_steps"), (int, float)) else "–"),
            (tr("设备", "Device"), status.get("device", "–")),
            (tr("状态", "State"), status.get("state", "–")),
            (tr("数据目录", "Data directory"), str(run_dir) if run_dir else "–"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=[tr("项", "Field"), tr("值", "Value")]),
            use_container_width=True, hide_index=True)


def log_panel() -> None:
    lines = st.session_state.get("log", [])
    st.markdown(f"**{tr('运行日志', 'Run log')}**")
    with st.container(height=300, border=True):
        if not lines:
            st.caption(tr("（等待 stdout……）", "(waiting for stdout…)"))
            return
        st.code(chr(10).join(lines[-240:]), language="text")


def test_panel(arch_key: str, run_dir: Path) -> None:
    st.markdown(f"### {tr('测试重排', 'Test rerank')}")
    if arch_key == "mlp":
        st.info(tr(
            "MLP Listwise Ranker 使用 32 维特征向量，不能直接对原始文本算 PPL。"
            "本面板只支持 GPT-2 / LSTM；MLP 的导出文件在运行目录里。",
            "MLP Listwise Ranker uses 32-dim feature vectors, so it cannot "
            "score raw text PPL. This panel supports GPT-2 / LSTM only; "
            "the MLP export is in the run directory."))
        return

    model, vocab = find_model(run_dir, arch_key)
    if model is None or vocab is None:
        st.warning(tr("还没有可用的 model.pt / vocab.txt；先完成一次训练。",
                      "No model.pt / vocab.txt yet; finish a training run first."))
        return

    st.caption(f"scorer: `{model.name}` · `{vocab.name}`")
    context = st.text_input(
        tr("前文上下文", "Preceding context"),
        placeholder=tr("例如：小王昨天", "e.g. Xiaowang yesterday"),
        key="score_context")
    candidates_text = st.text_area(
        tr("候选句子（每行一个）", "Candidate sentences (one per line)"),
        height=150, key="score_candidates",
        placeholder=tr(
            "他说的这个事情我也不太清楚\n她说的这个事情我也不太清楚",
            "I am not sure about this thing he said\n"
            "I am not sure about this thing she said"))
    if st.button(tr("测试重排", "Test rerank"), type="primary"):
        candidates = [line.strip() for line in candidates_text.splitlines()
                      if line.strip()]
        if not candidates:
            st.warning(tr("至少输入一个候选句子。", "Enter at least one candidate."))
            return
        if len(candidates) > 20:
            st.warning(tr("一次最多测试 20 条，已截断。",
                          "At most 20 candidates per run; truncated."))
            candidates = candidates[:20]
        try:
            scorer = load_scorer(
                arch_key, str(model), str(vocab),
                os.path.getmtime(model))
        except Exception as exc:  # noqa: BLE001
            st.error(f"{tr('加载模型失败', 'Failed to load model')}: {exc}")
            return

        rows = []
        progress = st.progress(0.0, text="scoring…")
        for index, candidate in enumerate(candidates, start=1):
            result = score_candidate(scorer, context, candidate)
            if result is None:
                rows.append({"candidate": candidate, "ppl": None,
                             "mean_nll": None, "chars": 0})
            else:
                rows.append({
                    "candidate": candidate,
                    "ppl": round(result["ppl"], 2),
                    "mean_nll": round(result["mean_nll"], 4),
                    "chars": result["scored_chars"],
                })
            progress.progress(index / len(candidates),
                              text=f"scoring {index}/{len(candidates)}")
        progress.empty()
        rows.sort(key=lambda r: (r["ppl"] is None, r["ppl"] or 1e18))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        best = next((r for r in rows if r["ppl"] is not None), None)
        if best:
            st.success(
                f"{tr('最高分候选', 'Top candidate')}：**{best['candidate']}**"
                f"（PPL {best['ppl']:.2f}）")
        labels = {
            "rank": tr("排名 (Rank)", "Rank"),
            "candidate": tr("候选句 (Candidate)", "Candidate"),
            "ppl": tr("困惑度 (PPL)", "PPL"),
            "mean_nll": tr("平均NLL (Mean NLL)", "Mean NLL"),
            "chars": tr("参与字数 (Chars)", "Chars"),
        }
        table = (pd.DataFrame(rows)[list(labels)]
                 .rename(columns=labels))
        st.dataframe(_highlight_rows(table, labels["rank"], labels["ppl"]),
                     use_container_width=True)

        with st.expander(tr(
                "指标说明",
                "Metric definitions")):
            st.markdown(tr(
                """- **PPL（困惑度）**：衡量模型对句子的不确定度；数值越低，表示句子越符合训练分布。
- **NLL（平均负对数似然）**：PPL 的底层统计量；数值越低越好。
- **Rank**：按模型分数排序后的名次；1 为最高。
- **参与字数 (Chars)**：参与打分的字符数；空白字符不计。""",
                """- **PPL (perplexity)**: measures the model's uncertainty over the sentence; lower values indicate it is closer to the training distribution.
- **NLL (mean negative log-likelihood)**: the underlying statistic for PPL; lower is better.
- **Rank**: position after sorting by model score; 1 is highest.
- **Chars**: number of scored characters; whitespace is excluded."""))

        st.caption(tr(
            "PPL 越低表示候选越符合训练分布；该分数不包含拼音、"
            "词典或用户选择证据。",
            "Lower PPL means the candidate is closer to the training "
            "distribution; this score does not include pinyin, dictionary, "
            "or user-selection evidence."))


def _playground_last_path(slot: str) -> Path:
    return PLAYGROUND_DIR / f"last_{slot}.json"


def read_playground_last(slot: str) -> dict:
    data = read_json_safe(_playground_last_path(slot))
    return data if isinstance(data, dict) else {}


def save_playground_last(slot: str, arch_key: str, model_path: Path,
                         vocab_path: Optional[Path]) -> None:
    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
    _playground_last_path(slot).write_text(
        json.dumps({"arch": arch_key, "model": str(model_path),
                    "vocab": str(vocab_path) if vocab_path else ""},
                   ensure_ascii=False),
        encoding="utf-8")


def save_playground_upload(upload) -> Path:
    """Persist an uploaded model/vocab to disk and return its path.

    The digest makes repeated uploads idempotent and lets load_scorer() reuse
    its cache instead of reloading the checkpoint on every Streamlit rerun.
    """
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    name = Path(upload.name).name
    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
    path = PLAYGROUND_DIR / f"{digest}_{name}"
    if not path.exists():
        path.write_bytes(data)
    return path


def _vocab_words_json(model_path: Path, vocab_path: Optional[Path]) -> str:
    """JSON word list for the scorer cache key (embedded vocab or sidecar)."""
    if vocab_path and Path(vocab_path).is_file():
        return ""
    return json.dumps(
        inspect_checkpoint_file(model_path)["words"] or [],
        ensure_ascii=False)


def _load_playground_entry(slot: str, arch_key: str, model_path: Path,
                           vocab_path: Optional[Path] = None,
                           restored: bool = False):
    words_json = _vocab_words_json(model_path, vocab_path)
    key = (arch_key, str(model_path), str(vocab_path or ""), words_json,
           model_path.stat().st_mtime)
    current = st.session_state.get(slot)
    if current and current.get("key") == key:
        if restored:
            current["restored"] = True
        return current
    try:
        with st.spinner(tr("加载模型中…", "Loading model…")):
            scorer = load_scorer(
                arch_key, str(model_path), str(vocab_path or ""),
                model_path.stat().st_mtime, vocab_words_json=words_json)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{tr('模型加载失败', 'Model load failed')}: {exc}")
        st.session_state[slot] = None
        return None
    entry = {
        "key": key,
        "arch": arch_key,
        "scorer": scorer,
        "vocab_size": len(scorer["vocab"]) + 1,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path) if vocab_path else "",
        "vocab_embedded": bool(words_json),
        "restored": restored,
    }
    st.session_state[slot] = entry
    return entry


def load_model_source(slot: str, arch_key: str, model_path: Path,
                      vocab_path: Optional[Path] = None,
                      persisted: bool = True):
    """Shared single-source loader (playground manual/auto, export tab)."""
    entry = _load_playground_entry(slot, arch_key, model_path, vocab_path)
    if entry is not None and persisted:
        save_playground_last(slot, arch_key, model_path, vocab_path)
    return entry


def ensure_playground_model(slot: str, arch_key: str, model_upload,
                            vocab_upload, last: dict):
    """Load A/B from uploads, or restore the last saved pair after a refresh."""
    if model_upload is None and vocab_upload is None:
        if last and last.get("arch") == arch_key:
            model_path = Path(last.get("model", ""))
            vocab_raw = last.get("vocab", "")
            vocab_path = Path(vocab_raw) if vocab_raw else None
            if model_path.is_file() and (vocab_path is None or vocab_path.is_file()):
                return _load_playground_entry(
                    slot, arch_key, model_path, vocab_path, restored=True)
        st.session_state[slot] = None
        return None
    if model_upload is None:
        st.session_state[slot] = None
        return None
    try:
        model_path = save_playground_upload(model_upload)
        vocab_path = (save_playground_upload(vocab_upload)
                      if vocab_upload is not None else None)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{tr('保存上传文件失败', 'Failed to save upload')}: {exc}")
        st.session_state[slot] = None
        return None
    return load_model_source(slot, arch_key, model_path, vocab_path)


def playground_model_type(key: str, default_arch: str) -> str:
    options = ["gpt2", "lstm"]
    index = 0 if default_arch == "gpt2" else 1
    return st.selectbox(
        tr("模型类型", "Model type"), options, index=index,
        format_func=lambda value: tr(
            "GPT-2 字符级 LM", "GPT-2 character LM") if value == "gpt2"
        else tr("LSTM 语言模型", "LSTM language model"),
        key=key,
        help=tr("上传的 .pt 是哪种架构就选哪种；选错会加载失败。",
                "Pick the architecture of the uploaded .pt; a mismatch fails to load."))


def _playground_slot_ui(slot: str, title_zh: str, title_en: str,
                        optional: bool) -> Optional[dict]:
    """One playground slot: auto-load latest / manual upload (smart vocab)."""
    lang = st.session_state.get("lang", "zh")
    st.markdown(f"#### {tr(title_zh, title_en)}")
    arch = playground_model_type(f"{slot}_arch",
                                 read_playground_last(slot).get("arch", "gpt2"))

    source = st.radio(
        tr("模型来源", "Model source"),
        [tr("自动加载最新训练", "Auto-load latest run"),
         tr("手动上传模型", "Manual upload")],
        index=0, horizontal=True, key=f"{slot}_source",
        help=tr("自动模式在 runs/ 里找最新的 model(.best).pt；"
                "词表优先用同名 vocab.txt，其次读 checkpoint 内置词表。",
                "Auto mode finds the newest model(.best).pt under runs/; "
                "vocab comes from the sidecar file or the checkpoint itself."))

    if source.startswith(tr("自动加载最新训练", "Auto-load latest run")):
        latest = find_latest_run(RUNS_DIR)
        using_demo = False
        if latest is None:
            latest = find_latest_run(DEMO_MODEL_DIR)
            using_demo = latest is not None
        if latest is None:
            st.info(tr("runs/ 里还没有可用的训练产物；先训练一次，或改用手动上传。",
                       "No usable run under runs/ yet; train first or upload manually."))
            st.session_state[slot] = None
            return None
        entry = load_model_source(slot, arch, latest["model"],
                                  latest["vocab"], persisted=False)
        if entry is not None:
            src = tr("内置示例", "Bundled demo") if using_demo else tr("自动加载", "Auto-loaded")
            vocab_note = (tr("词表来自 vocab.txt", "vocab from vocab.txt")
                          if latest["vocab"] else
                          tr("词表来自模型内置", "vocab from checkpoint"))
            st.success(tr(
                f"{src}：{latest['model'].name} · 词表大小 {entry['vocab_size']}（{vocab_note}）",
                f"{src}: {latest['model'].name} · vocab {entry['vocab_size']} ({vocab_note})"))
        return entry

    # ---- manual upload: single .pt first, vocab.txt only when needed ----
    model_upload = st.file_uploader(
        tr("模型文件（.pt/.pth/.ckpt/.bin/.safetensors）",
           "Model file (.pt/.pth/.ckpt/.bin/.safetensors)"),
        type=[suffix.lstrip(".") for suffix in CHECKPOINT_SUFFIXES],
        key=f"{slot}_model",
        help=tr("新版训练器的 .pt 内置词表，上传一个文件即可；"
                "旧版模型会在下一步要求补充 vocab.txt。",
                "New-trainer .pt files embed the vocab; legacy ones ask for "
                "vocab.txt in the next step."))
    if model_upload is None:
        st.session_state[slot] = None
        return None
    try:
        model_path = save_playground_upload(model_upload)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{tr('保存上传文件失败', 'Failed to save upload')}: {exc}")
        st.session_state[slot] = None
        return None

    info = inspect_checkpoint_file(model_path)
    if not info["loadable"]:
        st.error(tr("该文件不是可加载的 PyTorch checkpoint。",
                    "This file is not a loadable PyTorch checkpoint."))
        st.session_state[slot] = None
        return None

    vocab_upload = None
    if not info["words"]:
        st.warning(tr(
            "检测到旧版模型（无内置词表），请补充上传配套的 vocab.txt。",
            "Legacy model detected (no embedded vocab); upload the paired vocab.txt."))
        vocab_upload = st.file_uploader(
            tr("词表文件 (vocab.txt)", "Vocabulary (vocab.txt)"),
            type=["txt"], key=f"{slot}_vocab")
        if vocab_upload is None:
            st.session_state[slot] = None
            return None

    vocab_path = (save_playground_upload(vocab_upload)
                  if vocab_upload is not None else None)
    entry = load_model_source(slot, arch, model_path, vocab_path)
    if entry is not None:
        vocab_note = (tr("词表来自模型内置", "vocab from checkpoint")
                      if not vocab_path else tr("词表来自 vocab.txt", "vocab from vocab.txt"))
        st.success(tr(
            f"成功加载{tr('模型', 'model')}，词表大小：{entry['vocab_size']}（{vocab_note}）",
            f"Model loaded, vocab size: {entry['vocab_size']} ({vocab_note})"))
    return entry


def _highlight_rows(table: pd.DataFrame, rank_col: str, ppl_col: str,
                    green_max=None, red_min=None, protected_col=None):
    """Row tinting.  Defaults: green for Rank 1, red for PPL far above best.
    With explicit thresholds: green at PPL <= green_max, red at PPL >= red_min.
    Rows flagged by ``protected_col`` never get the red tint (custom dict)."""
    GREEN = "background-color: rgba(52,211,153,.16)"
    RED = "background-color: rgba(251,113,133,.16)"
    ppl_values = pd.to_numeric(table[ppl_col], errors="coerce")

    def _is_protected(row):
        if not protected_col or protected_col not in table.columns:
            return False
        value = str(row[protected_col]).strip()
        return value not in ("", "nan", "None", "否")

    def _style(row):
        ppl = ppl_values[row.name]
        if _is_protected(row):
            return [""] * len(row)
        if pd.notna(ppl):
            if red_min is not None and ppl >= red_min:
                return [RED] * len(row)
            if green_max is not None and ppl <= green_max:
                return [GREEN] * len(row)
            best = ppl_values.min()
            threshold = (best * 5) if pd.notna(best) and best > 0 else float("inf")
            if ppl > threshold:
                return [RED] * len(row)
            if int(row[rank_col]) == 1:
                return [GREEN] * len(row)
        return [""] * len(row)

    return table.style.apply(_style, axis=1)


def render_playground() -> None:
    st.markdown(f"### {tr('模型测试台', 'Model playground')}")
    st.caption(tr(
        "自动加载 runs/ 里最新的训练产物，或手动上传模型，直接测试候选句。"
        "测试台独立于左侧训练配置，不会启动训练。",
        "Auto-load the newest run under runs/ or upload a model, then score "
        "candidates. Independent of the training sidebar."))

    col_a, col_b = st.columns(2)
    with col_a:
        entry_a = _playground_slot_ui("play_a", "模型 A（必填）",
                                      "Model A (required)", optional=False)
    with col_b:
        entry_b = _playground_slot_ui("play_b", "模型 B（可选，A/B 对比）",
                                      "Model B (optional, A/B)", optional=True)

    # ---- model spec comparison ----
    spec_entries = [("A", entry_a)]
    if entry_b is not None:
        spec_entries.append(("B", entry_b))
    loaded_specs = [(name, ent) for name, ent in spec_entries if ent]
    if loaded_specs:
        with st.expander(tr("模型参数对比", "Model specs"), expanded=True):
            spec_rows = {
                tr("文件", "File"): [Path(ent["model_path"]).name for _, ent in loaded_specs],
                tr("架构", "Architecture"): [ent["arch"].upper() for _, ent in loaded_specs],
                tr("参数量", "Parameters"): [f"{ent['scorer']['meta']['params']:,}" for _, ent in loaded_specs],
                tr("层数", "Layers"): [str(ent["scorer"]["meta"]["layers"]) for _, ent in loaded_specs],
                tr("维度", "Dim"): [str(ent["scorer"]["meta"]["dim"]) for _, ent in loaded_specs],
                tr("头数", "Heads"): [str(ent["scorer"]["meta"]["heads"]) if ent["scorer"]["meta"]["heads"] else "–"
                                      for _, ent in loaded_specs],
                tr("上下文", "Context"): [str(ent["scorer"]["meta"]["block"]) for _, ent in loaded_specs],
                tr("词表", "Vocab"): [str(ent["vocab_size"]) for _, ent in loaded_specs],
                tr("文件大小", "File size"): [f"{ent['scorer']['meta']['file_mb']:.1f} MB" for _, ent in loaded_specs],
                tr("词表来源", "Vocab source"): [tr("模型内置", "checkpoint") if ent.get("vocab_embedded")
                                                else tr("vocab.txt", "vocab.txt") for _, ent in loaded_specs],
            }
            spec_table = pd.DataFrame(spec_rows, index=[name for name, _ in loaded_specs]).T
            st.dataframe(spec_table, use_container_width=True)
            if len(loaded_specs) == 2:
                pa = loaded_specs[0][1]["scorer"]["meta"]["params"]
                pb = loaded_specs[1][1]["scorer"]["meta"]["params"]
                if pa and pb:
                    st.caption(tr(f"参数差距：B 是 A 的 {pb / pa:.2f}×",
                                  f"Size ratio: B is {pb / pa:.2f}x A"))

    st.markdown(f"### {tr('测试', 'Test')}")

    modes = [tr("模型 A", "Model A")]
    if entry_b is not None:
        modes += [tr("模型 B", "Model B"), tr("A / B 对比", "A / B compare")]
    mode = st.radio(tr("测试方式", "Test mode"), modes, horizontal=True,
                    key="play_mode",
                    help=tr("先选单个模型测试，或同时跑两个模型做对比。",
                            "Test one model, or run both for a comparison."))
    primary = entry_b if mode.startswith(tr("模型 B", "Model B")) else entry_a
    comparing = (mode.startswith("A / B") or mode.startswith(tr("A / B", "A / B"))) and entry_b is not None

    with st.expander(tr("自定义词典保护（专有名词）", "Custom dictionary protection")):
        dict_up = st.file_uploader(
            tr("custom_dict.txt（每行一个专有名词）",
               "custom_dict.txt (one term per line)"),
            type=["txt"], key="play_custom_dict",
            help=tr("包含这些词的候选不会被标红；可选择性优先排序。",
                    "Candidates containing these terms are never flagged red; "
                    "optional priority sorting."))
        if dict_up is not None:
            terms = [ln.strip() for ln in dict_up.read().decode("utf-8-sig").splitlines()
                     if ln.strip()]
            st.session_state["custom_terms"] = sorted(set(terms))
        terms_now = st.session_state.get("custom_terms") or []
        if terms_now:
            st.caption(tr(f"已加载 {len(terms_now)} 个保护词。",
                          f"{len(terms_now)} protected terms loaded."))
        st.checkbox(tr("包含保护词的候选优先排序", "Sort protected candidates first"),
                    key="protect_first")

    thr_enabled = st.checkbox(tr("自定义高亮阈值", "Custom highlight thresholds"),
                              key="play_thr_enabled",
                              help=tr("开启后按 PPL 数值决定绿/红，而不是只标 Rank 1。",
                                      "Colour by PPL value instead of rank only."))
    green_thr, red_thr = 20.0, 300.0
    if thr_enabled:
        tc1, tc2 = st.columns(2)
        green_thr = tc1.slider(tr("PPL ≤ 标绿", "PPL <= green"),
                               min_value=1.0, max_value=500.0,
                               value=20.0, step=1.0, key="play_green_thr")
        red_thr = tc2.slider(tr("PPL ≥ 标红", "PPL >= red"),
                             min_value=1.0, max_value=2000.0,
                             value=300.0, step=10.0, key="play_red_thr")
    green_max = green_thr if thr_enabled else None
    red_min = red_thr if thr_enabled else None
    context = st.text_input(
        tr("前文上下文", "Preceding context"),
        placeholder=tr("例如：小王昨天", "e.g. Xiaowang yesterday"),
        key="play_context",
        help=tr("用户已输入的前文；模型据此评估候选句的条件概率。",
                "The text the user has already typed. The model uses it as context."))
    candidates_text = st.text_area(
        tr("候选句子（每行一个）", "Candidate sentences (one per line)"),
        height=150, key="play_candidates",
        placeholder=tr("他说的这个事情我也不太清楚\n她说的这个事情我也不太清楚",
                       "I am not sure about this thing he said\n"
                       "I am not sure about this thing she said"),
        help=tr("待评估的候选句，每行一条；结果按模型分数升序排列。",
                "The sentences to test, one per line. The model reranks them by naturalness."))

    if st.button(tr("测试", "Test"), type="primary", key="play_run"):
        if primary is None:
            st.warning(tr("先加载要测试的模型。", "Load the model you want to test first."))
            return
        candidates = [line.strip() for line in candidates_text.splitlines()
                      if line.strip()]
        if not candidates:
            st.warning(tr("至少输入一个候选句子。", "Enter at least one candidate."))
            return
        if len(candidates) > 20:
            st.warning(tr("一次最多测试 20 条，已截断。",
                          "At most 20 candidates per run; truncated."))
            candidates = candidates[:20]

        custom_terms = st.session_state.get("custom_terms") or []
        protect_first = bool(st.session_state.get("protect_first"))
        secondary = entry_b if (comparing and entry_a is primary) else (
            entry_a if (comparing and entry_b is primary) else None)

        rows = []
        progress = st.progress(0.0, text="scoring…")
        for index, candidate in enumerate(candidates, start=1):
            p_res = score_candidate(primary["scorer"], context, candidate)
            s_res = (score_candidate(secondary["scorer"], context, candidate)
                     if secondary is not None else None)
            p_ppl = round(p_res["ppl"], 2) if p_res else None
            s_ppl = round(s_res["ppl"], 2) if s_res else None
            row = {
                "candidate": candidate,
                "p_ppl": p_ppl,
                "p_nll": round(p_res["mean_nll"], 4) if p_res else None,
                "s_ppl": s_ppl,
                "s_nll": round(s_res["mean_nll"], 4) if s_res else None,
                "chars": p_res["scored_chars"] if p_res else 0,
                "protected": any(term and term in candidate
                                 for term in custom_terms),
            }
            if comparing and p_ppl is not None and s_ppl is not None:
                row["delta"] = round(p_ppl - s_ppl, 2)
                row["winner"] = ("A" if p_ppl < s_ppl
                                 else "B" if s_ppl < p_ppl else "=")
            rows.append(row)
            progress.progress(index / len(candidates),
                              text=f"scoring {index}/{len(candidates)}")
        progress.empty()

        if protect_first and custom_terms:
            rows.sort(key=lambda row: (
                not row["protected"], row["p_ppl"] is None,
                row["p_ppl"] or 1e18))
        else:
            rows.sort(key=lambda row: (
                row["p_ppl"] is None, row["p_ppl"] or 1e18))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank

        if custom_terms:
            for row in rows:
                row["protected"] = tr("是", "yes") if row["protected"] else ""

        primary_name = ("B" if primary is entry_b else "A")
        if comparing:
            labels = {
                "rank": tr("排名 (Rank)", "Rank"),
                "candidate": tr("候选句 (Candidate)", "Candidate"),
                "p_ppl": tr(f"模型{primary_name} (PPL)", f"Model {primary_name} (PPL)"),
                "s_ppl": tr("另一模型 (PPL)", "Other model (PPL)"),
                "delta": tr("PPL 差 (本-另一)", "PPL delta"),
                "winner": tr("更优", "Better"),
                "p_nll": tr("本模型 NLL", "NLL (this)"),
                "s_nll": tr("另一模型 NLL", "NLL (other)"),
                "chars": tr("参与字数 (Chars)", "Chars"),
            }
        else:
            labels = {
                "rank": tr("排名 (Rank)", "Rank"),
                "candidate": tr("候选句 (Candidate)", "Candidate"),
                "p_ppl": tr("困惑度 (PPL)", "PPL"),
                "p_nll": tr("平均NLL", "Mean NLL"),
                "chars": tr("参与字数 (Chars)", "Chars"),
            }
        if custom_terms:
            labels["protected"] = tr("词库保护", "Protected")
        table = pd.DataFrame(rows)[list(labels)].rename(columns=labels)
        st.dataframe(
            _highlight_rows(table, labels["rank"], labels["p_ppl"],
                            green_max=green_max, red_min=red_min,
                            protected_col=labels["protected"] if custom_terms else None),
            use_container_width=True)

        best = next((row for row in rows if row["p_ppl"] is not None), None)
        if best is not None:
            st.success(tr(
                f"模型 {primary_name} 最高分候选：**{best['candidate']}**（PPL {best['p_ppl']:.2f}）",
                f"Model {primary_name} best: **{best['candidate']}** "
                f"(PPL {best['p_ppl']:.2f})"))

        if comparing:
            valid = [row for row in rows
                     if row["p_ppl"] is not None and row["s_ppl"] is not None]
            if valid:
                avg_p = sum(r["p_ppl"] for r in valid) / len(valid)
                avg_s = sum(r["s_ppl"] for r in valid) / len(valid)
                wins_p = sum(1 for r in valid if r["p_ppl"] < r["s_ppl"])
                wins_s = sum(1 for r in valid if r["s_ppl"] < r["p_ppl"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(tr(f"模型{primary_name} 平均 PPL", f"Model {primary_name} avg PPL"),
                          f"{avg_p:.2f}")
                m2.metric(tr("另一模型 平均 PPL", "Other avg PPL"), f"{avg_s:.2f}")
                m3.metric(tr(f"{primary_name} 胜出", f"{primary_name} wins"),
                          f"{wins_p}/{len(valid)}")
                m4.metric(tr("另一模型 胜出", "Other wins"), f"{wins_s}/{len(valid)}")
                st.caption(tr(
                    f"平均差 {avg_p - avg_s:+.2f}（负值表示 {primary_name} 更贴合训练分布）；"
                    "PPL 差列为每句的逐条对比。",
                    f"Mean delta {avg_p - avg_s:+.2f} (negative favours {primary_name}); "
                    "the delta column compares each sentence."))
        st.caption(tr(
            "PPL 越低表示候选越符合训练分布；切换上方「测试方式」可单独测某个模型。",
            "Lower PPL means more natural; switch the test mode above to score a single model."))

    # ---- general tools: per-character NLL / batch scoring / generation ----
    if primary is not None:
        with st.expander(tr("工具：逐字 NLL（哪个字最意外）",
                            "Tool: per-character NLL (what surprised the model)"),
                         expanded=False):
            sentence = st.text_input(
                tr("要分析的句子", "Sentence to analyse"), key="tool_sentence",
                placeholder=tr("例如：今天天气不错", "e.g. the weather is nice today"))
            if st.button(tr("分析", "Analyse"), key="tool_char_run"):
                if not sentence.strip():
                    st.warning(tr("先输入一个句子。", "Enter a sentence first."))
                else:
                    details = char_nlls(primary["scorer"], context, sentence.strip())
                    if not details:
                        st.warning(tr("没有可分析的字符。", "No characters to analyse."))
                    else:
                        st.dataframe(pd.DataFrame(details), use_container_width=True)
                        chart = pd.DataFrame(details).set_index("char")["nll"]
                        st.bar_chart(chart)

        with st.expander(tr("工具：批量打分（每行一句，导出 CSV）",
                            "Tool: batch scoring (one sentence per line, CSV export)"),
                         expanded=False):
            batch_upload = st.file_uploader(
                tr("批量文本 .txt（每行一句）", "Batch text .txt (one sentence per line)"),
                type=["txt"], key="tool_batch_file")
            if batch_upload is not None and st.button(
                    tr("开始批量打分", "Run batch scoring"), key="tool_batch_run"):
                lines = [ln.strip() for ln in
                         batch_upload.getvalue().decode("utf-8-sig", errors="replace").splitlines()
                         if ln.strip()]
                if len(lines) > 2000:
                    st.warning(tr("一次最多 2000 行，已截断。",
                                  "At most 2000 lines per run; truncated."))
                    lines = lines[:2000]
                out_rows = []
                bar = st.progress(0.0, text="scoring…")
                for i, line in enumerate(lines, start=1):
                    res = score_candidate(primary["scorer"], context, line)
                    out_rows.append({
                        tr("句子", "Sentence"): line,
                        tr("PPL", "PPL"): round(res["ppl"], 2) if res else None,
                        tr("平均NLL", "Mean NLL"): round(res["mean_nll"], 4) if res else None,
                        tr("字数", "Chars"): res["scored_chars"] if res else 0,
                    })
                    if i % 20 == 0 or i == len(lines):
                        bar.progress(i / len(lines), text=f"scoring {i}/{len(lines)}")
                bar.empty()
                frame = pd.DataFrame(out_rows).sort_values(tr("PPL", "PPL"))
                st.dataframe(frame, use_container_width=True)
                st.download_button(
                    tr("下载打分结果 CSV", "Download scores CSV"),
                    data=frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name="zimo_batch_scores.csv", mime="text/csv",
                    key="tool_batch_download")
                st.caption(tr("PPL 最高的行通常是最不像训练语料的句子。",
                              "The highest-PPL rows are usually the least in-domain."))

        with st.expander(tr("工具：文本生成（验证模型学到什么）",
                            "Tool: text generation (see what the model learned)"),
                         expanded=False):
            prompt = st.text_input(tr("前缀", "Prompt"), key="tool_gen_prompt",
                                   placeholder=tr("例如：今天", "e.g. today"))
            g1, g2, g3 = st.columns(3)
            max_new = g1.number_input(tr("生成字数", "New chars"), 1, 200, 30,
                                      key="tool_gen_len")
            temperature = g2.slider(tr("温度（0=贪心）", "Temperature (0 = greedy)"),
                                    0.0, 1.5, 0.8, 0.1, key="tool_gen_temp")
            top_k = g3.number_input(tr("Top-k（0=不限）", "Top-k (0 = off)"),
                                    0, 100, 0, key="tool_gen_topk")
            if st.button(tr("生成", "Generate"), key="tool_gen_run"):
                if not prompt.strip():
                    st.warning(tr("先输入前缀。", "Enter a prompt first."))
                else:
                    text = generate_text(primary["scorer"], prompt.strip(),
                                         int(max_new), float(temperature),
                                         int(top_k))
                    st.code(prompt.strip() + text, language="text")
                    st.caption(tr(
                        "这是自回归采样结果，只用来看模型学到了什么，不代表可用输出。",
                        "Sampled continuation, useful only to inspect what the model learned."))

    # ---- Dimension 3: custom correction-accuracy eval ("定制考题") ----
    render_correction_eval(entry_a)


def render_correction_eval(entry_a: Optional[dict]) -> None:
    """Score a user-supplied (correct, typo) CSV: Top-1 correction accuracy."""
    with st.expander(tr("定制考题：纠错准确率评测", "Custom eval: correction accuracy")):
        st.caption(tr(
            "CSV 两列：正确句子, 错别字句子。逐对打分，统计模型把正确句排在错句前的比例。",
            "CSV with two columns: correct sentence, typo sentence.  Pairs are "
            "scored; accuracy = how often the correct sentence wins."))
        st.markdown(f"**{tr('自己打字试试', 'Type your own')}**")
        pair_cols = st.columns(2)
        good_text = pair_cols[0].text_input(
            tr("正确句子", "Correct sentence"), key="corr_good",
            placeholder=tr("例如：今天天气不错", "e.g. the weather is nice today"))
        bad_text = pair_cols[1].text_input(
            tr("错别字句子", "Typo sentence"), key="corr_bad",
            placeholder=tr("例如：今天天汽不错", "e.g. the weathr is nice"))
        if st.button(tr("比对这两句", "Compare these two"), key="corr_pair_run"):
            if entry_a is None:
                st.warning(tr("先在上方加载模型 A。", "Load model A first."))
            elif not good_text.strip() or not bad_text.strip():
                st.warning(tr("两句都要填。", "Fill in both sentences."))
            else:
                res_g = score_candidate(entry_a["scorer"], "", good_text.strip())
                res_b = score_candidate(entry_a["scorer"], "", bad_text.strip())
                p_g = res_g["ppl"] if res_g else None
                p_b = res_b["ppl"] if res_b else None
                m1, m2, m3 = st.columns(3)
                m1.metric(tr("正确句 PPL", "Correct PPL"),
                          f"{p_g:.2f}" if p_g is not None else "–")
                m2.metric(tr("错句 PPL", "Typo PPL"),
                          f"{p_b:.2f}" if p_b is not None else "–")
                if p_g is not None and p_b is not None:
                    m3.metric(tr("差值", "Delta"), f"{p_g - p_b:+.2f}")
                    if p_g < p_b:
                        st.success(tr(
                            f"判定正确：模型认为正确句更自然（差距 {p_b / max(p_g, 1e-6):.2f}×）。",
                            f"Correct: the model prefers the correct sentence "
                            f"({p_b / max(p_g, 1e-6):.2f}x)."))
                    else:
                        st.error(tr(
                            f"判定错误：模型反而更喜欢错句（错句 PPL 低 {p_g / max(p_b, 1e-6):.2f}×）。",
                            f"Missed: the model prefers the typo sentence "
                            f"({p_g / max(p_b, 1e-6):.2f}x)."))

        suggest_text = st.text_input(
            tr("要纠错的句子（模型逐字给建议）", "Sentence to correct (per-character hints)"),
            key="corr_suggest_text",
            placeholder=tr("例如：我爱泥", "e.g. a sentence with typos"))
        if st.button(tr("给出纠错建议", "Suggest corrections"), key="corr_suggest_run"):
            if entry_a is None:
                st.warning(tr("先在上方加载模型 A。", "Load model A first."))
            elif not suggest_text.strip():
                st.warning(tr("先输入一个句子。", "Enter a sentence first."))
            else:
                target = suggest_text.strip()
                hints = suggest_corrections(entry_a["scorer"], target)
                before = score_candidate(entry_a["scorer"], "", target)
                if not hints:
                    st.info(tr(
                        "模型没有找到明显的错字（每个字在当前上下文下都足够合理）。",
                        "No obvious typos found: every character is plausible here."))
                else:
                    corrected = apply_corrections(target, hints)
                    after = score_candidate(entry_a["scorer"], "", corrected)
                    st.dataframe(pd.DataFrame(hints).rename(columns={
                        "pos": tr("位置", "Pos"),
                        "char": tr("原字", "Char"),
                        "suggest": tr("建议", "Suggest"),
                        "ppl_before": tr("改前 PPL", "PPL before"),
                        "ppl_after": tr("改后 PPL", "PPL after"),
                        "p_suggest": tr("建议字概率", "P(suggest)"),
                    }), use_container_width=True)
                    st.code(f"{target}  →  {corrected}", language="text")
                    if before and after:
                        c1, c2, c3 = st.columns(3)
                        c1.metric(tr("修改前 PPL", "PPL before"), f"{before['ppl']:.2f}")
                        c2.metric(tr("修改后 PPL", "PPL after"), f"{after['ppl']:.2f}")
                        delta = after["ppl"] - before["ppl"]
                        c3.metric(tr("变化", "Delta"), f"{delta:+.2f}")
                        if delta < 0:
                            st.success(tr("建议让句子更符合训练分布。",
                                          "The suggestions make the sentence more in-domain."))
                        else:
                            st.warning(tr("建议没有改善 PPL，请人工判断。",
                                          "Suggestions did not improve PPL; judge manually."))

        eval_upload = st.file_uploader(
            tr("评测集 CSV", "Eval CSV"), type=["csv"], key="corr_csv")
        if eval_upload is None:
            st.caption(tr(
                "格式示例：第一行可以是表头（正确句子, 错别字句子）。",
                "Example: first row may be a header (correct, typo)."))
            return
        if entry_a is None:
            st.warning(tr("先在上方加载模型 A。", "Load model A first."))
            return
        if st.button(tr("一键评测", "Run eval"), key="corr_run"):
            try:
                frame = pd.read_csv(io.BytesIO(eval_upload.getvalue()),
                                    header=None, dtype=str)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{tr('CSV 解析失败', 'CSV parse failed')}: {exc}")
                return
            pairs = []
            for _, row in frame.iterrows():
                values = [str(v).strip() for v in row.tolist()[:2]]
                if len(values) < 2 or not values[0] or not values[1]:
                    continue
                if any("正确" in v or "correct" in v.lower() for v in values):
                    continue  # header row
                pairs.append((values[0], values[1]))
            if not pairs:
                st.warning(tr("没有解析到有效句对。", "No valid pairs parsed."))
                return
            if len(pairs) > 200:
                st.warning(tr("一次最多评测 200 对，已截断。",
                              "At most 200 pairs per run; truncated."))
                pairs = pairs[:200]

            wins = 0
            detail = []
            progress = st.progress(0.0, text="evaluating…")
            for index, (correct, typo) in enumerate(pairs, start=1):
                res_c = score_candidate(entry_a["scorer"], "", correct)
                res_t = score_candidate(entry_a["scorer"], "", typo)
                ppl_c = res_c["ppl"] if res_c else None
                ppl_t = res_t["ppl"] if res_t else None
                won = (ppl_c is not None and ppl_t is not None
                       and ppl_c < ppl_t)
                wins += int(won)
                detail.append({
                    "correct": correct, "typo": typo,
                    "ppl_correct": round(ppl_c, 2) if ppl_c else None,
                    "ppl_typo": round(ppl_t, 2) if ppl_t else None,
                    "correct_wins": tr("是", "yes") if won else tr("否", "no"),
                })
                progress.progress(index / len(pairs),
                                  text=f"evaluating {index}/{len(pairs)}")
            progress.empty()

            accuracy = wins / max(1, len(pairs))
            mc1, mc2 = st.columns(2)
            mc1.metric(tr("纠错准确率 (Top-1)", "Correction accuracy (Top-1)"),
                       f"{accuracy * 100:.1f}%")
            mc2.metric(tr("句对数", "Pairs"), str(len(pairs)))
            if accuracy < 0.5:
                st.warning(tr(
                    "准确率低于 50%：模型可能没有覆盖这个领域的纠错模式。",
                    "Below 50%: the model may not cover this correction domain."))
            st.dataframe(pd.DataFrame(detail), use_container_width=True)


def classic_dashboard() -> None:
    """Optional compatibility view: embed the standalone 8765 panel."""
    with st.expander(tr(
            "经典仪表盘（兼容视图）",
            "Classic dashboard (compatibility view)"), expanded=False):
        st.caption(tr(
            f"嵌入 {CLASSIC_DASHBOARD_URL} 的经典监控面板；它读取自己 --dir 指定的运行目录，"
            "需要该服务正在运行。原生监控始终可用。",
            f"Embeds the classic monitor at {CLASSIC_DASHBOARD_URL}; it reads its own "
            "--dir run directory and must be running. The native monitor is always available."))
        components.iframe(CLASSIC_DASHBOARD_URL, height=900, scrolling=True)


def _export_slot_paths(slot: str) -> tuple[Path, Path]:
    """Persisted model/vocab paths for an export slot."""
    last = read_playground_last(f"export_{slot}")
    return Path(last.get("model", "")), Path(last.get("vocab", ""))


def render_export_tab() -> None:
    st.markdown(f"### {tr('导出与量化', 'Export & Quantization')}")
    st.caption(tr(
        "把训练好的 .pt 转成 ONNX（FP32），可选 INT8 动态量化。"
        "导出与训练/测试台互相独立，产物在本地生成后可直接下载。",
        "Convert a trained .pt to ONNX (FP32), optionally INT8 dynamic "
        "quantization. Independent of training/playground; artifacts are "
        "generated locally and downloadable."))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### {tr('模型输入', 'Model input')}")
        arch = st.selectbox(
            tr("模型架构", "Model architecture"), ["gpt2", "lstm"],
            index=0,
            format_func=lambda key: (
                tr("GPT-2 字符级 LM", "GPT-2 character LM") if key == "gpt2"
                else tr("LSTM 语言模型", "LSTM language model")),
            key="export_arch",
            help=tr("上传的 .pt 是哪种架构就选哪种，否则无法实例化。",
                    "Pick the architecture of the uploaded .pt; a mismatch fails to load."))
        source = st.radio(
            tr("模型来源", "Model source"),
            [tr("自动读取最新训练模型", "Auto-read latest run"),
             tr("手动上传模型", "Manual upload")],
            index=0, horizontal=True, key="export_source",
            help=tr("自动模式使用 runs/ 里最新的 model(.best).pt；"
                    "词表来自同名 vocab.txt 或 checkpoint 内置词表。",
                    "Auto mode uses the newest model(.best).pt under runs/; "
                    "vocab comes from the sidecar or the checkpoint."))

        model_upload = None
        vocab_upload = None
        manual_model_path: Optional[Path] = None
        manual_vocab_path: Optional[Path] = None
        if not source.startswith(tr("自动读取最新训练模型", "Auto-read latest run")):
            model_upload = st.file_uploader(
                tr("模型文件（.pt/.pth/.ckpt/.bin/.safetensors）",
                   "Model file (.pt/.pth/.ckpt/.bin/.safetensors)"),
                type=[suffix.lstrip(".") for suffix in CHECKPOINT_SUFFIXES],
                key="export_model",
                help=tr("训练产出里的 model.pt / model_best.pt，或同结构的 safetensors。",
                        "model.pt / model_best.pt from a run, or an equivalent safetensors file."))
            if model_upload is not None:
                try:
                    manual_model_path = save_playground_upload(model_upload)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{tr('保存上传文件失败', 'Failed to save upload')}: {exc}")
                info = inspect_checkpoint_file(manual_model_path)
                if not info["words"]:
                    st.warning(tr(
                        "检测到旧版模型（无内置词表），请补充上传配套的 vocab.txt。",
                        "Legacy model detected (no embedded vocab); upload the paired vocab.txt."))
                    vocab_upload = st.file_uploader(
                        tr("词表文件 (vocab.txt)", "Vocabulary (vocab.txt)"),
                        type=["txt"], key="export_vocab",
                        help=tr("和模型同一次训练产出的 vocab.txt。",
                                "The vocab.txt produced by the same training run."))
                    if vocab_upload is not None:
                        manual_vocab_path = save_playground_upload(vocab_upload)

    precision = st.radio(
        tr("精度", "Precision"),
        [tr("原始精度 (FP32)", "FP32"),
         tr("动态量化 (INT8)", "Dynamic INT8")],
        index=0, horizontal=True, key="export_precision",
        help=tr("INT8 体积约 1/4，分数会有轻微偏差。",
                "INT8 is ~1/4 the size with slight score drift."))
    formats = st.multiselect(
        tr("导出格式", "Export formats"),
        [tr("ONNX（FP32/INT8）", "ONNX (FP32/INT8)"),
         tr("TorchScript（纯 PyTorch C++）", "TorchScript (PyTorch C++)"),
         tr("TFLite（实验，需 tensorflow + onnx-tf）",
            "TFLite (experimental; needs tensorflow + onnx-tf)"),
         tr("safetensors（权重 + config + 词表）",
            "safetensors (weights + config + vocab)"),
         tr("TJGPT（输入法引擎直读，零依赖）",
            "TJGPT (engine-native, zero dependency)")],
        default=[tr("ONNX（FP32/INT8）", "ONNX (FP32/INT8)")],
        key="export_formats",
        help=tr("ONNX 是默认；TorchScript 供 libtorch；TFLite 面向安卓/单片机，"
                "需要额外安装 tensorflow 和 onnx-tf。",
                "ONNX by default; TorchScript for libtorch; TFLite for "
                "Android/MCU (extra deps)."))
    want_onnx = any("ONNX" in f for f in formats)
    want_ts = any("TorchScript" in f for f in formats)
    want_tflite = any("TFLite" in f for f in formats)
    want_safetensors = any("safetensors" in f.lower() for f in formats)
    want_tjgpt = any("TJGPT" in f for f in formats)
    pkg_name = st.text_input(
        tr("模型包名", "Bundle name"), key="export_pkg",
        placeholder="zimo-lm-v1.0-news",
        help=tr("导出 zip 的文件名（不含 .zip）；只允许字母数字、点、横线、下划线。",
                "Zip file name without .zip; letters/digits/dot/dash/underscore only."))
    pkg_name = re.sub(r"[^A-Za-z0-9._-]", "", pkg_name.strip()) or "zimo-lm-bundle"

    if source.startswith(tr("自动读取最新训练模型", "Auto-read latest run")):
        latest = find_latest_run(RUNS_DIR)
        export_demo = False
        if latest is None or latest["model"].parent.name == "playground":
            latest = find_latest_run(DEMO_MODEL_DIR)
            export_demo = latest is not None
        if latest is None:
            st.info(tr("runs/ 里还没有可用的训练产物；先训练一次，或改用手动上传。",
                       "No usable run under runs/ yet; train first or upload manually."))
            return
        entry = load_model_source("export_loaded", arch, latest["model"],
                                  latest["vocab"], persisted=False)
        if entry is not None:
            vocab_note = (tr("词表来自 vocab.txt", "vocab from vocab.txt")
                          if latest["vocab"] else
                          tr("词表来自模型内置", "vocab from checkpoint"))
            prefix = (tr("内置示例", "Bundled demo") if export_demo
                      else tr("自动加载", "Auto-loaded"))
            st.success(tr(
                f"{prefix}：{latest['model'].name} · 词表大小 {entry['vocab_size']}（{vocab_note}）",
                f"{prefix}: {latest['model'].name} · vocab {entry['vocab_size']} ({vocab_note})"))
    else:
        if manual_model_path is None:
            st.info(tr("上传 .pt 后开始转换。", "Upload a .pt to start conversion."))
            return
        entry = load_model_source("export_loaded", arch, manual_model_path,
                                  manual_vocab_path)
    if entry is None:
        return
    st.success(tr(
        f"模型加载成功，词表大小：{entry['vocab_size']}",
        f"Model loaded, vocab size: {entry['vocab_size']}"))
    scorer = entry["scorer"]
    block = int(scorer["block"])

    if st.button(tr("开始转换", "Start conversion"), type="primary",
                 key="export_run"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PLAYGROUND_DIR / "export" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with st.spinner(tr("导出 ONNX…", "Exporting ONNX…")):
                fp32_path = export_onnx(
                    scorer["model"], arch, entry["vocab_size"], block,
                    out_dir / "model_fp32.onnx")
            st.session_state["export_fp32"] = str(fp32_path)

            int8_path = None
            if "INT8" in precision:
                with st.spinner(tr("INT8 量化中…", "Quantizing to INT8…")):
                    int8_path = quantize_onnx_int8(
                        fp32_path, out_dir / "model_int8.onnx")
            ts_path = None
            tflite_path = None
            safetensors_path = None
            if want_safetensors:
                with st.spinner(tr("导出 safetensors…", "Exporting safetensors…")):
                    ordered = [word for word, _ in sorted(
                        scorer["vocab"].items(), key=lambda kv: kv[1])]
                    arch_config = {
                        "arch": arch,
                        "block": int(block),
                        "dim": int(scorer["meta"]["dim"]),
                        "layers": int(scorer["meta"]["layers"]),
                    }
                    if arch == "gpt2":
                        arch_config["heads"] = int(scorer["meta"]["heads"])
                    else:
                        arch_config["hidden"] = int(scorer["meta"]["hidden"])
                    safetensors_path = export_safetensors(
                        scorer["model"], arch_config, ordered,
                        out_dir / "model.safetensors")
            if want_ts:
                with st.spinner(tr("导出 TorchScript…", "Exporting TorchScript…")):
                    ts_path = export_torchscript(
                        scorer["model"], arch, out_dir / "model_scripted.pt",
                        sample_len=min(block, 16))
            if want_tflite and want_onnx:
                with st.spinner(tr("转换 TFLite…", "Converting TFLite…")):
                    try:
                        tflite_path = export_tflite(fp32_path,
                                                    out_dir / "model.tflite")
                    except ImportError as exc:
                        st.warning(tr(
                            f"TFLite 转换需要额外依赖：{exc}。"
                            "请先 pip install tensorflow onnx-tf。",
                            f"TFLite conversion needs: {exc}. "
                            "Run pip install tensorflow onnx-tf first."))
            tjgpt_path = None
            if want_tjgpt and arch == "gpt2":
                with st.spinner(tr("导出 TJGPT…", "Exporting TJGPT…")):
                    ordered_words = [word for word, _ in sorted(
                        scorer["vocab"].items(), key=lambda kv: kv[1])]
                    tjgpt_path = write_tjgpt(
                        scorer["model"], ordered_words,
                        out_dir / "model.tjgpt",
                        block=int(block),
                        layers=int(scorer["meta"]["layers"]),
                        dim=int(scorer["meta"]["dim"]),
                        heads=int(scorer["meta"]["heads"]),
                        tied=False)
            elif want_tjgpt:
                st.warning(tr("TJGPT 目前只支持字符级 GPT-2（LSTM 暂未实现）。",
                              "TJGPT currently supports the char GPT-2 only."))
            st.session_state["export_tjgpt"] = str(tjgpt_path) if tjgpt_path else ""
            st.session_state["export_int8"] = str(int8_path) if int8_path else ""
            st.session_state["export_ts"] = str(ts_path) if ts_path else ""
            st.session_state["export_tflite"] = str(tflite_path) if tflite_path else ""
            st.session_state["export_safetensors"] = (
                str(safetensors_path) if safetensors_path else "")
            st.session_state["export_pkg"] = pkg_name
            st.session_state["export_meta"] = {
                "arch": arch, "vocab": entry["vocab_size"], "block": block,
                "fp32_mb": onnx_file_size_mb(fp32_path),
                "int8_mb": onnx_file_size_mb(int8_path) if int8_path else None,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"{tr('转换失败', 'Conversion failed')}: {exc}")

    fp32_value = st.session_state.get("export_fp32", "")
    int8_value = st.session_state.get("export_int8", "")
    meta = st.session_state.get("export_meta") or {}
    int8_path: Optional[Path] = None
    if not fp32_value:
        return
    fp32_path = Path(fp32_value)
    st.success(tr("转换完成。", "Conversion finished."))
    metric_cols = st.columns(3)
    metric_cols[0].metric("FP32", f"{meta.get('fp32_mb', '–')} MB")
    metric_cols[1].metric(
        "INT8", f"{meta.get('int8_mb', '–')} MB" if meta.get("int8_mb") else "–")
    metric_cols[2].metric(
        tr("词表", "Vocab"), str(meta.get("vocab", "–")))
    st.caption(tr(
        f"FP32：`{fp32_path}`" + (f" · INT8：`{int8_path}`" if int8_value else ""),
        f"FP32: `{fp32_path}`" + (f" · INT8: `{int8_path}`" if int8_value else "")))

    # ---- zip bundle: onnx + vocab.txt (vocab source aware) ----
    ts_value = st.session_state.get("export_ts", "")
    st_value = st.session_state.get("export_safetensors", "")
    tjgpt_value = st.session_state.get("export_tjgpt", "")
    tflite_value = st.session_state.get("export_tflite", "")
    zip_path = fp32_path.parent / f"{pkg_name}.zip"
    vocab_source = entry.get("vocab_path") or ""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(fp32_path, "model_fp32.onnx")
        if int8_value:
            bundle.write(int8_value, "model_int8.onnx")
        if ts_value and Path(ts_value).is_file():
            bundle.write(ts_value, "model_scripted.pt")
        if tflite_value and Path(tflite_value).is_file():
            bundle.write(tflite_value, "model.tflite")
        if st_value and Path(st_value).is_file():
            bundle.write(st_value, "model.safetensors")
        if tjgpt_value and Path(tjgpt_value).is_file():
            bundle.write(tjgpt_value, "model.tjgpt")
        if vocab_source and Path(vocab_source).is_file():
            bundle.write(vocab_source, "vocab.txt")
        else:
            words = extract_vocab_words(
                torch.load(str(entry["model_path"]), map_location="cpu",
                           weights_only=False)) or []
            bundle.writestr("vocab.txt", chr(10).join(words))
        bundle.writestr(
            "README.txt",
            "arch=" + str(meta.get("arch", "")) + chr(10)
            + "vocab=" + str(meta.get("vocab", "")) + chr(10)
            + "int8=" + ("yes" if int8_value else "no") + chr(10)
            + "input: int64 input_ids [batch, seq_len]" + chr(10)
            + "output: float32 logits [batch, seq_len, vocab]" + chr(10))
    st.session_state["export_zip"] = str(zip_path)

    zip_col, fp32_col, int8_col = st.columns(3)
    with zip_col:
        st.download_button(
            tr("⬇ 下载部署包 (.zip)", "⬇ Download bundle (.zip)"),
            data=zip_path.read_bytes(), file_name=f"{pkg_name}.zip",
            mime="application/zip", use_container_width=True,
            key="download_zip", type="primary")
    with fp32_col:
        st.download_button(
            tr("下载 FP32 ONNX", "Download FP32 ONNX"),
            data=fp32_path.read_bytes(), file_name="model_fp32.onnx",
            mime="application/onnx", use_container_width=True,
            key="download_fp32")
    if int8_value:
        int8_path = Path(int8_value)
        with int8_col:
            st.download_button(
                tr("下载 INT8 ONNX", "Download INT8 ONNX"),
                data=int8_path.read_bytes(), file_name="model_int8.onnx",
                mime="application/onnx", use_container_width=True,
                key="download_int8")

    with st.expander(tr("C++ 推理代码（onnxruntime，已按本次导出参数生成）",
                        "C++ inference code (onnxruntime, generated for this export)")):
        cpp_model = (Path(int8_value).name if int8_value
                     else fp32_path.name)
        st.code(cpp_snippet(cpp_model, int(meta.get("vocab", 0) or 0),
                            int(meta.get("block", 64) or 64),
                            bool(int8_value)), language="cpp")

    sample_text = st.text_input(
        tr("一致性检查文本（可选）", "Parity check text (optional)"),
        key="export_parity_text",
        help=tr("输入一段文本，比较 PyTorch 与 ONNX 导出模型的 PPL。",
                "Enter text to compare PPL between PyTorch and the ONNX export."))
    if sample_text and st.button(tr("运行一致性检查", "Run parity check"),
                                 key="export_parity"):
        words = [w for w in
                 Path(entry["vocab_path"]).read_text(encoding="utf-8")
                 .splitlines() if w]
        vocab_map = {w: i + 1 for i, w in enumerate(words)}
        ids = [vocab_map.get(ch, 0) for ch in sample_text if not ch.isspace()]
        try:
            torch_ppl = score_candidate(scorer, "", sample_text)["ppl"]
            rows = [{"model": "PyTorch", "ppl": round(torch_ppl, 3)}]
            rows.append({"model": "ONNX FP32",
                         "ppl": round(onnx_ppl(arch, fp32_path, ids, block), 3)})
            if int8_value:
                rows.append({"model": "ONNX INT8",
                             "ppl": round(onnx_ppl(arch, Path(int8_value), ids, block), 3)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{tr('一致性检查失败', 'Parity check failed')}: {exc}")


def render_training(controls: dict) -> None:
    proc = st.session_state.get("proc")
    run_dir_value = st.session_state.get("run_dir")
    arch_key = st.session_state.get("arch_key") or controls["arch_key"]
    running = is_running()

    if proc is None and not st.session_state.get("finished"):
        st.markdown(f"""
<div class="tj-card">
  <b>{tr("开始之前", "Before you start")}</b><br>
  {tr("1. 在左侧选择模型架构并上传数据。", "1. Choose an architecture and upload data in the sidebar.")}<br>
  {tr("2. GPT-2 / LSTM 接受 .txt（每行一句）；MLP 接受带 scores JSON 列的 .csv 或 .jsonl。",
      "2. GPT-2 / LSTM take .txt (one sentence per line); MLP takes .csv with a JSON scores column, or .jsonl.")}<br>
  {tr("3. 点击“开始训练”。训练器在后台运行，stdout 会实时流到这里。",
      "3. Click Start training. The trainer runs in the background and its stdout streams here.")}
</div>
""", unsafe_allow_html=True)
        return

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    monitor_col, side_col = st.columns([2.4, 1], gap="large")
    with monitor_col:
        st.markdown(f"### {tr('训练监控', 'Training monitor')}")
        if arch_key == "mlp":
            if running:
                st.info(tr("MLP 训练中；该 trainer 不输出 step 进度，请看日志。",
                           "MLP is training; this trainer does not emit step progress. See the log."))
            else:
                st.success(tr("MLP 训练结束；导出文件在运行目录。",
                              "MLP training finished; the exported file is in the run directory."))
        else:
            metric_cards(st.session_state.get("status", {}), running)
            monitor_charts()
            model_info(st.session_state.get("status", {}), Path(run_dir_value) if run_dir_value else None)
        log_panel()
        if not running and not st.session_state.get("error") and run_dir_value:
            prefix = (tr("已加载训练", "Loaded run")
                      if str(st.session_state.get("command", "")).startswith("loaded")
                      else tr("训练完成", "Training complete"))
            st.success(f"{prefix} · {tr('产物目录', 'artifacts')}：`{run_dir_value}`")

    with side_col:
        st.markdown(f"### {tr('运行信息', 'Run info')}")
        run_dir = Path(run_dir_value) if run_dir_value else None
        if run_dir:
            st.markdown(f"<div class='tj-code'>run dir<br>{run_dir}</div>",
                        unsafe_allow_html=True)
        if st.session_state.get("command"):
            with st.expander(tr("完整命令（可复制到远端终端）",
                                "Full command (copy to a remote shell)"),
                             expanded=False):
                st.code(st.session_state["command"], language="bash")
                st.caption(tr(
                    "右上角图标可复制。在租用的 GPU 机器上装好依赖后，"
                    "这条命令可脱离本页面直接在终端运行（加了 --resume 即续训）。",
                    "Use the copy icon. On a rented GPU box with dependencies "
                    "installed, this command runs standalone in a terminal "
                    "(--resume continues from the checkpoint)."))
        status = st.session_state.get("status", {})
        if status:
            st.markdown(f"<div class='tj-code'>"
                        f"device: {status.get('device', '–')}<br>"
                        f"vocab: {status.get('vocab', '–')}<br>"
                        f"tokens: {status.get('tokens', '–')}<br>"
                        f"val ppl: {status.get('val_ppl', '–')}"
                        f"</div>", unsafe_allow_html=True)
        if not running and st.session_state.get("error"):
            st.warning(tr("训练已停止/失败", "Training stopped/failed"))

        if not running and run_dir:
            st.markdown(f"### {tr('训练产物', 'Artifacts')}")
            st.caption(tr("运行目录", "Run directory"))
            st.code(str(run_dir), language="text")
            model, _vocab = find_model(run_dir, arch_key)
            if model is not None:
                st.caption(tr("模型文件", "Model file"))
                st.code(str(model), language="text")
            button_cols = st.columns(2)
            if button_cols[0].button(tr("打开运行目录", "Open run folder"),
                                     use_container_width=True,
                                     key="open_run_dir"):
                open_directory(run_dir)
            if model is not None and button_cols[1].button(
                    tr("打开模型目录", "Open model folder"),
                    use_container_width=True, key="open_model_dir"):
                open_directory(model.parent)
            if model is not None:
                st.download_button(
                    tr("下载模型", "Download model"),
                    data=model.read_bytes(), file_name=model.name,
                    mime="application/octet-stream",
                    use_container_width=True, key="download_model")

    if run_dir:
        classic_dashboard()

    if not running and run_dir:
        st.divider()
        test_panel(arch_key, run_dir)



def main() -> None:
    st.set_page_config(
        page_title="Zimo · Studio",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_state()
    auto_attach_run()
    sync_process()

    controls = sidebar()
    hero()

    if controls["start"]:
        start_training(
            controls["upload"], controls["arch_key"], controls["arch_kind"],
            controls["epochs"], controls["batch"], controls["tokens"],
            controls["use_gpu"], controls.get("arch_params"),
            controls.get("clean_opts"), controls.get("local_paths"),
            controls.get("max_vocab", 13000), controls.get("val_ratio", 0.02),
            controls.get("learning_rate", 3e-4), controls.get("warmup", 300))
    if controls["stop"]:
        stop_training()

    tab_train, tab_playground, tab_export = st.tabs([
        tr("训练控制台", "Training console"),
        tr("模型测试台", "Model playground"),
        tr("导出与量化", "Export & Quantization")])
    with tab_train:
        render_training(controls)
    with tab_playground:
        render_playground()
    with tab_export:
        render_export_tab()

    if is_running():
        time.sleep(0.7)
        st.rerun()

if __name__ == "__main__":
    main()
