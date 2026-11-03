# SPDX-License-Identifier: MIT
"""Checkpoint helpers: multi-format loading, vocabulary and config detection.

Supported containers
--------------------
* ``.pt`` / ``.pth`` / ``.ckpt`` / ``.bin`` - ``torch.save`` output.  Accepted
  shapes: an ``nn.Module``, a dict holding ``state`` / ``state_dict`` /
  ``model`` / ``model_state_dict``, or a bare state dict.
* ``.safetensors`` - ``safetensors.torch.load_file`` plus optional metadata
  (``config`` / ``vocab`` JSON strings) and sidecar ``vocab.txt`` /
  ``vocab.json`` / ``config.json``.

Architecture hyper-parameters come from the checkpoint when present, from a
sidecar ``config.json`` otherwise, and are finally inferred from tensor shapes.
Vocabulary comes from the checkpoint (``vocab`` key / metadata), then a
sidecar ``vocab.txt`` or ``vocab.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")
CHECKPOINT_STEMS = ("model_best", "model")
_STATE_KEYS = ("state", "state_dict", "model", "model_state_dict", "net",
               "weights", "params")
_CONFIG_KEYS = ("block", "seq", "seq_len", "dim", "hidden", "layers", "heads",
                "bptt", "tied")


def extract_vocab_words(payload: object) -> Optional[list[str]]:
    """Return an embedded word list from a checkpoint or metadata mapping."""
    if not isinstance(payload, dict):
        return None
    for key in ("vocab", "vocab_dict", "words", "itos"):
        raw = payload.get(key)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], str):
            return [str(word) for word in raw]
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_vocab_sidecar(directory: Path) -> Optional[list[str]]:
    txt = directory / "vocab.txt"
    if txt.is_file():
        words = [line for line in txt.read_text(encoding="utf-8").splitlines()
                 if line]
        if words:
            return words
    for name in ("vocab.json", "tokenizer.json"):
        candidate = directory / name
        if candidate.is_file():
            data = _read_json(candidate)
            words = extract_vocab_words(data)
            if words:
                return words
            if isinstance(data, dict):
                keys = [k for k in data.keys() if not k.startswith("__")]
                if keys and all(isinstance(k, str) for k in keys):
                    return keys
    return None


def _state_from_torch(obj: object) -> Optional[dict]:
    if isinstance(obj, torch.nn.Module):
        return dict(obj.state_dict())
    if isinstance(obj, dict):
        for key in _STATE_KEYS:
            inner = obj.get(key)
            if isinstance(inner, torch.nn.Module):
                return dict(inner.state_dict())
            if isinstance(inner, dict) and inner and all(
                    isinstance(v, torch.Tensor) for v in inner.values()):
                return {k: v for k, v in inner.items()}
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return {k: v for k, v in obj.items()}
    return None


def load_state_any(path: Path) -> dict:
    """Load any supported checkpoint.

    Returns ``{"state", "config", "vocab", "format", "loadable", "error"}``.
    """
    result = {"state": {}, "config": {}, "vocab": None,
              "format": "", "loadable": False, "error": ""}
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        result["format"] = "safetensors"
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - dependency hint
            result["error"] = f"safetensors 未安装：{exc}"
            return result
        try:
            state = load_file(str(path))
            metadata = {}
            with safe_open(str(path), framework="pt") as handle:
                metadata = handle.metadata() or {}
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
        result["state"] = state
        for key, value in metadata.items():
            if key in _CONFIG_KEYS:
                result["config"][key] = value
            elif key in ("config", "hparams"):
                try:
                    result["config"].update(json.loads(value))
                except (TypeError, json.JSONDecodeError):
                    pass
        result["vocab"] = extract_vocab_words(metadata)
    else:
        result["format"] = suffix.lstrip(".") or "pt"
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
        state = _state_from_torch(ckpt)
        if state is None:
            result["error"] = "文件里没有找到 model weights"
            return result
        result["state"] = state
        if isinstance(ckpt, dict):
            for key in _CONFIG_KEYS:
                if key in ckpt and isinstance(ckpt[key], (int, float, bool)):
                    result["config"][key] = ckpt[key]
            result["vocab"] = extract_vocab_words(ckpt)

    directory = path.parent
    for name in ("config.json", "hparams.json"):
        data = _read_json(directory / name)
        if data:
            for key in _CONFIG_KEYS:
                if key in data and isinstance(data[key], (int, float)):
                    result["config"].setdefault(key, data[key])
            result["vocab"] = result["vocab"] or extract_vocab_words(data)
            break
    result["vocab"] = result["vocab"] or _read_vocab_sidecar(directory)
    result["loadable"] = bool(result["state"])
    return result


def infer_config_from_state(state: dict) -> dict:
    """Recover arch hyper-parameters from tensor shapes."""
    keys = set(state.keys())
    config: dict = {}

    def shape(key: str) -> Optional[tuple]:
        tensor = state.get(key)
        return tuple(tensor.shape) if tensor is not None else None

    token = (shape("tok.weight") or shape("token_embedding.weight")
             or shape("wte.weight"))
    if token:
        vocab, dim = int(token[0]), int(token[1])
        config.update({"arch": "gpt2", "vocab": vocab, "dim": dim})
        position = (shape("pos.weight") or shape("position_embedding.weight")
                    or shape("wpe.weight"))
        if position:
            config["block"] = int(position[0])
        layers = 0
        for key in keys:
            parts = key.split(".")
            if (len(parts) > 2 and parts[0] in ("blocks", "layers", "h")
                    and parts[1].isdigit()):
                layers = max(layers, int(parts[1]) + 1)
        if layers:
            config["layers"] = layers
        if dim % 4 == 0:
            config["heads"] = 4
        elif dim % 8 == 0:
            config["heads"] = 8
        else:
            config["heads"] = 1
        return config

    embed = shape("embed.weight") or shape("lstm.embedding.weight")
    if embed:
        vocab, dim = int(embed[0]), int(embed[1])
        config.update({"arch": "lstm", "vocab": vocab, "dim": dim})
        weight_ih = shape("lstm.weight_ih_l0")
        if weight_ih:
            config["hidden"] = int(weight_ih[0]) // 4
        layers = 0
        for key in keys:
            if key.startswith("lstm.weight_ih_l"):
                tail = key.rsplit("l", 1)[-1]
                if tail.isdigit():
                    layers = max(layers, int(tail) + 1)
        config["layers"] = max(1, layers)
        config["block"] = 64
        return config
    return config


def inspect_checkpoint_file(path: Path) -> dict:
    """Compatibility wrapper: vocabulary presence + loadability + metadata."""
    info = load_state_any(path)
    words = info["vocab"] or _read_vocab_sidecar(Path(path).parent)
    config = dict(info["config"])
    if info["loadable"]:
        for key, value in infer_config_from_state(info["state"]).items():
            config.setdefault(key, value)
    return {
        "words": words,
        "loadable": info["loadable"],
        "format": info["format"],
        "config": config,
        "error": info["error"],
    }


def find_latest_run(runs_dir: Path) -> Optional[dict]:
    """Newest trained checkpoint under ``runs_dir`` plus its vocabulary."""
    best: Optional[tuple[float, Path]] = None
    if not runs_dir.is_dir():
        return None
    for path in runs_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        if path.stem not in CHECKPOINT_STEMS:
            continue
        # Export artifacts live under playground/export; they are not runs.
        if {"playground", "export"} & set(path.parts):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    if best is None:
        return None
    model_path = best[1]
    for name in ("vocab.txt", "vocab.json"):
        candidate = model_path.parent / name
        if candidate.is_file():
            return {"model": model_path, "vocab": candidate, "words": None}
    words = inspect_checkpoint_file(model_path)["words"]
    if not words:
        return None
    return {"model": model_path, "vocab": None, "words": words}
