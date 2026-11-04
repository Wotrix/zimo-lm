# SPDX-License-Identifier: MIT
"""Device detection that also understands Moore Threads (MUSA) GPUs.

PyTorch on Moore Threads hardware comes from ``torch_musa``, which registers a
``musa`` device backend and exposes ``torch.musa``.  ``torch.cuda`` stays
unavailable there, so every device check in this project goes through here.

    resolve_device("auto")   -> "musa" | "cuda" | "cpu"
    resolve_device("cuda")   -> falls back to musa when that is all we have
    backend_label()          -> human readable string for the UI
"""

from __future__ import annotations

from typing import Optional


def _torch():
    import torch
    return torch


def musa_available() -> bool:
    """True when torch_musa is installed and sees a device."""
    try:
        torch = _torch()
        musa = getattr(torch, "musa", None)
        if musa is not None and hasattr(musa, "is_available"):
            return bool(musa.is_available())
    except Exception:  # noqa: BLE001 - a broken driver must not crash the UI
        return False
    return False


def cuda_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def detect_backend() -> str:
    """Best available backend: musa > cuda > cpu."""
    if musa_available():
        return "musa"
    if cuda_available():
        return "cuda"
    return "cpu"


def resolve_device(name: Optional[str]) -> str:
    """Map a user-facing device string onto a usable torch device."""
    requested = (name or "auto").strip().lower()
    backend = detect_backend()
    if requested in ("auto", ""):
        return backend
    if requested == "musa":
        return "musa" if backend == "musa" else "cpu"
    if requested == "cuda":
        # A machine with only MUSA reports no CUDA; use what is actually there.
        if backend == "cuda":
            return "cuda"
        if backend == "musa":
            return "musa"
        return "cpu"
    return requested


def backend_label() -> str:
    """Short label for the UI: 'GPU · MUSA (MTT S80)' / 'GPU · CUDA (RTX 4090)'."""
    backend = detect_backend()
    if backend == "musa":
        try:
            name = _torch().musa.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "MUSA"
        return f"GPU · MUSA ({name})"
    if backend == "cuda":
        try:
            name = _torch().cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "CUDA"
        return f"GPU · CUDA ({name})"
    return "CPU"
