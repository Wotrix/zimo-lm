# SPDX-License-Identifier: MIT
"""TJGPT: self-contained deployment container for the Zimo character LM.

Why another format
------------------
``.pt`` / ``.safetensors`` need PyTorch (or a tensor reader) to run; ``.onnx``
needs onnxruntime.  TJGPT is a flat little-endian binary with a fixed tensor
order, so the C++ engine reads it and runs the forward pass with no
third-party dependency at all.

Layout (v1, little endian)
--------------------------
    magic[8]      "TJTGPT1" padded with a NUL byte
    u32 version   (=1)
    u32 vocab     vocabulary size, including id 0 = <unk>
    u32 block     context window
    u32 dim       hidden size
    u32 layers
    u32 heads
    u32 tied      1 when head.weight is tok.weight
    u32 vocab_bytes
    u8  vocab[vocab_bytes]        UTF-8, one token per line, ids 1..vocab-1
    f32 tok[vocab*dim]
    f32 pos[block*dim]
    per layer (L times):
        f32 ln1_w[dim]  f32 ln1_b[dim]
        f32 in_proj[3*dim*dim]  f32 in_proj_b[3*dim]
        f32 out_proj[dim*dim]   f32 out_proj_b[dim]
        f32 ln2_w[dim]  f32 ln2_b[dim]
        f32 fc_w[4*dim*dim]     f32 fc_b[4*dim]
        f32 proj_w[dim*4*dim]   f32 proj_b[dim]
    f32 lnf_w[dim] f32 lnf_b[dim]
    f32 head[vocab*dim]           only when tied == 0
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

MAGIC = b"TJTGPT1\x00"  # 8 bytes
VERSION = 1
HEADER_SIZE = 8 + 8 * 4  # magic + 8 u32 fields


def _state_dict(model: torch.nn.Module) -> dict:
    return {key: value.detach().to(torch.float32).cpu().contiguous()
            for key, value in model.state_dict().items()}


def write_tjgpt(model: torch.nn.Module, vocab: list[str], out_path: Path,
                *, block: int, layers: int, dim: int, heads: int,
                tied: bool = False) -> Path:
    """Serialise a GPTLM into the TJGPT container."""
    if not vocab:
        raise ValueError("vocabulary must not be empty")
    state = _state_dict(model)
    vocab_blob = ("\n".join(vocab) + "\n").encode("utf-8")
    vocab_size = len(vocab) + 1  # id 0 = <unk>

    out_path = Path(out_path)
    with out_path.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<8I", VERSION, vocab_size, block, dim, layers,
                             heads, 1 if tied else 0, len(vocab_blob)))
        fh.write(vocab_blob)

        def write(tensor: torch.Tensor) -> None:
            fh.write(tensor.numpy().astype("<f4", copy=False).tobytes())

        write(state["tok.weight"])
        write(state["pos.weight"])
        for index in range(layers):
            prefix = f"blocks.{index}."
            write(state[prefix + "ln1.weight"])
            write(state[prefix + "ln1.bias"])
            write(state[prefix + "attn.in_proj_weight"])
            write(state[prefix + "attn.in_proj_bias"])
            write(state[prefix + "attn.out_proj.weight"])
            write(state[prefix + "attn.out_proj.bias"])
            write(state[prefix + "ln2.weight"])
            write(state[prefix + "ln2.bias"])
            write(state[prefix + "fc.weight"])
            write(state[prefix + "fc.bias"])
            write(state[prefix + "proj.weight"])
            write(state[prefix + "proj.bias"])
        write(state["ln_f.weight"])
        write(state["ln_f.bias"])
        if not tied:
            write(state["head.weight"])
    return out_path


def read_tjgpt_header(path: Path) -> dict:
    """Parse just the header (used by tests and tooling)."""
    with Path(path).open("rb") as fh:
        magic = fh.read(8)
        if magic != MAGIC:
            raise ValueError(f"not a TJGPT file: magic={magic!r}")
        version, vocab, block, dim, layers, heads, tied, vocab_bytes = \
            struct.unpack("<8I", fh.read(32))
        if version != VERSION:
            raise ValueError(f"unsupported TJGPT version {version}")
        vocab_blob = fh.read(vocab_bytes)
    return {
        "version": version, "vocab": vocab, "block": block, "dim": dim,
        "layers": layers, "heads": heads, "tied": bool(tied),
        "vocab_bytes": vocab_bytes,
        "words": vocab_blob.decode("utf-8").splitlines(),
        "data_offset": HEADER_SIZE + vocab_bytes,
    }


def expected_size(meta: dict) -> int:
    """File size implied by the header (for validation)."""
    dim = meta["dim"]
    per_layer = (2 * dim                       # ln1
                 + 3 * dim * dim + 3 * dim     # in_proj
                 + dim * dim + dim             # out_proj
                 + 2 * dim                     # ln2
                 + 4 * dim * dim + 4 * dim     # fc
                 + dim * 4 * dim + dim)        # proj
    floats = (meta["vocab"] * dim + meta["block"] * dim
              + meta["layers"] * per_layer + 2 * dim)
    if not meta["tied"]:
        floats += meta["vocab"] * dim
    return meta["data_offset"] + floats * 4
