# SPDX-License-Identifier: MIT
"""ONNX export + INT8 dynamic quantization for the Zimo Studio test bench.

GPT-2 (GPTLM) exports as a full graph: positional embedding, causal mask and
attention all live inside forward(x), so the ONNX input is a single int64
``input_ids`` tensor of shape [batch, seq_len] and the output is ``logits``
[batch, seq_len, vocab].  Dynamic axes keep both seq_len and batch flexible.

LSTM (LSTMLM) exposes forward(x, hidden) -> (logits, hidden).  The Studio uses
stateless scoring (fresh hidden state per call), so the exported graph omits
the hidden state entirely: one int64 input, one logits output.  Looping over a
stateful LSTM in ONNX is not modelled here.

Quantization uses onnxruntime.quantization.quantize_dynamic with a MatMul/
Gemm weight-type pass (QDQ default ops set); attention MatMuls and all
Gemm/Linear weights become int8, which is where most of the size reduction
comes from for these small models.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class _ExportGPT(nn.Module):
    """Wrapper so ONNX sees a pure x -> logits function."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


class _ExportLSTM(nn.Module):
    """Stateless wrapper: hidden state is created inside forward."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def export_onnx(model: nn.Module, arch_key: str, vocab_size: int,
                block: int, out_path: Path,
                sample_len: int = 16) -> Path:
    """Export the scorer model to ONNX with dynamic sequence length."""
    if arch_key == "gpt2":
        wrapped = _ExportGPT(model)
    elif arch_key == "lstm":
        wrapped = _ExportLSTM(model)
    else:
        raise ValueError(f"unsupported architecture: {arch_key}")
    wrapped.eval()

    sample = torch.ones(1, sample_len, dtype=torch.long)
    os.makedirs(out_path.parent, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (sample,),
            str(out_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    _inline_external_data(out_path)
    return out_path


def _inline_external_data(onnx_path: Path) -> None:
    """torch>=2.12 writes weights as an external .data file by default; merge
    it back so the user downloads a single .onnx."""
    import onnx

    sidecar = Path(str(onnx_path) + ".data")
    if not sidecar.exists():
        return
    model = onnx.load(str(onnx_path), load_external_data=True)
    for tensor in model.graph.initializer:
        tensor.data_location = onnx.TensorProto.DEFAULT
        tensor.ClearField("external_data")
    onnx.save(model, str(onnx_path))
    sidecar.unlink(missing_ok=True)


def quantize_onnx_int8(src_path: Path, dst_path: Path) -> Path:
    """Dynamic INT8 quantization (weights only) via onnxruntime."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"MatMulConstBOnly": True},
    )
    return dst_path


def onnx_ppl(scorer_key: str, onnx_path: Path, ids: list[int],
             block: int) -> Optional[float]:
    """Mean NLL of ids under the exported ONNX model (parity helper)."""
    import math

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"])
    if len(ids) < 2:
        return None
    nlls = []
    block = max(2, int(block))
    for pos in range(1, len(ids)):
        start = max(0, pos - block + 1)
        window = ids[start:pos + 1]
        if len(window) < 2:
            continue
        x = torch.tensor([window[:-1]], dtype=torch.long)
        y = window[-1]
        logits = session.run(["logits"], {"input_ids": x.numpy()})[0]
        last = torch.tensor(logits[0, -1])
        log_probs = torch.log_softmax(last.float(), dim=-1)
        nlls.append(-log_probs[y].item())
    if not nlls:
        return None
    mean_nll = sum(nlls) / len(nlls)
    return math.exp(min(mean_nll, 20.0))


def onnx_file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def export_torchscript(model: nn.Module, arch_key: str, out_path: Path,
                       sample_len: int = 16) -> Path:
    """TorchScript trace of the same stateless wrapper used for ONNX."""
    wrapped = _ExportGPT(model) if arch_key == "gpt2" else _ExportLSTM(model)
    wrapped.eval()
    sample = torch.ones(1, sample_len, dtype=torch.long)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapped, sample)
    scripted.save(str(out_path))
    return out_path


def export_tflite(onnx_path: Path, out_path: Path) -> Path:
    """Experimental ONNX -> TFLite conversion.

    Requires heavyweight optional dependencies (tensorflow + onnx-tf).  The
    caller is expected to surface the ImportError message in the UI.
    """
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf  # noqa: F401 - needed by onnx-tf

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_rep.tf_func])
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    return out_path


def cpp_snippet(model_file: str, vocab_size: int, block: int,
                int8: bool) -> str:
    """Generate an onnxruntime C++ inference skeleton for this export."""
    model_line = ("model_int8.onnx" if int8 else "model_fp32.onnx")
    if model_file:
        model_line = model_file
    text = r'''// onnxruntime C++ inference skeleton for this Zimo export.
// deps: onnxruntime (C++ API), and the vocab.txt shipped in this bundle.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// 1) Load the vocabulary: vocab.txt line i (0-based) -> token id i + 1.
//    Unknown characters map to id 0 (<unk>).
static int char_to_id(const std::string& ch) { /* your vocab map */ return 0; }

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "zimo");
    Ort::SessionOptions opts;
    Ort::Session session(env, __MODEL_FILE__, opts);

    const int block = __BLOCK__;          // context window from training
    const int vocab = __VOCAB__;
    std::string context = "...";          // already-committed text (UTF-8 chars)
    std::string candidate = "...";        // candidate to score

    // 2) char ids for context + candidate (spaces ignored)
    std::vector<int64_t> ids;             // fill from your vocab map

    // 3) sliding-window scoring: mean NLL of the candidate chars
    double nll_sum = 0.0; int scored = 0;
    for (size_t pos = /*ctx_len*/ 0; pos + 1 < ids.size(); ++pos) {
        size_t start = (pos + 1 > (size_t)block) ? pos + 1 - block : 0;
        std::vector<int64_t> window(ids.begin() + start, ids.begin() + pos + 1);
        std::array<int64_t, 2> shape = {1, (int64_t)window.size()};
        auto input = Ort::Value::CreateTensor<int64_t>(
            window.data(), window.size() * sizeof(int64_t), shape.data(), 2);
        auto output = session.Run(Ort::RunOptions{nullptr},
                                  &"input_ids", &input, 1, &"logits", 1);
        float* data = output[0].GetTensorMutableData<float>();
        // last position logits over the vocab; pick the id of the next char
        int next_id = /* ids[pos + 1] */ 0;
        float max_logit = -1e30f; double sum_exp = 0.0;
        for (int v = 0; v < vocab; ++v) max_logit = std::max(max_logit, data[v]);
        for (int v = 0; v < vocab; ++v) sum_exp += std::exp(data[v] - max_logit);
        double log_prob = data[next_id] - max_logit - std::log(sum_exp);
        nll_sum += -log_prob; ++scored;
    }
    double mean_nll = nll_sum / std::max(1, scored);
    double ppl = std::exp(std::min(mean_nll, 20.0));
    std::printf("PPL = %.3f\n", ppl);
    return 0;
}

// Notes:
// - input name: "input_ids" (int64 [1, seq_len]); output: "logits" [1, seq_len, __VOCAB__]
// - __QUANT__
// - replace the three /* ... */ placeholders with your vocab map and ids.
'''
    return (text.replace("__MODEL_FILE__", model_line)
                .replace("__BLOCK__", str(block))
                .replace("__VOCAB__", str(vocab_size))
                .replace("__QUANT__",
                         "quantized: int8 weights" if int8
                         else "precision: fp32"))


def export_safetensors(model: nn.Module, config: dict, vocab: list[str],
                       out_path: Path) -> Path:
    """Save weights as safetensors with config+vocab in the metadata.

    The metadata makes the file self-contained: Zimo can reload it without a
    sidecar vocab.txt or config.json.
    """
    from safetensors.torch import save_file

    state = {key: value.detach().contiguous()
             for key, value in model.state_dict().items()}
    metadata = {
        "config": json.dumps({k: int(v) if isinstance(v, (int, bool)) else v
                              for k, v in config.items()}, ensure_ascii=False),
        "vocab": json.dumps(list(vocab), ensure_ascii=False),
    }
    save_file(state, str(out_path), metadata=metadata)
    return out_path
