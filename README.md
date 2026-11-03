# zimo-lm

> 中文名：**字模**。`zimo-lm` 是包名，`Zimo Studio` 是浏览器里的工作台。
>
> **一句话**：把一段中文语料，变成几 MB、能在端侧跑的候选重排模型——
> 训练、评测、量化、导出到 C++ 引擎，全在一个页面里完成。
>
> **In one line:** turn a Chinese corpus into a few-MB reranking model that
> runs on the edge - train, evaluate, quantize and export straight to a C++
> engine, all from one local page.

![Zimo Studio](docs/screenshot.png)

<sub>模型测试台：自动加载模型、并排对比结构参数（文件/层数/维度/头数/词表/体积）。</sub>

![Playground results](docs/screenshot-results.png)

<sub>候选打分：PPL 升序，Rank 1 标绿、离谱值标红；下方还有逐字 NLL、批量打分、文本生成与纠错工具。</sub>

**5 分钟跑起来：**

```bash
git clone https://github.com/Wotrix/zimo-lm.git && cd zimo-lm
./setup.sh          # Windows: setup.bat
./run.sh            # Windows: run.bat
# 打开 http://127.0.0.1:8769
```

不打字也能玩：仓库自带一个 4.3M 参数的示例模型（`examples/demo/`），
打开「模型测试台」输入几句候选就能看到打分、逐字 NLL 和纠错建议。
详见 [quickstart.md](quickstart.md)。

---

## 它是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 中文端侧小模型（1M–44M 参数）的一站式工作台 | 通用大模型训练器 |
| 字符级/词级语言模型 + 候选重排 | 拼音解码器（不负责按键转汉字） |
| 本地运行、数据不出机器 | 云服务 / 有 GPU 集群的方案 |
| 训练 → 评测 → INT8 → ONNX/TJGPT 全链路 | 算法研究框架（训练器刻意保持简单） |

**它帮到谁**：做中文输入法/键盘的独立开发者、树莓派与单片机玩家、
想讲清「模型怎么从语料变成部署文件」的教学场景。
**帮不到谁**：想训通用聊天模型、追 SOTA、要下载即用大模型的人。

---

## 为什么值得用

1. **闭环**：别的项目要么只有训练脚本，要么只有推理；这里从语料清洗一路到
   C++ 引擎直读的 `.tjgpt`。
2. **中文优先**：字符级切分，词表 4k–6k，不需要 BPE/分词器；中文语料开箱即用。
3. **部署格式打通**：`.pt` / `.safetensors` 归档，`.onnx` 给第三方或移动端，
   `.tjgpt` 给自带零依赖 C++ 运行时的引擎（已与 PyTorch 逐条对齐）。
4. **工程细节**：后台守护训练（断 SSH 不停）、断点续训、5 种 checkpoint 格式、
   Windows 文件锁防护、全中文说明书。

---

## 详细文档

- [quickstart.md](quickstart.md) — 安装、跑通、训练第一个模型
- [README 其余章节](#) — 模型选型指南、数据准备、部署矩阵、API/格式说明
- Studio 内「使用说明 / Docs」— 十二章规格书（模型类型、六训练器规格、
  五维自定义、部署平台）

A minimal, from-scratch **character-level GPT-2-style language model trainer**
for small Chinese LMs, with a live training dashboard, a self-contained static
report, corpus preparation, and an offline candidate-reranking evaluator.

Reference size **4M** parameters; minimum meaningful roughly **1M parameters +
10M tokens**; practical ceiling for LCCC roughly **25M parameters + 250–500M
tokens**.  Details and the full lower/upper bounds are below.

## Purpose
## Purpose

`zimo-lm` is a **language-model scorer for Chinese candidate text**:

- **Input**: already-committed context + a candidate word/phrase sequence.
- **Output**: average per-token log-probability, i.e. a naturalness/context
  score for that candidate sequence.
- **Use**: a feature or rerank signal in an input method when several candidate
  readings exist (for example `ta shuo de` -> 他/她/它, 的/得/地).
- **Not a generator**: it does not read pinyin, produce candidates, or replace
  the decoder.  It only judges text that another component produced.

Secondary use: a small, reproducible character-level Chinese LM baseline for
research, teaching, and ablations.  The dashboard and static report are
development tools; they are not part of a production input method.

The trainer itself is domain-agnostic: point it at any plain-text corpus and
you get a domain scorer; feed it click logs and you get a reranker.  See
"你能用它训练什么、部署到哪" below for ten concrete scenarios and the
deployment matrix (PC / Raspberry Pi / ESP32).

## 模型选型与数据准备指南

> **默认建议：没有用户选择日志时，使用 GPT-2。** 纯文本即可训练；4M 参考配置可在 CPU 上运行。

### 三大阵营

| 阵营 | 数据 | 模型 | 定位 | 核心用途 |
|---|---|---|---|---|
| **1. 无监督文本打分** | 纯文本 `.txt` | GPT-2 字符级 LM (`train_gpt.py`) | 字符级自回归语言模型 | 输出候选文本的 PPL / 平均对数似然，用于候选重排与纠错 |
| | | LSTM LM (`train_nnlm.py`) | 轻量循环语言模型 | 与 GPT-2 同类任务，体积更小、推理更快，长上下文能力较弱 |
| **2. 有监督用户日志重排** | 用户选择日志 | TJTR1 交叉编码器 (`train_tiny_transformer.py`) | Transformer Encoder 交叉重排模型 | 输入上下文与候选列表，直接学习用户选择 |
| | | TJMLP1 Listwise Ranker (`train_mlp.py`) | 32 维特征列表重排器 | 低延迟实时重排，适合按键/停顿路径 |
| **3. 传统低延迟兜底** | 分词语料 + 词表 | 3-gram 量化 LM (`train_quantized_lm.py`) | 统计语言模型 | 基于相邻词频的低延迟打分，用于停顿/兜底 |
| | | 词级 n-gram (`train_v3.py`) | 词级统计语言模型 | 旧版本兼容与迁移 |

### 选型规则

1. **只有纯文本**：选择 GPT-2（默认）；设备资源受限或要求更低延迟时选择 LSTM。
2. **有用户点击日志**：选择 TJTR1（精度优先）或 TJMLP1（延迟优先）。
3. **需要低延迟兜底或旧版兼容**：选择 3-gram 或词级 n-gram。

注意事项：

- 没有用户选择日志时，不要选择 TJTR1 / TJMLP1；纯文本无法学习用户偏好。
- 3-gram / 词级 n-gram 需要分词语料和词表，不能直接使用原始纯文本。
- LSTM / TJMLP1 在当前 Studio 中以 CPU 为主；GPU 选择会被忽略。

### 数据准备

| 模型 | 最小可运行 | 推荐规模 | 输入格式 |
|---|---|---|---|
| GPT-2 / LSTM | ~10M token | 40–80M token（4M 模型） | `.txt`，每行一句 |
| TJTR1 | 数千条点击事件 | 1 万条以上，覆盖真实候选分布 | JSONL / CSV：上下文 + 候选列表 + 选中项 |
| TJMLP1 | ≥1000 条点击事件 | 1 万条以上，带完整 32 维特征 | JSONL / CSV，`scores` 为 JSON 数组 |
| 3-gram | 数千万 token | 数亿 token 级分词语料 | 分词语料 + `vocab.tsv` |
| 词级 n-gram | 项目现有语料 | 项目词典 + 分词语料 | 分词语料 + 项目词典 |

纯文本示例（GPT-2 / LSTM）：

```text
今天 天气 不错
我 下午 要 去 医院
```

点击日志 CSV 的 `scores` 示例（TJTR1 / TJMLP1）：

```json
[
  {"word": "这个", "features": [0.6, 0.2, 0, 0.8, "...共 32 维..."]},
  {"word": "那个", "features": [0.4, 0.1, 0, 0.5, "...共 32 维..."]}
]
```

### Studio 支持范围

| 模型 | Studio 一键训练 | 说明 |
|---|---|---|
| GPT-2 | 支持 | 包内自带 `train_gpt.py` |
| LSTM | 支持（需 trainer） | 自动搜索同级项目 checkout，或设置 `ZIMO_TRAINER_ROOT` |
| TJMLP1 | 支持（需 trainer） | 需要点击日志 CSV/JSONL |
| TJTR1 | CLI | `train_tiny_transformer.py` |
| 3-gram | CLI | 需要分词语料 + 词表 |
| 词级 n-gram | CLI | 旧版兼容，需要项目词典 |

**选型结论：没有用户选择日志时，默认 GPT-2；有用户点击日志时再考虑 TJTR1 / TJMLP1；n-gram 用于低延迟兜底或旧版兼容。**

## 你能用它训练什么、部署到哪

这个工具箱的本质是一个**通用微型语言模型训练器**：输入任意纯文本（或用户点击日志），
产出一个可在 PC / 树莓派 / ESP32 上推理的小模型。输入法是它的源场景，不是边界。

### 四种可训练的产物（按你手里的数据）

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

### 部署目标

| 平台 | 方式 | 状态 |
|---|---|---|
| Windows / Linux / macOS | onnxruntime 或 PyTorch | 即用（导出台直出 ONNX） |
| 树莓派 | onnxruntime aarch64 + INT8（4.3MB） | 即用 |
| Android | ONNX Runtime Mobile | 需自行集成 |
| ESP32 | 3-gram C/mmap 即用形态；神经模型需 TFLite Micro 全整数量化 | 统计模型即用；神经模型需补转换链 |

### 三个典型组合

- **客服质检**：客服对话语料 → GPT-2 打分器（回答自然度分级）+ 客服操作日志 →
  TJMLP1（话术排序）
- **离线智能键盘**：本 README 的完整链路（训练 → 测试 → INT8 → 树莓派 / 手机）
- **课堂教学**：`examples/sample.txt` + 5 分钟冒烟训练，学生能看到从 loss 曲线
  到 ONNX 产物的每一步

> 详细的能力边界与平台差异见侧边栏"使用说明 / Docs"第五、六章。

## 部署格式：存储 vs 运行时

这四种格式解决的是不同问题，不是互相替代：

| 格式 | 用途 | 运行时依赖 | 谁能读 |
|---|---|---|---|
| `.pt` / `.pth` / `.ckpt` / `.bin` | 训练中间产物 | PyTorch | Zimo、训练脚本 |
| `.safetensors` | 权重交换 / 归档（安全、可 mmap） | PyTorch（或自写张量读取） | Zimo、HF 生态 |
| `.onnx`（FP32 / INT8） | 跨框架推理 | onnxruntime | PC、树莓派、移动端 |
| **`.tjgpt`** | **输入法引擎直读** | **无（自研 C++ 运行时）** | `tongjia-ime` 的 `TjgptModel` |

`.tjgpt` 是本项目新增的自包含容器：magic `TJTGPT1` + 超参数 + 词表 + 按固定顺序排列的
fp32 权重。C++ 侧零依赖读取并跑前向，已与 PyTorch 逐条对齐（同句 PPL 误差 < 1e-4）。

引擎里的运行时：`tongjia-ime/src/input/language/tjgpt_model.{h,cpp}`，
配套探针 `tongjia_tjgpt_probe`（`cmake --build build --target tongjia_tjgpt_probe`）：

```
tongjia_tjgpt_probe model.tjgpt --file pairs.txt   # 每行 "前文|候选"
```

> 引擎目前**没有** ONNX Runtime 依赖：ONNX 适合发给第三方/移动端，
> 自用部署走 TJGPT 可以省掉 ~15MB 的运行时。

## 数据来源：本地路径优先

训练在本机进行，所以语料输入框就是**本地路径**，没有额外的来源切换：

- 填文件或目录，目录会收集匹配类型的文件；支持 `.txt`、`.jsonl`、`.jsonl.gz`
  （LCCC 原包可直接用）；可用 `;` 分隔多个路径；留空则不加载语料
- 小样本仍可上传：收在「或上传小样本（可选）」折叠面板里，受 Streamlit
  `server.maxUploadSize` 限制（本项目已设为 4096 MB；官方默认 200 MB）；
  填了本地路径时以路径为准

大语料（如全量 LCCC 933 MB / 2.4 GB 解压 / 约 8.7 亿字符）请走本地路径，
并按 Token 预算截断；全量单轮在 CPU 上约 23 小时，且受内存限制建议分段 `--resume`。

## 高级自定义（v0.4.0）

Studio 面向愿意动手的开发者开放了五个维度的自定义：

| 维度 | 位置 | 能做什么 |
|---|---|---|
| 自定义模型结构 | 侧栏 高级参数 | **全部自由输入**：层数 1–64、维度 8–2048、头数 1–64、上下文 8–2048、词表上限、验证比例、学习率、Warmup、Token 预算（≤50 亿）、Batch（≤4096）；四个预设一键填入后仍可改；实时估算参数量与 INT8 体积 |
| 数据清洗 | 侧栏 上传框下方 数据清洗 | 只留汉字 / 去数字字母 / 64 字切分 / 去重 / 自定义正则删除 |
| 自定义评测 | 测试台 定制考题 | 上传 (正确句, 错别字句) CSV，一键算纠错 Top-1 准确率；可自定义 PPL 绿/红阈值 |
| 自定义导出 | 导出与量化 | ONNX（FP32/INT8）、TorchScript、TFLite（实验），自定义包名决定 zip 文件名，并即时生成 onnxruntime C++ 推理代码 |
| 自定义词典 | 测试台 自定义词典保护 | custom_dict.txt 保护专有名词不被标红，可优先排序，结果表新增保护列 |

参数量估算：`Params ≈ 12 × Layers × Dim² + Vocab × Dim`，INT8 体积约为参数的 1/4。

## Scope: lower bound and upper bound

### Lower bound: the minimum that is meaningful

| Level | Model | Data | Hardware | What it is good for |
|---|---|---|---|---|
| Smoke only | 1 layer / 32 dim / block 16 | < 1k tokens (`examples/sample.txt`) | CPU | Checking that the pipeline, checkpoint, dashboard, and report work. **Not a usable model.** |
| Minimum LM | ~1M parameters, block 64 | ≥ 10M tokens (≥10 tokens/param) | 4-core CPU, 8 GB RAM | A weak but real language-model scorer; useful only as a directional baseline. |
| Minimum rerank experiment | ~4M parameters | 40–80M tokens | CPU or 8 GB GPU | A 4M baseline that can be compared on a held-out candidate set. Below this, perplexity is not actionable. |

Below roughly **1M tokens**, do not train a neural LM: use an n-gram or a
pretrained model.  Below **10 tokens/parameter**, expect an undertrained model;
the loss may still fall, but the ranking signal is usually too weak to measure.

### Upper bound: the practical limit of this reference trainer

| Target | Example configuration | Parameters | Data (10–20 tokens/param) | Practical hardware | Status |
|---|---|---:|---:|---|---|
| **4M** | 6L / 192d / 4H, block 64 | 4.31M | 40–80M tokens | CPU or 8 GB GPU | Reference example; this is what the README trains. |
| **25M** | 12L / 384d / 6H, block 64 | ~24.5M | 250–500M tokens | 8–16 GB GPU | Practical next step; LCCC's ~612M characters can support it. CPU is too slow. |
| **50M** | 16L / 512d / 8H, block 64 | ~55M | 500M–1B tokens | 24 GB GPU | Technically trainable, but LCCC alone is borderline; needs more corpora. |
| **100M** | 20L / 640d / 8H, block 64 | ~104M | 1–2B tokens | 24–40 GB GPU + AMP/streaming | Possible in principle, **not recommended** with this reference trainer or LCCC alone. |
| **1B+** | — | — | — | multi-GPU | Out of scope; use FSDP/DeepSpeed/Megatron instead. |

Parameter counts assume the **4,251-token character vocabulary** from the LCCC
example.  A larger vocabulary (for example the default `--max-vocab 13000`)
increases the embedding and output-head parameters; with tied embeddings the
increase is roughly `vocab × dim`.

Additional hard limits of the current implementation:

- **Corpus size**: `train_gpt.py` loads all tokens into a Python list.  On a
  32 GB machine this is roughly 200–250M tokens; the full ~612M-character LCCC
  would need a memmap/streaming loader.  A few tens of millions of tokens are
  comfortable.
- **One process, one device**: no gradient accumulation, no multi-GPU, no
  FSDP; distributed training is out of scope.
- **No int8/ONNX/TFLite export**: the package trains and evaluates; deployment
  conversion is separate.

### Choosing a size

- Start with **4M**.  It is the cheapest configuration that can be compared on
  a real candidate set, and it runs on CPU.
- Move to **25M** only after the 4M model shows a measurable ranking gain and
  you can supply 250–500M tokens.  This is the natural ceiling for LCCC.
- **50M** needs extra corpora (and a licence review); **100M+** needs a
  different training stack.  Bigger is not automatically better when the data
  budget is fixed.

### What this package is not for

- Not a pinyin-to-Hanzi decoder, IME engine, or candidate generator.
- Not a chat, instruction-following, or factual model.
- Not a replacement for a cross-encoder when you have real candidate-choice
  labels: a plain LM scores text, it does not learn from selections.
- Not a distributed training framework and not a deployment runtime.

## What it is

- A decoder-only Transformer (pre-LN, causal self-attention, GELU MLP).
- Trained from scratch with next-token cross-entropy; not OpenAI GPT-2 weights
  and not a wrapper around a pretrained model.
- Character-level tokenisation: no word segmenter required.
- ~4.3M parameters in the reference configuration (6 layers, 192 dim, 4 heads,
  block 64, 4,251-token vocabulary).  This is the reference size, **not a cap**;
  see [Scope](#scope-lower-bound-and-upper-bound) for 25M/50M/100M trade-offs.
- Runs on CPU or CUDA; the only Python dependencies are PyTorch and NumPy.
- Includes a zero-dependency live dashboard and a static HTML report.

## Layout

```
train_gpt.py              GPT-2-style trainer (from scratch)
train_dashboard.py        live local dashboard (stdlib + Canvas, no CDN)
train_report.py           self-contained static HTML report
app.py                    Streamlit one-click training studio
prepare_corpus.py         JSONL/text -> character-token corpus
eval_gpt_rerank.py        offline LM candidate-reranking evaluation
examples/sample.txt       tiny smoke-test corpus
tests/test_smoke.py       end-to-end smoke tests
requirements-studio.txt   Streamlit/pandas extras
.streamlit/config.toml    dark studio theme
MODEL_CARD.md             intended use, data, limitations
DATA_CARD.md              corpus provenance and licensing notes
```

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (smoke test)

```bash
python train_gpt.py \
  --corpus examples/sample.txt --tokens 2000 --epochs 1 --max-steps 5 \
  --block 16 --batch 4 --dim 32 --layers 1 --heads 2 \
  --lr 1e-3 --warmup 10 --min-count 1 --max-vocab 2000 \
  --val-ratio 0 --device cpu --save-every 100 \
  --out out_smoke --metrics out_smoke/metrics.jsonl \
  --status out_smoke/status.json

# live dashboard
python train_dashboard.py --dir out_smoke --port 8765 \
  --title "GPT-2 Character LM" --subtitle "smoke run"

# static report, no server
python train_report.py --dir out_smoke --output report.html \
  --title "GPT-2 Character LM"
```

Open <http://127.0.0.1:8765> for live metrics, or `report.html` directly.

## Prepare a corpus

`prepare_corpus.py` accepts LCCC-style JSONL (one JSON array of utterances per
line), JSONL strings, or plain text:

```bash
# one JSON array per line: ["你好", "今天怎么样"]
python prepare_corpus.py --input base.jsonl.gz --output corpus.txt \
  --max-chars 15000000

# append a second source
python prepare_corpus.py --input large.jsonl.gz --output corpus.txt \
  --max-chars 15000000 --append

# plain text: one utterance per line
python prepare_corpus.py --input plain.txt --format text --output corpus.txt
```

Output is one utterance per line with characters separated by spaces:

```
你 好
今 天 怎 么 样
```

Punctuation is kept as its own token.  The trainer's vocabulary decides which
tokens survive (`--min-count` / `--max-vocab`).

### LCCC

The example configuration was trained on a sample of [LCCC](https://github.com/thu-coai/CDial-GPT)
(Chinese conversational corpus).  The corpus is **not bundled** here.  Check
the upstream repository for the authoritative license and attribution terms,
and see [DATA_CARD.md](DATA_CARD.md).  Do not commit the corpus or derived
weights without reviewing the upstream terms.

## Train

Example 4.3M model on ~30M characters (base + large sample):

```bash
python train_gpt.py \
  --corpus corpus.txt --tokens 30000000 --epochs 2 \
  --block 64 --batch 64 --dim 192 --layers 6 --heads 4 \
  --lr 3e-4 --warmup 300 --min-count 20 --max-vocab 13000 \
  --val-ratio 0.02 --device auto --save-every 500 \
  --out out --metrics out/metrics.jsonl --status out/status.json
```

`--device auto` selects CUDA when available.  On a GPU you can usually raise
`--batch` to 128 or 256; re-check the loss curve if you also change the LR.
The example model is small enough for an 8 GB GPU.

### Outputs

```
out/model.pt          fp32 weights (also model_best.pt after validation)
out/vocab.txt         vocabulary
out/metrics.jsonl     loss/PPL every 10 steps
out/status.json       live state for the dashboard
out/train.log         stdout/stderr
```

For the example 4.31M-parameter model, `model.pt` is about 17 MB in fp32.
Int8 deployment would be roughly 4.3 MB plus metadata; quantisation/export is
not included in this package.

### Resume

`--resume` loads `model.pt` and continues from its `epoch`/`step`.  The
checkpoint stores weights, not the AdamW moments, so a resumed run restarts
the optimiser state.  This is acceptable for short experiments; for exact
resumption, add optimizer state to the checkpoint.

## Live dashboard

```bash
python train_dashboard.py --dir out --port 8765 \
  --title "GPT-2 Character LM" --subtitle "LCCC · 6L x 192d"
```

The page shows a progress ring, loss/PPL/throughput charts, token throughput,
best validation PPL, model metadata, and a colourised log tail.  It reads
`status.json`, `metrics.jsonl`, and `train.log`; no external assets are used.
The title and subtitle are configurable and safely escaped.

## Static report

```bash
python train_report.py --dir out --output report.html \
  --title "GPT-2 Character LM" \
  --subtitle "6L x 192d · 4.31M parameters"
```

`report.html` embeds the run data and renders with inline CSS/Canvas.  It has
no server, no CDN, and no external files, so it can be committed, attached to a
release, or opened from disk.

## Streamlit studio (one-click trainer)

`app.py` is a local Streamlit UI for users who do not want to type CLI flags:

```bash
pip install -r requirements-studio.txt
streamlit run app.py
```

After editing `app.py` while the server is running, Streamlit shows a
"File change" prompt — click **Rerun** (or **Always rerun**) before testing;
the previous page state keeps running the old script until you do.

Put `app.py` at the project root (where `tools/lm/train_gpt.py` lives) or set
`ZIMO_TRAINER_ROOT` (or legacy `TINY_CHAR_LM_TRAINER_ROOT` /
`TONGJIA_TRAINER_ROOT`) to the checkout that contains the trainers.  With no
explicit root, the app also searches nearby project checkouts (for example
`../tongjia-ime`, `../tongjia-engine`) for
`tools/lm/train_gpt.py`, `tools/lm/train_nnlm.py`, and
`tools/ranker/train_mlp.py`.  GPT-2 ships in this package; LSTM and MLP need
one of those checkouts.

- **Model playground tab**: auto-load the newest run or upload a `.pt`
  (all-in-one checkpoints need no vocab.txt).  A spec sheet compares file,
  architecture, parameters, layers/dim/heads, context, vocab, size and vocab
  source.  Test mode picks Model A, Model B or A/B compare; compare mode adds
  a PPL-delta column, a winner column and win counts.  General tools:
  per-character NLL, batch scoring with CSV export, and sampled text
  generation, plus manual correction: type a correct/typo pair to get a
  verdict, or type one sentence and let the model propose per-character edits
  (each edit is accepted only if it lowers the whole-sentence PPL).  Uploaded
  models are saved under `runs/playground` and restored after a refresh.
- **Checkpoint formats**: loaders accept `.pt`, `.pth`, `.ckpt`, `.bin` and
  `.safetensors`.  A checkpoint may be an `nn.Module`, a dict under
  `state`/`state_dict`/`model`/`model_state_dict`, or a bare state dict.
  Vocabulary: embedded `vocab` key or safetensors metadata, else a sidecar
  `vocab.txt` / `vocab.json`.  Architecture: checkpoint keys, else
  `config.json`, else inferred from tensor shapes (layer count, dim, context;
  heads default to the largest of 4/8 that divides the dim).
  The export tab can also **write** `model.safetensors` with config and vocab
  embedded in its metadata, so a single file round-trips back into Zimo.
- **Export & Quantization tab** (label `Export & Quantization`, no emoji icons in the UI): convert a loaded model to ONNX with dynamic
  axes (`input_ids [batch, seq_len]` -> `logits`), optionally INT8 dynamic
  quantization via `onnxruntime.quantization.quantize_dynamic`.  A parity
  check compares PyTorch vs ONNX FP32 vs ONNX INT8 PPL on sample text.
  Requires `pip install onnx onnxruntime onnxscript` (see
  `requirements-studio.txt`).  ONNX output is written under
  `runs/playground/export/<timestamp>/` and downloadable in the UI; the
  "Download bundle (.zip)" button packages ONNX + vocab.txt + README.
- **In-app documentation**: a "使用说明 / Docs" button at the bottom of the
  sidebar opens a full spec-style manual dialog (bilingual, follows the
  language toggle).  For every trainer it documents the architecture
  (attention type, layers/dims, parameter count), input/output tensor shapes,
  the exact training signal (next-token CE vs listwise CE), inference
  behavior, uses and boundaries - plus the two scoring paradigms, the IME
  ranking pipeline position, workspace walkthroughs, metric definitions
  (including why PPL is not an acceptance metric), data preparation, FAQ,
  capability boundaries - plus three newer chapters: a model-type taxonomy
  (neural vs statistical, generative vs discriminative), general uses beyond
  IME (scoring/filtering/anomaly/generation, ASR/MT n-best reranking), and a
  deployment matrix for PC / Raspberry Pi / ESP32 (with the MCU memory math
  and the full-integer-quantization gap).  The sidebar footer shows the
  version (`v0.3.0`).
- **Theme**: a 深色 / 浅色 toggle restyles the whole app (native containers,
  cards, charts, logs) at runtime; dark is the startup default.
- **Language**: a 中文 / English toggle in the sidebar localizes the main UI.
  The model itself is character-level: Chinese, English, or mixed text are all
  accepted as training data.  A model learns whatever is in its corpus; the
  4M example is Chinese LCCC.  Train separate models for different language
  mixes instead of assuming one model is bilingual.
- **Sidebar**: architecture, local corpus path (or optional upload), CPU/GPU
  radio, collapsible advanced params (epochs, batch, token budget, custom
  architecture), start/stop, and a **Resume from checkpoint** panel.
- **Detached training**: the trainer is launched as an independent process
  (`runs/<id>/train.pid`, output appended to `train.log`).  Closing the browser
  or dropping SSH keeps it running; reopening the page auto-attaches and tails
  the log.  Every launch also writes `train_config.json` for reproducibility.
- **Monitor**: progress ring, best validation PPL, parameter/vocabulary/
  processed-token cards, Loss / Perplexity / throughput charts with hover
  tooltips, a model-and-data table, a run log, and the exact command.
- **Classic dashboard (compatibility)**: the Training console can embed the
  standalone 8765 monitor in an expander, preserving its original Canvas
  rendering.  That service reads its own `--dir`; set
  `ZIMO_CLASSIC_DASHBOARD_URL` if it is not on `http://127.0.0.1:8765`.
- **Outputs**: when a run finishes, the right panel shows the run directory and
  buttons to open the folder or download the model.  The sidebar's
  "Open existing run" box can adopt an older run directory (for example
  `runs/20260101_120000_gpt2`) so its model can be scored without
  retraining.
- **Test rerank**: after training, enter a context and one candidate per line;
  the studio scores each candidate with the trained GPT-2/LSTM and sorts by PPL.
  Both inputs have contextual tooltips, the result table uses bilingual
  headers, and a "metric definitions" expander explains
  PPL, mean NLL, rank, and the scored-character count.
- **MLP**: trains from selection events but does **not** show step progress
  (the trainer does not emit it) and does **not** support raw-text PPL; it
  exports `l1_ranker.bin` for the feature-based ranker.

Input formats:

- **GPT-2 / LSTM**: `.txt`, one sentence per line.  The app removes spaces and
  re-splits every line into character tokens before training.
- **MLP**: `.jsonl` in the native `selection_events` format, or `.csv` with
  columns `input, selected_text, selected_index, prev_committed, scores`, where
  `scores` is a JSON array of candidate dictionaries (word/features).  A flat
  candidate list is not enough.

Subprocess monitoring: the app starts `python -u <trainer>` with
`PYTHONUNBUFFERED=1`, pumps stdout from a reader thread into a queue, parses
`epoch/step/loss/ppl`, and merges the trainer's `metrics.jsonl`.  While the
process is alive it deliberately does **not** open `status.json`, so an
unpatched Windows trainer cannot be crashed by the UI's reader; `status.json`
is read once after exit for the final state.  Run the studio locally only: it
spawns processes and reads local files.

## Offline reranking evaluation

Perplexity is a training signal, not an acceptance metric.  `eval_gpt_rerank.py`
takes candidate lists from an IME harness and measures whether the LM improves
top-1 reranking:

```bash
python eval_gpt_rerank.py \
  --model out/model_best.pt --vocab out/vocab.txt \
  --cases cases.tsv --csv candidates.tsv
```

- `cases.tsv`: `input<TAB>gold` (one case per line).
- `candidates.tsv`: the harness CSV with at least `input`, `word`, `spanwords`,
  and `final` columns.
- The script reports top-1 under several blend weights; use a held-out case set
  and report correctness before/after, not only PPL.

## Limitations

The explicit lower/upper bounds are in
[Scope](#scope-lower-bound-and-upper-bound).  Implementation-level limits:

- The reference data loader reads all tokens into a Python list.  A few tens of
  millions of tokens are fine; a 600M-character corpus needs a streaming/memmap
  loader (for example, a `uint16` token array + chunked batches).
- `--resume` restores weights only, not optimizer/scheduler state.
- The validation split is the tail of the token stream; for domain-mixed
  corpora, split by source or dialogue instead.
- Character-level models do not know pinyin; they score text, they do not
  generate candidates from keystrokes.
- Small models trained on chat data are domain-specific.  Measure on your own
  task set before integrating.

## License

Code: **MIT** (see [LICENSE](LICENSE)).
Corpus and weights: governed by their upstream data licenses; see
[DATA_CARD.md](DATA_CARD.md) and [MODEL_CARD.md](MODEL_CARD.md).  No corpus,
dictionary, or pretrained weight is included in this repository.

## Citation

See [CITATION.cff](CITATION.cff).  If you use the trainer, cite the repository
and the upstream corpus you trained on.
