# 租 GPU 训练：从开机到出模型

面向「按小时租的云 GPU」。按顺序复制命令即可，预计 **20 分钟上手 + 2–3 小时训练**。

---

## 0. 选卡（重要）

| 类型 | 建议 | 说明 |
|---|---|---|
| NVIDIA（推荐） | RTX 3090 / 4090 / A10 / A100 | 生态最稳，`setup.sh` 全自动 |
| 摩尔线程 MUSA | MTT S80 / S4000 | 能用但要装 `torch_musa`，见第 6 节 |

**显存优先于算力**：本项目是 FP32 训练（没有 AMP），16 GB 起比较舒服；
8 GB 也能跑，把 `--batch` 降到 32–64。

> 训练不需要 CUDA Toolkit，只需要驱动 + PyTorch wheel（`setup.sh` 会处理）。

---

## 1. 拉代码并安装（5–10 分钟）

```bash
sudo apt-get update && sudo apt-get install -y git python3-venv   # 首次
git clone https://github.com/Wotrix/zimo-lm.git
cd zimo-lm

./setup.sh --check     # 先看会装什么
./setup.sh             # 自动识别显卡：有卡装 CUDA 版，无卡装 CPU 版
```

装完会打印：

```text
torch      : 2.x.x+cu124
cuda build : 12.4
available  : True
device     : NVIDIA GeForce RTX 4090
```

`available: True` 才继续。若为 False，看第 7 节。

---

## 2. 冒烟（2 分钟，必做）

```bash
.venv/bin/python train_gpt.py \
  --corpus examples/sample.txt \
  --tokens 60000 --epochs 1 \
  --dim 32 --layers 1 --heads 1 --block 16 \
  --min-count 1 --max-vocab 500 \
  --device cuda --out runs/smoke
```

成功的标志：

```text
[lm] tokens=358 vocab=216 windows=21
[lm] device=cuda:0 backend=cuda
[lm] epoch=0 val_loss=... val_ppl=...
[lm] done
```

出现 `backend=cuda`（或 `musa`）说明 GPU 真的被用上了。

---

## 3. 准备语料并上传

### 3.1 本地转成训练文本（一次性）

LCCC 的 `.jsonl.gz` 要先转成「每行一句、字符间空格」的文本：

```bash
# base 取 1.5 亿字符，large 取 1.5 亿字符，合成 3 亿
python prepare_corpus.py --input yuliao/lccc_base_train.jsonl.gz     --output corpus_base.txt --max-chars 150000000
python prepare_corpus.py --input yuliao/lccc_large.jsonl.gz     --output corpus_large.txt --max-chars 150000000
cat corpus_base.txt corpus_large.txt > corpus_300m.txt
wc -c corpus_300m.txt          # 约 1.2 GB（字符间有空格，体积约 4 字节/字）
```

小语料够用时可以只做基线，`--max-chars 50000000` 即可。

### 3.2 上传

```bash
# 本地执行
scp corpus_300m.txt user@<host>:~/zimo-lm/data/
```

> 也可以直接把 `.jsonl.gz` 传上去，在机器上用同一个脚本转换，省一次拷贝。
> 训练时只按 `--tokens` 读取，不会一次性载入全部；但 `--tokens` 越大内存越高
>（int64 数组 + 字符串列表）。

---

## 4. 训练（推荐两条路线）

### 4.1 基线：6 层 / 192 维（4.3M）· 100M token

```bash
.venv/bin/python train_gpt.py \
  --corpus data/lccc_train.txt \
  --tokens 100000000 --epochs 2 \
  --block 128 --batch 64 \
  --dim 192 --layers 6 --heads 4 \
  --min-count 20 --max-vocab 13000 \
  --lr 3e-4 --warmup 500 --val-ratio 0.02 --save-every 500 \
  --device cuda --out runs/base_6x192
```

RTX 4090 上约 **20–40 分钟**；这一步用来确认端到端可行，并拿到新基线 PPL。

### 4.2 正式：8 层 / 256 维（9.4M）· 300M token

```bash
.venv/bin/python train_gpt.py \
  --corpus data/lccc_train.txt \
  --tokens 300000000 --epochs 2 \
  --block 128 --batch 128 \
  --dim 256 --layers 8 --heads 8 \
  --min-count 20 --max-vocab 13000 \
  --lr 3e-4 --warmup 500 --val-ratio 0.02 --save-every 500 \
  --device cuda --out runs/base_8x256
```

RTX 4090 上约 **2–3 小时**；显存 8 GB 时把 `--batch` 降到 64。

> 断线不用怕：加 `--resume` 重跑同一条命令即可从 `model.pt` 继续。
> 用 Studio（`./zimo`）启动的话，训练本身就是独立后台进程。

---

## 5. 拿回产物

```bash
# 本地执行
scp -r user@<host>:~/zimo-lm/runs/base_8x256 ./
```

目录里需要的是：

```text
model_best.pt   权重（含词表和结构，单文件）
vocab.txt       词表
status.json     最终状态（val_ppl 等）
```

拿到本机后，Studio 里用「手动上传模型」加载，或在「导出与量化」里转
ONNX / safetensors / TJGPT。

---

## 6. 摩尔线程（MUSA）

代码已支持，但**依赖要你自己装对**：

```bash
# 1. 驱动 + MUSA Toolkit（用厂商给的最新版）
# 2. PyTorch 与 torch_musa 版本必须严格匹配，安装方式见摩尔线程官方文档
python -c "import torch, torch.musa; print(torch.musa.is_available())"
```

看到 `True` 后：

```bash
./setup.sh cpu                      # 装基础依赖（不要装 CUDA 版 torch）
.venv/bin/python train_gpt.py ... --device musa --out runs/musa_smoke
```

训练器会自动识别：`--device auto` 时优先 MUSA，其次 CUDA，最后 CPU；
即使传 `--device cuda`，只有 MUSA 的机器也会自动落到 MUSA。

**风险**：`torch_musa` 个别算子可能回退 CPU（表现为异常慢）。
如果冒烟测试的 `tokens_per_s` 明显低于同价位 NVIDIA，先换一张卡再投入长训练。

---

## 7. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `available: False` | 驱动没装好 / 容器没透传 GPU | `nvidia-smi` 能跑吗？Docker 加 `--gpus all` |
| `CUDA out of memory` | batch/block 太大 | `--batch` 减半，再降 `--block` |
| 训练 1 秒结束、`vocab=1` | 语料太小或 `min-count` 太高 | `--min-count` 降到 1–3，语料 ≥ 10 万 token |
| `embed_dim must be divisible by num_heads` | dim 不能被 heads 整除 | 256/8 ✅，384/12 ✅，512/12 ❌ |
| 中文变 `�` | 语料是 GBK | 训练器只按 UTF-8 读；先转码，或用 Studio 的本地路径模式 |
| 磁盘满 | 每 500 步存一次 checkpoint | `--save-every 2000`，或训练后清理旧 `model.pt` |

---

## 8. 时间与费用参考（按小时计费）

| 任务 | RTX 4090 | 说明 |
|---|---|---|
| setup + 冒烟 | 15 分钟 | 装依赖占大头 |
| 6层/192 · 100M token ×2 | 20–40 分钟 | 先跑这个 |
| 8层/256 · 300M token ×2 | 2–3 小时 | 正式模型 |
| 导出 INT8 + 验证 | 10 分钟 | 可在本机做 |

按 ¥2–4/小时算，**正式训练一次约 ¥6–12**。建议一次租 4 小时，够跑基线和正式两轮。

---

## 9. 租卡当天的顺序（照抄）

```bash
git clone https://github.com/Wotrix/zimo-lm.git && cd zimo-lm
./setup.sh
.venv/bin/python train_gpt.py --corpus examples/sample.txt --tokens 60000 \
  --epochs 1 --dim 32 --layers 1 --heads 1 --block 16 \
  --min-count 1 --max-vocab 500 --device cuda --out runs/smoke
# ↑ 看到 backend=cuda 再继续
scp <本地> data/lccc_train.txt user@host:~/zimo-lm/data/
.venv/bin/python train_gpt.py --corpus data/lccc_train.txt \
  --tokens 100000000 --epochs 2 --block 128 --batch 64 \
  --dim 192 --layers 6 --heads 4 --min-count 20 --max-vocab 13000 \
  --lr 3e-4 --warmup 500 --save-every 500 \
  --device cuda --out runs/base_6x192
```
