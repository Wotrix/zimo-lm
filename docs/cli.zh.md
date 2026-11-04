# 命令行训练指引

Studio 的每个选择都会拼成一条命令，在「运行信息 → 完整命令」里可以复制。
不想开浏览器时，直接用这份指引。

```bash
python train_gpt.py --help      # 全部中文参数说明
```

## 1. 最小可跑

```bash
python train_gpt.py \
  --corpus examples/sample.txt \
  --tokens 200000 --epochs 1 \
  --dim 64 --layers 2 --heads 2 --block 32 \
  --min-count 1 --max-vocab 2000 \
  --out runs/smoke
```

几十秒就能看到 loss。`--min-count 1` 是必须的：示例语料太小，默认的 20
会把所有低频字过滤掉，词表塌成 1。

## 2. 常规训练（CPU / 单卡）

```bash
python train_gpt.py \
  --corpus corpus.txt \
  --tokens 30000000 --epochs 2 \
  --block 64 --batch 32 \
  --dim 192 --layers 6 --heads 4 \
  --min-count 20 --max-vocab 13000 \
  --lr 3e-4 --warmup 300 --val-ratio 0.02 \
  --out runs/base
```

这是 4.31M 参数配置，30M token 在普通 CPU 上约 48 分钟。

## 3. GPU 训练

```bash
python train_gpt.py --corpus corpus.txt --tokens 100000000 \
  --dim 256 --layers 8 --heads 8 --block 128 --batch 128 \
  --device cuda --out runs/gpu
```

- `--device auto`（默认）会自动用 GPU；`cuda` 强制，`cpu` 强制
- 显存不够先降 `--batch`，再降 `--block`
- 大 batch 配大学习率；`--batch 128` 时 `--lr 5e-4` 通常比 3e-4 更快收敛

## 4. 断点续训

```bash
python train_gpt.py --corpus corpus.txt --out runs/base --resume
```

`--resume` 读取 `runs/base/model.pt`（含 `step` / `epoch`），从上次位置继续。
**语料路径和结构参数必须和上次一致**，否则形状对不上。

## 5. 多份语料一起训

方式一：先合并成一个文件（推荐，顺序可控）

```bash
cat a.txt b.txt c.txt > corpus.txt
# Windows PowerShell: Get-Content a.txt,b.txt,c.txt | Set-Content -Encoding utf8 corpus.txt
```

方式二：在 Studio 里多选上传 `.txt`，或本地路径里填目录（会自动收集
`.txt` / `.jsonl` / `.jsonl.gz`）。多份文件按选择顺序拼接。

> 编码：UTF-8（含 BOM）和 GBK/GB18030 都能读。JSONL 需是 UTF-8。

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--corpus` | `corpus.txt` | 每行一句；字符间有空格更佳 |
| `--tokens` | 15M | 最多读多少 token，越大越慢 |
| `--epochs` | 3 | 轮数；小语料 5–10，大语料 1–3 |
| `--block` | 64 | 上下文长度，决定「能看多远」 |
| `--batch` | 32 | CPU 32；GPU 64–256 |
| `--dim` | 128 | 隐藏维度，**必须能被 `--heads` 整除** |
| `--layers` | 3 | 层数；4M 配置用 6 |
| `--heads` | 4 | 注意力头数 |
| `--lr` | 3e-4 | 大模型降到 1e-4 |
| `--min-count` | 20 | 字符最低频次，**小语料必须调小** |
| `--max-vocab` | 13000 | 词表上限 |
| `--val-ratio` | 0.02 | 验证集比例，从尾部切 |
| `--warmup` | 300 | 线性升温步数 |
| `--save-every` | 500 | 每 N 步落盘，断线续训靠它 |
| `--device` | auto | `auto` / `cpu` / `cuda` |
| `--resume` | 关 | 从 `--out` 里的 `model.pt` 继续 |
| `--max-steps` | 0 | 最多多少步，调试时可设 50 |
| `--tied` | 关 | 共享输入词向量与输出头 |

## 产物

```text
<out>/
├── model.pt          最新权重（续训读它）
├── model_best.pt     验证集最优
├── vocab.txt         词表，一行一个字符
├── metrics.jsonl     逐步指标（step/loss/ppl，验证点带 val=1）
└── status.json       最终状态（state/step/val_ppl/tokens/params）
```

## 常见问题

**词表只有 1 个**（`vocab=1`）
语料太小或 `--min-count` 太高。先降到 `--min-count 1`，语料建议 ≥ 10 万 token。

**`embed_dim must be divisible by num_heads`**
`--dim` 不能被 `--heads` 整除。例如 12 头配 512 维可以（512/12 不行，用 8 头）。

**`CUDA out of memory`**
降 `--batch`（128 → 64 → 32），再降 `--block`。

**训练 1 秒就结束、PPL 恒为 1**
同上，词表塌了。

**中文变成 `�`**
语料是 GBK/ANSI。`train_gpt.py` 按 UTF-8 读；先用记事本另存为 UTF-8，
或走 Studio 的本地路径模式（它会自动识别 GBK）。

**断线后想接着跑**
加 `--resume` 指向同一个 `--out`。若 `model.pt` 不存在（还没到
`--save-every`），只能从头开始——大语料务必把 `--save-every` 设小一点。

## 与 Studio 的关系

Studio 只是把这些参数包成界面：训练在**独立后台进程**里跑（`train.pid` +
`train.log`），关掉浏览器不影响；「完整命令」面板里显示的就是等价的命令行。
