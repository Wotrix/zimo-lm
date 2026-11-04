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
| `--device` | auto | `auto` / `cpu` / `cuda` / `musa` |
| `--amp` | auto | `auto`（GPU 上用 bf16）/ `off` / `bf16` / `fp16`；bf16 与 FP32 相对误差 ~3e-5 |
| `--compile` | 关 | `torch.compile` 加速，质量不变，首次编译较慢 |
| `--resume` | 关 | 从 `--out` 里的 `model.pt` 继续 |
| `--max-steps` | 0 | 最多多少步，调试时可设 50 |
| `--tied` | 关 | 共享输入词向量与输出头 |

## 看训练进度

训练在后台跑（`nohup` 或 Studio 的守护进程），终端里不会自己打印。两种看法：

```bash
tail -f runs/base_8x256/train.log      # 原始日志
python monitor.py                      # 进度面板（默认取 runs/ 最新）
python monitor.py --run runs/base_8x256 -w     # 持续刷新，Ctrl+C 停止
python monitor.py --run runs/base_8x256 --no-color
```

面板是一张固定宽度的卡片（按中英文显示宽度对齐，不会错列）：

```text
┌────────────────────────────────────────────────────────────────┐
│ ZIMO · 训练监控                  RTX 4090 D  19:02:58          │
├────────────────────────────────────────────────────────────────┤
│  ██████████████░░░░░░░░░░░░░░░░░░░░░░  42.7%                   │
│  step  20,280 / 47,480        epoch 0                          │
│  loss  3.636                  ppl   37.9                       │
│  speed 133.8k tok/s           eta   42m                        │
│                                                                │
│  trend █▄▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                       │
│        6569 → 37.9  (-99%)                                     │
├────────────────────────────────────────────────────────────────┤
│  GPU  ▮▮▮▮▯▯▯▯▯▯▯▯▯▯▯▯▯▯   5.8/24 GB                            │
│       util  99%  ·  65°C  ·  397 W                             │
│  CPU  192 cores   load 17.6   disk 3.5/50 GB                   │
├────────────────────────────────────────────────────────────────┤
│  model gpt2 12x384 · 25.70M params · vocab 5,672               │
│  data  300.00M tokens · max_12x384                             │
│  ● running   Ctrl+C 只停止查看                                 │
└────────────────────────────────────────────────────────────────┘
```

只依赖标准库：`nvidia-smi` 存在就显示 GPU（型号/显存/温度/功耗），
没有就显示「未检测到 nvidia-smi（CPU 训练）」。摩尔线程机器上同样可用。

## 评测：拿留出语料算分

训练日志里的 val_ppl 是训练时切出来的 2%，用来选 checkpoint；**正式对比要用留出语料单独评**：

```bash
# 单个模型
python eval.py --model runs/base_8x256 --corpus data/valid.txt

# 多个模型对比（同一份语料、同一批 token）
python eval.py --model runs/base_6x192 --model runs/base_8x256                --corpus data/valid.txt --limit 2000000
```

输出：

```text
model                             arch                params      ppl      nll  bits/char
-----------------------------------------------------------------------------------------
base_6x192                        gpt2 6x192       4,596,096    41.72   3.7310      5.383
base_8x256                        gpt2 8x256       9,406,976    36.15   3.5877      5.176 ←

语料：data/valid.txt（2,000,000 tokens，设备 cuda）
```

指标含义：

| 指标 | 含义 |
|---|---|
| **PPL** | 困惑度，越低越好；跨语料不可比，同语料可比 |
| **NLL** | 每 token 平均负对数似然，PPL 的底层值 |
| **bits/char** | 每字符信息量（bits），跨模型/跨词表最公平 |

注意：`--corpus` 必须放**训练时没见过**的文本（LCCC 的 `lccc_base_valid.jsonl.gz`
就适合），否则分数虚高。

## 模型说明卡

```bash
python make_model_card.py --run runs/base_8x256        # 生成 runs/base_8x256/README.md
python make_model_card.py --index models/              # 为目录下所有模型生成总索引
```

卡片内容全部来自运行记录：结构（从 checkpoint 读）、训练配置（status.json /
train_config.json）、验证曲线（metrics.jsonl）、本机 CPU 延迟实测、INT8 体积估算。

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
