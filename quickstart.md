# Quickstart · 5 分钟跑通

## 0. 你需要什么

- Python 3.10 及以上（3.14 已验证）
- 约 1 GB 磁盘（依赖）+ 17 MB（示例模型）
- 不需要 GPU、不需要联网服务（首次装依赖需要网络）

## 1. 安装

**Windows**

```bat
setup.bat
```

**macOS / Linux**

```bash
chmod +x setup.sh run.sh
./setup.sh
```

脚本会创建 `.venv`、安装 `requirements-studio.txt`（streamlit / torch / pandas /
altair / onnx / onnxruntime / safetensors）。

## 2. 启动

```bat
run.bat        :: Windows
```

```bash
./run.sh       # macOS / Linux
```

浏览器打开 <http://127.0.0.1:8769>。

## 3. 先玩示例（不需要训练）

打开「**模型测试台**」：

1. 模型来源保持「自动加载最新训练」——`runs/` 为空时会自动用内置的
   `examples/demo/`（4.31M 参数，对话域）
2. 候选句填几行，例如：

   ```
   我爱你
   我耐你
   我爱泥
   留爱你
   ```

3. 点「测试」→ 按 PPL 排序，Rank 1 绿色、离谱值红色
4. 展开「工具：逐字 NLL」看每个字多"意外"
5. 展开「定制考题」→ 输入 `我爱泥` → 「给出纠错建议」→ 得到 `我爱你`

想批量验证：上传 `examples/demo_pairs.csv`（正确句 / 错别字句），一键算纠错准确率。

## 4. 训练你自己的第一个模型

侧栏：

1. **语料路径**：填一个 `.txt`（每行一句）或目录；也支持 `.jsonl` / `.jsonl.gz`
2. **模型架构**：先用预设「端侧 4M」
3. **高级参数**：Token 预算 100 万起步即可试跑
4. 点「开始训练」

训练在后台独立进程运行（`runs/<时间戳>_gpt2/`），关掉浏览器也不会中断；
重新打开页面会自动接管。

常见坑：

- 语料太小（< 5000 token 或 < 100 种字符）会被拦下——`min_count=20` 会把低频字滤空
- `维度` 必须能被 `注意力头数` 整除
- 训练中不要手动删 `runs/` 下的文件

## 5. 导出到端侧

「导出与量化」页：

| 格式 | 给谁用 |
|---|---|
| `.safetensors` | 归档 / 交换（安全、可 mmap） |
| `.onnx`（FP32/INT8） | onnxruntime：PC、树莓派、移动端 |
| `.tjgpt` | 自带 C++ 引擎直读（零依赖） |
| `.zip` | 上面全部 + `vocab.txt` + README |

INT8 体积约为 FP32 的 1/4，PPL 偏差通常在 0.2% 以内。

## 6. 想看全部功能

侧栏底部「使用说明 / Docs」有十二章规格书：模型类型、六个训练器的架构与训练信号、
五维自定义、三个部署平台、指标定义与常见问题。

## 还是要用命令行？

```bash
.venv/bin/python train_gpt.py \
  --corpus examples/sample.txt --tokens 1000000 --epochs 2 \
  --dim 128 --layers 4 --heads 4 --block 64 --out runs/cli_demo
```

完整参数见 `python train_gpt.py --help`。
