# 部署在线演示（可选）

Studio 可以在云端跑一个**只读演示版**：模型测试台 + 导出量化，使用仓库自带的
4.31M 示例模型。训练不在云端提供——它需要本地语料和持续的 CPU/GPU，
社区托管既不合适也违反其使用条款。

云端入口是 **`app_cloud.py`**：它设置 `ZIMO_CLOUD=1`，隐藏训练控制台、
隐藏本地路径输入，只保留两个 Tab。

---

## 方案 A：Streamlit Community Cloud（免费）

**限制**：约 1 GB 内存、CPU、无人访问时休眠、文件系统临时。
torch（CPU 版）装完约 800 MB，所以只适合演示。

### 步骤

1. 打开 <https://share.streamlit.io>，用 GitHub 登录
2. **Create app** → *Deploy a public app from GitHub*
3. 填写：

   | 字段 | 值 |
   |---|---|
   | Repository | `Wotrix/zimo-lm` |
   | Branch | `main` |
   | Main file path | **`app_cloud.py`** |
   | Advanced settings → Python version | **3.11**（3.13 可能缺 wheel） |

4. 点 **Deploy**，首次安装依赖约 5–10 分钟

### 部署后检查

- 侧栏显示「在线演示 · 端侧小模型工作台」
- 只有两个 Tab：模型测试台 / 导出与量化
- 测试台自动加载 `examples/demo/` 的示例模型

### 如果启动失败

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: torch` | requirements 没被读取 | 确认根目录有 `requirements.txt`，且 Main file path 是 `app_cloud.py` |
| 内存不足 / 容器重启 | 导出 ONNX 时峰值内存偏高 | 演示时只用测试台；或把 `onnx`、`onnxscript` 从 requirements 临时移除 |
| 页面白屏很久 | 首次装 torch 慢 | 等 10 分钟，Cloud 日志里能看到 pip 进度 |

---

## 方案 B：Hugging Face Spaces（推荐，内存 16 GB）

比社区云宽松得多，适合带 torch 的应用。

1. <https://huggingface.co/new-space> → SDK 选 **Streamlit**
2. 把仓库推上去：

   ```bash
   git remote add space https://huggingface.co/spaces/<你的用户名>/zimo-lm
   git push space main
   ```

3. 在 Space 的 `README.md` 顶部加 front-matter（HF 要求）：

   ```yaml
   ---
   title: Zimo Studio
   emoji: 🧩
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   sdk_version: 1.32.0
   app_file: app_cloud.py
   pinned: false
   ---
   ```

4. Space 会自动构建并给出公网地址

> 注意：HF 会把 `README.md` 当成 Space 卡片，所以直接推这个仓库时
> front-matter 会覆盖项目 README；更干净的做法是只推必要文件，或在 HF 上
> 维护一份带 front-matter 的 README。

---

## 云端能做什么 / 不能做什么

| 功能 | 云端演示 | 说明 |
|---|---|---|
| 候选打分 / A/B 对比 | ✅ | 用自带示例模型 |
| 逐字 NLL / 批量打分 / 文本生成 | ✅ | 纯推理，CPU 够用 |
| 纠错建议 / 自定义词典 | ✅ | 同上 |
| 导出 ONNX / safetensors / TJGPT | ✅ | 4M 模型约 16 MB，内存可承受 |
| **训练** | ❌ | 需要本地语料与长时间计算 |
| **本地路径语料** | ❌ | 云端没有你的磁盘 |
| **后台守护 / 断点续训** | ❌ | 容器重启即丢 |

---

## 想全功能在线用？

租一台带 GPU 的机器，直接在那边跑 `zimo`（或 `run.sh`），用 SSH 端口转发访问：

```bash
ssh -L 8769:127.0.0.1:8769 user@your-gpu-box
# 然后在服务器上： ./run.sh   （或 zimo）
# 本地浏览器打开 http://127.0.0.1:8769
```

这样训练、测试、导出都是完整功能，且不用把服务暴露到公网。
