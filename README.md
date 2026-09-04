# KV Cache Saturation and Preemption Benchmark

面向 **OpenAI 兼容流式 API 与 vLLM Prometheus 指标** 的 KV Cache 容量压测工具。它以固定长上下文 workload 逐档提升并发，记录 KV Cache 使用率、Waiting Requests 与 Preemption，并输出一张聚焦 KV Cache 饱和点的曲线图。

## 功能概览

- **阶梯并发扫描**：使用 `--concurrency` 指定一个档位或逗号分隔的并发矩阵。
- **闭环固定并发**：每个 worker 完成一个请求后立即发送下一个请求，稳定阶段的在途请求数接近指定并发。
- **长上下文压力**：默认每请求 8192 input tokens 与最多 512 output tokens，适合观察 KV Cache 容量边界。
- **真实流式指标**：基于第一个非空 SSE 文本 chunk 计算 TTFT；基于输出 token 数计算 TPOT 与吞吐。
- **vLLM 调度指标**：从 `/metrics` 采集 KV Cache 使用率、Running / Waiting Requests、Preemption 与 Queue Time。
- **避免 Prefix Cache 干扰**：默认在每个 Prompt 首部写入不同 nonce，避免重复完整前缀造成缓存命中。
- **单图输出**：只生成 KV Cache 平均/峰值曲线，并标出 85% 压力区与 95% 饱和区。
- **终端对比表**：逐档显示 `waiting_peak`、`preemptions`，结束后汇总所有并发档位。

## 项目结构

```text
kv-cache-saturation-preemption-benchmark/
├── kv_cache_saturation_preemption_benchmark.py  # 压测、Prometheus 采样与绘图 CLI
├── KV_Cache_容量压力测试_实验思路.md              # 实验设计与指标说明
├── requirements.txt                              # pip 安装依赖（可选）
├── results/                                      # 运行后生成的结果，默认忽略 Git
└── README.md
```

## 环境要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- OpenAI 兼容的流式推理服务；默认使用 `/v1/completions`
- 如需采集 KV Cache / Waiting / Preemption：vLLM 的 Prometheus `/metrics`
- 如需采集本机 GPU 利用率、显存和功耗：可执行 `nvidia-smi`

## 使用 uv 运行

脚本内置 PEP 723 依赖声明。首次执行时，uv 会自动创建隔离环境并安装依赖：

```powershell
uv run .\kv_cache_saturation_preemption_benchmark.py --help
```

也可使用传统 pip 安装：

```powershell
python -m pip install -r requirements.txt
```

## 运行测试

### 单一并发档

```powershell
uv run .\kv_cache_saturation_preemption_benchmark.py `
  --base-url http://HOST:PORT/v1 `
  --metrics-url http://HOST:PORT/metrics `
  --model MODEL `
  --concurrency 16 `
  --requests 200 `
  --input-tokens 8192 `
  --output-tokens 512 `
  --ignore-eos `
  --gpu-smi
```

### 执行 KV Cache 饱和扫描

```powershell
uv run .\kv_cache_saturation_preemption_benchmark.py `
  --base-url http://HOST:PORT/v1 `
  --metrics-url http://HOST:PORT/metrics `
  --model MODEL `
  --concurrency 1,2,4,8,16,32,64,96,128 `
  --requests 200 `
  --input-tokens 8192 `
  --output-tokens 512 `
  --ignore-eos `
  --gpu-smi
```

如果最高并发较高，建议保证请求数至少为最高并发的 10 倍；例如测试到 128 并发时使用 `--requests 1280`。这样每个 worker 至少会完成约 10 个请求，减少收尾阶段对结果的影响。

### 精确构造输入 Token

未设置 `--tokenizer` 时，脚本按 `--chars-per-token`（默认 4）估算输入与回退输出 token，并在结果中标记 `estimated_chars`。

传入本地 tokenizer 路径或 Hugging Face tokenizer 名称后，脚本使用 `transformers` 精确构造目标输入长度：

```powershell
--tokenizer MODEL
```

默认使用 `/v1/completions`，便于控制输入长度。若服务只暴露 Chat Completions 接口，加入：

```powershell
--api-mode chat
```

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--base-url` | OpenAI 兼容 API 根地址 | `http://127.0.0.1:8000/v1` |
| `--model` | 服务端模型名 | 必填 |
| `--concurrency` | 单个正整数，或逗号分隔的并发档位 | `1,2,4,8,16,32` |
| `--requests` | 每个并发档的总请求数 | `200` |
| `--input-tokens` | 每请求目标输入 token 数 | `8192` |
| `--output-tokens` | 每请求最大输出 token 数 | `512` |
| `--api-mode` | `completion` 或 `chat` | `completion` |
| `--tokenizer` | 用于精确 token 构造和计数的 tokenizer | 未设置 |
| `--chars-per-token` | 未设置 tokenizer 时的 token 估算比例 | `4.0` |
| `--temperature` | 采样温度 | `0.0` |
| `--ignore-eos` | 忽略 EOS，尽量生成到目标输出长度 | `False` |
| `--reuse-prompt` | 复用相同 Prompt，用于 Prefix Cache 对照 | `False` |
| `--no-stream-usage` | 不发送 `stream_options.include_usage`，兼容旧服务 | `False` |
| `--api-key` | API Key；未设置时读取 `OPENAI_API_KEY` | 未设置 |
| `--metrics-url` | vLLM Prometheus `/metrics` 地址 | 未设置 |
| `--metrics-interval-s` | Prometheus / GPU 采样间隔（秒） | `1.0` |
| `--gpu-smi` | 采集运行脚本机器的 GPU 指标 | `False` |
| `--connect-timeout-s` | HTTP 建连超时（秒） | `20` |
| `--request-timeout-s` | 单请求总超时（秒） | `3600` |
| `--output-dir` | 结果目录 | `results` |

## 结果输出

每轮测试创建 `results/YYYYMMDD_HHMMSS/`：

```text
results/YYYYMMDD_HHMMSS/
├── kv_cache_utilization.png     # 唯一可视化结果
├── matrix_summary.csv           # 所有并发档位的汇总表
├── matrix_summary.json          # 汇总 JSON
├── command.json                 # 本次执行的 CLI 参数
└── concurrency_N/
    ├── summary.json             # 单档位吞吐、延迟、KV / 调度 / GPU 汇总
    ├── requests.jsonl           # 逐请求 TTFT、TPOT、E2E 与 token 数
    └── metrics_samples.jsonl    # Prometheus / GPU 时序采样
```

终端在每个档位结束时输出：

```text
[完成] concurrency=64; ok=640/640; waiting_peak=18; preemptions=7
```

全部档位结束后输出并发对比表：

```text
Concurrency | Success | KV Avg | KV Peak | Waiting Peak | Preemptions
```

## KV Cache 曲线解读

- **蓝线 `KV avg`**：该并发档压测期间的平均 KV Cache 使用率。
- **红线 `KV peak`**：该档出现过的最高使用率，是判断容量饱和的主要依据。
- **85%–95%**：压力区。继续提高并发时，等待、尾延迟或抢占风险上升。
- **≥95%**：饱和区。新请求通常等待；若调度器需要释放运行请求占用的 KV Block，可能出现 Preemption。

建议结合曲线和终端表判断：

```text
KV Peak < 85%                     → KV Cache 充足
85% ≤ KV Peak < 95%               → KV Cache 压力区
KV Peak ≥ 95% + Waiting 增加      → 明显容量饱和
KV Peak ≥ 95% + Preemption 增加   → 已发生 KV Cache 抢占
```

KV Cache 到 100% 并不必然导致请求立即失败。vLLM 可以先让新请求等待，必要时抢占并稍后恢复部分运行请求；此时 TTFT、E2E 与尾延迟通常会恶化。

## 指标口径

| 指标 | 定义 |
| --- | --- |
| **KV Avg / KV Peak** | Prometheus 采样期间的 KV Cache 使用率平均值 / 峰值。 |
| **Waiting Peak** | `vllm:num_requests_waiting` 的采样峰值。 |
| **Preemptions** | `vllm:num_preemptions_total` 在该档压测期间的增量。 |
| **TTFT** | 从 HTTP 请求开始到收到第一个非空生成文本 SSE chunk 的时间。 |
| **TPOT** | `(请求完成时间 - 首 token 时间) / (输出 token - 1)`。 |
| **E2E** | 从发送请求到 SSE 流结束的端到端时间。 |
| **Output tok/s** | 所有成功请求输出 token 数 / 该并发档墙钟时间。 |

## 实验约束

横向比较前，请固定模型、vLLM 版本、GPU 数量、Tensor Parallel、dtype、KV Cache dtype、`max_model_len`、`gpu_memory_utilization`、Prefix Cache 与 block size。第一轮实验仅改变 `--concurrency`。

默认 Prompt 使用唯一 nonce 以规避 Prefix Cache 干扰。只有明确需要测试重复前缀缓存命中时，才添加 `--reuse-prompt`。

## License

本项目使用 [MIT License](LICENSE)。
