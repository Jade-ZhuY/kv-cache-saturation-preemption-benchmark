# KV Cache 容量压力测试 Benchmark

`kv_cache_saturation_preemption_benchmark.py` 面向 OpenAI 兼容的 vLLM 服务执行**闭环、固定并发**压测：每个 worker 完成一个请求后立即发送下一个请求。因此 `--concurrency` 表示在稳定阶段同时在飞的客户端请求数。

## 安装

```powershell
cd 'D:\benchmark\KV cache测试'
uv run .\kv_cache_saturation_preemption_benchmark.py --help
```

脚本已经声明 PEP 723 依赖；首次 `uv run` 会创建隔离环境并安装依赖，之后自动复用。也可以继续使用 `python -m pip install -r requirements.txt`。

如需严格按模型 tokenizer 构造 8192-token 输入，请传入 `--tokenizer`（本地模型目录或 Hugging Face 模型名）。未传入时会根据 `--chars-per-token`（默认 4）估算，并在结果中标明 `estimated_chars`。

## 单一并发档

```powershell
uv run .\kv_cache_saturation_preemption_benchmark.py `
  --base-url http://127.0.0.1:8000/v1 `
  --metrics-url http://127.0.0.1:8000/metrics `
  --model MODEL `
  --concurrency 16 `
  --requests 200 `
  --input-tokens 8192 `
  --output-tokens 512 `
  --tokenizer MODEL `
  --ignore-eos `
  --gpu-smi
```

## 第一轮实验矩阵

```powershell
uv run .\kv_cache_saturation_preemption_benchmark.py `
  --base-url http://127.0.0.1:8000/v1 `
  --metrics-url http://127.0.0.1:8000/metrics `
  --model MODEL `
  --concurrency 1,2,4,8,16,32,64 `
  --requests 200 `
  --input-tokens 8192 `
  --output-tokens 512 `
  --tokenizer MODEL `
  --ignore-eos `
  --gpu-smi
```

默认使用 `/v1/completions`，这样输入长度是直接可控的。若服务只启用了 Chat Completions，加入 `--api-mode chat`。

默认情况下每个请求都会在 prompt 首部加入不同 nonce，避免 Prefix Cache 命中把本实验变成前缀缓存命中测试。只有有意测试 Prefix Cache 时才传 `--reuse-prompt`。

## 输出

每次执行创建 `results/YYYYMMDD_HHMMSS/`：

- `matrix_summary.csv`：并发档位的横向对比表，可直接用于寻找 knee point。
- `matrix_summary.json`：同一汇总的结构化版本。
- `kv_cache_utilization.png`：唯一的可视化结果。KV Cache 平均/峰值曲线中标出 85% 压力区与 95% 饱和区。
- `concurrency_N/summary.json`：单档位的吞吐、TTFT/TPOT/E2E 分位数、KV/调度/GPU 汇总。
- `concurrency_N/requests.jsonl`：逐请求的 TTFT、TPOT、E2E、实际输入/输出 token、异常。
- `concurrency_N/metrics_samples.jsonl`：压测期间的时序采样，便于画 KV Cache、Waiting、Running 曲线。

当传入 `--metrics-url` 时，脚本兼容常见的 vLLM Prometheus 指标名，采集：

- KV Cache 使用率的平均值/峰值；
- Running / Waiting requests 的平均值/峰值；
- Preemption counter 增量；
- 若服务暴露 `request_queue_time_seconds_sum/count`，则计算平均 Queue Time；
- `--gpu-smi` 时本机 GPU 利用率、显存和功耗的平均值/峰值。

指标原名会写入 `summary.json` 的 `server_metrics.source_metric_names`。某个版本的 vLLM 未暴露某指标时，该字段留空，不会伪造数值。

终端显示每个并发档的开始/完成状态、`waiting_peak` 和 `preemptions`；全部完成后输出并发对比表（Concurrency、KV Avg、KV Peak、Waiting Peak、Preemptions），并自动生成唯一图表 `kv_cache_utilization.png`。

## 指标口径

- **TTFT**：从 HTTP 请求开始到收到第一个非空生成文本 SSE chunk 的时间。
- **TPOT**：`(请求完成时间 - 首 token 时间) / (输出 token - 1)`；优先使用服务端流式 `usage`，否则使用 tokenizer/字符估算。
- **E2E**：从发送请求到 SSE 流结束。
- **吞吐**：所有成功请求的 token 数除以该并发档从开始到结束的墙钟时间。

建议在比较结果前固定 vLLM 版本、模型、TP、dtype、KV cache dtype、`max_model_len`、`gpu_memory_utilization`、Prefix Cache、block size 和 GPU 数量；第一轮仅改变 `--concurrency`。
