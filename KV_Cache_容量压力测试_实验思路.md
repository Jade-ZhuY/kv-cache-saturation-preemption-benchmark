# KV Cache 容量压力测试：第一组实验思路

## 1. 实验目标

本实验的核心不是单纯查看 KV Cache 占用了多少显存，而是回答一个更工程化的问题：

> **在当前模型、当前 GPU、当前 vLLM 配置下，随着并发请求数不断增加，KV Cache 什么时候开始成为系统瓶颈？当 KV Cache 接近饱和后，TTFT、TPOT、吞吐、排队与调度分别会发生什么变化？**

因此，这一组实验本质上是在寻找当前部署配置下的：

- **KV Cache Saturation Point（KV Cache 饱和点）**
- **Knee Point（性能拐点）**
- **较合理的稳定并发区间**

最终希望得到的不只是某个并发下的 TPS，而是能够回答：

> 当前模型在某个上下文长度下，最多可以稳定承载多少并发；继续提高并发以后，系统是因为算力、KV Cache，还是调度开始成为瓶颈。

---

## 2. 实验基本原理

模型在推理过程中会为每一个已经处理过的 token 保存对应的 K/V 状态。

可以粗略理解为：

```text
一个请求的历史 token 越多
        ↓
这个请求占用的 KV Cache 越多
```

如果同时存在多个活跃请求，那么 KV Cache 的总占用大致与所有活跃请求当前已经处理的 token 数量之和相关：

```text
Active KV Tokens
≈ 所有 Active Request 已处理 Token 数量之和
```

例如，假设每个请求最终需要维护约 8K token 的 KV：

```text
1 个请求  × 8K ≈ 8K tokens KV
4 个请求  × 8K ≈ 32K tokens KV
16 个请求 × 8K ≈ 128K tokens KV
32 个请求 × 8K ≈ 256K tokens KV
```

因此，在其他条件不变时：

```text
Concurrency ↑
      ↓
同时活跃的 Sequence ↑
      ↓
KV Cache 占用 ↑
      ↓
逐渐接近 KV Cache 容量上限
```

当 KV Cache 仍然充足时，增加并发通常有利于提升 batching 效率和 GPU 利用率。

当 KV Cache 接近饱和后，可能出现：

```text
KV Cache 接近 100%
        ↓
新的请求无法立即获得足够 KV Block
        ↓
Waiting / Preemption 增加
        ↓
Queue Time 增加
        ↓
TTFT / E2E 延迟恶化
        ↓
吞吐增长趋缓，甚至下降
```

这就是本实验要观察的完整过程。

---

## 3. 实验设计原则

### 3.1 一次只改变一个变量

第一轮实验只改变：

> **Concurrency（并发数）**

其他所有配置必须保持一致，例如：

- 模型不变
- vLLM 版本不变
- GPU 数量不变
- Tensor Parallel 配置不变
- 模型 dtype 不变
- KV Cache dtype 不变
- `max_model_len` 不变
- `gpu_memory_utilization` 不变
- Prompt 长度不变
- Output 长度不变
- 请求总数不变
- Benchmark 数据分布不变

不要在这一轮同时修改：

- FP8 KV Cache
- Prefix Cache
- KV Offload
- Block Size
- GPU Memory Utilization
- Context Length

否则出现性能变化后，很难判断到底是哪一个因素导致的。

---

## 4. 第一轮实验 Workload 设计

为了让 KV Cache 真正产生压力，Prompt 不能太短。

如果使用：

```text
Input  = 100 tokens
Output = 50 tokens
```

即使并发很高，也可能仍然无法把 KV Cache 压到高水位。

因此第一轮建议使用一个相对较长、但又不会轻易超过模型上下文限制的 workload，例如：

```text
Input Length  = 8192 tokens
Output Length = 512 tokens
```

一个请求在完成时大约需要维护：

```text
8192 + 512 = 8704 tokens
```

对应的 KV Cache。

这样随着并发增加，会更容易观察到 KV Cache 从低占用到高占用、再到饱和的全过程。

---

## 5. 第一版实验矩阵

建议先使用下面这组并发：

| 实验编号 | Input Length | Output Length | Concurrency |
|---|---:|---:|---:|
| A1 | 8192 | 512 | 1 |
| A2 | 8192 | 512 | 2 |
| A3 | 8192 | 512 | 4 |
| A4 | 8192 | 512 | 8 |
| A5 | 8192 | 512 | 16 |
| A6 | 8192 | 512 | 32 |
| A7 | 8192 | 512 | 64（可选） |

每一档并发建议使用相同的请求总数，例如：

```text
200 requests
```

这样不同实验之间更容易横向比较。

如果 32 并发已经出现明显的 KV Cache 饱和，则不一定必须继续跑 64。

如果 32 并发仍然非常轻松，则可以继续提高到 64、128，直到找到明显拐点。

---

## 6. 每一组实验需要记录的指标

第一组实验不能只看 TPS，至少要同时记录以下指标。

### 6.1 KV Cache 相关指标

- KV Cache Usage
- KV Cache 使用率峰值
- KV Cache 使用率平均值
- 可缓存 token 数量（如果能够获取）
- KV Block 使用情况（如果开启相关指标）

核心问题：

> 随着并发增加，KV Cache 是否逐渐接近 100%？

---

### 6.2 调度相关指标

- Running Requests
- Waiting Requests
- Preemption 次数
- Queue Time

核心问题：

> KV Cache 接近饱和以后，是否开始出现请求等待或抢占？

---

### 6.3 延迟指标

至少记录：

- TTFT P50
- TTFT P95
- TTFT P99
- TPOT / ITL P50
- TPOT / ITL P95
- TPOT / ITL P99
- E2E Latency

核心问题：

> 随着 KV Cache 压力增大，是 TTFT 先恶化，还是 TPOT 也明显变差？

---

### 6.4 吞吐指标

- Request Throughput
- Input Token Throughput
- Output Token Throughput
- Total Token Throughput

核心问题：

> 并发增加后，吞吐是否仍然线性增长？什么时候开始进入平台期？

---

### 6.5 GPU 资源指标

- GPU Utilization
- GPU Memory Usage
- HBM 使用情况
- GPU Power（可选）

核心问题：

> 当前瓶颈到底是 GPU 算力已经饱和，还是 KV Cache 容量先饱和？

---

## 7. 预期会看到的三个阶段

### 阶段一：KV Cache 充足

可能出现在：

```text
Concurrency = 1 / 2 / 4
```

典型现象：

```text
KV Cache Usage：较低
Preemption：0
Waiting Requests：0
GPU Utilization：逐渐提高
Throughput：明显提升
TTFT：变化较小
```

此时增加并发通常是有利的，因为 batching 更充分，GPU 利用率更高。

可以理解为：

```text
Concurrency 1
GPU：████████░░░░░░░░

Concurrency 4
GPU：██████████████░░
```

---

### 阶段二：理想高负载区

可能出现在：

```text
Concurrency = 8 / 16
```

典型现象：

```text
KV Cache Usage：60% ~ 85%
GPU Utilization：接近高位
Preemption：接近 0
Waiting：较少
Throughput：仍然增长
TTFT：仍然可接受
```

这个区间通常是最值得关注的，因为它可能就是当前配置下较合理的生产运行区间。

理想状态是：

> GPU 已经被充分利用，但 KV Cache 还没有进入明显争抢状态。

---

### 阶段三：KV Cache 饱和区

当并发继续增加，例如：

```text
Concurrency = 32 / 64
```

可能出现：

```text
KV Cache Usage：95% ~ 100%
Waiting Requests：明显增加
Preemption：开始出现或快速增加
Queue Time：明显增加
P95/P99 TTFT：快速恶化
Throughput：增长趋缓甚至下降
```

可能出现类似现象：

```text
Concurrency：16 → 32

Output Throughput：
1000 tok/s → 1100 tok/s

但 P99 TTFT：
100 ms → 800 ms
```

虽然并发翻倍，但吞吐几乎没有提升，尾延迟却显著恶化。

这往往意味着系统已经越过了合理工作区间。

---

## 8. 最关键的观察：寻找 Knee Point

本实验最终需要找到性能曲线的“拐点”。

### 8.1 KV Cache Usage 曲线

```text
KV Cache Usage
100% |                         ●
 90% |                    ●
 80% |                ●
 60% |           ●
 40% |       ●
 20% |   ●
     +---------------------------
       1  2  4  8  16 32 64
             Concurrency
```

---

### 8.2 Throughput 曲线

```text
Throughput
   ↑
   │                 ●────●
   │             ●
   │         ●
   │     ●
   │ ●
   └────────────────────────→
     1 2 4 8 16 32 64
```

重点寻找：

```text
开始：并发增加 → 吞吐快速提升
后来：并发增加 → 吞吐几乎不再提升
```

---

### 8.3 TTFT 曲线

```text
TTFT
 ↑
 │                         ●
 │                    ●
 │
 │             ●
 │ ●──●──●──●
 └────────────────────────→
    1 2 4 8 16 32 64
```

重点关注：

> P95/P99 TTFT 是否在某个并发点之后突然开始上升。

---

### 8.4 Preemption 曲线

```text
Preemption
 ↑
 │                    █████
 │                █████
 │
 │
 │__________________________
    1 2 4 8 16 32 64
```

如果：

- KV Cache Usage 接近 100%
- Preemption 开始增加
- Waiting Requests 增加
- P99 TTFT 快速上升
- Throughput 开始进入平台期

这些现象在相近的并发位置同时出现，那么就可以认为找到了一个非常明确的 KV Cache 压力拐点。

---

## 9. 不要把“吞吐不再增长”直接等同于 KV Cache 瓶颈

这是本实验中最重要的判断原则之一。

### 情况一：算力瓶颈

例如：

```text
GPU Utilization ≈ 100%
KV Cache Usage ≈ 50%
Preemption = 0
Waiting 很少
```

此时即使吞吐不再增长，也更可能说明：

> GPU Compute 已经成为主要瓶颈。

---

### 情况二：KV Cache / 调度瓶颈

例如：

```text
KV Cache Usage ≈ 100%
Preemption 明显增加
Waiting Requests 增加
Queue Time 增加
P99 TTFT 快速恶化
```

此时才更有证据说明：

> 系统进入了 KV Cache Pressure 区域。

因此最终判断必须基于多个指标联合分析，而不能只看 TPS。

推荐使用以下证据链：

```text
Throughput 增长停止
        +
KV Cache Usage 接近 100%
        +
Preemption 增加
        +
Waiting Requests 增加
        +
Queue Time / TTFT 恶化
        ↓
KV Cache / Scheduler 出现明显压力
```

---

## 10. 第一轮实验最终应该输出什么结论

不要只写：

> Concurrency=32 时 Output TPS 为 xxx。

更完整的结论应该类似：

> 在 8K Input + 512 Output 的 workload 下，随着 Concurrency 从 1 提升到 16，GPU batching 效率逐渐提高，系统吞吐持续增长，KV Cache 使用率稳步上升，但没有明显 Preemption 或 Waiting。
>
> 当 Concurrency 提升到 32 后，KV Cache 使用率接近饱和，Waiting Requests 和 Preemption 开始明显增加，同时 P95/P99 TTFT 快速恶化，而 Output Throughput 增长趋于平缓。
>
> 因此，在当前模型、硬件和 vLLM 配置下，8～16 可以视为该 workload 下较合理的稳定并发区间，32 附近开始进入明显的 KV Cache 压力区域。

这种结论才具有实际容量规划价值。

---

## 11. 第一轮实验结果记录模板

建议每组实验记录如下：

| Concurrency | KV Cache Avg | KV Cache Peak | GPU Util | Output TPS | TTFT P50 | TTFT P95 | TTFT P99 | TPOT P50 | TPOT P95 | Queue Time | Waiting Peak | Preemption |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |  |  |  |  |  |  |
| 16 |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 |  |  |  |  |  |  |  |  |  |  |  |  |
| 64 |  |  |  |  |  |  |  |  |  |  |  |  |

---

## 12. 第一轮做完后的第二轮交叉验证

第一轮固定：

```text
Context = 8K
Concurrency = 1 → 32/64
```

找到大致的稳定并发区和饱和区以后，再进行第二轮实验。

第二轮改变 Context Length：

```text
1K
4K
8K
16K
32K
```

每一个 Context Length 再分别测试不同 Concurrency。

理论上应该看到：

```text
Context Length ↑
        ↓
单个 Request 占用 KV Cache ↑
        ↓
可稳定承载的并发 ↓
```

最终可以得到一张非常有价值的容量规划表：

| Context Length | 稳定并发 | KV Cache 开始明显压力的并发 |
|---:|---:|---:|
| 1K |  |  |
| 4K |  |  |
| 8K |  |  |
| 16K |  |  |
| 32K |  |  |

这张表可以直接回答：

> 当前模型在不同上下文长度下，大约可以稳定承载多少并发请求。

---

## 13. 本实验的核心逻辑总结

整个第一组实验可以总结为：

```text
固定模型和所有推理配置
        ↓
固定 Input / Output Length
        ↓
逐步提高 Concurrency
        ↓
观察 KV Cache Usage
        ↓
观察 GPU Utilization
        ↓
观察 Waiting / Preemption / Queue
        ↓
观察 TTFT / TPOT / Throughput
        ↓
寻找性能 Knee Point
        ↓
判断是 Compute Bottleneck
还是 KV Cache / Scheduler Bottleneck
        ↓
得到当前 Workload 的稳定并发区间
```

第一组实验真正要回答的问题只有一个：

> **随着并发不断提升，KV Cache 是怎样从“充足”逐渐走向“饱和”的，以及这个过程中系统性能和调度行为发生了什么变化。**

只要围绕这个问题设计、采集和分析数据，这一组实验就是完整且有工程意义的。
