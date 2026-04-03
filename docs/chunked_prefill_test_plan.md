# Chunked Prefill + Continuous Batching 测试执行说明

本文档说明在可运行环境中如何验证当前实现，以及每条命令“预期看到什么输出”才算通过。

## 1. 快速单元测试（无需 GPU）

```bash
pytest -q tests/test_scheduler_chunked.py
```

### 预期通过输出

- 结尾出现：
  - `4 passed`（当前仓库基线）
  - 总耗时，如 `in 0.0Xs`

示例：

```text
....                                                                     [100%]
4 passed in 0.03s
```

### 不符合预期的输出

- `FAILED tests/test_scheduler_chunked.py::...`
- `ERROR collecting ...`
- `ModuleNotFoundError`（表示依赖环境不完整或 PYTHONPATH 不对）

## 2. 静态可导入检查

```bash
python -m compileall nanovllm tests
```

### 预期通过输出

- 能看到 `Listing ...` / `Compiling ...`，并且进程退出码为 0。

### 不符合预期的输出

- `SyntaxError`
- `IndentationError`
- 命令退出码非 0

## 3. 代码风格/补丁完整性检查

```bash
git diff --check
```

### 预期通过输出

- 无输出，且退出码为 0。

### 不符合预期的输出

- 出现 trailing whitespace、冲突标记等 diff 问题。

## 4. GPU 端到端功能验证（需要可用模型与 CUDA）

> 该步骤依赖本地模型权重、PyTorch CUDA、flash-attn 等运行时。

```bash
python example.py
```

### 预期行为

- 能正常初始化引擎并完成生成；
- 进度条中 `Prefill` / `Decode` 两个吞吐字段持续刷新，并出现 `Mixed=x/y`；
- 程序末尾打印 `Scheduling Stats`，其中 `Mixed steps (prefill+decode)` 大于 0（通常不是 0）；
- 程序末尾打印 `Total inference time: ...s`，用于不同参数组合下的对比；
- 最终得到非空文本输出。

> 若 `Mixed steps` 为 0，通常不是功能错误，而是 workload 太短或 token budget 太宽松，导致 prefill 在很少 step 内一次跑完。可尝试：
>
> - 降低 `max_num_batched_tokens`（如 128）
> - 降低 `max_prefill_chunk_size`（如 32）
> - 增加一个超长 prompt 与多个短 prompt 混合输入

### 不符合预期行为

- CUDA / NCCL 初始化失败；
- flash-attn 导入或运行失败；
- 生成中断、死循环、或返回空结果（在输入非空情况下）。

## 5. 当前测试覆盖了什么

- 同 step 的 decode + prefill chunk 混合调度；
- prefill 已完成请求可同 step 直接 decode；
- block 边界处 KV block append 行为；
- token budget 上限与同 step 不重复调度同一序列。

## 6. 还没覆盖什么（建议补充）

- 真正 GPU 上 `run_mixed` 数值对齐（与 Phase A/原 decode 路径对比）；
- prefix cache 命中 + chunked prefill 的端到端正确性；
- preempt/recompute 场景下长序列稳定性；
- 大批量并发下吞吐与延迟回归基线。

## 7. 参数含义与约束说明

- `max_model_len`：单条序列允许的最大长度上限（prompt + completion），初始化时会再被模型的 `max_position_embeddings` 截断。
- `max_num_batched_tokens`：**每个调度 step 的 query token 预算**（prefill chunk token + decode token 总和），不是单条请求长度上限。

为什么现在不再强制 `max_num_batched_tokens >= max_model_len`：

- 在 chunked prefill 下，长序列本来就是分多 step 推进；允许较小的 `max_num_batched_tokens` 可以更容易形成 prefill/decode 交叉，也利于显存受限场景。

潜在隐患（需要认知）：

- 预算过小会让 prefill 拆得很碎，step 数增加，调度/launch 开销变大，吞吐可能下降；
- 若远小于常见 prompt 长度，TTFT 可能上升（首 token 要等更多 prefill step）；
- 预算配置不当时，吞吐与延迟会出现明显 trade-off（建议结合业务流量做压测）。

## 8. `Mixed=x/y` 指标如何解读

- `y`：当前已经执行的调度 step 总数（`num_steps`）。
- `x`：其中“同时存在 prefill 与 decode”的 step 数（`num_mixed_steps`）。
- 即：`Mixed=x/y` 表示“截至当前，混合 step 占比为 `x / y`”。

在真实运行中，常见轨迹是：

1. **开头 `0/1`**：第一步通常只有 prefill（还没有可 decode 的序列），所以混合步为 0。
2. **中段 `x` 和 `y` 一起增长**：有的序列已进入 decode，而长序列仍在 prefill，出现 prefill+decode 并行混合。
3. **后段 `x` 停住、`y` 继续涨**：prefill 已基本结束，只剩 decode 尾巴，因此后续 step 不再计入混合步。

所以像 `0/1 -> 105/106 -> 105/386 -> 105/490` 这种模式是符合预期的，表示：

- 早期进入了较长一段混合阶段；
- 后期进入纯 decode 收尾阶段（`x` 不再增长）。

## 9. 如何做“实现前后”速度对比

仓库提供两个可直接运行的脚本，prompt 与 sampling 参数保持一致：

- `python example.py`：chunked + continuous batching 配置（`max_prefill_chunk_size=32`, `max_num_batched_tokens=128`）
- `python example_baseline_prefill_first.py`：近似 prefill-first 基线配置（`max_prefill_chunk_size=4096`, `max_num_batched_tokens=4096`）

说明：`baseline_prefill_first_like` 是“行为近似”，不是“代码完全回到改造前版本”。
它的含义是把 chunk 与预算放大，让调度在大多数 step 上更接近“先 prefill、后 decode”的形态；
但底层仍运行当前代码（包含 mixed runner、统计逻辑等）。

建议对比以下字段：

- `Total inference time`
- `Mixed ratio`
- `Prefill tokens scheduled`
- `Decode tokens scheduled`

## 10. 如果“实现后更慢”是否正常

结论：**有可能正常**，尤其在离线小批量场景。

常见原因：

- chunked prefill 把一次大 prefill 拆成了更多 step，会增加调度与 kernel launch 次数；
- continuous batching 的价值更偏在线场景（降低长尾/提升并发体验），不一定在单次离线吞吐上占优；
- 如果 `max_prefill_chunk_size` 太小、`max_num_batched_tokens` 太小，会进一步增加 step 数与管理开销。

建议做“公平对比”时同时观察：

- `Total inference time`
- `Mixed ratio`
- `Total steps`
- `Prefill/Decode tokens scheduled`

若你看到“实现后 60s，实现前 32s”，通常说明当前参数更偏 latency fairness 而非纯吞吐。可尝试逐步调大：

- `max_prefill_chunk_size`（例如 32 -> 64 -> 128 -> 256）
- `max_num_batched_tokens`（例如 128 -> 256 -> 512 -> 1024）

再重新比较总耗时与 mixed ratio，找到你业务更合适的折中点。

## 11. 不调参也能量化“是否有用”的基准测试

直接运行：

```bash
python benchmark_chunked_effectiveness.py
```

> 脚本会自动以“两个子进程”分别运行 baseline 与 optimized，避免 PyTorch 分布式默认进程组重复初始化报错。

该脚本固定两组配置（baseline-like 与 chunked+continuous），在同一组长短混合请求上输出：

- `total_time`（总耗时）
- `decode_throughput`（解码吞吐）
- `ttft_p50 / ttft_p95`（首 token 延迟）
- `finish_p95`（请求完成尾延迟）
- `finish_short_mean / finish_long_mean`（短请求与长请求完成时间）
- `mixed_ratio`（混合步占比）

如果优化“有用”，常见表现是：

- `mixed_ratio` 明显上升；
- `ttft_p95`、`finish_short_mean` 或 `finish_p95` 至少有一项改善；
- 即便总吞吐不升，也能看到公平性/尾延迟指标改善。
