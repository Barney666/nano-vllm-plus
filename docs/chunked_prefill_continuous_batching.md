# Chunked Prefill + Continuous Batching 设计与落地指南（基于 nano-vllm 当前实现）

> 目标：在不破坏当前 prefix cache / paged KV cache 逻辑的前提下，把“prefill 全量一次性跑完”改成“可分块 prefill”，并且允许 prefill 与 decode 在同一个 step 内共同组成 batch（continuous batching）。

## 1. 当前实现的调度特点（现状）

当前 `Scheduler.schedule()` 采用二选一模型：

1. 若 `waiting` 可调度，则仅做 prefill（并返回 `is_prefill=True`）。
2. 否则才做 decode（返回 `is_prefill=False`）。

这意味着：

- prefill 与 decode 互斥执行，不能混合；
- 长 prompt 会占满一个 step 的 token budget，decode 延迟抖动更大；
- `ModelRunner.run` 也只支持纯 prefill 或纯 decode 两种模式。

## 2. 目标行为定义

### 2.1 Chunked Prefill

对每条请求维护“prefill 进度指针”，每次最多推进 `chunk_size` 个未缓存 prompt token，而不是一次把剩余 prompt token 全跑完。

### 2.2 Continuous Batching

一个调度 step 内同时允许：

- 一部分 running 序列执行 decode（每条 1 token query）；
- 一部分 waiting/running 序列执行 prefill chunk（每条若干 token query）。

共同受 `max_num_batched_tokens` 和 `max_num_seqs` 约束。

## 3. 数据结构改造建议

### 3.1 Sequence 扩展字段

建议在 `Sequence` 增加：

- `num_prefilled_tokens`: 已经完成 prefill 的 prompt token 数；
- `stage`: `PREFILLING | DECODING | FINISHED`（或通过现有 status + 指针推导）；
- `next_prefill_end(chunk_size)`: 计算本轮 chunk 终点。

注意：当前已有 `num_cached_tokens`（用于 prefix cache 命中），不要混淆：

- `num_cached_tokens` = 通过缓存“无需算”的前缀；
- `num_prefilled_tokens` = 已经真实跑过（或等价已覆盖）的 prompt 进度。

初始化时可设置：

- `num_prefilled_tokens = num_cached_tokens`（如果有 prefix 命中，prefill 起点直接跳过）。

### 3.2 新增调度输出结构

将 `schedule()` 返回值从 `(seqs, is_prefill)` 改为“混合批次计划”：

- `decode_seqs: list[Sequence]`
- `prefill_seqs: list[Sequence]`
- `prefill_chunk_lens: list[int]`（与 prefill_seqs 对齐）

这样 `ModelRunner` 可以一次前向里同时处理 decode + prefill。

## 4. Scheduler 设计（核心）

建议把 `schedule()` 拆成三个阶段：

1. **先放 decode**：优先保障已有 running 请求的 TTFT/TBT 稳定；
2. **再放 prefill chunk**：在剩余 token budget 填充 waiting 请求；
3. **内存校验与抢占**：沿用当前 block manager 逻辑，必要时 preempt decode 尾部请求。

### 4.1 预算模型

- `decode` 每条固定消耗 1 query token；
- `prefill chunk` 每条消耗 `chunk_len` query token；
- 一个 step 的总 query token 不超过 `max_num_batched_tokens`。

> 推荐先给 decode 预留最小配额（如 `decode_reserve = min(len(running), max_num_seqs//2)`），再把余量给 prefill。

### 4.2 chunk 切分策略

每条 prefill 请求本轮 chunk：

- `remaining = num_prompt_tokens - num_prefilled_tokens`
- `chunk_len = min(remaining, max_prefill_chunk_size, token_budget_left)`

当 `chunk_len == remaining` 时，本条请求 prefill 完成，下一轮即可进入 decode。

### 4.3 抢占策略

建议保留当前“无法 append 时抢占”思路，但细化优先级：

1. 先抢占最近加入且 decode 价值较低的序列（LIFO）；
2. 尽量避免抢占 prefill 已接近完成的序列；
3. 被抢占序列回到 waiting 时保留 `num_prefilled_tokens`，避免重复算。

## 5. ModelRunner 改造（一次前向混合）

## 5.1 新接口

新增例如：

```python
run_mixed(decode_seqs, prefill_seqs, prefill_chunk_lens) -> token_ids_for_decode
```

语义：

- decode 分支：和现在一致，每条产生 1 个 next token（用于采样）；
- prefill 分支：只写 KV cache，不参与采样输出（最后一个 token 的 logits 可忽略）。

### 5.2 输入拼接

构造统一的 `input_ids/positions/slot_mapping`：

- 前半段放 prefill chunk token（varlen）；
- 后半段放 decode token（每条 1 token）。

同时维护两套元信息：

- prefill 需要 `cu_seqlens_q/cu_seqlens_k`；
- decode 需要 `context_lens` + `block_tables`。

> 实现上可采用“两次 attention 调用 + 一次外层前向”或“按模式拆 batch 后分别前向”；nano-vllm 代码简洁优先，建议先走“分别前向（prefill 一次 + decode 一次）但在同一调度 step 完成”。先正确，再合并 kernel。

### 5.3 渐进式落地建议

为降低改造风险，可分两期：

- **Phase A（推荐先做）**：调度层 continuous batching，执行层仍分开 `run_prefill_chunk` 与 `run_decode` 两次调用；
- **Phase B**：把两次调用融合为一次 `run_mixed`，减少 launch 开销。

## 6. Attention/Context 层改造点

当前 `context` 是“单模式”（`is_prefill` 布尔）。混合批次要么：

1. 一次 step 分两次 run（A 方案）→ 不改 context 结构；
2. 真正单次 run_mixed（B 方案）→ context 需要表达“batch 内不同 token 属于哪种模式”。

若做 B，建议：

- `token_mode` 张量（0=prefill,1=decode）；
- 分段元信息（prefill 段 cu_seqlens、decode 段 context_lens/block_tables）；
- attention 内按段调用对应 flash-attn API。

## 7. 与 BlockManager 的一致性要求

关键不变量：

1. prefill chunk 只写“新覆盖的槽位”，不能覆写已完成块；
2. `num_prefilled_tokens` 推进时，同步保证 block_table 已分配到位；
3. 从 prefill 切到 decode 前，最后一个 token 必须已经进入 cache。

建议在调试阶段加入断言：

- `num_cached_tokens <= num_prefilled_tokens <= num_prompt_tokens`
- `len(block_table) == num_blocks`（已分配状态下）
- decode 前 `num_prefilled_tokens == num_prompt_tokens`

## 8. 关键伪代码

```python
def schedule():
    budget = max_num_batched_tokens
    decode_seqs, prefill_seqs, chunk_lens = [], [], []

    # 1) decode first
    while running and len(all_seqs) < max_num_seqs and budget >= 1:
        seq = pick_decode_candidate()
        if can_decode(seq):
            decode_seqs.append(seq)
            budget -= 1

    # 2) prefill chunks
    while waiting and len(all_seqs) < max_num_seqs and budget > 0:
        seq = waiting[0]
        ensure_allocated(seq)
        chunk = min(seq.remaining_prompt(), max_prefill_chunk_size, budget)
        if chunk == 0:
            break
        prefill_seqs.append(seq)
        chunk_lens.append(chunk)
        budget -= chunk
        move_to_running_if_needed(seq)

    return decode_seqs, prefill_seqs, chunk_lens
```

## 9. 参数建议（首版默认值）

可在 `Config` 增加：

- `enable_chunked_prefill: bool = True`
- `max_prefill_chunk_size: int = 512`（0.6B/8GB 可从 256~1024 试）
- `decode_first_ratio: float = 0.5`（控制 decode 预留）

调优建议：

- 延迟优先：减小 chunk（128~256）；
- 吞吐优先：增大 chunk（512~2048）；
- 若 prompt 很短，chunked prefill 影响可忽略。

## 10. 验证与基准

### 10.1 正确性

- 与未分块版本对比：相同随机种子下输出一致（允许浮点微小差异）；
- 前缀缓存命中场景：验证 `num_cached_tokens` 生效，prefill 起点正确；
- 抢占恢复：被抢占请求恢复后不会重复 prefill 已完成部分。

### 10.2 性能指标

- TTFT（first token latency）
- ITL/TBT（token 间隔）
- 总吞吐（tok/s）
- GPU occupancy / kernel launch 次数

> 经验上，连续批处理通常显著改善在线场景尾延迟；chunked prefill 主要缓解“超长 prompt 阻塞 decode”。

## 11. 推荐实施顺序（最稳妥）

1. 在 `Sequence` 增加 prefill 进度字段与断言。
2. Scheduler 支持 prefill chunk（但仍与 decode 互斥），先保证 chunk 正确。
3. Scheduler 支持同 step 内 decode + prefill（Phase A：两次 run）。
4. 性能稳定后再做 `run_mixed`（Phase B：一次 run）。

这个顺序可以最大限度复用当前实现，减少一次性重构风险。
