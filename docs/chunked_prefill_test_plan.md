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
- 最终得到非空文本输出。

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
