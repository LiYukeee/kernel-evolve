---
name: optimize-decision
description: "分析 profiling 数据和性能指标，做出优化决策（目标选择、饱和检测、ceiling 检查）"
---

# 优化决策 Skill

基于性能指标和 profiling 数据，决定下一步优化方向。

## 输入参数

- `model_time`：原始模型耗时 (ms)
- `model_new_time`：当前优化模型耗时 (ms)
- `speedup`：当前加速比
- `best_speedup`：历史最佳加速比
- `base_speedup`：当前基准加速比
- `profiling_top`：Top-10 kernel 热点列表（Name, Self CUDA, Self CUDA %, CUDA total, # Calls）
- `op_plan`：当前 OP 计划状态（各目标的 status）
- `current_target`：当前正在优化的目标（可为空）
- `retry_count`：当前目标连续回滚次数
- `stagnant_count`：当前目标连续边际收益不足次数
- `prev_speedup`：上一轮 speedup（用于边际计算）

## 决策流程

### 1. Amdahl ceiling 检查
```
target_self_cuda_ms = 当前目标 kernel 的 Self CUDA time 总和
target_ceiling = model_time / (model_new_time - target_self_cuda_ms)
```
若 `target_ceiling - best_speedup < 0.03` → 标记 `done`，选新目标

### 2. 饱和检测
```
marginal_gain_rel = (speedup - prev_speedup) / prev_speedup
```
- `marginal_gain_rel < 0.01` → `stagnant_count += 1`
- `marginal_gain_rel >= 0.01` → `stagnant_count = 0`
- `stagnant_count >= 2` → 标记 `saturated`，选新目标

### 3. 热点漂移检查
若另一 pending 目标的 `Self CUDA %` 比当前目标高 > 3% → 暂停当前（保持 pending），切换

### 4. 目标选择优先级（从 profiling 热点中选取 pending 目标）
1. 可融合算子组（matmul+add+activation 的 epilogue 融合等）
2. memory-bound 独立 OP（残差加法、cat、shuffle、transpose）
3. 独立 activation / pooling / normalization
4. einsum / attention 后处理

### 5. 跳过规则
- trivial OP 永远跳过
- library_backed 默认低优先级（除非小维度/可融合 epilogue/layout 开销）
- 大尺寸标准 GEMM/Conv 不手写

## 输出

```
decision: continue / switch / done_all
next_target: (目标 ID 和描述)
strategy: (推荐优化策略)
updated_state: {stagnant_count, retry_count, op_status_changes}
```
