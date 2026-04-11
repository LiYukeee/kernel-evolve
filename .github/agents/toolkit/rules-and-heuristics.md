---
filename: rules-and-heuristics.md
last_updated: 2026-04-08
schema_version: 1.0
purpose: 决策规则、优先级与回滚
related_files:
  - kernel.agent.md
  - op-taxonomy.md
  - optimization-toolkit.md
---

# OP 选择优先级

从 `output/op_plan.json` 的 `pending` 中选择 `current_target`：

1. `fusion_group` 或 `attention_kernel`：同优先级，优先 `ops` 数量更大的目标。
2. `memory_bound`：无 1 类目标时，结合 profiler 选 `cuda_time_total` 最大者。
3. `activation` 或 `pooling`：无 1/2 类目标时选择热点。
4. `library_backed`、standalone `normalizer`、`trivial`：不选。

无可选目标时进入 Phase 3（CUDA Graph）。

# profiling 解读要点

- 必看字段：`cuda_time_total`、调用次数、热点函数名。
- 选择目标以 GPU 热点优先，不以 CPU 时间为主。
- 若多个候选接近，优先可融合和可复用代码路径。

# base/best 更新规则

- `best_kernel`：只要当前 speedup 超过历史最高，立即更新。
- `base_kernel`：满足任一条件才更新：
  - 相比上一 base 的倍率提升 >= 1.1
  - 绝对 speedup 提升 >= 0.1

# rollback 规则

以下任意条件触发回滚到 `round_BASE.py`：

1. 正确性失败。
2. 编译或运行崩溃。
3. 新 speedup 低于当前 base speedup。

回滚后在本轮记录中显式标注 `rollback: true`。

**重要：rollback 不等于放弃目标**。回滚后该目标保持 `pending` 状态，`retry_count` +1。仅当 `retry_count >= max_retries`（3）时才标记为 `skip`。

# Amdahl ceiling 计算

在每轮分析阶段，对当前目标计算其理论最大加速比天花板（假设它被完全消除）：

```
target_self_cuda_ms = 当前目标 kernel 的 Self CUDA time（从 profiling top-10 获取）
target_ceiling = model_time / (model_new_time - target_self_cuda_ms)
```

若 `target_ceiling - best_speedup < 0.03`，则即使极限优化也只能再提升 <3%，标记目标为 `done`，切换下一目标。

> **注意**：`target_self_cuda_ms` 应使用 `Self CUDA` 总和（= 单次耗时 × 调用次数），不是单次耗时。

# 饱和检测

对同一目标的连续多轮优化，跟踪 **边际收益**：

```
marginal_gain_rel = (new_speedup - prev_speedup) / prev_speedup
```

- 若某轮在同一目标上 `marginal_gain_rel < 0.01`（总 speedup 提升 < 1%）：`stagnant_count += 1`
- 若 `marginal_gain_rel >= 0.01`：`stagnant_count = 0`（有效提升重置计数）
- 若 `stagnant_count >= 2`：目标标记为 `saturated`（等同 `done`），**必须**切换到下一个热点目标

rollback 轮不计入 `stagnant_count`，但也不重置它。

# 热点漂移检查

每轮 profiling 后，比较当前目标与其他 pending 目标的 `Self CUDA %`：

- 若存在另一 pending 目标的 `Self CUDA %` **显著高于**当前目标（差值 > 3%），应暂停当前目标（保持 `pending`），切换到更高价值目标
- 这确保每轮的努力都花在当前最大的瓶颈上

# 重试与目标切换

## rollback 后重试

当某目标被 rollback 但仍有重试次数时：

1. 分析上一次失败的根因（launch overhead、非 coalesced 访存、寄存器压力等）。
2. 采用明确不同的策略重试，不得重复相同方案。
3. 在 JSON 记录中注明上次失败原因和本次策略差异。
4. 参考 `toolkit/optimization-toolkit.md#迭代精炼策略` 选择改进方向。

## 成功后的目标切换决策

当某目标的 CUDA kernel 已成功获得加速时，**不得无限期继续优化**。必须按以下优先级检查：

1. **Amdahl ceiling 检查**：若触发 → 标记 `done`，切换
2. **饱和检测**：若触发 → 标记 `saturated`，切换
3. **热点漂移检查**：若触发 → 暂停（保持 `pending`），切换
4. 以上均未触发：可继续深度优化

# 早停建议

- 仅当所有 `pending` 目标均已变为 `done` 或 `skip` 时，才进入 Phase 3。
- 不得因连续 rollback 而提前终止探索，应换策略重试或切换目标。
- **不得因为手写 CUDA kernel 困难而转向 `torch.compile` 等编译器捷径**。

---

关联参考：
- 分类来源：`op-taxonomy.md`
- 工具执行：`optimization-toolkit.md`
- JSON 记录：`json-schemas.md`
