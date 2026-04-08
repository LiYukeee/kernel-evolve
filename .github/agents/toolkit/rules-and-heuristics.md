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
  - 相比上一 base 的倍率提升 >= 1.3
  - 绝对 speedup 提升 >= 0.3

# rollback 规则

以下任意条件触发回滚到 `round_BASE.py`：

1. 正确性失败。
2. 编译或运行崩溃。
3. 新 speedup 低于当前 base speedup。

回滚后在本轮记录中显式标注 `rollback: true`。

# 早停建议

- 连续多轮仅微小收益可提前终止探索。
- 当剩余目标都处于低收益类别时优先转入 Phase 3。

---

关联参考：
- 分类来源：`op-taxonomy.md`
- 工具执行：`optimization-toolkit.md`
- JSON 记录：`json-schemas.md`
