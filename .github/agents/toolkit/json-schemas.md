---
filename: json-schemas.md
last_updated: 2026-04-08
schema_version: 1.0
purpose: output 目录 JSON 模板与字段约束
related_files:
  - kernel.agent.md
  - rules-and-heuristics.md
  - op-taxonomy.md
---

# 字段通用约束

- 所有轮次记录必须包含：`round`、`type`、`kernel_file`、`run_output`（Phase3 除外）。
- `kernel_file` 统一指向 `output/round_NNN.py` 或最终 `model_new.py`。
- 布尔字段统一使用 JSON 布尔值，不使用字符串。

# output/op_plan.json

```json
{
  "op_list": [
    {"id": "g1", "type": "fusion_group", "ops": ["conv1", "bn1", "relu"], "status": "pending", "retry_count": 0}
  ]
}
```

约束：
- `status` 仅允许 `pending`/`in_progress`/`done`/`skip`。
- `retry_count`：该目标被 rollback 的次数，初始为 0，达到 `max_retries`（默认 3）时标记为 `skip`。

# round_000_baseline.json

```json
{
  "round": 0,
  "type": "baseline",
  "kernel_file": "output/round_000.py",
  "note": "PyTorch functional baseline，无 CUDA kernel",
  "run_output": "..."
}
```

# round_NNN_repair.json

```json
{
  "round": 3,
  "type": "repair",
  "error_summary": "...",
  "fix_description": "...",
  "kernel_file": "output/round_003.py",
  "run_output": "..."
}
```

# round_NNN_opt.json

```json
{
  "round": 4,
  "type": "optimize",
  "target_op_id": "g2",
  "target_ops": ["conv2", "bn2"],
  "technique": "conv_bn_relu_fusion",
  "speedup_prev": "1.22x",
  "bottleneck": "cuda_time_total hotspot",
  "strategy": "...",
  "op_status_after": {"g2": "done", "m1": "pending"},
  "kernel_file": "output/round_004.py",
  "run_output": "...",
  "speedup_new": "1.38x",
  "rollback": false
}
```

# phase3_cuda_graph.json

```json
{
  "phase": 3,
  "technique": "cuda_graph",
  "speedup_before": "1.65x",
  "speedup_after": "1.71x",
  "rollback": false,
  "run_output": "..."
}
```

# summary.json

```json
{
  "total_rounds": 10,
  "best_speedup": "1.71x",
  "best_round": 8,
  "final_kernel_file": "model_new.py",
  "rounds": [
    {"round": 0, "type": "baseline", "speedup": "1.00x"},
    {"round": 8, "type": "optimize", "speedup": "1.71x", "rollback": false}
  ]
}
```

---

关联参考：
- 分类定义：`op-taxonomy.md`
- 决策规则：`rules-and-heuristics.md`
- 主流程：`kernel.agent.md`
