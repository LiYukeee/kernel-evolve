---
name: save-round
description: "保存当前轮次快照并写入轮次 JSON 记录"
---

# 保存轮次记录 Skill

每轮优化结束后，保存代码快照和元数据记录。

## 输入参数

- `round_num`：当前轮次号（如 `1`、`2`，补零为 `001`、`002`）
- `record_type`：记录类型，可选 `baseline`、`repair`、`optimize`
- `record_data`：JSON 记录内容（字段详见 `toolkit/json-schemas.md`）
- `run_output`：（可选）`bash run.sh` 的完整输出，用于回填 `run_output` 字段

## 执行步骤

1. **确保目录**：确保 `output/` 目录存在
2. **保存代码快照**：`cp model_new.py output/round_{NNN}.py`（NNN 为三位补零轮次号）
3. **写 JSON 记录**：创建 `output/round_{NNN}_{record_type}.json`，内容为 `record_data`
   - 必须包含字段：`round`、`type`、`kernel_file`
   - 若提供 `run_output`，回填到 JSON 的 `run_output` 字段
4. **可选保存 profiling**：若为 `optimize` 类型且存在 `output/profile_latest.txt`，可复制为 `output/round_{NNN}_profile.txt`

## JSON 模板参考

- baseline：`{"round": 0, "type": "baseline", "kernel_file": "output/round_000.py", "run_output": "..."}`
- repair：`{"round": N, "type": "repair", "error_summary": "...", "fix_description": "...", "kernel_file": "...", "run_output": "..."}`
- optimize：`{"round": N, "type": "optimize", "target_op_id": "...", "technique": "...", "speedup_new": "...", "rollback": false, "kernel_file": "...", "run_output": "..."}`
