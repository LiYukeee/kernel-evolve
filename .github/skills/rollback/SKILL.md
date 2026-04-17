---
name: rollback
description: "回滚 model_new.py 到指定基准轮次的快照"
---

# 回滚 Skill

当性能回退或验证失败时，将 `model_new.py` 回滚到安全基准。

## 输入参数

- `base_round`：要回滚到的基准轮次号
- `current_round`：当前轮次号（用于在 JSON 记录中标注 rollback）

## 执行步骤

1. 将基准代码恢复：`cp output/round_{base_round_NNN}.py model_new.py`
2. 在当前轮次的 JSON 记录中标注 `"rollback": true`
3. 返回回滚结果

## 回滚触发条件（任一）

- 正确性测试失败
- 编译或运行时崩溃
- 新 speedup < 当前 base speedup（性能回退）

## 注意事项

- 回滚**不等于放弃目标**：目标保持 `pending` 状态
- `retry_count += 1`
- 若 `retry_count >= 3`：标记该目标为 `skip`，下一轮选新目标
