---
name: update-state
description: "更新 base/best 状态变量，维护优化过程的全局状态"
---

# 状态更新 Skill

每轮评估后更新全局状态变量。

## 输入参数

- `current_round`：当前轮次号
- `new_speedup`：本轮加速比
- `base_round`：当前基准轮次
- `base_speedup`：当前基准加速比
- `best_round`：历史最佳轮次
- `best_speedup`：历史最佳加速比
- `rollback`：本轮是否回滚

## 更新规则

### best 更新（只要超过历史最高即更新）
```
if new_speedup > best_speedup:
    best_round = current_round
    best_speedup = new_speedup
```

### base 更新（满足任一条件）
```
ratio_improvement = new_speedup / base_speedup
abs_improvement = new_speedup - base_speedup

if ratio_improvement >= 1.1 or abs_improvement >= 0.1:
    base_round = current_round
    base_speedup = new_speedup
```

### rollback 时不更新 base/best
回滚轮次不改变 base 和 best 状态。

## 输出

返回更新后的状态：
```
base_round: N
base_speedup: x.xx
best_round: M
best_speedup: x.xx
```
