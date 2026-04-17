---
name: generate-summary
description: "生成最终优化总结报告 summary.json"
---

# 生成总结 Skill

在所有轮次完成后，生成最终总结报告。

## 输入参数

- `total_rounds`：总轮数
- `best_round`：最佳轮次
- `best_speedup`：最佳加速比
- `rounds_history`：各轮记录摘要列表

## 执行步骤

1. 将最佳轮次的代码回写到 `model_new.py`：`cp output/round_{best_round_NNN}.py model_new.py`
2. 运行 `bash run.sh` 做最终确认
3. 生成 `output/summary.json`：

```json
{
  "total_rounds": N,
  "best_speedup": "x.xxx",
  "best_round": K,
  "final_kernel_file": "model_new.py",
  "rounds": [
    {"round": 0, "type": "baseline", "speedup": "1.00x"},
    {"round": 1, "type": "optimize", "target": "...", "speedup": "...", "rollback": false},
    ...
  ]
}
```

## 注意事项

- 回写前确认最佳轮次的快照文件存在
- 最终确认必须通过正确性测试
- 若最终确认失败，回退到上一个已知通过的快照
