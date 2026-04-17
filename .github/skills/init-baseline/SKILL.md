---
name: init-baseline
description: "初始化优化环境：建立 baseline、分析 model.py、创建 OP 计划"
---

# 初始化 Skill

优化开始前的完整初始化流程。

## 输入参数

- `rounds`：用户指定的优化轮数（默认 10）

## 执行步骤

### 1. 读取环境信息
- 读取 `.github/agents/toolkit/system_info.md`，提取 GPU 型号、显存、带宽等
- 记录硬件参数供后续策略参考

### 2. 分析参考实现
- 读取 `model.py`
- 分析 `forward()` 中的完整计算流程：算子序列、输入输出 shape、dtype
- 识别所有 OP 并按 `toolkit/op-taxonomy.md` 分类

### 3. 建立等价 baseline
1. 复制 `model.py` → `model_new.py`，类名 `Model` → `ModelNew`
2. 确保 `output/` 目录存在
3. 运行 `bash run.sh`，确认输出 `✅ 正确性测试通过`
4. 保存快照：`cp model_new.py output/round_000.py`

### 4. 创建 OP 计划
- 基于分析结果生成 `output/op_plan.json`
- 按 `toolkit/fusion-patterns.md` 识别可融合组
- 所有目标初始状态为 `pending`

### 5. 写 baseline 记录
- 创建 `output/round_000_baseline.json`

### 6. 初始化状态变量
```
base_round = 0
best_round = 0
best_speedup = 1.00x
total_rounds = {rounds}
```

## 输出

初始化完成后返回：
- baseline 性能数据
- OP 计划摘要
- 初始状态变量
