---
name: run-eval
description: "运行 run.sh 进行评估，并解析输出结果"
---

# 运行评估 Skill

根据指定模式运行 `bash run.sh` 并解析输出。

## 输入参数

- `mode`：运行模式，可选 `full`（默认）、`quick`、`correctness`

## 执行步骤

1. 在终端执行 `bash run.sh {mode}`
2. 等待命令完成，捕获完整输出
3. 解析输出，按以下规则判定结果：

### 结果判定

| 输出特征 | 判定 |
|---|---|
| 包含 `Error` / `Traceback` / `❌` / 非 0 退出码 | `status: FAIL`，提取错误摘要 |
| 包含 `✅ 正确性测试通过` | `status: PASS` |

### 性能数据提取（`quick` / `full` 模式）

从 `FINAL_SPEED_RESULT` 行提取：
- `model_time`：原始模型耗时 (ms)
- `model_new_time`：优化模型耗时 (ms)
- `speedup`：加速比 (`model_time / model_new_time`)

### Profiling 数据（仅 `full` 模式）

- 运行后读取 `output/profile_latest.txt`
- 提取 Top-10 kernel 的：`Name`、`Self CUDA`、`Self CUDA %`、`CUDA total`、`# of Calls`

## 输出格式

返回结构化结果：
```
status: PASS/FAIL
error_summary: (仅 FAIL 时)
model_time: x.xxx ms
model_new_time: x.xxx ms
speedup: x.xxx
profiling_top: (仅 full 模式，kernel 热点列表)
```
