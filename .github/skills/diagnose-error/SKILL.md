---
name: diagnose-error
description: "诊断编译/运行时/数值错误，生成修复方案"
---

# 错误诊断 Skill

当 `bash run.sh` 输出包含错误时，诊断错误类型并指导修复。

## 输入参数

- `error_output`：完整的错误输出文本

## 诊断流程

### 1. 错误分类

| 错误类型 | 特征 | 检查重点 |
|---|---|---|
| **编译错误** | `error:` 关键字、`nvcc` 错误 | CUDA 语法、头文件缺失、函数签名不匹配 |
| **链接错误** | `undefined reference`、`unresolved symbol` | 函数声明/定义不一致、缺少 extern "C" |
| **运行时崩溃** | `CUDA error`、`Segmentation fault`、`illegal memory access` | tensor shape/dtype 不匹配、越界访问、launch 配置错误 |
| **编译超时** | `编译超时` | kernel 过于复杂、模板展开过多 |
| **数值错误** | `❌ 正确性测试失败`、`最大偏差` | float 累加顺序、边界条件、reduction 正确性 |

### 2. 修复方向

- **编译/链接**：检查 CUDA 代码语法，确认所有使用的 CUDA API 函数签名
- **运行时**：打印中间 tensor shape/dtype 验证，检查 grid/block 配置是否覆盖所有元素
- **数值**：检查 reduction 的原子操作或顺序依赖，考虑使用 Kahan summation 或 double 精度中间累加
- **超时**：简化 kernel 结构，减少模板实例化

## 输出

```
error_type: compile / link / runtime / timeout / numerical
error_summary: (错误的一句话摘要)
root_cause: (根因分析)
fix_suggestion: (具体修复建议)
```
