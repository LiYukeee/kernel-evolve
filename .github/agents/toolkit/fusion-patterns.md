---
filename: fusion-patterns.md
last_updated: 2026-04-08
schema_version: 1.0
purpose: 融合模式识别与降级策略
related_files:
  - kernel.agent.md
  - op-taxonomy.md
  - optimization-toolkit.md
---

# 模式与降级

## 模式 1: conv/linear 链

- 典型序列：`conv -> bn -> relu`、`conv -> bn -> relu -> pool`、`linear -> gelu`。
- 识别要点：相邻算子间数据依赖无分叉，且中间 tensor 只被后继消费。
- 降级策略：若存在分叉或 shape 动态变化，退化为子序列融合或 standalone。

## 模式 2: attention 链

- 典型序列：`q@k.T -> scale -> mask -> softmax -> @v`。
- 识别要点：同一 head 维度与时序维度连续，mask/scale 语义明确。
- 降级策略：若 layout 或 dtype 不满足，先保持 PyTorch 路径，记录待处理。

## 模式 3: einsum/SSM 链

- 典型序列：多个 `einsum` + element-wise + `cumsum/scan`。
- 识别要点：可闭合为单一块状计算，且中间结果仅局部复用。
- 降级策略：仅融合前半段或后半段，避免一次合并过大导致回归。

## 模式 4: depthwise epilogue

- 典型序列：`depthwise_conv(backbone) -> bn -> relu6`。
- 识别要点：depthwise 主干保持 library kernel，仅融合后处理 epilogue。
- 降级策略：若 relu6 与量化路径冲突，改为 standalone epilogue kernel。

# 融合识别检查单

1. 先构建调用顺序，再做相邻性判断。
2. 每个候选组检查依赖分叉与旁路写回。
3. 对 residual add 单独归类 `memory_bound`，默认不并入 conv 主链。
4. 识别失败时必须落一个明确降级路径，不可空置。
5. 结果同步回 `output/op_plan.json`。

# 反例

- shortcut 来源不同的 `out + identity` 不直接并入 conv 核心计算。
- 包含随机行为的算子链不做激进融合。
- 需要显式中间输出供后续多处消费的链路不做整段融合。

---

关联参考：
- 分类定义：`op-taxonomy.md`
- 具体实现：`optimization-toolkit.md`
- 优先级规则：`rules-and-heuristics.md`
