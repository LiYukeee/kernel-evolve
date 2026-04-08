---
filename: optimization-toolkit.md
last_updated: 2026-04-08
schema_version: 1.0
purpose: 优化工具包与实现提示
related_files:
  - kernel.agent.md
  - fusion-patterns.md
  - rules-and-heuristics.md
---

# 工具选择检查单

1. 先确定 `current_target` 类型。
2. `fusion_group`/`attention_kernel` 优先走 Toolkit A。
3. `memory_bound`/standalone `activation`/`pooling` 走 Toolkit B。
4. 本轮只改目标相关路径，其余保持不变。

# Toolkit A: 融合优先

## A1 conv/linear + norm + activation

- 目标：减少中间 tensor 写回。
- 常见技巧：epilogue 内联 affine 与激活，必要时合并 pooling。
- 风险：融合过深导致寄存器压力上升。

## A2 attention fusion

- 目标：避免显式存储大 attention 矩阵。
- 路径：Flash Attention 风格的分块与 online softmax。
- 风险：head_dim 与 layout 不匹配时实现复杂度高。

## A3 einsum/SSM fusion

- 目标：减少多次 global memory round-trip。
- 路径：融合 einsum 与 element-wise，scan 使用并行前缀和。
- 风险：一次融合范围过大易引发稳定性问题。

## A4 depthwise epilogue

- 目标：保留 backbone，优化后处理。
- 路径：向量化 epilogue（如 relu6 截断）。
- 风险：与量化/后续 layout 约束冲突。

# Toolkit B: standalone 优化

- 向量化访存：优先 `float4`，要求地址与长度满足对齐约束。
- warp reduction：用于 softmax/pooling 等局部归约。
- permutation 优化：ChannelShuffle 等重排场景优先确保 coalesced 访问。
- block size：优先尝试 128/256，结合 occupancy 调整。

# 通用实现注意

- 编译统一通过 `torch.utils.cpp_extension.load_inline`。
- 仅替换当前目标 OP，避免无关改动干扰评估。
- 关注数值一致性与边界条件，必要时先守正确性再追性能。

# CUDA Graph 阶段

触发条件：`op_plan` 无可选高价值 `pending` 目标。

步骤：
1. 先 warm-up，再捕获稳定前向图。
2. `replay` 前仅更新静态输入缓存。
3. 运行评估；若性能下降则回滚。

---

关联参考：
- 分类定义：`op-taxonomy.md`
- 模式识别：`fusion-patterns.md`
- 决策阈值：`rules-and-heuristics.md`
- JSON 记录：`json-schemas.md`
