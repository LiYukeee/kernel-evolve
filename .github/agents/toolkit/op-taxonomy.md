---
filename: op-taxonomy.md
last_updated: 2026-04-08
schema_version: 1.0
purpose: OP 分类与处理策略
related_files:
  - kernel.agent.md
  - fusion-patterns.md
  - rules-and-heuristics.md
---

# 分类总表

| type | 典型算子 | 默认策略 | 备注 |
|---|---|---|---|
| `library_backed` | `nn.Linear` `nn.Conv2d` `nn.LSTM` `nn.GRU` `nn.MultiheadAttention` | standalone 跳过，仅作为融合 backbone | 依赖 cuBLAS/cuDNN |
| `normalizer` | `BatchNorm` `LayerNorm` `GroupNorm` | standalone 跳过，优先并入融合 epilogue | 推理阶段可参数折叠 |
| `activation` | `ReLU` `ReLU6` `GELU` `Sigmoid` `Tanh` `Softmax` `SiLU` | 优先融合；不可融合时 standalone 优化 | 常见 memory-bound |
| `pooling` | `MaxPool2d` `AvgPool2d` `AdaptiveAvgPool2d` | 优先融合；不可融合时 standalone 优化 | 可做 warp reduction |
| `memory_bound` | 残差加法 `torch.cat` `ChannelShuffle` `unfold` `rearrange` | 可单独替换 | 对带宽敏感 |
| `attention_kernel` | `q@k.T` `softmax` `att@v` `scaled_dot_product_attention` `einsum` 链 | 优先整体 fusion_group 化 | 常见高收益目标 |
| `trivial` | `Dropout(p=0)` `Identity` `view` `reshape` `contiguous` | 永远跳过 | no-op 或近似 no-op |
| `fusion_group` | 可融合序列 | 最高优先级 | 先做组，再做点 |

# 初始化分类检查单

1. 扫描范围覆盖 `forward()` 与所有子模块，不只顶层。
2. 先标记 `trivial` 与 `library_backed`，避免误选。
3. 对 `activation`/`pooling` 先尝试归并到前置 backbone。
4. 对 attention/einsum 序列优先聚合为 `fusion_group`。
5. 写入 `output/op_plan.json` 时保证每个条目有 `id` `type` `ops` `status`。

# 跳过规则

- `library_backed` standalone **默认低优先级**，但以下情况可作为目标：极小维度导致 launch overhead 占比高、推理期可做参数折叠（如 Conv+BN）、存在可融合 epilogue（如 BN+activation 内联）、profiling 显示 layout 转换开销显著。
- `normalizer` standalone 不优先优化，但推理期应优先考虑参数折叠（如 BN 折叠进 Conv），折叠后可完全消除该 OP。
- `trivial` 不参与任何优化轮次。
- **降级策略**：当某个目标的 CUDA kernel 多次失败时，降级为 `skip` 并转向下一个目标，**不得降级为 `torch.compile` 等编译器方案**。

# 推理模式注意事项

- BN 可折叠为 affine：`scale = gamma / sqrt(var + eps)`，`bias = beta - mean * scale`。
- `Dropout(p=0)` 按 identity 处理，不应引入额外 kernel。
- 仅在语义等价前提下做分类合并。

---

关联参考：
- 识别模式：`fusion-patterns.md`
- 决策优先级：`rules-and-heuristics.md`
- 优化执行：`optimization-toolkit.md`
