# CUDA Kernel 自动优化 Agent

本项目是一个运行在 **GitHub Copilot** 之上的自治 Agent，能够把一份普通的 PyTorch
参考实现（`model.py`）逐步改写成由**手写 CUDA kernel** 加速的优化版本
（`model_new.py`），整个过程闭环自动迭代，直到达到目标加速比或用满指定的优化轮数。

在 VS Code Copilot Chat 中选择 `kernel` agent，输入一句话即可启动，例如：

> 优化 model.py，10 轮

剩下的 profiling、写 kernel、调试、回滚、记录，全部由 agent 自动完成。

---

## 这个项目做什么

给定一个 [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
风格的 [model.py](model.py)（暴露 `Model` / `get_inputs` / `get_init_inputs`），
`kernel` agent 会：

1. 建立数值与延迟 **baseline**（`output/round_000_baseline.json`）。
2. 根据 profiling 结果选出当前最耗时的算子或可融合的模式。
3. 用 `torch.utils.cpp_extension.load_inline` 在 `model_new.py` 中加载一份
   **手写的 CUDA / C++ kernel** 替换它。
4. 重新执行 [run.sh](run.sh) 进行**正确性校验**、**性能测量**（1000 次迭代）
   以及**性能 profiling**（`output/profile_latest.txt`）。
5. 每一轮都把代码快照（`output/round_NNN.py`）和结构化记录（同名 JSON）写盘。
6. 一旦本轮性能不及历史最佳，自动**回滚**到上一份通过的快照。
7. 最终生成 [output/summary.json](output/summary.json)：完整列出每一轮做了
   什么、加速比变化、成功的策略以及失败回滚的原因。

Agent 的写权限被严格限制在 `model_new.py` 与 `output/*` 之内，使整个优化过程
安全且可复现。

---

## 工作原理

### 1. 闭环优化循环

```
   ┌────────────┐    profile     ┌────────────────────┐
   │ model_new  │ ─────────────► │ optimize-decision  │
   │            │                │ （挑下一个目标）   │
   └─────▲──────┘                └─────────┬──────────┘
         │                                  │ 设计 kernel
         │ 回滚 / 修复                      ▼
   ┌─────┴───────┐  失败   ┌──────────────────────┐
   │ save-round  │◄────────│ run-eval (run.sh)    │
   │ update-state│  通过   │ 正确性 + 性能         │
   └─────────────┘         └──────────────────────┘
```

每轮迭代由一个小型状态机驱动，关键状态变量包括：`base_round`、`best_round`、
`best_speedup`、`current_target`，以及 `retry` / `stagnant` / `ceiling` 等计数器。

### 2. 基于 Skill 的能力组合

Agent 的推理过程被拆分成多个**单一职责的 Skill**（位于
`.github/skills/*`），每个 Skill 都有 `SKILL.md` 描述输入输出：

| Skill              | 职责                                                   |
| ------------------ | ------------------------------------------------------ |
| `init-baseline`    | 首轮 baseline，并基于 profiler 生成 OP 优化计划        |
| `run-eval`         | 调用 `bash run.sh [full|quick|correctness]`           |
| `diagnose-error`   | 区分编译 / 运行时 / 数值错误并给出修复建议             |
| `optimize-decision`| 根据 profiling 决定 `continue` / `switch` / `done_all` |
| `update-state`     | 维护 `base` / `best` 等全局状态                        |
| `save-round`       | 保存代码快照 + 写入轮次 JSON                           |
| `rollback`         | 把 `model_new.py` 回退到上一份安全快照                 |
| `generate-summary` | 生成 `output/summary.json`                             |

主流程因此可以保持简洁，复杂细节由各 Skill 内部完成。

### 3. 只允许手写 CUDA

`torch.compile`、`torch._dynamo`、`torch.jit`、Triton 等任何自动代码生成或
图编译加速路径都被**明令禁止**。允许的唯一加速手段是
`torch.utils.cpp_extension.load_inline`，迫使 agent 真正去思考 kernel 设计：
分块 / 向量化访存（`float4`）/ 共享内存 / warp 级归约 / epilogue 融合 /
CUDA Graph 捕获等。具体策略参考 `.github/agents/toolkit/` 下的
`op-taxonomy.md`、`fusion-patterns.md`、`optimization-toolkit.md` 等文档。

### 4. 仅针对 inference

优化目标限定在 `model.eval()` + `torch.no_grad()` 路径，可以安全地做更激进
的变换：把 BatchNorm 折进上一层 Conv、彻底删掉 Dropout、为反复使用的张量
预分配持久 workspace 等。

### 5. 持久化记录与安全回滚

每轮都会同时落盘 `round_NNN.py` 快照与一份结构化 JSON（schema 见
`toolkit/json-schemas.md`）。一旦检测到性能回退，agent 会自动恢复上一份通过
的快照，并按 `retry` 计数决定是否换一种思路或将该目标标记为 `skip`，绝不会
让进度悄悄丢失。

---

## 如何运行

### 环境准备

* Linux + NVIDIA GPU（仓库默认按 **sm_120 / RTX 5090** 配置，见
  [run.sh](run.sh)，按需修改 `TORCH_CUDA_ARCH_LIST` 与 `CUDA_HOME`）。
* CUDA 12.x 工具链，并通过 `CUDA_HOME` 可见。
* 装好 PyTorch + CUDA 的 Python 环境（默认路径写死在
  `/home/<user>/miniconda3/envs/CudaForge/bin/python`，请改成你自己的解释器）。
* VS Code，安装支持自定义 agent 的 **GitHub Copilot Chat**。

### 1. 准备 `model.py`

从 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 中挑一个
任务（或自行编写），把文件以 `model.py` 放在工程根目录，必须暴露：

```python
class Model(nn.Module):
    def __init__(self, *args): ...
    def forward(self, *inputs): ...

def get_inputs():        # 返回 forward(*inputs) 用到的张量列表
    ...

def get_init_inputs():   # 返回构造 Model(*args) 用到的参数列表
    ...
```

### 2. 先跑通 baseline

```bash
bash run.sh correctness   # 仅做数值校验
bash run.sh quick         # 100 次迭代的快速测速
bash run.sh               # full 模式：1000 次迭代 + profiling
```

这会创建 / 刷新 `model_new.py`（最初行为与 `model.py` 等价），并产出
`output/profile_latest.txt`。

### 3. 启动 agent

在 VS Code Copilot Chat 的 agent 选择器中切换到 **`kernel`**，发送：

> 优化 model.py，10 轮

Agent 将：

* 先跑一次完整 profiling，
* 在最多 *N* 轮内迭代优化，
* 每轮写出 `output/round_NNN.py` 与 `output/round_NNN_*.json`，
* 最后产出 `output/summary.json`，包含最佳加速比和完整历史
  （成功的策略、被回滚的方案及原因）。

### 4. 查看结果

* 最佳 kernel：`model_new.py`（始终对应当前 `best_round`）
* 每轮代码：`output/round_NNN.py`
* 每轮日志：`output/round_NNN_*.json`
* 最终报告：`output/summary.json`
* 最近一次 profiling：`output/profile_latest.txt`

英文版文档见 [README.md](README.md)。
