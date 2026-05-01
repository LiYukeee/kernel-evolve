# CUDA Kernel Auto-Optimizer Agent

An autonomous **GitHub Copilot agent** that turns a plain PyTorch reference
implementation (`model.py`) into a heavily-optimized version (`model_new.py`)
backed by **hand-written CUDA kernels**, iterating closed-loop until a target
speed-up is reached or the budget of optimization rounds is exhausted.

The agent is invoked from VS Code Copilot Chat with a single sentence such as
`优化 model.py，10 轮` ("optimize model.py for 10 rounds"). Everything else –
profiling, kernel writing, debugging, rollback, bookkeeping – is automated.

---

## What this project does

Given a PyTorch module exposed as `Model` in [model.py](model.py) (the layout
follows [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
conventions: `Model`, `get_inputs`, `get_init_inputs`), the `kernel` agent will:

1. Establish a numerical & latency **baseline** (`output/round_000_baseline.json`).
2. Repeatedly choose the next most expensive op (or fusion opportunity) from
   profiling data.
3. Write a custom CUDA/C++ kernel and load it through
   `torch.utils.cpp_extension.load_inline` inside `model_new.py`.
4. Re-run [run.sh](run.sh) to validate **correctness**, measure **latency**
   (1000 iterations) and collect **profiling traces**
   (`output/profile_latest.txt`).
5. Snapshot every round to `output/round_NNN.py` and a sibling JSON record.
6. Roll back automatically if a round regresses the best speed-up.
7. Emit a final [output/summary.json](output/summary.json) summarising every
   round, what worked, and what failed (and why).

The agent is restricted to a tight set of files (`model_new.py` and `output/*`),
making the optimization process safe and fully reproducible.

---

## How it works (principles)

### 1. Closed-loop optimization

```
   ┌────────────┐    profile     ┌────────────────────┐
   │ model_new  │ ─────────────► │ optimize-decision  │
   │            │                │ (pick next target) │
   └─────▲──────┘                └─────────┬──────────┘
         │                                  │ kernel design
         │ rollback / repair                ▼
   ┌─────┴───────┐  fail   ┌──────────────────────┐
   │ save-round  │◄────────│ run-eval (run.sh)    │
   │ update-state│  pass   │ correctness + perf   │
   └─────────────┘         └──────────────────────┘
```

Each iteration is a single transition in a small state machine governed by
five variables: `base_round`, `best_round`, `best_speedup`, `current_target`,
and counters for `retry`/`stagnant`/`ceiling`.

### 2. Skill-based composition

The agent's reasoning is decomposed into reusable, single-purpose **Skills**
(`.github/skills/*`), each with a `SKILL.md` describing inputs/outputs:

| Skill              | Responsibility                                            |
| ------------------ | --------------------------------------------------------- |
| `init-baseline`    | first run, build OP plan from profiler                    |
| `run-eval`         | invoke `bash run.sh [full|quick|correctness]`             |
| `diagnose-error`   | classify build / runtime / numerical failures             |
| `optimize-decision`| pick `continue` / `switch` / `done_all` from profile data |
| `update-state`     | move `base`/`best` markers                                |
| `save-round`       | snapshot code + write round JSON                          |
| `rollback`         | revert `model_new.py` to last good snapshot               |
| `generate-summary` | produce `output/summary.json`                             |

This lets a high-level workflow stay short while delegating the messy details.

### 3. Hand-written CUDA only

`torch.compile`, `torch._dynamo`, `torch.jit`, Triton and any other
auto-codegen path are **forbidden**. The only allowed acceleration mechanism
is `torch.utils.cpp_extension.load_inline`, which forces the agent to actually
*think* about kernels: tiling, vectorized loads (`float4`), shared memory,
warp-level reductions, fused epilogues, CUDA Graph capture, etc. Strategy
references live in `.github/agents/toolkit/` (`op-taxonomy.md`,
`fusion-patterns.md`, `optimization-toolkit.md`, …).

### 4. Inference-only guarantees

Optimization targets only `model.eval()` + `torch.no_grad()`. This unlocks
aggressive transformations such as folding BatchNorm into the preceding Conv,
dropping Dropout, or pre-allocating persistent workspace tensors.

### 5. Persistent memory & rollback safety

Every round writes both a `round_NNN.py` snapshot and a structured JSON record
(schema in `toolkit/json-schemas.md`). When a regression is detected the agent
restores the latest passing snapshot, increments a retry counter, and either
continues with a different strategy or marks the target `skip` — never
silently abandoning progress.

---

## How to run

### Prerequisites

* Linux + NVIDIA GPU (project ships configured for **sm_120 / RTX 5090** in
  [run.sh](run.sh); change `TORCH_CUDA_ARCH_LIST` and `CUDA_HOME` as needed).
* CUDA 12.x toolkit reachable from `CUDA_HOME`.
* A Python env with PyTorch + CUDA (default path expects
  `/home/<user>/miniconda3/envs/CudaForge/bin/python`; edit `python=...` in
  `run.sh` to point at your interpreter).
* VS Code with **GitHub Copilot Chat** that supports custom agents.

### 1. Drop in a `model.py`

Pick any task from [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
(or write your own) and place it at the workspace root as `model.py`. It must
expose:

```python
class Model(nn.Module):
    def __init__(self, *args): ...
    def forward(self, *inputs): ...

def get_inputs():        # -> list[Tensor] used as forward(*inputs)
    ...

def get_init_inputs():   # -> list passed to Model(*args)
    ...
```

### 2. Sanity-check the baseline

```bash
bash run.sh correctness   # numerical check only
bash run.sh quick         # 100-iter timing
bash run.sh               # full: 1000-iter + profiling
```

This will create / refresh `model_new.py` (initially identical behaviour) and
write `output/profile_latest.txt`.

### 3. Launch the agent

In VS Code Copilot Chat, switch the agent selector to **`kernel`** and send a
prompt such as:

> 优化 model.py，10 轮
> *("Optimize model.py for 10 rounds")*

The agent will:

* run an initial profiling pass,
* iterate up to *N* optimization rounds,
* write `output/round_NNN.py` + `output/round_NNN_*.json` per round,
* and finalize `output/summary.json` with the best speed-up and the full
  history (what worked, what was rolled back, and why).

### 4. Inspect results

* Best kernel:    `model_new.py` (always reflects the current `best_round`)
* Per-round code: `output/round_NNN.py`
* Per-round log:  `output/round_NNN_*.json`
* Final report:   `output/summary.json`
* Latest profile: `output/profile_latest.txt`

A Chinese version of this document is available in
[README.zh-CN.md](README.zh-CN.md).
