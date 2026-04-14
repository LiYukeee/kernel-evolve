"""
读取加载Model和ModelNew
1. 进行测试，比较两者的输出是否一致
2. 进行性能测试，比较两者的运行时间和资源消耗
3. 进行ModelNew的profile测试，分析其性能瓶颈和优化空间
"""

import torch
import time
import signal
import sys
import shutil
import os
from model import Model, get_inputs, get_init_inputs
import numpy as np
import subprocess
import os

COMPILE_TIMEOUT = 300  # 秒，超过此时间视为编译卡死
WARM_UP_TIMES = 20

def autoChooseCudaDevice():
    try:
        cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
        os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
        print(f'--- Auto chose CUDA device {os.environ["CUDA_VISIBLE_DEVICES"]} ---')
    except:
        print('--- Failed to auto choose CUDA device, using default ---')

# 必须在任何 CUDA 初始化（包括 load_inline）之前设置 CUDA_VISIBLE_DEVICES
autoChooseCudaDevice()
# ---------- 带超时的 ModelNew 编译导入 ----------
class _CompileTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _CompileTimeoutError()

def _clear_compile_cache(ext_name: str = "softmax_v2"):
    """清除 torch cpp_extension 针对指定扩展的编译缓存目录。"""
    try:
        from torch.utils.cpp_extension import get_default_build_root
        build_root = get_default_build_root()
    except Exception:
        build_root = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    # torch 会在 build_root/<python_ver>_<cuda_ver>/<ext_name>/ 下存放缓存
    # 用 glob 匹配以避免硬编码 Python/CUDA 版本号
    cleared = False
    if os.path.isdir(build_root):
        for sub in os.listdir(build_root):
            cache_dir = os.path.join(build_root, sub, ext_name)
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"已清除编译缓存：{cache_dir}")
                cleared = True
    if not cleared:
        print("未找到对应的编译缓存目录，无需清除。")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(COMPILE_TIMEOUT)
try:
    print(f"正在编译 ModelNew（超时阈值 {COMPILE_TIMEOUT}s）...")
    from model_new import ModelNew
    signal.alarm(0)  # 编译成功，取消闹钟
    print("ModelNew 编译完成。")
except _CompileTimeoutError:
    signal.alarm(0)
    print(f"❌ ModelNew 编译超时（>{COMPILE_TIMEOUT}s），进程可能已卡死。")
    print("正在清除编译缓存以便下次重新编译...")
    _clear_compile_cache()
    print("请修复 CUDA kernel 代码后重新运行。")
    sys.exit(1)
except Exception as _compile_exc:
    signal.alarm(0)
    print(f"❌ ModelNew 编译失败：{_compile_exc}")
    print("正在清除编译缓存以便下次重新编译...")
    _clear_compile_cache()
    print("请修复上述错误后重新运行。")
    sys.exit(1)
# ------------------------------------------------

def test_correctness(model, model_new, inputs):
    print("正在进行正确性测试...")
    with torch.no_grad():
        output = model(*inputs)
        output_new = model_new(*inputs)

    # 分块比较，避免 allclose 一次性分配与 output 等大的临时 tensor
    max_diff = 0.0
    chunk = 256
    for i in range(0, output.shape[0], chunk):
        diff = (output[i:i+chunk] - output_new[i:i+chunk]).abs().max().item()
        max_diff = max(max_diff, diff)
    del output, output_new
    torch.cuda.empty_cache()

    if max_diff <= 1e-5:
        print("✅ 正确性测试通过：Model 和 ModelNew 的输出一致。")
        return True
    else:
        print("❌ 正确性测试失败：Model 和 ModelNew 的输出不一致。")
        print(f"最大偏差: {max_diff}")
        return False

def _bench_single(m, inputs, name, iterations):
    """对单个模型进行 warmup + 计时，返回平均每次耗时（秒）。"""
    print(f"正在进行 {name} Warmup...")
    with torch.no_grad():
        for _ in range(WARM_UP_TIMES):
            m(*inputs)
    torch.cuda.synchronize()
    print(f"正在测试 {name}...")
    with torch.no_grad():
        start_time = time.time()
        for _ in range(iterations):
            m(*inputs)
        torch.cuda.synchronize()
    t = (time.time() - start_time) / iterations
    print(f"{name} 平均耗时: {t * 1000:.6f} ms")
    return t

def test_performance(model, model_new, model_compile, inputs, iterations=1000):
    try:
        print(f"正在进行性能测试 (迭代次数: {iterations})...")

        model.eval()
        model_new.eval()

        model_time         = _bench_single(model,         inputs, "Model",            iterations)
        model_new_time     = _bench_single(model_new,     inputs, "ModelNew",         iterations)
        model_compile_time = _bench_single(model_compile, inputs, "Model (compiled)", iterations)

        baseline = model_time
        speedup  = baseline / model_new_time
        print(f"\n加速比汇总（以 Model 为基准）:")
        print(f"  ModelNew:              {speedup:.2f}x")
        print(f"  Model (compiled):      {baseline / model_compile_time:.2f}x")

        return model_time, model_new_time, model_compile_time, speedup
    except Exception as e:
        print(f"❌ 性能测试过程中出现错误: {e}")
        return None

def profile_model_new(model_new, inputs, profile_output=None):
    print("正在对 ModelNew 进行 Profiling...")
    with torch.no_grad(), torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            model_new(*inputs)
    
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(table)
    if profile_output:
        os.makedirs(os.path.dirname(profile_output) or ".", exist_ok=True)
        with open(profile_output, "w") as f:
            f.write(table)
        print(f"Profiling 结果已保存到 {profile_output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "quick", "correctness"], default="full",
                        help="full=1000iter+profiling(default), quick=100iter无profiling, correctness=仅正确性")
    args = parser.parse_args()

    TEST_ITERATIONS = 100 if args.mode == "quick" else 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. 初始化
    init_inputs = [x for x in get_init_inputs()]
    model = Model(*init_inputs).to(device)
    model_new = ModelNew(*init_inputs).to(device)

    # 同步权重：Model 与 ModelNew 结构一致，直接按同名参数复制
    model_new.load_state_dict(model.state_dict())

    # torch.compile 版本（首次调用时触发追踪/编译，warmup 阶段会完成）
    print("正在创建 model_compile (torch.compile)...")
    model_compile = torch.compile(model)

    inputs = [x.to(device) for x in get_inputs()]

    # 1. 正确性测试 -> 2. 性能测试 -> 3. Profiling
    if test_correctness(model, model_new, inputs):
        if args.mode == "correctness":
            print("CORRECTNESS_ONLY: PASS")
        else:
            perf_result = test_performance(
                model, model_new, model_compile,
                inputs, iterations=TEST_ITERATIONS
            )
            if perf_result is not None:
                model_time, model_new_time, model_compile_time, speedup = perf_result
                del model, model_compile
                torch.cuda.empty_cache()
                if args.mode == "full":
                    profile_model_new(model_new, inputs,
                                      profile_output="output/profile_latest.txt")
                    torch.cuda.empty_cache()
                print(
                    f"FINAL_SPEED_RESULT: "
                    f"model={model_time * 1000:.6f} ms, "
                    f"model_new={model_new_time * 1000:.6f} ms, "
                    f"model_compile={model_compile_time * 1000:.6f} ms, "
                    f"speedup(compile/base)={model_time / model_compile_time:.2f}x, "
                    f"speedup(new/base)={speedup:.2f}x"
                )

