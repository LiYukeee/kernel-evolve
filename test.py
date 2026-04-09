"""
读取加载Model和ModelNew
1. 进行测试，比较两者的输出是否一致
2. 进行性能测试，比较两者的运行时间和资源消耗
3. 进行ModelNew的profile测试，分析其性能瓶颈和优化空间
"""

import torch
import time
from model import Model, get_inputs, get_init_inputs
from model_new import ModelNew

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

def test_performance(model, model_new, inputs, iterations=1000):
    try:
        print(f"正在进行性能测试 (迭代次数: {iterations})...")
        
        # 启用推理模式
        model.eval()
        model_new.eval()

        # Warm up: 预热 GPU，确保时钟频率稳定并初始化算子
        print("正在进行 Warmup...")
        with torch.no_grad():
            for _ in range(50):
                model(*inputs)
                model_new(*inputs)
        torch.cuda.synchronize()

        # Test Model
        print("正在测试 Model...")
        with torch.no_grad():
            start_time = time.time()
            for _ in range(iterations):
                model(*inputs)
            torch.cuda.synchronize()
        model_time = (time.time() - start_time) / iterations
        print(f"Model 平均耗时: {model_time * 1000:.6f} ms")

        # Test ModelNew
        print("正在测试 ModelNew...")
        with torch.no_grad():
            start_time = time.time()
            for _ in range(iterations):
                model_new(*inputs)
            torch.cuda.synchronize()
        model_new_time = (time.time() - start_time) / iterations
        print(f"ModelNew 平均耗时: {model_new_time * 1000:.6f} ms")
        
        speedup = model_time / model_new_time
        print(f"加速比: {speedup:.2f}x")
        return model_time, model_new_time, speedup
    except Exception as e:
        print(f"❌ 性能测试过程中出现错误: {e}")
        return None

def profile_model_new(model_new, inputs):
    print("正在对 ModelNew 进行 Profiling...")
    with torch.no_grad(), torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./output/log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            model_new(*inputs)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. 初始化
    init_inputs = [x for x in get_init_inputs()]
    model = Model(*init_inputs).to(device)
    model_new = ModelNew(*init_inputs).to(device)

    # 同步权重：Model 与 ModelNew 结构一致，直接按同名参数复制
    model_new.load_state_dict(model.state_dict())

    inputs = [x.to(device) for x in get_inputs()]

    # 1. 正确性测试 -> 2. 性能测试 -> 3. Profiling
    if test_correctness(model, model_new, inputs):
        perf_result = test_performance(model, model_new, inputs)
        if perf_result is not None:
            model_time, model_new_time, speedup = perf_result
            del model
            torch.cuda.empty_cache()
            profile_model_new(model_new, inputs)
            torch.cuda.empty_cache()
            print(
                f"FINAL_SPEED_RESULT: model={model_time * 1000:.6f} ms, "
                f"model_new={model_new_time * 1000:.6f} ms, speedup={speedup:.2f}x"
            )


