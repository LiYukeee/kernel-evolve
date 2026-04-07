"""
读取加载Model和ModelNew
1. 进行测试，比较两者的输出是否一致
2. 进行性能测试，比较两者的运行时间和资源消耗
3. 进行ModelNew的profile测试，分析其性能瓶颈和优化空间
"""

import torch
import time
from model import Model, get_inputs
from model_new import ModelNew

def test_correctness(model, model_new, inputs):
    print("正在进行正确性测试...")
    with torch.no_grad():
        output = model(*inputs)
        output_new = model_new(*inputs)
    
    if torch.allclose(output, output_new, atol=1e-5):
        print("✅ 正确性测试通过：Model 和 ModelNew 的输出一致。")
        return True
    else:
        print("❌ 正确性测试失败：Model 和 ModelNew 的输出不一致。")
        print(f"最大偏差: {(output - output_new).abs().max().item()}")
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
        return True
    except Exception as e:
        print(f"❌ 性能测试过程中出现错误: {e}")
        return False

def profile_model_new(model_new, inputs):
    print("正在对 ModelNew 进行 Profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./output/log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            model_new(*inputs)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. 初始化
    model = Model().to(device)
    model_new = ModelNew().to(device)
    inputs = [x.to(device) for x in get_inputs()]

    # 1. 正确性测试 -> 2. 性能测试 -> 3. Profiling
    if test_correctness(model, model_new, inputs) and \
       test_performance(model, model_new, inputs):
        profile_model_new(model_new, inputs)


