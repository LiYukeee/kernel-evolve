# System Information

## Hardware
- Primary GPU: NVIDIA GeForce RTX 5090
- GPU Architecture: Blackwell (GB202)
- GPU Memory: 32GB GDDR7
- Memory Bandwidth: 1792 GB/s
- FP32 TFLOPS: 104.8
- TF32 Tensor Core TFLOPS: 209.6 (419.2 with sparsity)
- BFLOAT16 Tensor Core TFLOPS: 419.2 (838.4 with sparsity)
- FP16 Tensor Core TFLOPS: 419.2 (838.4 with sparsity)
- FP8 Tensor Core TFLOPS: 838.4 (1676.8 with sparsity)
- INT8 Tensor Core TOPS: 838.4 (1676.8 with sparsity)
- Register File Size: 64K 32-bit registers per SM
- Maximum Registers Per Thread: 255
- Maximum Thread Blocks Per SM: 24
- Shared Memory Capacity Per SM: 100 KB
- Maximum Shared Memory Per Thread Block: 99 KB

## Optimization Hints (Agent-Readable)
- Use RTX 5090 as the default tuning target unless the user explicitly changes device.
- Prefer stable benchmarking on a fixed GPU (avoid cross-device drift between rounds).
- Use this rough bandwidth lower bound for memory-bound kernels:
	- time_ms >= bytes_moved / (1792e9) * 1e3
- For CUDA kernel tuning, prioritize:
	- memory coalescing
	- launch configuration search (block size 128/256/512)
	- reducing intermediate global-memory round trips via safe fusion
	- occupancy balance by controlling register pressure and shared memory usage

## RTX5090 Strategy Notes
- If kernel is clearly memory-bound, prioritize fewer global reads/writes over extra arithmetic optimization.
- If kernel uses too many registers and hurts occupancy, reduce per-thread work and test smaller unroll factors.
- Keep per-block shared memory under practical limits to avoid active-block collapse on SM.
- For GEMM/Conv main body, prefer library kernels unless profiling shows small-shape overhead or avoidable layout conversion.

## Notes
- If runtime environment differs from this file, follow runtime detection results and record the mismatch in round logs.
