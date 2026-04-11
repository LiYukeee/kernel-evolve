# env setup
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python=/home/liyk/miniconda3/envs/CudaForge/bin/python

# 支持参数: full(默认,1000iter+profiling) / quick(100iter,无profiling) / correctness(仅正确性)
# full 模式会将 profiling 结果持久化到 output/profile_latest.txt
MODE=${1:-full}

cd "$(dirname "$0")"
$python test.py --mode "$MODE"