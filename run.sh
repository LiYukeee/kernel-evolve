# env setup
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python=/home/liyk/miniconda3/envs/CudaForge/bin/python

# run test
cd "$(dirname "$0")"
$python test.py