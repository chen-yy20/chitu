#!/bin/bash

# 如果未传入参数，默认使用2个GPU
num_gpus=${1:-2}
echo "PYTHONPATH: $PYTHONPATH"
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# 计算cp_size并确保最小值为1
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

# 请自行调整模型配置
model="Wan2.1-T2V-1.3B"
ckpt_dir="/home/zlq/diffusion/Wan2.1-main/Wan2.1-T2V-1.3B"

# 设置分布式训练环境变量（单节点多GPU）
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_GRAPH_REGISTER=0

# 获取脚本所在目录的绝对路径，然后找到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 从 chitu/diffusion/run_wan_demo_local.sh 向上两级到项目根目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "Running Chitu Diffusion with torchrun"
echo "Number of GPUs: $num_gpus"
echo "CP Size: $cp_size"
echo "Model: $model"
echo "Checkpoint Dir: $ckpt_dir"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "=========================================="

# 使用 torchrun 直接运行（单节点，多GPU）
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Test file should be at: $PROJECT_ROOT/chitu/diffusion/test_generate.py"

# 设置 PYTHONPATH，将项目根目录添加到 Python 模块搜索路径
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PROJECT_ROOT"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi
echo "PYTHONPATH set to: $PYTHONPATH"

cd "$PROJECT_ROOT"
if [ ! -f "./chitu/diffusion/test_generate.py" ]; then
    echo "ERROR: Cannot find test_generate.py at $PROJECT_ROOT/chitu/diffusion/test_generate.py"
    exit 1
fi

torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./chitu/diffusion/test_generate.py \
    models=$model \
    models.ckpt_dir=$ckpt_dir \
    infer.diffusion.cp_size=$cp_size \
    cache.enabled=true \
    cache.teacache_thresh=0.08 \
    infer.diffusion.up_limit=2

