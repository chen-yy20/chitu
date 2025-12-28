# Chitu-Diffusion
目前处于测试和开发阶段。欢迎感兴趣的同学加入团队。
测试的模型为Wan2.1-T2V, 会陆续补充支持新的模型。

# Setup
 > Python3.12, cuda 12.4
## Environment

按照`chitu/diffusion/requirements.txt`安装。

Flash Attention建议用wheel安装：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.1.post2

## Model Checkpoint
目前的模型：Wan2.1-T2V-1.3B

建议使用huggingface-cli安装，国内使用hf-mirror.

```
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts
```

# Run Demo
**模型架构参数**(层数、注意力头数等)是静态的，在`chitu/config/models/Wan2.1-T2V-1.3B.yaml`中进行设置。

**用户参数**(生成步数、形状等)是动态的，`Chitu`提供`DiffusionUserParams`以请求为单位进行设置。

**系统参数**(并行度、算子、加速算法等)，在`Chitu`的launch args中设置。

测试脚本：`chitu/diffusion/test_generate.py`
单卡/分布式启动：`bash run_wan_demo.sh <num_gpus>`

```
num_gpus=$1
echo $PYTHONPATH
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# 优先设置cfg并行为2，其余为序列并行 (up_limit为ulysses并行度，其余为ring-attention)
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

model="Wan2.1-T2V-1.3B"
ckpt_dir="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-1.3B"

./script/srun_multi_node.sh 1 $num_gpus ./chitu/diffusion/test_generate.py models=$model models.ckpt_dir=$ckpt_dir \
    infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=8
```