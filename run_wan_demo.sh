num_gpus=$1
echo $PYTHONPATH
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# 计算cp_size并确保最小值为1
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

# model="Wan2.1-T2V-1.3B"
# ckpt_dir="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-1.3B"

model="Wan2.1-T2V-14B"
ckpt_dir="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-14B"

# model="Wan2.2-T2V-A14B"
# ckpt_dir="/home/zhongrx/cyy/model/Wan22-t2v-a14b"

./script/srun_multi_node.sh 1 $num_gpus ./chitu/diffusion/test_generate.py models=$model models.ckpt_dir=$ckpt_dir \
    infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=2 infer.diffusion.low_memory=false