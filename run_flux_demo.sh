num_gpus=$1
echo $PYTHONPATH
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# 计算cp_size并确保最小值为1
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

model="FLUX.2-klein-4B"
ckpt_dir="/home/dataset/FLUX.2-klein-4B"

./script/srun_multi_node.sh 1 $num_gpus ./chitu/diffusion/test_generate.py models=$model models.ckpt_dir=$ckpt_dir \
    infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=2 infer.diffusion.low_memory=false