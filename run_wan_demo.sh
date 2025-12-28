num_gpus=$1
echo $PYTHONPATH
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# 计算cp_size并确保最小值为1
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

./script/srun_multi_node.sh 1 $num_gpus ./chitu/diffusion/test_generate.py models=Wan2.1-T2V-1.3B models.ckpt_dir="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-1.3B" \
    infer.seed=42 \
    infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=8