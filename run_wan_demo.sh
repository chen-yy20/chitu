echo $PYTHONPATH
export CHITU_DEBUG=1
./script/srun_multi_node.sh 1 1 ./chitu/diffusion/test_generate.py models=Wan2.1-T2V-1.3B models.ckpt_dir="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-1.3B" infer.attn_type=flash_attn \
    infer.seed=42 \
    infer.diffusion.cp_size=1 infer.diffusion.up_limit=8