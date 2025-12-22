import hydra
import torch
import time
import os
import random
import logging
from logging import getLogger

from chitu.diffusion.chitu_diffusion_main import (
    chitu_init,
    chitu_generate,
    warmup_diffusion_engine,
)

# from chitu.task import UserRequest, TaskPool, Task
from chitu.diffusion.task import DiffusionUserRequest, DiffusionTask, DiffusionTaskPool, DiffusionParams

from chitu.global_vars import get_timers
from chitu.schemas import ServeConfig
from chitu.utils import get_config_dir_path, gen_req_id

logger = getLogger(__name__)

# Default wan params
# 需要注意区分：系统参数 / 模型参数 config / 用户参数 params
default_diffusion_params = DiffusionParams(
    seed=42,
    frame_num=81,
    size=(832,480),
    negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
    sample_shift=5.0,
    guidance_scale=7.5,
    num_inference_steps=50,
    sample_solver='unipc',
)


# T2V prompts
msgs = [
    [{"role": "user", "content": "A cat walking on the grass."}],
    # [{"role": "user", "content": "Two beautiful asian girls."}],
    # [{"role": "user", "content": "一名宇航员在火星上拍照。"}],
]

def gen_reqs(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    # TODO: 请求应该包含prompt，size等信息，同时对最大负载进行控制
    reqs: list[DiffusionUserRequest] = []
    for i in range(num_reqs):
        req = DiffusionUserRequest(
            message = msgs[i % len(msgs)],
            request_id = f"{gen_req_id()}",
            params=default_diffusion_params,
        )
        reqs.append(req)
    return reqs

def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    warmup_diffusion_engine(args)
    logger.info("chitu warmup engine.")

    # 重复执行
    for i in range(1):
        reqs = gen_reqs(
            # num_reqs=args.infer.max_reqs,
            num_reqs=len(msgs),
            max_new_tokens=args.request.max_new_tokens,
            frequency_penalty=args.request.frequency_penalty,
            is_vl=hasattr(args.models, "vision_config"),
        )
        logger.info(f'{reqs=}')
        
        # 已有的request加入
        for req in reqs:
            DiffusionTaskPool.add(DiffusionTask(req.request_id, req))
            
        logger.info(f"------ batch {i} ------")
        t_start = time.time()
        
        timers("overall").start()
        tokens = 0
        while len(DiffusionTaskPool.pool) > 0:
            chitu_generate()

        print("GPU memory used : ", torch.cuda.memory_allocated())
        timers("overall").stop()
        t_end = time.time()
        logger.info(f"Tokens generate : {tokens}")
        logger.info(f"Time cost {t_end - t_start}")

        for i, req in enumerate(reqs):
            logger.info(f"Response in rank {rank}: reqs[{i}].output={req.output}")

        timers.log()

@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", get_config_dir_path()),
    config_name=os.getenv("CONFIG_NAME", "serve_config"),
)
def main(args: ServeConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    logger.info(f"Run with args: {args}")

    chitu_init(args, logging_level=logging.INFO)
    logger.info("initialized chitu.")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    timers = get_timers()
    logger.debug("finish init")
    
    run_normal(args, timers)


if __name__ == "__main__":
    main()

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits