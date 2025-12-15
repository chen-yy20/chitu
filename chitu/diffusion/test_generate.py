import hydra
import torch
import time
import os
import random
import logging
from logging import getLogger

from chitu.task import UserRequest, TaskPool, Task
from chitu.diffusion.chitu_diffusion_main import (
    chitu_init,
    chitu_run,
    chitu_start,
    chitu_terminate,
    chitu_is_terminated,
    warmup_engine,
)
from chitu.global_vars import get_timers
from chitu.schemas import ServeConfig
from chitu.utils import get_config_dir_path, gen_req_id

logger = getLogger(__name__)

# T2V prompts
prmpts = [
    [{"role": "user", "content": "A cat walking on the grass."}],
    [{"role": "user", "content": "Two beautiful asian girls."}],
    [{"role": "user", "content": "一名宇航员在火星上拍照。"}],
]

# TI2V prompts
prmpts_vl = []

def gen_reqs(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    # 请求应该包含prompt，size等信息
    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        if is_vl:
            req = UserRequest(
                prmpts_vl[i % len(prmpts_vl)],
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=1,
            )
        else:
            req = UserRequest(
                prmpts[i % len(prmpts)],
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=1,
            )
        reqs.append(req)
    return reqs

def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    warmup_engine(args)
    logger.info("chitu warmup engine.")

    for i in range(1):
        reqs = gen_reqs(
            num_reqs=args.infer.max_reqs,
            max_new_tokens=args.request.max_new_tokens,
            frequency_penalty=args.request.frequency_penalty,
            is_vl=hasattr(args.models, "vision_config"),
        )
        logger.info(f'{reqs=}')
        for req in reqs:
            TaskPool.add(Task(req.request_id, req, stop_with_eos=True))
        logger.info(f"------ batch {i} ------")
        t_start = time.time()
        timers("overall").start()
        tokens = 0
        while len(TaskPool.pool) > 0:
            tokens += 1
            chitu_run()

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
    exit()
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