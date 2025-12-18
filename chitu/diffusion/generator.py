# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.distributed
from typing import Optional
import numpy as np
import torch.cuda.amp as amp

from logging import getLogger
from chitu.global_vars import get_global_args, get_slot_handle, get_timers
from chitu.diffusion.backend import DiffusionBackend
from chitu.backend import BackendState
from chitu.diffusion.task import DiffusionTask, DiffusionTaskType
from chitu.executor import TasksDispatcher
from chitu.distributed.parallel_state import (
    get_cfg_group,
    get_cp_group,
    get_up_group,
    get_world_group
)
from chitu.diffusion.modules.solvers.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from chitu.diffusion.modules.solvers.fm_solvers_unipc import FlowUniPCMultistepScheduler


logger = getLogger(__name__)


class SequenceDispatcher(TasksDispatcher):
    pass # context parallelism support

class Generator:

    @classmethod
    def build(cls, args) -> "Generator":
        return cls(args)
    
    def __init__(self, args):
        self.timers = get_timers()
        self.rank = torch.distributed.get_rank()
        self.local_rank = int
        self.sp_size = args.infer.diffusion.cp_size
        self.cfg_size = get_cfg_group().group_size

    def step(self, task: Optional[DiffusionTask]) -> torch.Tensor:
        # 调度器会给generator task，翻译成kernel -> 运行 -> 正确放置输出 -> 回收对应内存
        
        task_type = task.task_type if task is not None else None

        if task_type == DiffusionTaskType.TextEncode:
            out = self.text_encode_step(task)
        elif task_type == DiffusionTaskType.VAEEncode:
            out = self.vae_encode_step(task)
        elif task_type == DiffusionTaskType.VAEDecode:
            out = self.vae_decode_step(task)
        elif task_type == DiffusionTaskType.Denoise:
            out = self.denoise_step(task)
        else:
            raise NotImplementedError    
        
        task.update_stage_and_buffer(out)
        
        return out
        
            
    def text_encode_step(self, task: DiffusionTask) -> torch.Tensor:
        # TODO: 支持offload和t5 cpu
        # payload 是本次task需要处理的数据的抽象
        device = torch.cuda.current_device()
        if task.buffer.text_embeddings is None:
            payload = task.req.get_prompt()
        else:
            payload = task.req.get_n_prompt()
        logger.info(f"[text_encode_step] task_id={task.task_id}, txt={payload}")
        out = DiffusionBackend.text_encoder(payload, device=device)
        logger.info(f"[text_encode_step] context shape: {out.shape}")
        return out
    
    def vae_encode_step(self, task: DiffusionTask):
        pass
    
    def vae_decode_step(self, task: DiffusionTask):
        pass
        
    
    def denoise_step(self, task: DiffusionTask):
        logger.info("Enter Denoise Stage!")
        self._pre_denoising(task)
        exit()
        
        
    def _pre_denoising(self, task: DiffusionTask):
        """
        Before denoising loop, prepare latents, timesteps and solver for one task.
        TODO: Control Devices
        """
        device = torch.cuda.current_device()
        # Prepare latents
        if task.buffer.latents is None:
            F = task.req.params.frame_num
            size = task.req.params.size
            vae_stride = DiffusionBackend.args.models.vae.stride
            target_shape = (DiffusionBackend.vae.model.z_dim, (F - 1) // vae_stride[0] + 1,
                            size[1] // vae_stride[1],
                            size[0] // vae_stride[2])

            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(task.req.params.seed)
            latents = torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=device,
                        generator=seed_g
                    )
        
        # Prepare Solver and Timestep
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(): 
            if task.req.params.sample_solver == 'unipc': # 求解器
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=task.req.params.num_inference_steps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    task.req.params.num_inference_steps, 
                    device=device,
                    shift=task.req.params.sample_shift
                    )
                timesteps = sample_scheduler.timesteps
            elif task.req.params.sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    task.req.params.num_inference_steps, 
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(task.req.params.num_inference_steps, task.req.params.sample_shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")
            
            task.buffer.latents = latents
            task.buffer.timesteps = timesteps
            logger.info(f"[Pre Denoise] Init {latents.shape=} {timesteps=}")
            exit()
        
    
