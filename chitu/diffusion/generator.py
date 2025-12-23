# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import torch.distributed
from typing import Optional
import numpy as np
import torch.amp as amp

from logging import getLogger
from chitu.global_vars import get_global_args, get_slot_handle, get_timers
from chitu.diffusion.backend import DiffusionBackend
from chitu.backend import BackendState
from chitu.task import SerializedPackedTasksPayloadType
from chitu.diffusion.task import DiffusionTask, DiffusionTaskType, DiffusionTaskPool, DiffusionTaskStatus
from chitu.executor import TasksDispatcher
from chitu.distributed.parallel_state import (
    get_cfg_group,
    get_cp_group,
    get_up_group,
    get_world_group
)
from chitu.diffusion.modules.samplers.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from chitu.diffusion.modules.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from chitu.diffusion.utils.wan_utils import cache_video


logger = getLogger(__name__)

class CfgDispatcher(TasksDispatcher):
    def __init__(self):
        super().__init__()
        self.cfg_group = get_cfg_group()
        self.rank = self.cfg_group.global_rank
        self.local_rank = self.cfg_group.local_rank

        self.cfg_main_rank = self.cfg_group.rank_list[0]
        self.is_main_rank = self.cfg_group.is_first_rank

        pass

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
        # dispatcher
        task.status = DiffusionTaskStatus.Running

        if task_type == SerializedPackedTasksPayloadType.TerminateBackend:
            DiffusionBackend.state = BackendState.Terminated
            return None


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
    
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_step(self, task: DiffusionTask):
        logger.info(f"Step {task.buffer.current_step}: Enter Denoise Stage!")
        if task.buffer.latents is None or task.buffer.timesteps is None:
            self._pre_denoising(task)

        latent_model_input = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]

        if task.do_cfg:
            noise_pred_cond = DiffusionBackend.model(
                latent_model_input,
                t=timestep,
                context=task.buffer.text_embeddings,
                seq_len=task.buffer.seq_len
            )
            noise_pred_uncond = DiffusionBackend.model(
                latent_model_input,
                t=timestep,
                context=task.buffer.negative_embeddings,
                seq_len=task.buffer.seq_len
            )
            noise_pred = noise_pred_uncond + \
                task.req.params.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            logger.info(f"do cfg: {noise_pred.shape=}")
        else:
            noise_pred = DiffusionBackend.model(
                latent_model_input,
                t=timestep,
                context=task.buffer.text_embeddings,
                seq_len=task.buffer.seq_len
            )

        sampled_latents = task.buffer.sampler.step(
            noise_pred.unsqueeze(0),
            timestep,
            task.buffer.latents.unsqueeze(0),
            return_dict=False,
            generator=task.buffer.seed_g
        )[0].squeeze(0)

        logger.info(f"[Denoise Step] {task.buffer.current_step}/"
                    f"{task.req.params.num_inference_steps} "
                    f"timestep: {timestep} latents shape: {sampled_latents.shape}, {sampled_latents.dtype=}")

        return sampled_latents
    
    def vae_decode_step(self, task: DiffusionTask):
        payload = [task.buffer.latents]
        if torch.distributed.get_rank() == 0:
            logger.info(f"Step {task.buffer.current_step}: Enter VAE Decode Stage!")
            video = DiffusionBackend.vae.decode(payload)[0]
            self._save_image(task, video)
            return video
        return None



    # TODO: CPU/GPU overlap
    def _save_image(self, task: DiffusionTask, video: torch.Tensor):
        os.makedirs(task.req.params.save_dir, exist_ok=True)
        save_name = task.req.get_prompt()[:20].replace(" ", "_").replace(".", "") \
                    + f"_{task.task_id}.mp4"
        save_path = os.path.join(task.req.params.save_dir, save_name)
        logger.info(f"Saving video: {video.shape=} {video.dtype=}")
        cache_video(
            tensor=video[None],
            save_file=save_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        logger.info(f"[Succeed] Task {task.task_id} video saved to {save_path}")

        
        
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
            
            patch_size = DiffusionBackend.args.models.transformer.patch_size
            seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (patch_size[1] * patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size
        
        # Prepare Solver and Timestep
        if task.req.params.sample_solver == 'unipc': # 求解器
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
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
                num_train_timesteps=1000, 
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(task.req.params.num_inference_steps, task.req.params.sample_shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
            
        task.buffer.seed_g = seed_g
        task.buffer.latents = latents
        task.buffer.timesteps = timesteps
        task.buffer.seq_len = seq_len
        task.buffer.sampler = sample_scheduler

        DiffusionBackend.model.to(device)

        logger.info(f"[Pre Denoise] Init {latents.shape=} {timesteps=}")
        
    
