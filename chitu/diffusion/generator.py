# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import torch.distributed as dist
from typing import Optional, List, Tuple
from functools import partial
from contextlib import contextmanager
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
from chitu.diffusion.utils.shared_utils import SequencePadder


logger = getLogger(__name__)

class DiffusionTaskDispatcher(TasksDispatcher):
    def __init__(self):
        super().__init__()
        self.group = get_world_group()
        self.rank = self.group.global_rank
        self.local_rank = self.group.local_rank

        self.main_rank = 0
        self.is_main_rank = self.group.is_first_rank
        logger.info("init diffusion task dispatcher.")


    def dispatch_metadata(self, task: Optional[DiffusionTask] = None) -> tuple[DiffusionTaskType, DiffusionTask]:
        if self.is_main_rank:
            assert task is not None
            # 发送方：序列化任务并获取大小
            task_tensor = task.serialize(
                device="cpu" if DiffusionBackend.use_gloo else self.local_rank
            )
            task_type = task.task_type
            task_size = torch.tensor([task_tensor.size()[0]], dtype=torch.int64,
                                    device="cpu" if DiffusionBackend.use_gloo else self.local_rank)
            
            # 第一阶段：广播任务大小
            logger.info(f"Rank {self.rank}: Broadcasting task size {task_size.item()}")
            dist.broadcast(
                tensor=task_size,
                src=self.main_rank,
            )
            
        else:
            # 接收方：创建空的size tensor用于接收
            task_size = torch.zeros(1, dtype=torch.int64,
                                device="cpu" if DiffusionBackend.use_gloo else self.local_rank)
            
            # 第一阶段：接收任务大小
            dist.broadcast(
                tensor=task_size,
                src=self.main_rank,
            )
            
            # 第二阶段：根据接收到的大小创建空buffer
            logger.info(f"Rank {self.rank}: Received task size {task_size.item()}, creating buffer")
            task_tensor = DiffusionTask.create_empty_serialization(
                size=task_size.item(),  # 注意这里要用 .item() 获取标量值
                device="cpu" if DiffusionBackend.use_gloo else self.local_rank
            )
        
        logger.info(f"Rank {self.rank} | {task_tensor.shape=} {task_size[0]=} {task_tensor.dtype=} {task_tensor.device=}, ready to broadcast.")
        
        # 第二阶段：广播实际的任务数据
        dist.broadcast(
            tensor=task_tensor,
            src=self.main_rank,
        )
        
        if not self.is_main_rank:
            # 接收方：反序列化任务
            task = DiffusionTask.deserialize(task_tensor)
            task_type = task.task_type
        
        logger.info(f"Rank {self.rank}: {task=}")
        return task_type, task

    def send_payload(self, *args, **kwargs):
        return
    
    def recv_payload(self, *args, **kwargs):
        return
    

class CfgDispatcher():
    def __init__(self):
        super().__init__()
        self.group = get_cfg_group()
        self.rank = self.group.global_rank
        self.local_rank = self.group.local_rank

    def all_gather_cfg_noise_preds(self, local_noise_pred: torch.Tensor):

        gathered_preds = [torch.empty_like(local_noise_pred) for _ in range(2)]
        dist.all_gather(
            tensor_list=gathered_preds,
            tensor=local_noise_pred,
            group=self.group.gpu_group,
        )
        return gathered_preds[0], gathered_preds[1]

class ContextParallelDispatcher():
    def __init__(self):
        super().__init__()
        self.group = get_cp_group()
        self.cp_size = self.group.group_size
        self.rank = self.group.global_rank
        self.local_rank = self.group.local_rank
        self.rank_in_group = self.group.rank_in_group

    def dispatch(self, tokens: torch.Tensor):
        return SequencePadder.split_sequence_padding(tokens, 
                                                     split_num=self.cp_size,
                                                     split_dim=1, 
                                                     name='x')[self.rank_in_group]
    
    def gather(self, tokens: torch.Tensor):
        tokens_list = [torch.empty_like(tokens) for _ in range(self.cp_size)]
        dist.all_gather(tensor_list=tokens_list, 
                        tensor=tokens, 
                        group=self.group.gpu_group)
        return SequencePadder.remove_sequence_padding_and_concat(tokens_list, 
                                                                 gather_dim=1,
                                                                 name='x')
    
    def wrap_model_compute_with_cp(self):
        """替换DiffusionBackend.model.model_compute方法，添加CP支持"""
        
        def create_wrapped_forward(model_instance):
            original_forward = model_instance.model_compute
            def wrapped_compute(tokens, **kwargs):
                tokens = self.dispatch(tokens)
                if "seq_lens" in kwargs.keys():
                    kwargs["seq_lens"] = torch.tensor([tokens.size(1)])
                x = original_forward(tokens, **kwargs)
                x = self.gather(x)
                return x
            return wrapped_compute

        if DiffusionBackend.args.models.name in ["Wan2.2-T2V-A14B"]:
            # pack two models
            DiffusionBackend.high_noise_model.model_compute = create_wrapped_forward(DiffusionBackend.high_noise_model)
            DiffusionBackend.low_noise_model.model_compute = create_wrapped_forward(DiffusionBackend.low_noise_model)
        elif DiffusionBackend.args.models.name in ["Wan2.1-T2V-1.3B"]:
            DiffusionBackend.model.model_compute = create_wrapped_forward(DiffusionBackend.model)
        else:
            raise NotImplementedError("Unsupported model for CP.")

        # DiffusionBackend.model.rope_apply = partial(rope_apply_with_cp, 
        #                                             cp_size=self.group.group_size, 
        #                                             cp_rank=self.rank_in_group)
            

class Generator:
    @classmethod
    def build(cls, args) -> "Generator":
        return cls(args)
    
    def __init__(self, args):
        self.timers = get_timers()
        self.rank = torch.distributed.get_rank()
        self.local_rank: int = self.rank % 8 
        self.task_dispatchers: List[TasksDispatcher] = []
        self.cp_size = args.infer.diffusion.cp_size
        self.cfg_size = get_cfg_group().group_size
        if self.cp_size > 1:
            self.cp_dispatcher = ContextParallelDispatcher()
            self.cp_dispatcher.wrap_model_compute_with_cp()
        if self.cfg_size == 2:
            self.cfg_dispatcher = CfgDispatcher()

        # Ensure FIFO
        self.current_task = None # 通过这个储存当前任务的中间状态


    def step(self, task: Optional[DiffusionTask]) -> torch.Tensor:
        # 调度器会给generator task，翻译成kernel -> 运行 -> 正确放置输出 -> 回收对应内存
        # Prepare Payload
        if self.current_task is None: # 生成的最开始，将任务分发到workers
            task_type, task = DiffusionTaskDispatcher().dispatch_metadata(task)
            self.current_task = task
        else:
            task = self.current_task # 接续当前任务

        torch.cuda.synchronize()
        dist.barrier()
        task_type = task.task_type if task is not None else None
        assert self.current_task.task_id == task.task_id # 确保任务逐个完成，避免产生太多中间状态
        
        task.status = DiffusionTaskStatus.Running

        if task_type == DiffusionTaskType.Terminate:
            DiffusionBackend.state = BackendState.Terminated
            task.status = DiffusionTaskStatus.Completed
            return None

        # Execute
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
        
        self._update_task_stage_and_buffer(task, out)

        return out
        
            
    def text_encode_step(self, task: DiffusionTask) -> torch.Tensor:
        # TODO: 支持offload和t5 cpu
        # payload 是本次task需要处理的数据的抽象
        device = torch.cuda.current_device()
        if self.cfg_size == 1:
            if task.buffer.text_embeddings is None:
                payload = task.req.get_prompt()
            else:
                payload = task.req.get_n_prompt()
        elif self.cfg_size == 2:
            if get_cfg_group().rank_in_group == 0:
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

        assert task.buffer.latents is not None and task.buffer.timesteps is not None

        latent_model_input = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]

    
        # FIXME: reduce branch
        if DiffusionBackend.args.models.name in ["Wan2.2-T2V-A14B"]:
            if timestep >= DiffusionBackend.args.models.transformer.boundary * DiffusionBackend.args.models.transformer.num_train_timesteps:
                noise_guidance_scale = DiffusionBackend.args.models.transformer.high_guide_scale
                model = DiffusionBackend.high_noise_model
            else:
                noise_guidance_scale = DiffusionBackend.args.models.transformer.low_guide_scale
                model = DiffusionBackend.low_noise_model
        else:
            model = DiffusionBackend.model
            noise_guidance_scale = task.req.params.guidance_scale

        if task.do_cfg: # Wan的cfg是做两次
            if self.cfg_size == 2:
                if get_cfg_group().rank_in_group == 0:
                    context = task.buffer.text_embeddings
                else:
                    context = task.buffer.negative_embeddings

                cfg_partial_noise_pred = model(
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=task.buffer.seq_len
                )
                noise_pred_cond, noise_pred_uncond = self.cfg_dispatcher.all_gather_cfg_noise_preds(cfg_partial_noise_pred)
            else:
                noise_pred_cond = model(
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.text_embeddings,
                    seq_len=task.buffer.seq_len
                )
                noise_pred_uncond = model(
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.negative_embeddings,
                    seq_len=task.buffer.seq_len
                )
            noise_pred = noise_pred_uncond + \
                noise_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model(
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
        # Prepare latents on rank 0, only data in rank 0 would be used.
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
                        target_shape[1] / self.cp_size) * self.cp_size
    
        # Prepare Solver and Timestep on main rank
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
        
        task.buffer.sampler = sample_scheduler
        task.buffer.seed_g = seed_g
        task.buffer.latents = latents
        task.buffer.timesteps = timesteps
        task.buffer.seq_len = seq_len
        
        # FIXME: Reduce specified branch
        if DiffusionBackend.args.models.name in ["Wan2.2-T2V-A14B"]:
            DiffusionBackend.low_noise_model.to(device)
            DiffusionBackend.high_noise_model.to(device)    
        else:
            DiffusionBackend.model.to(device)

        logger.info(f"[Pre Denoise] Init {latents.shape=} {timesteps=}")


    def _update_task_stage_and_buffer(self, task: DiffusionTask, tokens: torch.Tensor):
        has_img = task.req.init_image is not None
        if task.status == DiffusionTaskStatus.Running:
            task.status = DiffusionTaskStatus.Pending
        else:
            logger.warning(f"Task {task.task_id} - Status: {task.status}")
        
        logger.info(f"[Task] current stage: {task.task_type}")
        
        # 处理Text Encode阶段
        if task.task_type == DiffusionTaskType.TextEncode:
            if self.cfg_size == 1:
                if task.buffer.text_embeddings is None:
                    # 首次text encode (正向prompt)
                    task.buffer.text_embeddings = tokens
                    if task.do_cfg:
                        # CFG模式需要第二次encode negative prompt
                        return True
                else:
                    # 第二次text encode（仅CFG模式）
                    if task.do_cfg:
                        task.buffer.negative_embeddings = tokens
            elif self.cfg_size == 2:
                if get_cfg_group().rank_in_group == 0:
                    task.buffer.text_embeddings = tokens
                else:
                    task.buffer.negative_embeddings = tokens
                
            # Text Encode完成，转换到下一阶段
            if has_img:
                task.task_type = DiffusionTaskType.VAEEncode 
            else:
                task.task_type = DiffusionTaskType.Denoise
                self._pre_denoising(task)
            
            # 如果转换到Denoise阶段，初始化denoise相关参数
            if task.task_type == DiffusionTaskType.Denoise:
                task.buffer.current_step = 0
                task.num_inference_steps = task.req.params.num_inference_steps
                
            logger.debug(f"Task {task.task_id} transitioned to {task.task_type}")
            return True
        
        # 处理VAE Encode阶段
        elif task.task_type == DiffusionTaskType.VAEEncode:
            task.buffer.latents = tokens  # 保存编码后的latents
            
            # 转换到Denoise阶段
            task.task_type = DiffusionTaskType.Denoise
            
            # 初始化denoise相关参数            
            logger.debug(f"Task {task.task_id} transitioned to {task.task_type}")
            return True
        
        # 处理Denoise阶段（关键：需要多次执行）
        elif task.task_type == DiffusionTaskType.Denoise:
            # 更新当前去噪后的latents
            task.buffer.latents = tokens
            task.buffer.current_step += 1
            
            logger.debug(f"Task {task.task_id} denoise step {task.buffer.current_step}/{task.num_inference_steps}")
            
            # 检查是否完成所有denoise步骤
            if task.buffer.current_step >= task.num_inference_steps:
                # 所有denoise步骤完成，转换到VAE Decode
                task.task_type = DiffusionTaskType.VAEDecode
                logger.debug(f"Task {task.task_id} completed denoising, transitioned to {task.task_type}")
                return True
            else:
                # 还需要继续denoise，保持当前阶段但状态改为Pending等待下次调度
                logger.debug(f"Task {task.task_id} continuing denoise step {task.buffer.current_step}/{task.num_inference_steps}")
                return True
        
        # 处理VAE Decode阶段（最终阶段）
        elif task.task_type == DiffusionTaskType.VAEDecode:
            task.buffer.generated_image = tokens  # 保存最终生成的图像
            logger.debug(f"Task {task.task_id} completed all stages")
            task.status = DiffusionTaskStatus.Completed
            self.current_task = None
            return False  # 完成所有阶段，不需要继续调度
        
        # 未知阶段
        else:
            logger.error(f"Unknown task type: {task.task_type}")
            task.req.finish_reason = "error"
            task.status = DiffusionTaskStatus.Failed
            return False
        
    
