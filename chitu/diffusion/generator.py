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
from tqdm import tqdm

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
from chitu.diffusion.utils.flux2_utils import (
    batched_prc_img,
    batched_prc_txt,
    get_schedule,
    scatter_ids,
    save_image_as_png,
)


logger = getLogger(__name__)

from contextlib import contextmanager

@contextmanager
def device_scope(model: torch.nn.Module):
    # low_mem = getattr(DiffusionBackend.args.infer.diffusion, "low_memory", False)
    original_device = None
    
    if model is not None and torch.cuda.is_available():
        # 记录原始设备
        original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else None
        model.to(torch.cuda.current_device())
    
    try:
        DiffusionBackend.memory_used(f"Loaded model to {torch.cuda.current_device()}")
        yield model
    finally:
        if model is not None:
            # 恢复到原始设备，而不是强制CPU
            target_device = original_device if original_device is not None else "cpu"
            model.to(target_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            DiffusionBackend.memory_used(f"Offloaded model to {target_device}")

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

        for model in DiffusionBackend.model_pool:
            model.model_compute = create_wrapped_forward(model)

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
        # payload 是本次task需要处理的数据的抽象
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

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            with device_scope(DiffusionBackend.text_encoder.model):
                out = DiffusionBackend.text_encoder([payload])
            ctx, ctx_ids = batched_prc_txt(out.to(torch.bfloat16))
            task.buffer.ctx_ids = ctx_ids
            logger.info(f"[text_encode_step] FLUX2 ctx shape: {ctx.shape}, ctx_ids shape: {ctx_ids.shape}")
            return ctx

        with device_scope(DiffusionBackend.text_encoder.model):        
            out = DiffusionBackend.text_encoder(payload, torch.cuda.current_device())
            
        logger.info(f"[text_encode_step] context shape: {out.shape}")
        return out
    
    def vae_encode_step(self, task: DiffusionTask):
        pass
    
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_step(self, task: DiffusionTask):
        assert task.buffer.latents is not None and task.buffer.timesteps is not None

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            """Single Euler denoising step for FLUX2 (guidance-distilled, no CFG)."""
            x = task.buffer.latents
            t_curr = task.buffer.timesteps[task.buffer.current_step]
            t_prev = task.buffer.timesteps[task.buffer.current_step + 1]

            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

            pred = DiffusionBackend.active_model(
                x=x,
                x_ids=task.buffer.x_ids,
                timesteps=t_vec,
                ctx=task.buffer.text_embeddings,
                ctx_ids=task.buffer.ctx_ids,
                guidance=task.buffer.guidance_vec,
            )

            sampled_latents = x + (t_prev - t_curr) * pred
            return sampled_latents

        latent_model_input = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]

        if DiffusionBackend.guidance_scale > 0: # Wan的cfg是做两次
            if self.cfg_size == 2:
                if get_cfg_group().rank_in_group == 0:
                    context = task.buffer.text_embeddings
                else:
                    context = task.buffer.negative_embeddings

                cfg_partial_noise_pred = DiffusionBackend.active_model(
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=task.buffer.seq_len
                )
                noise_pred_cond, noise_pred_uncond = self.cfg_dispatcher.all_gather_cfg_noise_preds(cfg_partial_noise_pred)
            else:
                noise_pred_cond = DiffusionBackend.active_model(
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.text_embeddings,
                    seq_len=task.buffer.seq_len
                )
                noise_pred_uncond = DiffusionBackend.active_model(
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.negative_embeddings,
                    seq_len=task.buffer.seq_len
                )
            noise_pred = noise_pred_uncond + \
                DiffusionBackend.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = DiffusionBackend.active_model(
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

        # logger.info(f"[Denoise Step] {task.buffer.current_step}/"
        #             f"{task.req.params.num_inference_steps} "
        #             f"timestep: {timestep} latents shape: {sampled_latents.shape}, {sampled_latents.dtype=}")

        return sampled_latents

        
    
    def vae_decode_step(self, task: DiffusionTask):
        target_decode_device = 0 # TODO: Flexible vae device
        if torch.distributed.get_rank() == target_decode_device:
            payload = [task.buffer.latents]
            logger.info(f"Step {task.buffer.current_step}: Enter VAE Decode Stage!")

            if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
                x = torch.cat(scatter_ids(task.buffer.latents, task.buffer.x_ids)).squeeze(2)
                with device_scope(DiffusionBackend.vae.model):
                    img = DiffusionBackend.vae.decode(x).float()
                self._save_image(task, img)
                return img

            with device_scope(DiffusionBackend.vae.model):
                video = DiffusionBackend.vae.decode(payload)[0]
            self._save_video(task, video)
            return video
        return None

    def _save_video(self, task: DiffusionTask, video: torch.Tensor):
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

    def _save_image(self, task: DiffusionTask, img: torch.Tensor):
        os.makedirs(task.req.params.save_dir, exist_ok=True)
        save_name = task.req.get_prompt()[:20].replace(" ", "_").replace(".", "") \
                    + f"_{task.task_id}.png"
        save_path = os.path.join(task.req.params.save_dir, save_name)
        logger.info(f"Saving image: {img.shape=} {img.dtype=}")
        save_image_as_png(img[0], save_path)
        logger.info(f"[Succeed] Task {task.task_id} image saved to {save_path}")
        
    def _pre_denoising(self, task: DiffusionTask):
        """
        Before denoising loop, prepare latents, timesteps and solver for one task.
        TODO: Control Devices
        """

        device = torch.cuda.current_device()

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            """Prepare FLUX2 latents (packed 2D), Euler schedule, and guidance vector."""

            height, width = task.req.params.size
            shape = (1, 128, height // 16, width // 16)

            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(task.req.params.seed)

            randn = torch.randn(shape, generator=seed_g, dtype=torch.bfloat16, device=device)
            x, x_ids = batched_prc_img(randn)

            timesteps = get_schedule(task.req.params.num_inference_steps, x.shape[1])

            guidance = DiffusionBackend.args.models.sampler.guidance_scale[0]
            guidance_vec = torch.full((x.shape[0],), guidance, device=device, dtype=x.dtype)

            task.buffer.seed_g = seed_g
            task.buffer.latents = x
            task.buffer.x_ids = x_ids
            task.buffer.timesteps = timesteps
            task.buffer.guidance_vec = guidance_vec

            DiffusionBackend.switch_active_model(flush=True)
            logger.info(f"[Pre Denoise FLUX2] x={x.shape}, x_ids={x_ids.shape}, "
                        f"timesteps={len(timesteps)} steps, guidance={guidance}")
            return

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
                num_train_timesteps=DiffusionBackend.args.models.sampler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                task.req.params.num_inference_steps, 
                device=device,
                shift=DiffusionBackend.args.models.sampler.sample_shift
                )
            timesteps = sample_scheduler.timesteps
        elif task.req.params.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=DiffusionBackend.args.models.sampler.num_train_timesteps, 
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(task.req.params.num_inference_steps, DiffusionBackend.args.models.sampler.sample_shift)
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
        
        DiffusionBackend.switch_active_model(flush=True)

        # logger.info(f"[Pre Denoise] Init {latents.shape=} {timesteps=}")


    def _update_task_stage_and_buffer(self, task: DiffusionTask, tokens: torch.Tensor):

        is_main_process = dist.get_rank() == 0
        has_img = task.req.init_image is not None
        
        if task.status == DiffusionTaskStatus.Running:
            task.status = DiffusionTaskStatus.Pending
        elif is_main_process:
            logger.warning(f"Task {task.task_id} - Status: {task.status}")
        
        # 处理Text Encode阶段
        if task.task_type == DiffusionTaskType.TextEncode:
            if is_main_process:
                logger.info(f"[Task] current stage: {task.task_type}")
                
            if self.cfg_size == 1:
                if task.buffer.text_embeddings is None:
                    # 首次text encode (正向prompt)
                    task.buffer.text_embeddings = tokens
                    if DiffusionBackend.do_cfg:
                        # CFG模式需要第二次encode negative prompt
                        return True
                else:
                    # 第二次text encode（仅CFG模式）
                    if DiffusionBackend.do_cfg:
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
                
            if is_main_process:
                logger.debug(f"Task {task.task_id} transitioned to {task.task_type}")
            return True
        
        # 处理VAE Encode阶段
        elif task.task_type == DiffusionTaskType.VAEEncode:
            if is_main_process:
                logger.info(f"[Task] current stage: {task.task_type}")
            task.buffer.latents = tokens  # 保存编码后的latents
            
            # 转换到Denoise阶段
            task.task_type = DiffusionTaskType.Denoise
            
            if is_main_process:
                logger.debug(f"Task {task.task_id} transitioned to {task.task_type}")
            return True
        
        # 处理Denoise阶段（关键：需要多次执行）
        elif task.task_type == DiffusionTaskType.Denoise:
            # 只在主进程的开始denoise时显示阶段信息和初始化进度条
            if task.buffer.current_step == 0:
                if is_main_process:
                    logger.info(f"[Task] current stage: {task.task_type}")
                    # 将进度条作为task的属性
                    task.progress_bar = tqdm(
                        total=task.req.params.num_inference_steps,
                        desc=f"Task {task.task_id} Denoising",
                        unit="steps",
                        dynamic_ncols=True,  # 自适应终端宽度
                        leave=True  # 保持进度条显示
                    )
            
            # 更新当前去噪后的latents
            task.buffer.latents = tokens
            task.buffer.current_step += 1
            
            # 主进程更新进度条
            # 主进程更新进度条
            if is_main_process and hasattr(task, 'progress_bar') and task.progress_bar is not None:
                task.progress_bar.update(1)
                task.progress_bar.refresh()  # 强制刷新显示

            # 检查是否完成所有denoise步骤
            if task.buffer.current_step >= task.req.params.num_inference_steps:
                # 主进程关闭进度条
                if is_main_process and hasattr(task, 'progress_bar') and task.progress_bar is not None:
                    task.progress_bar.close()
                    task.progress_bar = None

                # 所有denoise步骤完成，转换到VAE Decode
                task.task_type = DiffusionTaskType.VAEDecode
                if is_main_process:
                    logger.debug(f"Task {task.task_id} completed denoising, transitioned to {task.task_type}")
                return True
            elif DiffusionBackend.boundary is not None and task.buffer.current_step >= DiffusionBackend.boundary * task.req.params.num_inference_steps:
                # 检查是否需要切换模型
                DiffusionBackend.switch_active_model(flush=False)
            else:
                # 还需要继续denoise，保持当前阶段但状态改为Pending等待下次调度
                if is_main_process:
                    logger.debug(f"Task {task.task_id} continuing denoise")
                return True
        
        # 处理VAE Decode阶段（最终阶段）
        elif task.task_type == DiffusionTaskType.VAEDecode:
            if is_main_process:
                logger.info(f"[Task] current stage: {task.task_type}")
                # 确保进度条被正确关闭
                if hasattr(task, 'progress_bar') and task.progress_bar is not None:
                    task.progress_bar.close()
                    task.progress_bar = None
                    
            task.buffer.generated_image = tokens  # 保存最终生成的图像
            if is_main_process:
                logger.debug(f"Task {task.task_id} completed all stages")
            task.status = DiffusionTaskStatus.Completed
            self.current_task = None
            return False  # 完成所有阶段，不需要继续调度
        
        # 未知阶段
        else:
            if is_main_process:
                if self.progress_bar:
                    self.progress_bar.close()
                logger.error(f"Unknown task type: {task.task_type}")
            task.req.finish_reason = "error"
            task.status = DiffusionTaskStatus.Failed
            return False
        
    
