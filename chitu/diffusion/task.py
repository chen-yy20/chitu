# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
import torch
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Any, Optional, Union, Dict, List, Deque
from pathlib import Path
from collections import deque

from chitu.task import TaskLoad
from chitu.diffusion.backend import DiffusionBackend

logger = getLogger(__name__)


import time
import torch
from enum import Enum
from logging import getLogger
from typing import Optional, List
from dataclasses import dataclass

logger = getLogger(__name__)


class DiffusionTaskType(Enum):
    TextEncode = 1
    VAEEncode = 2
    Denoise = 3
    VAEDecode = 4


class DiffusionTaskStatus(Enum):
    Pending = 1     # 任务创建，等待执行
    Running = 2     # 任务执行中
    Completed = 3   # 任务完成
    Failed = 4      # 任务失败


@dataclass
class DiffusionParams:
    """Diffusion生成参数"""
    size: tuple[int, int] = (512, 512)
    frame_num: int = 81
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    # 调度器参数
    sample_solver: str = "ddpm"
    # 其他参数
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    sample_shift: float = 5.0,
    # clip_skip: int = 1
    # strength: float = 1.0  # for img2img
    save_dir: Optional[str] = "./output"  # 输出保存路径


class DiffusionUserRequest:
    """用户请求封装"""
    
    def __init__(
        self,
        message,
        request_id,
        txt_emb = None,
        img_emb = None,
        latents = None,
        init_image: Optional[torch.Tensor] = None,  # for img2img
        mask: Optional[torch.Tensor] = None,        # for inpainting
        params: Optional[DiffusionParams] = None,
    ):
        self.message = message
        self.request_id = request_id
        self.txt_emb = txt_emb
        self.img_emb = img_emb
        self.latents = latents
        self.init_image = init_image
        self.mask = mask
        self.params = params if params is not None else DiffusionParams()
        
        # 时间戳
        self.created_time = time.monotonic()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # 结果
        self.finish_reason: Optional[str] = None

    def get_role(self):
        return self.message[0]["role"] if isinstance(self.message, list) and len(self.message) > 0 else "none"
    
    def get_prompt(self):
        return self.message[0]["content"] if isinstance(self.message, list) and len(self.message) > 0 else ""

    def get_n_prompt(self):
        if self.params.negative_prompt is not None:
            return self.params.negative_prompt
        return ""


    def __repr__(self):
        return f"DiffusionUserRequest(id={self.request_id}, message={self.message})"

@dataclass
class DiffusionTaskBuffer:
    """存储扩散任务的缓冲区数据"""
    # Text encode buffers
    text_embeddings: Optional[torch.Tensor] = field(default=None)
    negative_embeddings: Optional[torch.Tensor] = field(default=None)
    seq_len: Optional[int] = field(default=None)
    
    # Denoise buffers
    seed_g: Optional[torch.Generator] = field(default=None)
    sampler: Optional[Any] = field(default=None)
    latents: Optional[torch.Tensor] = field(default=None)
    timesteps: Optional[List[int]] = field(default=None)
    current_step: int = field(default=0)
    denoised_latents: Optional[torch.Tensor] = field(default=None)
    
    # VAE Decode buffers
    generated_image: Optional[torch.Tensor] = field(default=None)



class DiffusionTask:
    """Diffusion生成任务"""
    
    def __init__(
        self,
        task_id: str,
        req: DiffusionUserRequest,
        priority: int = 1,
    ):
        logger.debug(f"Create DiffusionTask {task_id} with priority {priority}")
        
        # 基本信息
        self.task_id = task_id
        self.task_type = DiffusionTaskType.TextEncode  # T2V task is always text encode
        self.req = req
        self.priority = priority
        self.status = DiffusionTaskStatus.Pending
        self.num_inference_steps = req.params.num_inference_steps
        self.do_cfg = req.params.guidance_scale > 0
        
        # 时间戳
        self.created_time = time.perf_counter_ns()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.sched_ts: Optional[int] = None  # 调度时间戳，与原LLM代码保持一致
        
        # 调度相关（保持与原代码一致性）
        self.waiting = False  # 是否在等待状态
        self.sched_group_id: Optional[str] = None  # 调度组ID

        # 输入输出
        self.input_data: Optional[torch.Tensor] = None
        self.output_data: Optional[torch.Tensor] = None
        
        # 任务特定数据
        self.buffer = DiffusionTaskBuffer()
        
        # 错误信息
        self.error_message: Optional[str] = None

    def start(self):
        """开始执行任务"""
        self.status = DiffusionTaskStatus.Running
        self.start_time = time.monotonic()
        if self.req.start_time is None:
            self.req.start_time = self.start_time
        logger.debug(f"Task {self.task_id} started")

    def complete(self, output_data: Optional[torch.Tensor] = None):
        """完成任务"""
        self.status = DiffusionTaskStatus.Completed
        self.end_time = time.monotonic()
        if output_data is not None:
            self.output_data = output_data
        logger.debug(f"Task {self.task_id} completed")

    def fail(self, error_message: str):
        """任务失败"""
        self.status = DiffusionTaskStatus.Failed
        self.end_time = time.monotonic()
        self.error_message = error_message
        self.req.finish_reason = "error"
        logger.error(f"Task {self.task_id} failed: {error_message}")

    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.status in [DiffusionTaskStatus.Completed, DiffusionTaskStatus.Failed]

    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        return self.status == DiffusionTaskStatus.Running

    def need_remove(self) -> bool:
        """检查任务是否需要从池中移除（保持与原LLM代码一致性）"""
        return self.status in [DiffusionTaskStatus.Completed, DiffusionTaskStatus.Failed]

    def can_schedule(self) -> bool:
        """检查任务是否可以被调度"""
        return (
            self.status == DiffusionTaskStatus.Pending 
            and not self.waiting
            and not self.need_remove()
        )

    def wait(self, handle=None):
        """设置任务为等待状态（保持与原LLM代码一致性）"""
        self.waiting = True
        logger.debug(f"Task {self.task_id} is now waiting")

    def unwait(self):
        """取消任务等待状态（保持与原LLM代码一致性）"""
        if self.waiting:
            self.waiting = False
            logger.debug(f"Task {self.task_id} is no longer waiting")

    def get_execution_time(self) -> Optional[float]:
        """获取执行时间（秒）"""
        if self.start_time is None:
            return None
        end_time = self.end_time if self.end_time is not None else time.monotonic()
        return end_time - self.start_time

    def update_denoise_progress(self, current_step: int, denoised_latents: torch.Tensor):
        """更新去噪进度（仅用于Denoise任务）"""
        if self.task_type != DiffusionTaskType.Denoise:
            return
        self.current_step = current_step
        self.denoised_latents = denoised_latents

    def get_denoise_progress(self) -> float:
        """获取去噪进度百分比"""
        if self.task_type != DiffusionTaskType.Denoise:
            return 0.0
        return self.current_step / max(self.num_inference_steps, 1)

    def update_stage_and_buffer(self, tokens) -> bool:
        """转换到下一个处理阶段并更新缓冲区
        
        Args:
            tokens: 当前阶段的输出tokens/latents
        
        Returns:
            bool: 是否需要继续调度（False表示已完成所有阶段）
        """
        has_img = self.req.init_image is not None
        
        logger.info(f"[Task] current stage: {self.task_type}")
        
        # 处理Text Encode阶段
        if self.task_type == DiffusionTaskType.TextEncode:
            if self.buffer.text_embeddings is None:
                # 首次text encode (正向prompt)
                self.buffer.text_embeddings = tokens
                if self.do_cfg:
                    # CFG模式需要第二次encode negative prompt
                    return True
            else:
                # 第二次text encode（仅CFG模式）
                if self.do_cfg:
                    self.buffer.negative_embeddings = tokens
                
            # Text Encode完成，转换到下一阶段
            self.task_type = DiffusionTaskType.VAEEncode if has_img else DiffusionTaskType.Denoise
            self.status = DiffusionTaskStatus.Pending
            
            # 如果转换到Denoise阶段，初始化denoise相关参数
            if self.task_type == DiffusionTaskType.Denoise:
                self.buffer.current_step = 0
                self.num_inference_steps = self.req.params.num_inference_steps
                
            logger.debug(f"Task {self.task_id} transitioned to {self.task_type}")
            return True
        
        # 处理VAE Encode阶段
        elif self.task_type == DiffusionTaskType.VAEEncode:
            self.buffer.latents = tokens  # 保存编码后的latents
            
            # 转换到Denoise阶段
            self.task_type = DiffusionTaskType.Denoise
            self.status = DiffusionTaskStatus.Pending
            
            # 初始化denoise相关参数            
            logger.debug(f"Task {self.task_id} transitioned to {self.task_type}")
            return True
        
        # 处理Denoise阶段（关键：需要多次执行）
        elif self.task_type == DiffusionTaskType.Denoise:
            # 更新当前去噪后的latents
            self.buffer.latents = tokens
            self.buffer.current_step += 1
            
            logger.debug(f"Task {self.task_id} denoise step {self.buffer.current_step}/{self.num_inference_steps}")
            
            # 检查是否完成所有denoise步骤
            if self.buffer.current_step >= self.num_inference_steps:
                # 所有denoise步骤完成，转换到VAE Decode
                self.task_type = DiffusionTaskType.VAEDecode
                self.status = DiffusionTaskStatus.Pending
                logger.debug(f"Task {self.task_id} completed denoising, transitioned to {self.task_type}")
                return True
            else:
                # 还需要继续denoise，保持当前阶段但状态改为Pending等待下次调度
                self.status = DiffusionTaskStatus.Pending
                logger.debug(f"Task {self.task_id} continuing denoise step {self.buffer.current_step}/{self.num_inference_steps}")
                return True
        
        # 处理VAE Decode阶段（最终阶段）
        elif self.task_type == DiffusionTaskType.VAEDecode:
            self.buffer.generated_image = tokens  # 保存最终生成的图像
            self.req.finish_reason = "completed"
            logger.debug(f"Task {self.task_id} completed all stages")
            self.status = DiffusionTaskStatus.Completed
            return False  # 完成所有阶段，不需要继续调度
        
        # 未知阶段
        else:
            logger.error(f"Unknown task type: {self.task_type}")
            self.req.finish_reason = "error"
            self.status = DiffusionTaskStatus.Failed
            return False

    def get_denoise_progress(self) -> float:
        """获取去噪进度百分比"""
        if self.task_type != DiffusionTaskType.Denoise:
            return 0.0
        return self.buffer.current_step / max(self.num_inference_steps, 1)

    def get_pipeline_progress(self) -> float:
        """获取整个流水线的进度百分比"""
        # 定义各个阶段的权重
        stage_weights = {
            DiffusionTaskType.TextEncode: 0.1,    # 10%
            DiffusionTaskType.VAEEncode: 0.1,     # 10% (仅img2img)
            DiffusionTaskType.Denoise: 0.7,      # 70%
            DiffusionTaskType.VAEDecode: 0.1,    # 10%
        }
        
        base_progress = 0.0
        
        # 计算已完成阶段的进度
        if self.task_type == DiffusionTaskType.TextEncode:
            if self.status == DiffusionTaskStatus.Completed:
                base_progress = stage_weights[DiffusionTaskType.TextEncode]
            elif self.status == DiffusionTaskStatus.Running:
                # Text encode阶段内部进度
                if self.do_cfg and self.buffer.text_embeddings is not None:
                    base_progress = stage_weights[DiffusionTaskType.TextEncode] * 0.75
                else:
                    base_progress = stage_weights[DiffusionTaskType.TextEncode] * 0.5
                    
        elif self.task_type == DiffusionTaskType.VAEEncode:
            base_progress = stage_weights[DiffusionTaskType.TextEncode]
            if self.status == DiffusionTaskStatus.Completed:
                base_progress += stage_weights[DiffusionTaskType.VAEEncode]
            elif self.status == DiffusionTaskStatus.Running:
                base_progress += stage_weights[DiffusionTaskType.VAEEncode] * 0.5
                
        elif self.task_type == DiffusionTaskType.Denoise:
            base_progress = stage_weights[DiffusionTaskType.TextEncode]
            if self.req.init_image is not None:
                base_progress += stage_weights[DiffusionTaskType.VAEEncode]
            
            # 去噪进度更细粒度
            denoise_progress = self.get_denoise_progress()
            base_progress += stage_weights[DiffusionTaskType.Denoise] * denoise_progress
            
        elif self.task_type == DiffusionTaskType.VAEDecode:
            base_progress = stage_weights[DiffusionTaskType.TextEncode]
            if self.req.init_image is not None:
                base_progress += stage_weights[DiffusionTaskType.VAEEncode]
            base_progress += stage_weights[DiffusionTaskType.Denoise]
            
            if self.status == DiffusionTaskStatus.Completed:
                base_progress += stage_weights[DiffusionTaskType.VAEDecode]
            elif self.status == DiffusionTaskStatus.Running:
                base_progress += stage_weights[DiffusionTaskType.VAEDecode] * 0.5
        
        return min(base_progress, 1.0)

    def is_final_stage(self) -> bool:
        """检查是否为最后一个处理阶段"""
        return self.task_type == DiffusionTaskType.VAEDecode

    def get_current_stage_name(self) -> str:
        """获取当前阶段的可读名称"""
        stage_names = {
            DiffusionTaskType.TextEncode: "Text Encoding",
            DiffusionTaskType.VAEEncode: "VAE Encoding", 
            DiffusionTaskType.Denoise: "Denoising",
            DiffusionTaskType.VAEDecode: "VAE Decoding"
        }
        return stage_names.get(self.task_type, "Unknown")

    def __repr__(self):
        return (
            f"DiffusionTask(id={self.task_id}, type={self.task_type}, "
            f"status={self.status}, priority={self.priority})"
        )


class DiffusionTaskPool:
    pool: dict[str, DiffusionTask] = {}
    id_list: list[str] = []
    pending_queue: deque[DiffusionTask] = Deque()

    def __bool__(self):
        return len(self.pool) > 0

    def __len__(self):
        return len(self.pool)

    @classmethod
    def reset(cls):
        cls.pool = {}
        cls.id_list = []

    @classmethod
    def is_empty(cls):
        return len(cls.pool) == 0

    @classmethod
    def all_finished(cls):
        return len(cls.pool) == 0

    @classmethod
    def add(cls, task: DiffusionTask):
        if task.task_id in cls.pool:
            return False  # Task already exists, failed to add
        cls.pool[task.task_id] = task
        cls.id_list.append(task.task_id)
        return True

    @classmethod
    def enqueue(cls, task: DiffusionTask):
        cls.pending_queue.append(task)

    @classmethod
    def add_all_queued(cls):
        while cls.pending_queue:
            cls.add(cls.pending_queue.popleft())

    @classmethod
    def remove(cls, task_id: str):
        assert task_id in cls.pool, "Task not found in pool"
        if cls.pool.pop(task_id) is None:
            raise ValueError(f"Task {task_id} not found in pool")
        cls.id_list.remove(task_id)
        if len(cls.pool) == 0:
            TaskLoad.clear()