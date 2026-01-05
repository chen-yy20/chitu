# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.distributed as dist
import pickle
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Any, Optional, Union, Dict, List, Deque, Tuple
from pathlib import Path
from collections import deque

from chitu.task import TaskLoad
from chitu.diffusion.backend import DiffusionBackend
from chitu.distributed.parallel_state import get_cfg_group

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
    Terminate = 5


class DiffusionTaskStatus(Enum):
    Pending = 1     # 任务创建，等待执行
    Running = 2     # 任务执行中
    Completed = 3   # 任务完成
    Failed = 4      # 任务失败


@dataclass
class DiffusionUserParams:
    """Diffusion生成参数"""
    role: str = "user"
    size: tuple[int, int] = (512, 512)
    frame_num: int = 81
    prompt: str = None
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
        request_id,
        params: DiffusionUserParams = None,
        init_image: Optional[torch.Tensor] = None,  # for i2v
        # txt_emb = None,
        # img_emb = None,
        # latents = None,
        # mask: Optional[torch.Tensor] = None,        # for inpainting
        
    ):
        self.request_id = request_id
        self.params = params
        self.init_image = init_image

    def get_role(self):
        return self.params.role
    
    def get_prompt(self):
        return self.params.prompt

    def get_n_prompt(self):
        if self.params.negative_prompt is not None:
            return self.params.negative_prompt
        return ""

    def __repr__(self):
        return f"DiffusionUserRequest(id={self.request_id}, request={self.params})"

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
    
    def __init__(
        self,
        task_id: str,
        task_type: DiffusionTaskType = None,
        req: Optional[DiffusionUserRequest] = None,
        buffer: Optional[DiffusionTaskBuffer] = None,
        signal_data: Optional[Dict] = None, # 系统信号携带的数据

    ):
        logger.debug(f"Create DiffusionTask {task_id}")
        
        # 基本信息
        self.task_id = task_id
        self.task_type = DiffusionTaskType.TextEncode if task_type is None else task_type # T2V task is always text encode
        self.status = DiffusionTaskStatus.Pending

        self.req = req
        self.do_cfg = req.params.guidance_scale > 0 if req is not None else False
        self.buffer = DiffusionTaskBuffer() if buffer is None else buffer
         # 系统信号数据
        self.signal_data = signal_data or {}
        
        # 错误信息
        self.error_message: Optional[str] = None

    @classmethod
    def create_terminate_signal(
        cls, 
        task_id: str = None,
        reason: str = "Normal shutdown"
    ) -> 'DiffusionTask':
        """
        创建终止信号任务
        
        Args:
            task_id: 任务ID，如果为None则自动生成
            reason: 终止原因
            
        Returns:
            DiffusionTask: 终止信号任务
        """
        if task_id is None:
            task_id = f"terminate_signal_{int(time.time() * 1000)}"
        
        return cls(
            task_id=task_id,
            task_type=DiffusionTaskType.Terminate,
            req=None,  # 终止信号没有用户请求
            buffer=None,  # 终止信号不需要buffer
            signal_data={'reason': reason, 'timestamp': time.time()}
        )

    def is_terminate_signal(self) -> bool:
        """检查是否为终止信号"""
        return self.task_type == DiffusionTaskType.Terminate

    def is_completed(self) -> bool:
        """检查任务是否完成"""
        if self.is_terminate_signal():
            return True
        return self.status in [DiffusionTaskStatus.Completed, DiffusionTaskStatus.Failed]

    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        if self.is_terminate_signal():
            return False
        return self.status == DiffusionTaskStatus.Running


    def update_stage_and_buffer(self, tokens) -> bool:
        """转换到下一个处理阶段并更新缓冲区
        
        Args:
            tokens: 当前阶段的输出tokens/latents
        
        Returns:
            bool: 是否需要继续调度（False表示已完成所有阶段）
        """
        if self.is_terminate_signal():
            logger.debug(f"Terminate signal {self.task_id} processed")
            self.status = DiffusionTaskStatus.Completed
            return False
        
        has_img = self.req.init_image is not None
        if self.status == DiffusionTaskStatus.Running:
            self.status = DiffusionTaskStatus.Pending
        else:
            logger.warning(f"Task {self.task_id} - Status: {self.status}")
        
        logger.info(f"[Task] current stage: {self.task_type}")
        
        # 处理Text Encode阶段
        if self.task_type == DiffusionTaskType.TextEncode:
            if get_cfg_group().group_size == 1:
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
            elif get_cfg_group().group_size == 2:
                if get_cfg_group().rank_in_group == 0:
                    self.buffer.text_embeddings = tokens
                else:
                    self.buffer.negative_embeddings = tokens
                
            # Text Encode完成，转换到下一阶段
            self.task_type = DiffusionTaskType.VAEEncode if has_img else DiffusionTaskType.Denoise
            
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
                logger.debug(f"Task {self.task_id} completed denoising, transitioned to {self.task_type}")
                return True
            else:
                # 还需要继续denoise，保持当前阶段但状态改为Pending等待下次调度
                logger.debug(f"Task {self.task_id} continuing denoise step {self.buffer.current_step}/{self.num_inference_steps}")
                return True
        
        # 处理VAE Decode阶段（最终阶段）
        elif self.task_type == DiffusionTaskType.VAEDecode:
            self.buffer.generated_image = tokens  # 保存最终生成的图像
            logger.debug(f"Task {self.task_id} completed all stages")
            self.status = DiffusionTaskStatus.Completed
            return False  # 完成所有阶段，不需要继续调度
        
        # 未知阶段
        else:
            logger.error(f"Unknown task type: {self.task_type}")
            self.status = DiffusionTaskStatus.Failed
            return False


    def __repr__(self):
        return (
            f"DiffusionTask(id={self.task_id}, type={self.task_type}, "
            f"status={self.status}"
        )
    
    # ================= 通信相关 ====================
    def serialize(self, device: str = "cpu") -> torch.Tensor:
        """将DiffusionTask序列化为torch.Tensor"""
        try:
            # 1. 准备基本数据
            serializable_data = {
                # 基本信息
                'task_id': self.task_id,
                'task_type': self.task_type,
                'status': self.status,
                'do_cfg': self.do_cfg,
                'error_message': self.error_message,
                'is_terminate_signal': self.is_terminate_signal(),
                'signal_data': self.signal_data,  # 终止信号数据
            }
            
            # 2. 处理用户请求数据（仅当req不为None时）
            if self.req is not None:
                serializable_data['user_params'] = {
                    'role': self.req.params.role,
                    'size': self.req.params.size,
                    'frame_num': self.req.params.frame_num,
                    'prompt': self.req.params.prompt,
                    'negative_prompt': self.req.params.negative_prompt,
                    'seed': self.req.params.seed,
                    'sample_solver': self.req.params.sample_solver,
                    'num_inference_steps': self.req.params.num_inference_steps,
                    'guidance_scale': self.req.params.guidance_scale,
                    'sample_shift': self.req.params.sample_shift,
                    'save_dir': self.req.params.save_dir,
                }
            else:
                serializable_data['user_params'] = None
            
            # 3. 处理Buffer数据（仅当buffer不为None时）
            if self.buffer is not None:
                serializable_data['buffer_metadata'] = {
                    'seq_len': self.buffer.seq_len,
                    'current_step': self.buffer.current_step,
                    'timesteps': self.buffer.timesteps,
                }
            else:
                serializable_data['buffer_metadata'] = None
            
            # 4. 序列化tensor数据（仅当buffer存在时）
            tensor_data = {}
            if self.buffer is not None:
                if self.buffer.text_embeddings is not None:
                    tensor_data['text_embeddings'] = self.buffer.text_embeddings.detach().clone()
                if self.buffer.negative_embeddings is not None:
                    tensor_data['negative_embeddings'] = self.buffer.negative_embeddings.detach().clone()
                if self.buffer.latents is not None:
                    tensor_data['latents'] = self.buffer.latents.detach().clone()
                if self.buffer.denoised_latents is not None:
                    tensor_data['denoised_latents'] = self.buffer.denoised_latents.detach().clone()
                if self.buffer.generated_image is not None:
                    tensor_data['generated_image'] = self.buffer.generated_image.detach().clone()
            
            # 5. 打包所有数据
            full_data = {
                'metadata': serializable_data,
                'tensors': tensor_data
            }
            
            # 6. 使用pickle序列化
            serialized_bytes = pickle.dumps(full_data)
            
            # 7. 转换为torch.Tensor并移到目标设备
            serialized_array = torch.frombuffer(serialized_bytes, dtype=torch.uint8)
            final_tensor = serialized_array.to(device)
            
            logger.debug(f"Serialized {'terminate signal' if self.is_terminate_signal() else 'task'} {self.task_id}, size: {len(serialized_array)} bytes")
            return final_tensor
            
        except Exception as e:
            logger.error(f"Failed to serialize task {self.task_id}: {e}")
            raise

    @staticmethod
    def deserialize(serialized_tensor: torch.Tensor) -> 'DiffusionTask':
        """从torch.Tensor反序列化DiffusionTask"""
        try:
            # 1. 移到CPU进行反序列化
            tensor_cpu = serialized_tensor.cpu()
            
            # 2. 转换回bytes并反序列化
            serialized_bytes = tensor_cpu.byte().numpy().tobytes()
            full_data = pickle.loads(serialized_bytes)
            metadata = full_data['metadata']
            tensor_data = full_data['tensors']
            
            # 3. 重建用户请求对象（如果存在）
            user_request = None
            if metadata['user_params'] is not None:
                user_params = DiffusionUserParams(**metadata['user_params'])
                user_request = DiffusionUserRequest(
                    request_id=metadata['task_id'],
                    params=user_params,
                )
            
            # 4. 重建buffer（如果存在）
            buffer = None
            if metadata['buffer_metadata'] is not None:
                buffer = DiffusionTaskBuffer()
                buffer_meta = metadata['buffer_metadata']
                buffer.seq_len = buffer_meta['seq_len']
                buffer.current_step = buffer_meta['current_step']
                buffer.timesteps = buffer_meta['timesteps']
                
                # 恢复buffer中的tensor
                buffer.text_embeddings = tensor_data.get('text_embeddings')
                buffer.negative_embeddings = tensor_data.get('negative_embeddings')
                buffer.latents = tensor_data.get('latents')
                buffer.denoised_latents = tensor_data.get('denoised_latents')
                buffer.generated_image = tensor_data.get('generated_image')
            
            # 5. 重建任务对象
            task = DiffusionTask(
                task_id=metadata['task_id'],
                task_type=metadata['task_type'],
                req=user_request,
                buffer=buffer,
                signal_data=metadata.get('signal_data', {})
            )
            
            # 6. 恢复任务状态
            task.status = metadata['status']
            task.do_cfg = metadata['do_cfg']
            task.error_message = metadata['error_message']
            
            logger.debug(f"Deserialized {'terminate signal' if task.is_terminate_signal() else 'task'} {task.task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Failed to deserialize task: {e}")
            raise

    def __repr__(self):
        signal_info = f" reason={self.signal_data.get('reason', 'N/A')}" if self.is_terminate_signal() else ""
        return (
            f"DiffusionTask(id={self.task_id}, type={self.task_type}, "
            f"status={self.status}{signal_info})"
        )

    @staticmethod
    def create_empty_serialization(size: int, device: str = "cpu") -> torch.Tensor:
        """创建一个空的序列化张量，用于接收广播数据"""
        empty_tensor = torch.zeros(size, dtype=torch.uint8, device=device)
        logger.info(f"Created {empty_tensor.shape=}")
        return empty_tensor


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
    def all_finished(cls) -> bool:
        if len(cls.pool) == 0:
            return True
        return all(task.is_completed() for task in cls.pool.values())

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