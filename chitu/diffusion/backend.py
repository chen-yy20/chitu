# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import gc
import itertools
import functools
import os
import time
import re
import copy
import torch.cuda.amp as amp

from argparse import Namespace
from collections import deque
from enum import Enum
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Deque, Optional, Iterable
import safetensors.torch as st
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from safetensors.torch import safe_open
from tqdm import tqdm
from chitu.attn_backend import (
    FlashAttnBackend,
    FlashInferBackend,
    FlashMLABackend,
    NpuAttnBackend,
    RefAttnBackend,
    TritonAttnBackend,
    NpuAttnBackend,
    HybridAttnBackend,
)
from chitu.cache_manager import DenseKVCacheManager, PagedKVCacheManager, GlobalLocalMap
from chitu.custom_gguf import *
from chitu.device_type import is_ascend, is_muxi
from chitu.distributed.parallel_state import (
    get_world_group,
    initialize_parallel_groups,
    initialize_diffusion_parallel_groups
)
from chitu.hybrid_device import CPUParameter
from chitu.models.registry import ModelType, get_model_class
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
    utils,
)
from chitu.tokenizer import ChatFormat, ChatFormatHF, Tokenizer, TokenizerHF, Processor
from chitu.utils import (
    compute_layer_dist_in_pipe,
    parse_dtype,
    try_import_opt_dep,
    ceil_div,
)

# from chitu.distributed.moe_token_dispatcher import init_token_dispatcher
from chitu.backend import BackendState
if TYPE_CHECKING:
    from chitu.diffusion.generator import Generator
    from chitu.diffusion.scheduler import DiffusionScheduler
    # from chitu.diffusion.task import BatchResult

numa, has_numa = try_import_opt_dep("numa", "cpu")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


logger = getLogger(__name__)


class DiffusionBackend:
    # init once
    model = None
    formatter = None
    args = None
    # --- cache_manager related (not used in the current code)
    curr_req_ids = None
    cache_type = ""
    # ---
    use_gloo = True
    group_gloo = None

    # components
    scheduler: Optional["DiffusionScheduler"] = None

    # mutable
    # ongoing_reqs: list["OngoingRequests"] = []
    state = BackendState.Running
    # last_batch_results: Deque["BatchResult"] = deque()
    indexer_cache_manager = None
    
    # diffusion
    generator: Optional["Generator"] = None
    text_encoder = None
    cache_manager = None
    vae = None

    @staticmethod
    def convert_config(args):
        """
        Yaml config can only represent lists, but some places need tuples.
        This method aims to convert all lists in args to tuples recursively.
        """ 
        def _convert_value(value):
            if isinstance(value, list):
                # 递归转换列表中的元素，然后转为元组
                return tuple(_convert_value(item) for item in value)
            elif isinstance(value, dict):
                # 递归转换字典
                return {k: _convert_value(v) for k, v in value.items()}
            elif hasattr(value, '__dict__'):
                # 对于有属性的对象（如 Namespace），递归转换其属性
                converted = copy.copy(value)
                for attr_name in dir(value):
                    if not attr_name.startswith('_'):  # 跳过私有属性
                        try:
                            attr_value = getattr(value, attr_name)
                            if not callable(attr_value):  # 跳过方法
                                setattr(converted, attr_name, _convert_value(attr_value))
                        except:
                            pass  # 如果无法访问或设置属性，跳过
                return converted
            else:
                # 其他类型直接返回
                return value
        
        return _convert_value(args)


    @staticmethod
    def _build_model_architecture(args, attn_backend):
        try:
            model_type = ModelType(args.type)
        except ValueError:
            raise ValueError(
                f"Model type '{args.type}' is not supported. "
                f"Available types: {[t.value for t in ModelType]}"
            )

        model_cls = get_model_class(model_type)
        logger.info(f"Building model with args: {args.transformer}")

        # # 从 args.transformer 构建正确的模型参数
        if args.type == "diff-wan":
            model_kwargs = args.transformer
        else:
            ValueError(f"Unsupported model type: {args.type}")
        
        # 创建模型实例
        model = model_cls(model_type=args.task, **model_kwargs)
        
        return model
    
    @staticmethod 
    def _load_checkpoint(model, args):
        """Load Wan model checkpoint from safetensors file."""
        
        ckpt_path = os.path.join(args.models.ckpt_dir, "diffusion_pytorch_model.safetensors")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        logger.info(f"Loading Wan checkpoint from: {ckpt_path}")
        
        # 加载权重
        checkpoint = st.load_file(ckpt_path, device="cpu")
        
        # 处理设备转移
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if next(model.parameters()).device.type == 'meta':
            model.to_empty(device=device)
        
        # 加载权重
        model_dict = model.state_dict()
        filtered_dict = {k: v.to(device) for k, v in checkpoint.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        
        missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
        
        if missing:
            logger.warning(f"Missing {len(missing)} keys")
        if unexpected:
            logger.warning(f"Unexpected {len(unexpected)} keys")
            
        logger.info(f"Loaded {len(filtered_dict)}/{len(model_dict)} parameters")

    # FIXME: When cache type is "skew", gloo backend cannot be used.
    @staticmethod
    def _init_distributed(args):
        """
        Initialize distributed training environment with tensor and pipeline parallelism.

        Arguments:
            args: Configuration object with distributed parameters
        """
        is_router_process = os.environ.get("CHITU_ROUTER_PROCESS", "0") == "1"
        if is_router_process:
            # Router process: as independent subprocess, skip CUDA device binding
            logger.info(f"[Router] Router subprocess skip CUDA device binding")
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Bind process to GPU. Please put it before init_process_group
        if args.infer.op_impl != "cpu":
            torch.cuda.set_device(local_rank)

        if not torch.distributed.is_initialized():
            if args.infer.op_impl == "cpu":
                torch.distributed.init_process_group("gloo")
            else:
                torch.distributed.init_process_group("nccl")
        if DiffusionBackend.use_gloo:
            DiffusionBackend.group_gloo = torch.distributed.new_group(backend="gloo")

        model_parallel_size = args.infer.tp_size
        pipeline_parallel_size = args.infer.pp_size
        non_expert_data_parallel_size = args.infer.dp_size
        expert_parallel_size = args.infer.ep_size

        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        assert model_parallel_size == 1, "DiffusionBackend only supports model_parallel_size=1"
        assert pipeline_parallel_size == 1, "DiffusionBackend only supports pipeline_parallel_size=1"
        assert expert_parallel_size == 1, "DiffusionBackend only supports expert_parallel_size=1"

        # Diffusion Parallelism
        non_expert_data_parallel_size = 1 # TODO: support batch generation with data parallelism
        cfg_size = 2 if (args.infer.diffusion.guidance_scale > 1 and world_size >= 2) else 1
        up_limit = args.infer.diffusion.up_limit
        context_parallel_size = args.infer.diffusion.cp_size

        assert (
            world_size
            == non_expert_data_parallel_size * cfg_size * context_parallel_size
        ), f"World size not match: {world_size} != {non_expert_data_parallel_size} * {cfg_size} * {context_parallel_size}"

        initialize_diffusion_parallel_groups(
            cfg_size= cfg_size,
            up_limit=up_limit,
            cp_size=context_parallel_size,
        )

    @staticmethod
    def _setup_environment(args):
        """
        Set up random seed, default dtype, and check prerequisites.

        Arguments:
            args: Configuration with seed and dtype settings
        """
        torch.manual_seed(args.infer.seed)

        # Set default_dtype
        if args.float_16bit_variant == "float16":
            torch.set_default_dtype(torch.float16)
        elif args.float_16bit_variant == "bfloat16":
            torch.set_default_dtype(torch.bfloat16)
        else:
            raise NotImplementedError(f"Unsupported float_16bit_variant {args.dtype}")


    @staticmethod
    def _init_text_encoder(args):
        """
        Initialize the multimodal processor for vision-language models.

        Arguments:
            args: Configuration with model settings

        Returns:
            Initialized processor or None if not a multimodal model
        """

        if args.models.name == "Wan2.1-T2V-1.3B":
            from chitu.diffusion.modules.encoders.t5 import T5EncoderModel
            logger.info(f"Initializing T5 encoder for {args.models.name}")

            text_encoder = T5EncoderModel(
                    text_len=args.models.encoder.text_len,
                    # device=torch.device('cpu'),
                    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'),
                    checkpoint_path=os.path.join(args.models.ckpt_dir, args.models.encoder.t5_checkpoint),
                    tokenizer_path=os.path.join(args.models.ckpt_dir, args.models.encoder.t5_tokenizer),
                )
            logger.info(f"Initialized T5 encoder for {args.models.name}")
        else:
            text_encoder = None

        logger.info(f"Initialized multimodal processor for {args.models.name}")
        return text_encoder

    @staticmethod
    def _init_vae(args):
        """
        Initialize the VAE model for diffusion.

        Arguments:
            args: Configuration with model settings
        """
        if args.models.name in ["Wan2.1-T2V-1.3B"]:
            from chitu.diffusion.modules.vaes.wan_vae import WanVAE
            logger.info(f"Initializing Wan VAE for {args.models.name}")

            vae = WanVAE(
                    vae_pth=os.path.join(args.models.ckpt_dir, args.models.vae.checkpoint),
                    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'),
                )
            logger.info(f"Initialized Wan VAE for {args.models.name}")
        else:
            vae = None
        return vae
    
    @staticmethod
    def _init_cache_manager():
        # TODO
        pass

    @staticmethod
    def _get_attention_backend_type(args):
        if args.infer.attn_type == "auto":
            if is_ascend():
                return NpuAttnBackend
            elif args.infer.op_impl == "cpu":
                return RefAttnBackend
            elif "deepseek-v3" in args.models.type:
                return FlashMLABackend
            else:
                return HybridAttnBackend
        elif args.infer.attn_type == "cpu":
            return RefAttnBackend
        elif args.infer.attn_type == "flash_attn":
            return FlashAttnBackend
        elif args.infer.attn_type == "flash_mla":
            return FlashMLABackend
        elif args.infer.attn_type == "flash_infer":
            return FlashInferBackend
        elif args.infer.attn_type == "triton":
            return TritonAttnBackend
        elif args.infer.attn_type == "npu":
            return NpuAttnBackend
        elif args.infer.attn_type == "ref":
            return RefAttnBackend
        else:
            raise ValueError(f"Unknown attn type {args.infer.attn_type}")

    @staticmethod
    def _init_attention_backend(attn_backend_type):
        # Yes, use `type` instead of `isinstance` here, because `AttnBackend`s inherit each other
        if attn_backend_type is FlashInferBackend:
            assert isinstance(DiffusionBackend.cache_manager, PagedKVCacheManager)
            return attn_backend_type(DiffusionBackend.cache_manager.get_max_num_blocks())
        else:
            return attn_backend_type()

    @staticmethod
    def _move_one_module_to_device(
        m: torch.nn.Module, non_blocking: bool = True, ignore_not_loaded: bool = False
    ):
        # NOTE: m._parameters contains parameters in this module (non-recursive),
        # while m.parameters() returns all parameters in this module and its submodules
        # (recursive).
        for key in m._parameters:
            param = m._parameters[key]
            if param is not None:
                if not isinstance(param, CPUParameter):
                    if param.device == torch.device("meta"):
                        if not ignore_not_loaded:
                            assert False, f"Unexpected unloaded parameter {key}"
                        else:
                            continue
                    if is_muxi():
                        # Work around a muxi bug that convert from NHWC to NCHW for whatever
                        # 4-D tensor even its not a convolution weight.
                        param.data = param.data.cuda(
                            non_blocking=non_blocking
                        ).contiguous()
                    else:
                        param.data = param.data.cuda(non_blocking=non_blocking)
        for key in m._buffers:
            buffer = m._buffers[key]
            if buffer is not None:
                if buffer.device == torch.device("meta"):
                    # Buffers are expected possibly not to be loaded, so buffer.device may be "meta"
                    m._buffers[key] = torch.empty(
                        buffer.shape, dtype=buffer.dtype, device="cuda"
                    )
                elif is_muxi():
                    # Work around a muxi bug that convert from NHWC to NCHW for whatever
                    # 4-D tensor even its not a convolution weight.
                    m._buffers[key] = buffer.cuda(
                        non_blocking=non_blocking
                    ).contiguous()
                else:
                    m._buffers[key] = buffer.cuda(non_blocking=non_blocking)

    @staticmethod
    def _build_and_setup_model(args, attn_backend):
        """
        Build model architecture, load checkpoints, and apply quantization.

        Arguments:
            args: Configuration with model settings
            attn_backend: The initialized attention backend
∑
        Returns:
            Fully set up model
        """
        # convert args.models lists to tuples
        args.models = DiffusionBackend.convert_config(args.models)

        if not args.debug.skip_model_load:
            # Build the model. Don't allocate memory yet.
            with torch.device("meta"):
                model = DiffusionBackend._build_model_architecture(args.models, attn_backend)

            # Load model parameters
            DiffusionBackend._load_checkpoint(model, args)

        else:
            # Use initialized weights
            model = DiffusionBackend._build_model_architecture(args.models, attn_backend)

        model.eval().requires_grad_(False)

        DiffusionBackend.model = model
        DiffusionBackend.args = args

        gc.collect()
        torch.cuda.empty_cache()



    @staticmethod
    def build(args):
        """
        Build and initialize the model, tokenizer, cache manager, and other components required for inference.

        Arguments:
            args: Configuration object containing model and training related configurations.
        """
        # Initialize distributed environment
        DiffusionBackend._init_distributed(args)

        # Setup environment and basic configuration
        DiffusionBackend._setup_environment(args)

        # Initialize tokenizer and formatter
        DiffusionBackend.text_encoder = DiffusionBackend._init_text_encoder(args)
        DiffusionBackend.vae = DiffusionBackend._init_vae(args)

        attn_backend_type = DiffusionBackend._get_attention_backend_type(args)

        # TODO: Initialize feature cache manager
        # DiffusionBackend.cache_manager = DiffusionBackend._init_cache_manager(args, attn_backend_type)
        # DiffusionBackend.cache_type = args.infer.cache_type

        # Initialize attention backend
        attn_backend = DiffusionBackend._init_attention_backend(attn_backend_type)
       
        DiffusionBackend._build_and_setup_model(args, attn_backend)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(
            f"rank {local_rank} Backend initialized with CUDA mem at {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )
        logger.info(
            f"Using {len(c10d._pg_map)} communication gruops. If this number is too high, there may be too much memory reserved for underlying communication libraries."
        )

        return DiffusionBackend

    @staticmethod
    def stop():
        setattr(DiffusionBackend, "model", None)
        setattr(DiffusionBackend, "cache_manager", None)
        gc.collect()
        torch.cuda.empty_cache()



def memory_used():
    logger.debug(
        f"gpu memory usage: {torch.cuda.memory_allocated()/(1024**3)} GB"
    )  # torch.cuda.max_memory_allocated()/(1024**3)) #, torch.cuda.memory_reserved()/(1024**3))
    import resource

    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.debug(f"cpu memory usage: {memory_usage / 1024} MB")
