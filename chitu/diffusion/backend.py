# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import gc
import itertools
from functools import partial
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
from chitu.distributed.parallel_state import get_cp_group
from chitu.diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend, DiffusionAttention_with_CP

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
    high_noise_model = None # Wan2.2
    low_noise_model = None # Wan2.2
    formatter = None
    args = None
    # --- cache_manager related (not used in the current code)
    curr_req_ids = None
    cache_type = ""
    # ---
    use_gloo = False
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
    def _build_model_architecture(args, attn_backend, rope_impl):
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
        # if args.type in ["diff-wan", "diff-wan-22"]:
        try:
            model_kwargs = args.transformer
        except:
            raise ValueError(f"Unsupported model type: {args.type}")
        
        # 创建模型实例
        model = model_cls(model_type=args.task, attn_backend=attn_backend, rope_impl=rope_impl, **model_kwargs)
        
        return model
    
    # hmx: refactored to support multi-part checkpoint loading
    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, path: str, args: Any):
        """load multi-part checkpoint(*.safetensors) from a directory or a single file"""
        path = os.path.expanduser(path)
        
        checkpoint_files = []
        
        if os.path.isfile(path):
            # 单个文件的情况
            checkpoint_files = [path]
        elif os.path.isdir(path):
            # 目录的情况，直接查找所有 .safetensors 文件
            checkpoint_files = sorted(glob(os.path.join(path, "*.safetensors")))
        else:
            # 可能是 HuggingFace 风格的分片模型路径（不存在的单文件）
            base_dir = os.path.dirname(path)
            base_name = os.path.basename(path)
            
            # 检查是否有对应的索引文件
            index_file = os.path.join(base_dir, f"{base_name}.index.json")
            
            if os.path.exists(index_file):
                # 有索引文件，读取分片文件列表
                import json
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                
                # 获取所有分片文件名并排序
                shard_files = set(index_data.get('weight_map', {}).values())
                checkpoint_files = sorted([
                    os.path.join(base_dir, shard_file) 
                    for shard_file in shard_files 
                    if shard_file.endswith('.safetensors')
                ])
            else:
                # 没有索引文件，尝试查找匹配的分片文件
                base_pattern = base_name.replace('.safetensors', '')
                pattern = os.path.join(base_dir, f"{base_pattern}-*-of-*.safetensors")
                checkpoint_files = sorted(glob(pattern))
                
                # 如果还是没找到，尝试在目录中查找所有 safetensors 文件
                if not checkpoint_files:
                    checkpoint_files = sorted(glob(os.path.join(base_dir, "*.safetensors")))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found for path: {path}")
        
        logger.info(f"Loading checkpoint from {len(checkpoint_files)} file(s): {checkpoint_files}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if next(model.parameters()).device.type == 'meta':
            model.to_empty(device=device)
        
        model_dict = model.state_dict()
        all_loaded_keys = set()
        
        for ckpt_file in checkpoint_files:
            if not os.path.exists(ckpt_file):
                logger.warning(f"Checkpoint file does not exist: {ckpt_file}")
                continue
                
            logger.info(f"Loading checkpoint part: {ckpt_file}")
            try:
                checkpoint = st.load_file(ckpt_file, device="cpu")
            except Exception as e:
                logger.error(f"Failed to load {ckpt_file}: {e}")
                continue
                
            filtered_dict = {k: v.to(device) for k, v in checkpoint.items() 
                            if k in model_dict and v.shape == model_dict[k].shape}
            
            _, unexpected = model.load_state_dict(filtered_dict, strict=False)

            all_loaded_keys.update(filtered_dict.keys())

            if unexpected:
                logger.warning(f"Unexpected {len(unexpected)} keys in part {ckpt_file}")
            
        # Check for missing keys after iterating through all checkpoint files
        missing = set(model_dict.keys()) - all_loaded_keys
        if missing:
            logger.warning(f"Missing {len(missing)} keys after loading all parts: {list(missing)[:10]}...")
            
        logger.info(f"Loaded {len(all_loaded_keys)}/{len(model_dict)} parameters from checkpoint files.")

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

        if "Wan" in args.models.name:
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
        if "Wan" in args.models.name:
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
        """
        初始化缓存管理器
        支持多种缓存策略（TeaCache等）
        """
        from chitu.diffusion.utils.teacache_utils import (
            init_teacache,
            TeaCacheConfig,
        )
        from chitu.diffusion.utils.cache_manage_utils import (
            enable_cache_for_backend,
        )
        
        # 从配置中读取缓存设置
        cache_config_obj = getattr(DiffusionBackend.args, "cache", None)
        if cache_config_obj is None:
            logger.info("[CacheManager] Cache not enabled in config, skip initialization")
            return
        
        # 检查是否启用缓存
        if hasattr(cache_config_obj, "enabled") and not cache_config_obj.enabled:
            logger.info("[CacheManager] Cache is disabled in config, skip initialization")
            return
        
        # 从配置对象中读取参数（支持 StaticConfig 和普通对象）
        def get_cache_attr(obj, attr, default):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            elif hasattr(obj, "get"):
                return obj.get(attr, default)
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            return default
        
        # 初始化TeaCache
        cache_config = TeaCacheConfig(
            enabled=get_cache_attr(cache_config_obj, "enabled", True),
            teacache_thresh=get_cache_attr(cache_config_obj, "teacache_thresh", 0.2),
            use_ret_steps=get_cache_attr(cache_config_obj, "use_ret_steps", False),
            sample_steps=get_cache_attr(cache_config_obj, "sample_steps", 50),
            task=get_cache_attr(cache_config_obj, "task", "t2v"),
            model=get_cache_attr(cache_config_obj, "model", "wan2.1-1.3B"),
            enable_cfg_separate_cache=get_cache_attr(cache_config_obj, "enable_cfg_separate_cache", True),
            max_cache_size=get_cache_attr(cache_config_obj, "max_cache_size", 100),
        )
        
        init_teacache(cache_config)
        
        # 在模型初始化后启用缓存
        # 注意：这里需要在模型构建完成后调用 enable_cache_for_backend()
        logger.info("[CacheManager] Cache manager initialized")

    @staticmethod
    def _init_attention_backend(args):
        attn = DiffusionAttnBackend()

        if args.infer.diffusion.cp_size > 1:
            attn = DiffusionAttention_with_CP(attn, args.infer.diffusion.up_limit)
        
        DiffusionBackend.attn = attn
        return attn
    
    @staticmethod
    def _get_rope_implementation(args):
        if args.infer.diffusion.cp_size > 1:
            from chitu.diffusion.utils.wan_utils import rope_apply_with_cp
            return partial(rope_apply_with_cp, cp_size=get_cp_group().group_size, cp_rank=get_cp_group().rank_in_group)
        
        return None

    # hmx: refactored for Wan2.2 because it has two noise models
    @staticmethod
    def _build_and_setup_model(args, attn_backend, rope_impl):
        """
        Build model architecture, load checkpoints, and apply quantization.

        Arguments:
            args: Configuration with model settings
            attn_backend: The initialized attention backend
            rope_impl: The rope implementation for the model
        """
        # convert args.models lists to tuples
        args.models = DiffusionBackend.convert_config(args.models)
        DiffusionBackend.args = args

        if args.models.name in ["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"]:
            ckpt_path = os.path.join(args.models.ckpt_dir, "diffusion_pytorch_model.safetensors")
            DiffusionBackend.model = DiffusionBackend._build_and_setup_single_model(
                args, 
                ckpt_path,
                attn_backend, 
                rope_impl
            )
        elif args.models.name in ["Wan2.2-T2V-A14B"]:
            # build high noise model
            high_ckpt_path = os.path.join(args.models.ckpt_dir, args.models.high_noise_checkpoint)
            DiffusionBackend.high_noise_model = DiffusionBackend._build_and_setup_single_model(
                args, 
                high_ckpt_path, 
                attn_backend, 
                rope_impl
            )
            # build low noise model
            low_ckpt_path = os.path.join(args.models.ckpt_dir, args.models.low_noise_checkpoint)
            DiffusionBackend.low_noise_model = DiffusionBackend._build_and_setup_single_model(
                args, 
                low_ckpt_path, 
                attn_backend, 
                rope_impl
            )
        else:
            raise ValueError(f"Unsupported model name: {args.models.name}")

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _build_and_setup_single_model(args, ckpt_path, attn_backend, rope_impl):
        """
        Build single model architecture, load checkpoints, and apply quantization.

        Arguments:
            args: Configuration with model settings
            ckpt_path: Checkpoint path for the model
            attn_backend: The initialized attention backend
            rope_impl: The rope implementation for the model

        Returns:
            Fully set up single model
        """

        if not args.debug.skip_model_load:
            # Build the model. Don't allocate memory yet.
            with torch.device("cuda"): # FIXME: support meta device
                model = DiffusionBackend._build_model_architecture(args.models, attn_backend, rope_impl)

            # Load model parameters
            DiffusionBackend._load_checkpoint(model, ckpt_path, args)

        else:
            # Use initialized weights
            model = DiffusionBackend._build_model_architecture(args.models, attn_backend, rope_impl)

        model.eval().requires_grad_(False)
        return model




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

        # TODO: feature cache manager

        # Initialize attention backend
        attn_backend = DiffusionBackend._init_attention_backend(args)
        rope_impl = DiffusionBackend._get_rope_implementation(args)
       
        DiffusionBackend._build_and_setup_model(args, attn_backend, rope_impl)

        # 初始化缓存管理器（如果启用）
        DiffusionBackend._init_cache_manager()
        
        # 为模型启用缓存（如果已初始化）
        try:
            from chitu.diffusion.utils.cache_manage_utils import enable_cache_for_backend
            enable_cache_for_backend()
        except Exception as e:
            logger.warning(f"[CacheManager] Failed to enable cache for backend: {e}")

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
        setattr(DiffusionBackend, "high_noise_model", None)
        setattr(DiffusionBackend, "low_noise_model", None)
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
