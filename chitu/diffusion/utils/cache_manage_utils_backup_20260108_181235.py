"""
统一的缓存管理系统，支持多种缓存策略（TeaCache等）
参考 cache-dit 的设计，实现无侵入式的缓存集成
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Tuple, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from logging import getLogger
import unittest.mock
from contextlib import ExitStack

logger = getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置基类"""
    enabled: bool = True
    cache_type: str = "teacache"  # teacache, none, etc.
    max_cache_size: int = 100  # 最大缓存条目数
    clear_on_new_task: bool = True  # 新任务时清空缓存
    task: str = 't2v'  # t2v, i2v
    model: str = 'wan2.1-1.3B'


class CacheStrategy(ABC):
    """缓存策略抽象基类"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_store: Dict[str, Any] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
        }
    
    @abstractmethod
    def should_use_cache(
        self,
        timestep: int,
        latent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> bool:
        pass
    
    @abstractmethod
    def get_cache_key(
        self,
        timestep: int,
        latent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        is_cfg: bool = False,
        **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    def get_cached_value(self, cache_key: str, **kwargs) -> Optional[torch.Tensor]:
        pass
    
    @abstractmethod
    def store_cache(
        self,
        cache_key: str,
        value: torch.Tensor,
        timestep: int,
        **kwargs
    ):
        pass
    
    def clear_cache(self):
        self.cache_store.clear()
        logger.info(f"[{self.__class__.__name__}] Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0.0
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache_store),
        }


class CacheManager:
    """统一的缓存管理器，支持多种缓存策略"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.strategies: Dict[str, CacheStrategy] = {}
        self.active_strategy: Optional[str] = None
        self._hooks_registered = False
    
    def register_strategy(self, name: str, strategy: CacheStrategy):
        self.strategies[name] = strategy
        logger.info(f"[CacheManager] Registered cache strategy: {name}")
    
    def set_active_strategy(self, name: str):
        if name not in self.strategies:
            raise ValueError(f"Cache strategy '{name}' not registered. Available: {list(self.strategies.keys())}")
        self.active_strategy = name
        logger.info(f"[CacheManager] Activated cache strategy: {name}")
    
    def get_active_strategy(self) -> Optional[CacheStrategy]:
        if self.active_strategy is None:
            return None
        return self.strategies.get(self.active_strategy)
    
    def clear_all_cache(self):
        for strategy in self.strategies.values():
            strategy.clear_cache()
        logger.info("[CacheManager] All cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "active_strategy": self.active_strategy,
            "strategies": {}
        }
        for name, strategy in self.strategies.items():
            stats["strategies"][name] = strategy.get_stats()
        return stats
    
    def enable_cache_for_model(
        self,
        model: nn.Module,
    ):
        """
        为模型启用缓存（无侵入式）
        参考 cache-dit 的实现方式：通过包装 forward 并在执行时临时替换 blocks
        
        注意：
        - PyTorch 的 Module.__call__ 不只是调用 forward，还负责 hooks、autocast、no_grad 等
        - 替换 __call__ 会绕过这些机制，导致 hooks 失效、compile/DDP/AMP 行为异常
        - 应该 wrap forward 方法，这样 model(...) 仍走 PyTorch 正常调用链
        """
        if not hasattr(model, "forward"):
            logger.warning(f"[CacheManager] Model does not have 'forward' method, skip cache hook")
            return
        
        # 检查模型是否有 blocks 属性
        if not hasattr(model, "blocks"):
            logger.warning(f"[CacheManager] Model does not have 'blocks' attribute, skip cache hook")
            return
        
        strategy = self.get_active_strategy()
        if strategy is None or not strategy.config.enabled:
            return
        
        # 检查策略是否支持 blocks 替换方式（如 TeaCache）
        if not hasattr(strategy, 'create_cached_blocks'):
            # 如果不支持 blocks 替换，使用通用的缓存逻辑
            logger.warning(f"[CacheManager] Strategy {strategy.__class__.__name__} does not support blocks replacement, using generic cache logic")
            return
        
        # 创建 cached blocks
        original_blocks = model.blocks
        cached_blocks = strategy.create_cached_blocks(original_blocks, model)
        
        # 保存原始 forward 方法的引用
        original_forward = model.forward
        
        # 用于捕获 e 和 e0 的变量（e 是 time_embedding 的输出，e0 是 time_projection 的输出）
        captured_e = [None]
        captured_e0 = [None]
        
        # Hook 来捕获 e（从 time_embedding 的输出）
        def capture_e_hook(module, input, output):
            captured_e[0] = output
        
        # Hook 来捕获 e0（从 time_projection 的输出，需要 unflatten）
        def capture_e0_hook(module, input, output):
            # output 是 time_projection 的输出，需要 unflatten 才能得到 e0
            # 但这里我们只捕获原始输出，实际的 e0 会在 forward 中计算
            # 或者我们可以在这里计算 e0
            if hasattr(model, 'dim'):
                dim = model.dim
                captured_e0[0] = output.unflatten(1, (6, dim))
            else:
                captured_e0[0] = output
        
        hooks = []
        if hasattr(model, 'time_embedding'):
            hook = model.time_embedding.register_forward_hook(capture_e_hook)
            hooks.append(hook)
        if hasattr(model, 'time_projection'):
            hook = model.time_projection.register_forward_hook(capture_e0_hook)
            hooks.append(hook)
        
        @wraps(original_forward)
        def cached_forward(*args, **kwargs):
            if strategy is None or not strategy.config.enabled:
                # 缓存未启用，直接调用原始 forward
                return original_forward(*args, **kwargs)
            
            # 提取 t（timestep）参数，保存到 model 的内部状态，供 TeaCacheBlocks 使用
            t = None
            if len(args) > 1:
                t = args[1]  # 通常 t 是第二个位置参数
            elif 't' in kwargs:
                t = kwargs['t']
            elif 'timestep' in kwargs:
                t = kwargs['timestep']
            
            # 保存 t 到 model 的内部状态，供 TeaCacheBlocks 使用
            # is_cfg 不再需要，因为 TeaCacheBlocks 会使用 get_cfg_group() 来判断
            if t is not None:
                model._last_timestep = t
            
            # 重置捕获的 e 和 e0
            captured_e[0] = None
            captured_e0[0] = None
            
            # 使用 mock.patch 临时替换 blocks
            with unittest.mock.patch.object(model, 'blocks', cached_blocks):
                # 调用原始 forward，此时 blocks 已被替换
                # 在 forward 执行过程中，hooks 会捕获 e 和 e0
                result = original_forward(*args, **kwargs)
            
            # 清理临时状态（可选）
            if hasattr(model, '_last_timestep'):
                delattr(model, '_last_timestep')
            
            return result
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
        
        model.forward = cached_forward
        self._hooks_registered = True
        logger.info(f"[CacheManager] Cache hook registered for {model.__class__.__name__}.forward (blocks replacement mode)")
    
    def disable_cache_for_model(
        self,
        model: nn.Module,
    ):
        logger.warning(f"[CacheManager] Cannot fully restore original forward method for {model.__class__.__name__}")


_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def enable_cache_for_backend():
    manager = get_cache_manager()
    strategy = manager.get_active_strategy()
    
    if strategy is None:
        logger.warning("[CacheManager] No active cache strategy, skip enabling cache")
        return
    
    from chitu.diffusion.backend import DiffusionBackend
    
    if DiffusionBackend.model is not None:
        manager.enable_cache_for_model(DiffusionBackend.model)
    
    if DiffusionBackend.high_noise_model is not None:
        manager.enable_cache_for_model(DiffusionBackend.high_noise_model)
    
    if DiffusionBackend.low_noise_model is not None:
        manager.enable_cache_for_model(DiffusionBackend.low_noise_model)
    
    logger.info("[CacheManager] Cache enabled for DiffusionBackend models")


def clear_cache_on_new_task():
    manager = get_cache_manager()
    strategy = manager.get_active_strategy()
    if strategy is not None and strategy.config.clear_on_new_task:
        strategy.clear_cache()
        logger.debug("[CacheManager] Cache cleared for new task")

