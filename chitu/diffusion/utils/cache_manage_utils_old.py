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
        """判断是否应该使用缓存"""
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
        """生成缓存键"""
        pass
    
    @abstractmethod
    def get_cached_value(self, cache_key: str, **kwargs) -> Optional[torch.Tensor]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def store_cache(
        self,
        cache_key: str,
        value: torch.Tensor,
        timestep: int,
        **kwargs
    ):
        """存储缓存值"""
        pass
    
    def clear_cache(self):
        """清空所有缓存"""
        self.cache_store.clear()
        logger.info(f"[{self.__class__.__name__}] Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0.0
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache_store),
        }


class CacheManager:
    """
    统一的缓存管理器
    管理所有缓存策略的注册、调度和生命周期
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.strategies: Dict[str, CacheStrategy] = {}
        self.active_strategy: Optional[str] = None
        self._hooks_registered = False
    
    def register_strategy(self, name: str, strategy: CacheStrategy):
        """注册缓存策略"""
        self.strategies[name] = strategy
        logger.info(f"[CacheManager] Registered cache strategy: {name}")
    
    def set_active_strategy(self, name: str):
        """设置当前激活的缓存策略"""
        if name not in self.strategies:
            raise ValueError(f"Cache strategy '{name}' not registered. Available: {list(self.strategies.keys())}")
        self.active_strategy = name
        logger.info(f"[CacheManager] Activated cache strategy: {name}")
    
    def get_active_strategy(self) -> Optional[CacheStrategy]:
        """获取当前激活的缓存策略"""
        if self.active_strategy is None:
            return None
        return self.strategies.get(self.active_strategy)
    
    def clear_all_cache(self):
        """清空所有策略的缓存"""
        for strategy in self.strategies.values():
            strategy.clear_cache()
        logger.info("[CacheManager] All cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有策略的统计信息"""
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
        通过包装模型的 forward 方法实现，而不是 __call__
        
        注意：
        - PyTorch 的 Module.__call__ 不只是调用 forward，还负责 hooks、autocast、no_grad 等
        - 替换 __call__ 会绕过这些机制，导致 hooks 失效、compile/DDP/AMP 行为异常
        - 应该 wrap forward 方法，这样 model(...) 仍走 PyTorch 正常调用链
        """
        if not hasattr(model, "forward"):
            logger.warning(f"[CacheManager] Model does not have 'forward' method, skip cache hook")
            return
        
        # 保存原始 forward 方法的引用
        original_forward = model.forward
        
        @wraps(original_forward)
        def cached_forward(*args, **kwargs):
            strategy = self.get_active_strategy()
            if strategy is None or not strategy.config.enabled:
                return original_forward(*args, **kwargs)
            
            # 从参数中提取关键信息
            # 模型调用: model(latent_model_input, t=timestep, context=context, seq_len=seq_len)
            latent = args[0] if len(args) > 0 else kwargs.get("latent_model_input")
            timestep = kwargs.get("t") or kwargs.get("timestep")
            context = kwargs.get("context")
            seq_len = kwargs.get("seq_len")
            
            # 判断是否为CFG步骤
            # 让上游显式传 is_cfg（最干净的方式）
            is_cfg = kwargs.get("is_cfg", False)
            
            if timestep is None or latent is None:
                # 参数不完整，直接调用原方法
                return original_forward(*args, **kwargs)
            
            # 对于 TeaCache，需要在 forward 内部实现逻辑
            # 因为 TeaCache 需要访问模型的内部状态（e0、blocks等）
            # 所以我们需要让 strategy 有机会在 forward 执行过程中介入
            # 这里我们使用 hook 机制，让 strategy 可以访问模型的内部状态
            
            # 检查是否是 TeaCache 策略
            if hasattr(strategy, 'implement_forward_logic'):
                # TeaCache 需要实现自己的 forward 逻辑
                # 因为它需要访问 e0、blocks 等模型内部状态
                return strategy.implement_forward_logic(
                    model=model,
                    original_forward=original_forward,
                    args=args,
                    kwargs=kwargs,
                    timestep=timestep,
                    latent=latent,
                    context=context,
                    seq_len=seq_len,
                    is_cfg=is_cfg
                )
            
            # 对于其他策略，使用通用的缓存逻辑
            # 生成缓存键
            cache_key = strategy.get_cache_key(
                timestep=timestep,
                latent=latent,
                context=context,
                is_cfg=is_cfg,
                seq_len=seq_len,
                **kwargs
            )
            
            # 尝试获取缓存（统一使用 **kwargs 传递参数）
            cached_value = strategy.get_cached_value(cache_key, is_cfg=is_cfg, **kwargs)
                
            if cached_value is not None:
                logger.debug(f"[CacheManager] Using cached value for timestep {timestep}, is_cfg={is_cfg}")
                return cached_value
            
            # 缓存未命中，执行原始计算
            result = original_forward(*args, **kwargs)
            
            # 判断是否应该缓存结果
            if strategy.should_use_cache(
                timestep=timestep,
                latent=latent,
                context=context,
                is_cfg=is_cfg,
                **kwargs
            ):
                # 统一使用 **kwargs 传递参数
                strategy.store_cache(
                    cache_key=cache_key,
                    value=result,
                    timestep=timestep,
                    is_cfg=is_cfg,
                    **kwargs
                )
            
            return result
        
        # 替换 forward 方法（而不是 __call__）
        model.forward = cached_forward
        self._hooks_registered = True
        logger.info(f"[CacheManager] Cache hook registered for {model.__class__.__name__}.forward")
    
    def disable_cache_for_model(
        self,
        model: nn.Module,
    ):
        """禁用模型的缓存（恢复原方法）"""
        # 注意：这里无法完全恢复，因为原方法引用已丢失
        # 实际应用中可能需要保存原方法的引用
        logger.warning(f"[CacheManager] Cannot fully restore original forward method for {model.__class__.__name__}")


# 全局缓存管理器实例
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def enable_cache_for_backend():
    """
    为 DiffusionBackend 启用缓存
    应该在 backend 初始化后调用
    """
    manager = get_cache_manager()
    strategy = manager.get_active_strategy()
    
    if strategy is None:
        logger.warning("[CacheManager] No active cache strategy, skip enabling cache")
        return
    
    from chitu.diffusion.backend import DiffusionBackend
    
    # 为所有模型启用缓存（包装 forward 方法）
    if DiffusionBackend.model is not None:
        manager.enable_cache_for_model(DiffusionBackend.model)
    
    if DiffusionBackend.high_noise_model is not None:
        manager.enable_cache_for_model(DiffusionBackend.high_noise_model)
    
    if DiffusionBackend.low_noise_model is not None:
        manager.enable_cache_for_model(DiffusionBackend.low_noise_model)
    
    logger.info("[CacheManager] Cache enabled for DiffusionBackend models")


def clear_cache_on_new_task():
    """在新任务开始时清空缓存"""
    manager = get_cache_manager()
    strategy = manager.get_active_strategy()
    if strategy is not None and strategy.config.clear_on_new_task:
        strategy.clear_cache()
        logger.debug("[CacheManager] Cache cleared for new task")
