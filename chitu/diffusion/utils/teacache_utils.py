"""
TeaCache (Timestep Embedding Aware Cache) 实现
基于时间步嵌入的相似度判断是否使用缓存
参考 cache-dit 的实现方式：通过包装 blocks 而不是完整 forward
"""
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from logging import getLogger
import hashlib
import torch.amp as amp
import unittest.mock

from chitu.diffusion.utils.cache_manage_utils import (
    CacheConfig,
    CacheStrategy,
    CacheManager,
    get_cache_manager,
    enable_cache_for_backend,
    clear_cache_on_new_task,
)
from chitu.distributed.parallel_state import get_cfg_group

logger = getLogger(__name__)


@dataclass
class TeaCacheConfig(CacheConfig):
    """TeaCache 特定配置"""
    cache_type: str = "teacache"
    teacache_thresh: float = 0.95  # 相似度阈值，用于判断是否使用缓存
    enable_cfg_separate_cache: bool = True  # 是否为CFG和非CFG分别缓存
    coefficients: Optional[List[float]] = field(default=None)  # TeaCache 系数
    use_ret_steps: bool = False  # 是否使用 ret_steps
    sample_steps: int = 50  # 采样步数
    ret_steps: Optional[int] = field(default=None)  # 重试步数，会在 __post_init__ 中计算
    cutoff_steps: Optional[int] = field(default=None)  # 截止步数，会在 __post_init__ 中计算
    
    def __post_init__(self):
        """初始化后根据 task 和 model 设置 coefficients, ret_steps, cutoff_steps"""
 
        # 根据 task 和 model 设置默认值
        if self.task == 't2v':
            if self.use_ret_steps:
                if '1.3B' in self.model:
                    self.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
                elif '14B' in self.model:
                    self.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]             
                self.ret_steps = 5 * 1      
                self.cutoff_steps = self.sample_steps * 1
            else:
                if '1.3B' in self.model:
                    self.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
                elif '14B' in self.model:
                    self.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]     
                self.ret_steps = 1 * 1           
                self.cutoff_steps = self.sample_steps * 1 - 1

        elif self.task == 'i2v':
            if self.use_ret_steps:
                if '1.3B' in self.model:
                    self.coefficients = [ 2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
                elif '14B' in self.model:
                    self.coefficients = [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]
                self.ret_steps = 5       
                self.cutoff_steps = self.sample_steps * 1
            else:
                if '1.3B' in self.model:
                    self.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
                elif '14B' in self.model:
                    # i2v 模式下 14B 没有提供非 ret_steps 的系数，使用默认值
                    self.coefficients =  [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
                self.ret_steps = 1 * 1           
                self.cutoff_steps = self.sample_steps * 1  - 1


class TeaCacheStrategy(CacheStrategy):
    """
    TeaCache (Timestep Embedding Aware Cache) 实现
    基于时间步嵌入的相似度判断是否使用缓存
    参考原始实现：teacache_origin.py
    """
    
    def __init__(self, config: TeaCacheConfig):
        super().__init__(config)
        self.config: TeaCacheConfig = config
        
        # TeaCache 特定的状态变量
        # cnt 用于跟踪当前去噪步骤（timestep），每个timestep递增一次
        # 注意：在CFG并行模式下，condition和uncondition是并行计算的，属于同一个timestep
        # 所以cnt表示去噪步骤数，不是CFG调用次数
        self.cnt: int = 0  # 计数器，用于跟踪当前去噪步骤
        self.num_steps: int = config.sample_steps  # 总去噪步数
        
        # CFG 分离缓存的状态
        # 在CFG并行模式下：
        # - rank_in_group == 0 (is_cfg=False): condition步，使用 *_condition 状态
        # - rank_in_group == 1 (is_cfg=True): unconditional步，使用 *_uncondition 状态
        self.accumulated_rel_l1_distance_condition: float = 0.0  # condition步累积的相对L1距离
        self.accumulated_rel_l1_distance_uncondition: float = 0.0  # unconditional步累积的相对L1距离
        self.previous_e0_condition: Optional[torch.Tensor] = None  # 上一次condition步的 e0
        self.previous_e0_uncondition: Optional[torch.Tensor] = None  # 上一次uncondition步的 e0
        self.previous_residual_condition: Optional[torch.Tensor] = None  # 上一次condition步的残差
        self.previous_residual_uncondition: Optional[torch.Tensor] = None  # 上一次uncondition步的残差
        
       
    def get_cache_key(
        self,
        timestep: int,
        latent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        is_cfg: bool = False,
        seq_len: Optional[int] = None,
        **kwargs
    ) -> str:
        """生成缓存键"""
        # TeaCache 使用去噪步骤计数器和CFG标志生成键
        # is_cfg=False: condition步 (rank_in_group == 0)
        # is_cfg=True: unconditional步 (rank_in_group == 1)
        cfg_suffix = "_uncondition" if is_cfg else "_condition"
        seq_len_suffix = f"_seq{seq_len}" if seq_len is not None else ""
        key = f"teacache_step{self.cnt}_t{timestep}{cfg_suffix}{seq_len_suffix}"
        return key
    
    def should_use_cache(
        self,
        timestep: int,
        latent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        is_cfg: bool = False,
        e0: Optional[torch.Tensor] = None,
        **kwargs
    ) -> bool:
        """
        判断是否应该使用缓存
        这是 TeaCache 的核心逻辑，基于时间步嵌入的相似度判断
        
        使用 get_cfg_group() 来判断是否为 condition 步骤
        """
        if not self.config.enabled:
            return False

        # 使用项目中的接口判断是否是 condition 分支
        try:
            cfg_group = get_cfg_group()
            if cfg_group is not None and cfg_group.group_size == 2:
                # CFG 并行模式：使用 rank_in_group 判断
                is_condition_step = (cfg_group.rank_in_group == 0)
            else:
                # 非 CFG 并行模式或串行 CFG：默认使用 condition
                is_condition_step = True
        except Exception:
            # 如果获取失败，默认使用 condition
            is_condition_step = True
        
        if is_condition_step:
            # Condition步（rank_in_group == 0）
            if self.cnt < self.config.ret_steps or self.cnt >= self.config.cutoff_steps:
                # 前 ret_steps 步或超过 cutoff_steps 后，必须计算
                return False
            
            # 在 ret_steps 和 cutoff_steps 之间，根据累积距离判断
            if e0 is not None and self.previous_e0_condition is not None:
                # 计算相对L1距离
                modulated_inp = e0 if self.config.use_ret_steps else kwargs.get("e", e0)
                if modulated_inp is None:
                    return False
                    
                rel_l1_distance = ((modulated_inp - self.previous_e0_condition).abs().mean() / 
                                  self.previous_e0_condition.abs().mean()).cpu().item()
                
                # 使用多项式系数重新缩放
                rescale_func = np.poly1d(self.config.coefficients)
                self.accumulated_rel_l1_distance_condition += rescale_func(rel_l1_distance)
                
                # 判断是否超过阈值
                if self.accumulated_rel_l1_distance_condition < self.config.teacache_thresh:
                    return True  # 使用缓存
                else:
                    self.accumulated_rel_l1_distance_condition = 0
                    return False  # 重新计算
        else:
            # Uncondition步（rank_in_group == 1）
            if self.cnt < self.config.ret_steps or self.cnt >= self.config.cutoff_steps:
                # 前 ret_steps 步或超过 cutoff_steps 后，必须计算
                return False
            
            # 在 ret_steps 和 cutoff_steps 之间，根据累积距离判断
            if e0 is not None and self.previous_e0_uncondition is not None:
                # 计算相对L1距离
                modulated_inp = e0 if self.config.use_ret_steps else kwargs.get("e", e0)
                if modulated_inp is None:
                    return False
                    
                rel_l1_distance = ((modulated_inp - self.previous_e0_uncondition).abs().mean() / 
                                  self.previous_e0_uncondition.abs().mean()).cpu().item()
                
                # 使用多项式系数重新缩放
                rescale_func = np.poly1d(self.config.coefficients)
                self.accumulated_rel_l1_distance_uncondition += rescale_func(rel_l1_distance)
                
                # 判断是否超过阈值
                if self.accumulated_rel_l1_distance_uncondition < self.config.teacache_thresh:
                    return True  # 使用缓存
                else:
                    self.accumulated_rel_l1_distance_uncondition = 0
                    return False  # 重新计算
        
        # 默认不缓存（需要计算）
        return False
    
    def get_cached_value(self, cache_key: str, **kwargs) -> Optional[torch.Tensor]:
        """
        获取缓存值
        
        Args:
            cache_key: 缓存键（TeaCache 不使用这个参数，保留以兼容接口）
            **kwargs: 其他参数（不使用，保留以兼容接口）
        """
        # 使用项目中的接口判断是否是 condition 分支
        try:
            cfg_group = get_cfg_group()
            if cfg_group is not None and cfg_group.group_size == 2:
                # CFG 并行模式：使用 rank_in_group 判断
                is_condition_step = (cfg_group.rank_in_group == 0)
            else:
                # 非 CFG 并行模式或串行 CFG：默认使用 condition
                is_condition_step = True
        except Exception:
            # 如果获取失败，默认使用 condition
            is_condition_step = True
        
        if is_condition_step:
            # Condition步（rank_in_group == 0）
            if self.previous_residual_condition is not None:
                self.cache_stats["hits"] += 1
                logger.debug(f"[TeaCache] Cache hit for condition step {self.cnt}")
                return self.previous_residual_condition.clone()
        else:
            # Uncondition步（rank_in_group == 1）
            if self.previous_residual_uncondition is not None:
                self.cache_stats["hits"] += 1
                logger.debug(f"[TeaCache] Cache hit for uncondition step {self.cnt}")
                return self.previous_residual_uncondition.clone()
        
        self.cache_stats["misses"] += 1
        return None
    
    def store_cache(
        self,
        cache_key: str,
        value: torch.Tensor,
        timestep: int,
        e0: Optional[torch.Tensor] = None,
        original_x: Optional[torch.Tensor] = None,
        is_cfg: bool = False,
        **kwargs
    ):
        """
        存储缓存值
        注意：TeaCache 存储的是残差（residual），而不是完整的结果
        
        Args:
            cache_key: 缓存键
            value: 计算后的值（blocks的输出）
            timestep: 时间步
            e0: 时间嵌入 e0
            original_x: 原始输入 x（用于计算残差）
            is_cfg: 是否为CFG步骤（保留以兼容接口，但不再使用）
        """
        # 使用项目中的接口判断是否是 condition 分支
        try:
            cfg_group = get_cfg_group()
            if cfg_group is not None and cfg_group.group_size == 2:
                # CFG 并行模式：使用 rank_in_group 判断
                is_condition_step = (cfg_group.rank_in_group == 0)
            else:
                # 非 CFG 并行模式或串行 CFG：默认使用 condition
                is_condition_step = True
        except Exception:
            # 如果获取失败，默认使用 condition
            is_condition_step = True
        
        if is_condition_step:
            # Condition步（rank_in_group == 0）：存储残差和 e0
            if original_x is not None:
                residual = value - original_x
                self.previous_residual_condition = residual.clone()
            else:
                # 如果没有提供 original_x，假设 value 就是残差
                self.previous_residual_condition = value.clone()
            
            if e0 is not None:
                self.previous_e0_condition = e0.clone()
        else:
            # Uncondition步（rank_in_group == 1）：存储残差和 e0
            if original_x is not None:
                residual = value - original_x
                self.previous_residual_uncondition = residual.clone()
            else:
                # 如果没有提供 original_x，假设 value 就是残差
                self.previous_residual_uncondition = value.clone()
            
            if e0 is not None:
                self.previous_e0_uncondition = e0.clone()
        
        self.cache_stats["stores"] += 1
        
        # 注意：cnt 的更新应该在每次去噪步骤完成后进行，而不是在每次CFG调用时
        # 这里不更新cnt，cnt应该在generator的denoise_step中更新
        
        logger.debug(f"[TeaCache] Stored cache for step {self.cnt}, timestep={timestep}, is_condition={is_condition_step}")
    
    def clear_cache(self):
        """清空所有缓存"""
        super().clear_cache()
        # 重置 TeaCache 特定的状态
        self.cnt = 0
        self.accumulated_rel_l1_distance_condition = 0
        self.accumulated_rel_l1_distance_uncondition = 0
        self.previous_e0_condition = None
        self.previous_e0_uncondition = None
        self.previous_residual_condition = None
        self.previous_residual_uncondition = None
        logger.info("[TeaCache] Cache and state cleared")
    
    def increment_step(self):
        """增加去噪步骤计数器（在每个timestep完成后调用）"""
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
            # 重置累积距离
            self.accumulated_rel_l1_distance_condition = 0
            self.accumulated_rel_l1_distance_uncondition = 0


class TeaCacheBlocks(torch.nn.Module):
    """
    TeaCache Blocks 包装类
    参考 cache-dit 的 CachedBlocks_Pattern_Base 实现方式
    通过包装 blocks 而不是完整 forward 来实现 TeaCache 逻辑
    """
    
    def __init__(
        self,
        original_blocks: torch.nn.ModuleList,
        strategy: TeaCacheStrategy,
        model: torch.nn.Module,
    ):
        super().__init__()
        self.original_blocks = original_blocks
        self.strategy = strategy
        self.model = model
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        TeaCache blocks forward 逻辑
        在 blocks 循环中判断是否使用缓存
        
        参考 teacache_origin.py 的实现方式：
        - 直接计算 e 和 e0，就像原始实现那样
        - 从 model 的内部状态获取 t（timestep）
        
        Args:
            x: 输入 hidden states
            **kwargs: 其他参数，包括 e (即 e0), is_cfg 等
        
        Returns:
            处理后的 hidden states
        """
        # 延迟计算 e 和 e0：只在需要判断缓存时才计算
        # 如果不需要判断（ret_steps 之前或 cutoff_steps 之后），跳过计算
        e0 = None
        e = None
        need_e0_for_cache = True
        
        # 快速检查：如果不在缓存范围内，不需要计算 e0
        if self.strategy.cnt < self.strategy.config.ret_steps or \
           self.strategy.cnt >= self.strategy.config.cutoff_steps:
            need_e0_for_cache = False
        
        # 只在需要时才计算 e 和 e0
        if need_e0_for_cache:
            if hasattr(self.model, '_last_timestep') and self.model._last_timestep is not None:
                t = self.model._last_timestep
                
                # 确保 t 是 1-D 张量（sinusoidal_embedding_1d 需要 1-D 输入）
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, dtype=torch.float32, device=x.device)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                elif t.dim() > 1:
                    t = t.flatten()
                
                # 直接计算 e 和 e0
                from chitu.models.diffusion.model_wan import sinusoidal_embedding_1d
                
                with amp.autocast(device_type="cuda", dtype=torch.float32):
                    freq_dim = getattr(self.model, 'freq_dim', 256)
                    e = self.model.time_embedding(
                        sinusoidal_embedding_1d(freq_dim, t).float())
                    e0 = self.model.time_projection(e).unflatten(1, (6, getattr(self.model, 'dim', 2048)))
                    assert e.dtype == torch.float32 and e0.dtype == torch.float32
            else:
                # 如果 model 没有保存 t，从 kwargs 中获取 e0（作为后备）
                e0 = kwargs.get("e", None)
                e = e0
        
        # 判断是否为 condition 分支：使用项目中的接口（缓存结果避免重复调用）
        # 参考 generator.py 的实现：get_cfg_group().rank_in_group == 0 表示 condition 步
        if not hasattr(self, '_cached_is_condition_step') or not hasattr(self, '_cached_cfg_rank'):
            try:
                cfg_group = get_cfg_group()
                if cfg_group is not None and cfg_group.group_size == 2:
                    self._cached_cfg_rank = cfg_group.rank_in_group
                    self._cached_is_condition_step = (cfg_group.rank_in_group == 0)
                else:
                    self._cached_cfg_rank = 0
                    self._cached_is_condition_step = True
            except Exception:
                self._cached_cfg_rank = 0
                self._cached_is_condition_step = True
        
        is_condition_step = self._cached_is_condition_step
        cfg_rank = self._cached_cfg_rank
        
        # 如果没有 e0，无法进行 TeaCache 判断，直接调用原始 blocks
        if e0 is None:
            for block in self.original_blocks:
                x = block(x, **kwargs)
            return x
        
        # TeaCache 核心逻辑：判断是否使用缓存
        should_use_cache = False
        
        if is_condition_step:
            # Condition 步
            if self.strategy.cnt < self.strategy.config.ret_steps or \
               self.strategy.cnt >= self.strategy.config.cutoff_steps:
                should_use_cache = False
                self.strategy.accumulated_rel_l1_distance_condition = 0
            else:
                if self.strategy.previous_e0_condition is not None:
                    modulated_inp = e0 if self.strategy.config.use_ret_steps else e
                    if modulated_inp is not None:
                        # 优化：避免 .cpu().item()，在 GPU 上计算
                        prev_mean = self.strategy.previous_e0_condition.abs().mean()
                        rel_l1_distance = ((modulated_inp - self.strategy.previous_e0_condition).abs().mean() / prev_mean)
                        # 只在需要时才移到 CPU
                        rel_l1_distance_cpu = rel_l1_distance.cpu().item()
                        rescale_func = np.poly1d(self.strategy.config.coefficients)
                        self.strategy.accumulated_rel_l1_distance_condition += rescale_func(rel_l1_distance_cpu)
                        
                        if self.strategy.accumulated_rel_l1_distance_condition < self.strategy.config.teacache_thresh:
                            should_use_cache = True
                        else:
                            should_use_cache = False
                            self.strategy.accumulated_rel_l1_distance_condition = 0
                    else:
                        should_use_cache = False
                else:
                    should_use_cache = False
            
            # 更新 previous_e0_condition
            if e0 is not None:
                self.strategy.previous_e0_condition = (e0 if self.strategy.config.use_ret_steps else e).clone()
        else:
            # Uncondition 步
            if self.strategy.cnt < self.strategy.config.ret_steps or \
               self.strategy.cnt >= self.strategy.config.cutoff_steps:
                should_use_cache = False
                self.strategy.accumulated_rel_l1_distance_uncondition = 0
            else:
                if self.strategy.previous_e0_uncondition is not None:
                    modulated_inp = e0 if self.strategy.config.use_ret_steps else e
                    if modulated_inp is not None:
                        # 优化：避免 .cpu().item()，在 GPU 上计算
                        prev_mean = self.strategy.previous_e0_uncondition.abs().mean()
                        rel_l1_distance = ((modulated_inp - self.strategy.previous_e0_uncondition).abs().mean() / prev_mean)
                        # 只在需要时才移到 CPU
                        rel_l1_distance_cpu = rel_l1_distance.cpu().item()
                        rescale_func = np.poly1d(self.strategy.config.coefficients)
                        self.strategy.accumulated_rel_l1_distance_uncondition += rescale_func(rel_l1_distance_cpu)
                        
                        if self.strategy.accumulated_rel_l1_distance_uncondition < self.strategy.config.teacache_thresh:
                            should_use_cache = True
                        else:
                            should_use_cache = False
                            self.strategy.accumulated_rel_l1_distance_uncondition = 0
                    else:
                        should_use_cache = False
                else:
                    should_use_cache = False
            
            # 更新 previous_e0_uncondition
            if e0 is not None:
                self.strategy.previous_e0_uncondition = (e0 if self.strategy.config.use_ret_steps else e).clone()
        
        # 执行 blocks 计算或使用缓存
        if should_use_cache:
            # 使用缓存的残差（不需要 clone ori_x）
            if is_condition_step and self.strategy.previous_residual_condition is not None:
                x = x + self.strategy.previous_residual_condition
                self.strategy.cache_stats["hits"] += 1
                # 使用 debug 级别减少日志开销
                logger.debug(f"[TeaCache] ✓ Cache HIT for condition step {self.strategy.cnt} (rank={cfg_rank})")
            elif not is_condition_step and self.strategy.previous_residual_uncondition is not None:
                x = x + self.strategy.previous_residual_uncondition
                self.strategy.cache_stats["hits"] += 1
                logger.debug(f"[TeaCache] ✓ Cache HIT for uncondition step {self.strategy.cnt} (rank={cfg_rank})")
            else:
                # 缓存不存在，需要计算
                should_use_cache = False
                logger.debug(f"[TeaCache] ✗ Cache MISS for {'condition' if is_condition_step else 'uncondition'} step {self.strategy.cnt} (rank={cfg_rank}), computing blocks...")
                ori_x = x.clone()  # 只在需要计算残差时才 clone
                for block in self.original_blocks:
                    x = block(x, **kwargs)
                
                # 存储残差
                residual = x - ori_x
                if is_condition_step:
                    self.strategy.previous_residual_condition = residual.clone()
                else:
                    self.strategy.previous_residual_uncondition = residual.clone()
                self.strategy.cache_stats["stores"] += 1
                self.strategy.cache_stats["misses"] += 1
        else:
            # 计算 blocks（不在缓存范围内或阈值未满足）
            # 只在需要存储残差时才 clone
            ori_x = x.clone()
            logger.debug(f"[TeaCache] → Computing {'condition' if is_condition_step else 'uncondition'} step {self.strategy.cnt} (rank={cfg_rank})")
            for block in self.original_blocks:
                x = block(x, **kwargs)
            
            # 存储残差
            residual = x - ori_x
            if is_condition_step:
                self.strategy.previous_residual_condition = residual.clone()
            else:
                self.strategy.previous_residual_uncondition = residual.clone()
            self.strategy.cache_stats["stores"] += 1
            self.strategy.cache_stats["misses"] += 1
        
        return x
    
    def create_cached_blocks_modulelist(self) -> torch.nn.ModuleList:
        """
        创建一个 ModuleList，包含这个 CachedBlocks 实例
        用于替换原始的 model.blocks
        """
        return torch.nn.ModuleList([self])


def create_teacache_blocks(
    original_blocks: torch.nn.ModuleList,
    strategy: TeaCacheStrategy,
    model: torch.nn.Module,
) -> torch.nn.ModuleList:
    """
    创建 TeaCache blocks 包装器
    
    Args:
        original_blocks: 原始的 blocks ModuleList
        strategy: TeaCacheStrategy 实例
        model: 模型实例
    
    Returns:
        包装后的 blocks ModuleList
    """
    cached_blocks = TeaCacheBlocks(original_blocks, strategy, model)
    return cached_blocks.create_cached_blocks_modulelist()


# 为 TeaCacheStrategy 添加 create_cached_blocks 方法
def _create_cached_blocks(self, original_blocks: torch.nn.ModuleList, model: torch.nn.Module) -> torch.nn.ModuleList:
    """创建 cached blocks（由 CacheManager 调用）"""
    return create_teacache_blocks(original_blocks, self, model)

# 将方法添加到 TeaCacheStrategy 类
TeaCacheStrategy.create_cached_blocks = _create_cached_blocks


def init_teacache(config: Optional[TeaCacheConfig] = None) -> CacheManager:
    """
    初始化 TeaCache 并注册到全局管理器
    这是主要的入口函数，用于启用 TeaCache
    
    Args:
        config: TeaCacheConfig 配置对象，如果为 None 则使用默认配置
    
    Returns:
        CacheManager: 配置好的缓存管理器
    """
    manager = get_cache_manager()
    config = config or TeaCacheConfig()
    strategy = TeaCacheStrategy(config)
    manager.register_strategy("teacache", strategy)
    manager.set_active_strategy("teacache")
    return manager


def enable_teacache(
    enabled: bool = True,
    teacache_thresh: float = 0.2,
    use_ret_steps: bool = False,
    sample_steps: int = 50,
    task: str = 't2v',
    model: str = 'wan2.1-1.3B',
    enable_cfg_separate_cache: bool = True,
    max_cache_size: int = 100,
) -> CacheManager:
    """
    启用 TeaCache 的便捷函数
    
    Args:
        enabled: 是否启用缓存
        teacache_thresh: TeaCache 阈值，用于判断是否使用缓存
        use_ret_steps: 是否使用 ret_steps
        sample_steps: 采样步数
        task: 任务类型（t2v 或 i2v）
        model: 模型名称
        enable_cfg_separate_cache: 是否为CFG和非CFG分别缓存
        max_cache_size: 最大缓存条目数
    
    Example:
        >>> from chitu.diffusion.utils.teacache_utils import enable_teacache
        >>> enable_teacache(enabled=True, teacache_thresh=0.2, use_ret_steps=False)
    """
    config = TeaCacheConfig(
        enabled=enabled,
        teacache_thresh=teacache_thresh,
        use_ret_steps=use_ret_steps,
        sample_steps=sample_steps,
        task=task,
        model=model,
        enable_cfg_separate_cache=enable_cfg_separate_cache,
        max_cache_size=max_cache_size,
    )
    
    manager = init_teacache(config)
    enable_cache_for_backend()
    
    return manager


def get_cache_stats():
    """获取缓存统计信息"""
    manager = get_cache_manager()
    return manager.get_stats()


def clear_cache():
    """清空所有缓存"""
    manager = get_cache_manager()
    manager.clear_all_cache()
