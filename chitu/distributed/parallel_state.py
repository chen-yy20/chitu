# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Any, List, Dict

import torch
from logging import getLogger


from chitu.distributed.comm_group import CommGroup
from chitu.device_type import is_ascend

logger = getLogger(__name__)

_PARALLEL_GROUPS_INITIALIZED = False

_WORLD_GROUP: Optional[CommGroup] = None
_TP_GROUP: Optional[CommGroup] = None
_PP_GROUP: Optional[CommGroup] = None
_DP_GROUP: Optional[CommGroup] = None
_EP_GROUP: Optional[CommGroup] = None

_PP_PAIR_GROUP_DICT: dict[tuple[int, int], Any] = {}  # Compatible with NPU platforms


def get_global_var(name):
    var = globals().get(name)
    assert var is not None, f"global var {name} not initialized."
    return var


def get_world_group() -> CommGroup:
    return get_global_var("_WORLD_GROUP")


def get_tp_group() -> CommGroup:
    return get_global_var("_TP_GROUP")


def get_pp_group() -> CommGroup:
    return get_global_var("_PP_GROUP")


def get_dp_group() -> CommGroup:
    return get_global_var("_DP_GROUP")


def get_ep_group() -> CommGroup:
    return get_global_var("_EP_GROUP")


def get_tp_size() -> int:
    """return 1 if TP not initialized"""
    global _TP_GROUP
    if _TP_GROUP is None:
        return 1
    return _TP_GROUP.group_size


def get_etp_size() -> int:
    if get_ep_size() == 1:
        return get_tp_size()
    return 1


def get_etp_group() -> CommGroup:
    if get_ep_size() == 1:
        return get_tp_group()
    raise NotImplementedError


def get_dp_size() -> int:
    """return 1 if DP not initialized"""
    global _DP_GROUP
    if _DP_GROUP is None:
        return 1
    return _DP_GROUP.group_size


def get_ep_size() -> int:
    """return 1 if EP not initialized"""
    global _EP_GROUP
    if _EP_GROUP is None:
        return 1
    return _EP_GROUP.group_size


def get_pp_pair_group(
    rank0: int, rank1: int
) -> Optional[torch.distributed.ProcessGroup]:
    return _PP_PAIR_GROUP_DICT.get((rank0, rank1), None)


def get_cpu_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    return get_global_var("_TP_GROUP").cpu_group


def initialize_world_group(rank: int, local_rank: int, world_size: int):
    global _WORLD_GROUP
    assert _WORLD_GROUP is None

    _WORLD_GROUP = CommGroup([list(range(world_size))], rank, local_rank)
    # logger.info(f"world group: {_WORLD_GROUP}")


def initialize_tp_group(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    rank: int,
    local_rank: int,
    world_size: int,
):
    global _TP_GROUP
    assert _TP_GROUP is None

    num_tp_groups = world_size // tp_size

    rank_list = []
    for i in range(num_tp_groups):
        rank_list.append(list(range(i * tp_size, (i + 1) * tp_size)))

    _TP_GROUP = CommGroup(rank_list, rank, local_rank)


def initialize_pp_group(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    rank: int,
    local_rank: int,
    world_size: int,
):
    global _PP_GROUP
    assert _PP_GROUP is None

    num_pp_groups = world_size // pp_size
    num_dp_groups = world_size // dp_size

    rank_list = []
    for i in range(dp_size):
        for j in range(num_pp_groups // dp_size):
            rank_list.append(
                list(
                    range(
                        i * num_dp_groups + j,
                        (i + 1) * num_dp_groups,
                        num_pp_groups // dp_size,
                    )
                )
            )

    _PP_GROUP = CommGroup(rank_list, rank, local_rank)

    if is_ascend():
        assert len(_PP_PAIR_GROUP_DICT) == 0
        if pp_size < 2:
            return
        ranks = [i * tp_size for i in range(pp_size)]
        for i in range(pp_size):
            next_i = (i + 1) % pp_size
            rank_pair = [ranks[i], ranks[next_i]]
            pg = torch.distributed.new_group(rank_pair)
            _PP_PAIR_GROUP_DICT[(ranks[i], ranks[next_i])] = pg
            _PP_PAIR_GROUP_DICT[(ranks[next_i], ranks[i])] = pg


def initialize_dp_group(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    rank: int,
    local_rank: int,
    world_size: int,
):
    global _DP_GROUP
    assert _DP_GROUP is None

    num_DP_GROUPs = world_size // dp_size

    rank_list = []
    for i in range(num_DP_GROUPs):
        rank_list.append(list(range(i, world_size, num_DP_GROUPs)))

    _DP_GROUP = CommGroup(rank_list, rank, local_rank)


def initialize_ep_group(ep_size: int, rank: int, local_rank: int, world_size: int):
    global _EP_GROUP
    assert _EP_GROUP is None

    dp_size = get_dp_size()
    tp_size = get_tp_size()
    pp_size = get_pp_group().group_size

    assert (dp_size * tp_size) % ep_size == 0
    num_EP_GROUPs = world_size // ep_size

    if ep_size > 1:
        if pp_size == 1:
            pp_rank_list = [list(range(world_size))]
        else:
            pp_rank_list = [[] for _ in range(pp_size)]
            for i in range(world_size):
                pp_stage = i % (world_size // dp_size) // tp_size
                pp_rank_list[pp_stage].append(i)
        rank_list = []
        for pp_stage in range(pp_size):
            for i in range(num_EP_GROUPs // pp_size):
                rank_list.append(
                    pp_rank_list[pp_stage][i * ep_size : (i + 1) * ep_size]
                )
        _EP_GROUP = CommGroup(rank_list, rank, local_rank)
    else:
        _EP_GROUP = CommGroup([[idx] for idx in range(world_size)], rank, local_rank)


def initialize_parallel_groups(
    tp_size: int, pp_size: int, dp_size: int = 1, ep_size: int = 1
):
    global _PARALLEL_GROUPS_INITIALIZED
    assert not _PARALLEL_GROUPS_INITIALIZED

    logger.info(
        f"initialize_parallel_groups: {tp_size=}, {pp_size=}, {dp_size=} {ep_size=}"
    )
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    initialize_world_group(rank, local_rank, world_size)
    initialize_tp_group(tp_size, pp_size, dp_size, rank, local_rank, world_size)
    initialize_pp_group(tp_size, pp_size, dp_size, rank, local_rank, world_size)
    initialize_dp_group(tp_size, pp_size, dp_size, rank, local_rank, world_size)
    initialize_ep_group(ep_size, rank, local_rank, world_size)

    _PARALLEL_GROUPS_INITIALIZED = True


def parallel_groups_initialized():
    return _PARALLEL_GROUPS_INITIALIZED


def destroy_parallel_groups():
    get_tp_group().destroy()
    get_pp_group().destroy()
    get_world_group().destroy()
    get_dp_group().destroy()
    # Currently we don't destroy ep_group as it is a copy of tp/dp
    # get_ep_group().destroy()


# CFG and Context Parallelism support for diffusion models
_CP_GROUP: Optional[CommGroup] = None
_CFG_GROUP: Optional[CommGroup] = None
_UP_GROUP_DICT: Optional[Dict[int, CommGroup]] = None

def get_cp_group() -> CommGroup:
    return get_global_var("_CP_GROUP")

def get_cfg_group() -> CommGroup:
    return get_global_var("_CFG_GROUP")

def get_up_group(size: int) -> CommGroup:
    global _UP_GROUP_DICT
    if _UP_GROUP_DICT is None or size not in _UP_GROUP_DICT:
        raise ValueError(f"UP group of size {size} not initialized.")
    return _UP_GROUP_DICT[size]

def initialize_cfg_group(cfg_size: int, rank: int, local_rank: int, world_size: int):
    global _CFG_GROUP
    assert _CFG_GROUP is None
    
    if cfg_size == 1:
        # No CFG parallelism
        _CFG_GROUP = CommGroup([[idx] for idx in range(world_size)], rank, local_rank)
    elif cfg_size == 2:
        # CFG parallelism with pairs
        assert world_size % 2 == 0, "World size must be even for CFG parallelism"
        half_size = world_size // 2
        rank_list = []
        for i in range(half_size):
            rank_list.append([i, i + half_size])
        _CFG_GROUP = CommGroup(rank_list, rank, local_rank)
    else:
        raise ValueError("CFG size can only be 1 or 2")

def initialize_cp_group(cp_size: int, cfg_size: int, rank: int, local_rank: int, world_size: int):
    global _CP_GROUP
    assert _CP_GROUP is None
    
    if cfg_size == 2:
        # With CFG parallelism
        half_size = world_size // 2
        rank_list = [
            list(range(0, half_size)),           # First half
            list(range(half_size, world_size))   # Second half
        ]
    else:
        # No CFG parallelism - all ranks in single group
        rank_list = [list(range(world_size))]
    
    _CP_GROUP = CommGroup(rank_list, rank, local_rank)

def initialize_up_groups(up_dividers: List[int], up_limit: int, cfg_size: int, rank: int, local_rank: int, world_size: int):
    global _UP_GROUP_DICT
    assert _UP_GROUP_DICT is None
    
    _UP_GROUP_DICT = {}
    
    # Get CP group size
    cp_group = get_cp_group()
    cp_group_size = cp_group.group_size
    
    # Get CP group ranks
    if cfg_size == 2:
        half_size = world_size // 2
        cp_group_ranks = [
            list(range(0, half_size)),
            list(range(half_size, world_size))
        ]
    else:
        cp_group_ranks = [list(range(world_size))]
    
    for divider in up_dividers:
        if cp_group_size % divider != 0:
            continue
            
        up_stride = cp_group_size // divider
        if up_stride > up_limit:
            continue
            
        rank_list = []
        for cp_ranks in cp_group_ranks:
            for i in range(0, len(cp_ranks), up_stride):
                group = cp_ranks[i:i+up_stride]
                if group:
                    rank_list.append(group)
        
        if rank_list:
            _UP_GROUP_DICT[up_stride] = CommGroup(rank_list, rank, local_rank)

def initialize_diffusion_parallel_groups(
    cfg_size: int,
    cp_size: int,
    up_limit: int = 8,
):
    global _PARALLEL_GROUPS_INITIALIZED
    assert not _PARALLEL_GROUPS_INITIALIZED
    
    logger.info(
        f"initialize_diffusion_parallel_groups: {cfg_size=}, {cp_size=}, {up_limit=}"
    )
    
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    
    # Initialize groups in order
    initialize_cfg_group(cfg_size, rank, local_rank, world_size)
    initialize_cp_group(cp_size, cfg_size, rank, local_rank, world_size)
    # up_dividers = [1, 2, 4] # DiTango Support
    up_dividers = [1]
    initialize_up_groups(up_dividers, up_limit, cfg_size, rank, local_rank, world_size)
    
    # Debug logging
    if rank == 0:
        logger.info(f"CFG groups initialized: {get_cfg_group().rank_list}")
        logger.info(f"CP groups initialized: {get_cp_group().rank_list}")
        for size, up_group in _UP_GROUP_DICT.items():
            logger.info(f"UP group size {size}: {up_group.rank_list}")