# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import operator
import os
from logging import getLogger
import psutil

import torch
import torch.distributed

from chitu.backend import BackendState

from chitu.device_type import is_nvidia
from chitu.global_vars import (
    get_global_args,
    set_global_variables,
    set_quant_variables,
    set_backend_variables,
)
# from chitu.task import (
#     PackedTasks,
#     PackedTasksBase,
#     SerializedPackedTasksPayloadType,
#     BatchResult,
#     Task,
#     TaskPool,
#     TaskType,
#     UserRequest,
#     MockFixedLengthedUserRequest,
#     DPTaskCollector,
# )
from chitu.utils import (
    gen_req_id,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
    ceil_div,
)
from chitu.schemas.utils import ModelConfigResolver
from chitu.utils import ceil_div
from chitu.distributed.parallel_state import get_dp_group
from chitu.logging_utils import setup_chitu_logging

from chitu.diffusion.task import DiffusionTask, DiffusionTaskPool
from chitu.diffusion.backend import DiffusionBackend as Backend
from chitu.diffusion.generator import Generator
from chitu.diffusion.scheduler import DiffusionScheduler

numa, has_numa = try_import_opt_dep("numa", "cpu")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")


logger = getLogger(__name__)


def init_logger(logging_level=logging.INFO):
    setup_chitu_logging()

    base_name = __name__.split(".")[0]
    base_logger = getLogger(base_name)
    base_logger.setLevel(logging_level)

    if base_logger.handlers:
        for handler in base_logger.handlers[:]:
            base_logger.removeHandler(handler)

    root_logger = getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            base_logger.addHandler(handler)


def init_cache_static():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)


def warmup_diffusion_engine(args):
    # TODO: Why we need warmup and how?
    pass


def check_checkpoint_path(args):
    '''
    Check path: VAE, Text Encoder, Transformers
    '''

    if args.models.ckpt_dir is None:
        raise ValueError(
            f"No checkpoint path provided. You can set it in command line by adding `models.ckpt_dir=<path>`. The model {args.models.name} can be downloaded from {args.models.source}"
        )


def chitu_init(args, logging_level=None):
    '''
    1. Init logger 
    2. Set parameters and environment variables
    3. Load models from checkpoint
    '''

    debug = os.getenv("CHITU_DEBUG", "0") == "1"

    if (
        is_nvidia()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() <= (2, 21, 5)
    ):
        os.environ["NCCL_NVLS_NCHANNELS"] = "32"

    if logging_level is None:
        logging_level = logging.DEBUG if debug else logging.INFO
    init_logger(logging_level)

    # Deal with legacy arguments
    if hasattr(args.infer, "soft_fp8") and args.infer.soft_fp8:
        logger.warning(
            "Argument `infer.soft_fp8=True` is deprecated. Use `infer.raise_lower_bit_float_to=bfloat16` instead."
        )
        args.infer.raise_lower_bit_float_to = "bfloat16"
    if hasattr(args, "dtype") and args.dtype is not None:
        logger.warning(
            "Argument `dtype` is deprecated. Use `float_16bit_variant` instead."
        )
        args.float_16bit_variant = args.dtype
    if hasattr(args.infer, "do_load") and not args.infer.do_load:
        logger.warning(
            "Argument `infer.do_load=False` is deprecated. Use `debug.skip_model_load=True` instead."
        )
        args.debug.skip_model_load = True

    # No chunked prefill in diffusion models

    # args.infer.has_schedule_overlap = (
    #     args.infer.dp_size <= 1 and args.infer.pp_size <= 1
    # )

    # Bind process to CPU NUMA
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if args.infer.bind_process_to_cpu == "auto":
        if not has_cpuinfer and not has_numa:
            args.infer.bind_process_to_cpu = "none"
        elif not has_numa:
            logger.warning(
                "'cpuinfer' is found but 'numa' is missing. Disabling NUMA binding. "
                "For better CPU inference performance, please refer to README.md and "
                "install the full '[cpu]' optional dependency."
            )
            args.infer.bind_process_to_cpu = "none"
        elif not numa.available():
            logger.warning(
                "NUMA is not support on this OS or hardware platform. Disabling NUMA binding."
            )
            args.infer.bind_process_to_cpu = "none"
        elif numa.get_max_node() + 1 < local_world_size:
            logger.info("Disable NUMA binding due to insufficient NUMA nodes.")
            args.infer.bind_process_to_cpu = "none"
        else:
            args.infer.bind_process_to_cpu = "numa"
    if args.infer.bind_process_to_cpu == "numa":
        numa.bind({local_rank})
    elif args.infer.bind_process_to_cpu == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported infer.bind_process_to_cpu={args.infer.bind_process_to_cpu}"
        )

    # TODO: Support cuda graph

    # Check checkpoint exists
    check_checkpoint_path(args)

    # Parse model configuration, supporting dynamic reading from config.json files
    # Uses $(config.json:field_name) syntax, e.g., n_heads: "$(config.json:head_dim)"
    model_resolver = ModelConfigResolver()
    args.models = model_resolver.process_config_dict(args.models, args.models.ckpt_dir)

    set_quant_variables(args)
    set_backend_variables(args)
    set_global_variables(args, debug=debug)

    args = get_global_args()
    Backend.build(args)
    
    # Naive Diffusion scheduler
    rank = torch.distributed.get_rank()
    if rank == 0:
        scheduler = DiffusionScheduler.build(args.infer.diffusion)
        Backend.scheduler = scheduler
    
    generator = Generator.build(args)
    
    Backend.generator = generator
    
    # TODO: Support batched generation
    # PackedTasks.configure(max_num_tasks=args.infer.max_reqs)
    
    logger.info("Chitu has been initialized")


@torch.inference_mode()
def chitu_run_normal():
    # 这个是rank0的执行，包括获取任务，preprocess，step，postprocess
    # TODO: scheduler 和 generator.step的联动方式
    # scheduler 给出本轮可以计算的task_ids
    task_ids = Backend.scheduler.schedule()
    logger.info(f"[run] scheduled task_ids={task_ids}")
    
    # 再基于task_ids给出打包
    if task_ids:
        # compute
        logger.debug(f"Processing {task_ids}")
        task = DiffusionTaskPool.pool[task_ids[0]]
        out = Backend.generator.step(task)
        logger.debug(f"[run] executor.step returned. {out.shape=}")
        # postprocess        
    else:
        logger.debug("No tasks scheduled in this round.")


@torch.inference_mode()
def chitu_generate():
    rank = torch.distributed.get_rank()
    if rank != 0:
        # 不需要传入任务，dispatcher会完成
        Backend.generator.step(None) 
        return

    # 其他只需要step，rank0则需要preprocess+step+postprocess
    chitu_run_normal()

# async def start_enhanced_scheduler_service(rank: int, dp_config, args):
#     # only main rank of dp group start enhanced scheduler service
#     dp_id = args.dp_config.dp_id
#     if rank != 0:
#         logger.warning(
#             f"[Enhanced Scheduler {dp_id}] only main rank of dp group start Enhanced Scheduler service"
#         )
#         return

#     """Start Enhanced Scheduler service, listen to ZMQ requests"""
#     import zmq
#     import zmq.asyncio
#     import msgpack
#     import time

#     logger.warning(f"[Enhanced Scheduler {dp_id}] Starting...")

#     # Initialize ZMQ
#     context = zmq.asyncio.Context()

#     # Receive request socket
#     request_socket = context.socket(zmq.PULL)
#     request_port = dp_config.scheduler_base_port
#     request_address = f"tcp://{dp_config.scheduler_base_host}:{request_port}"
#     request_socket.bind(request_address)
#     logger.warning(
#         f"[Enhanced Scheduler {dp_id}] Listening to requests: {request_address}"
#     )

#     # Send statistics socket
#     stats_socket = context.socket(zmq.PUSH)
#     stats_address = f"tcp://{dp_config.router.host}:{dp_config.router.stats_port}"  # Router stats port
#     stats_socket.connect(stats_address)
#     logger.warning(
#         f"[Enhanced Scheduler {dp_id}] connected to stats service: {stats_address}"
#     )

#     # Start DP Token Manager
#     try:
#         from chitu.dp_token_sender import start_dp_token_manager

#         dp_id = get_global_args().dp_config.dp_id
#         router_token_address = f"tcp://{dp_config.router.host}:{dp_config.router.token_port}"  # Token Router listen address

#         logger.warning(
#             f"[Enhanced Scheduler {dp_id}] Starting DP Token Manager, group ID={dp_id}"
#         )
#         await start_dp_token_manager(dp_id, router_token_address)
#         logger.warning(
#             f"[Enhanced Scheduler {dp_id}] DP Token Manager started successfully"
#         )
#     except Exception as e:
#         logger.error(
#             f"[Enhanced Scheduler {dp_id}] DP Token Manager failed to start: {e}"
#         )
#         # print stack trace
#         import traceback

#         logger.error(
#             f"[Enhanced Scheduler {dp_id}] DP Token Manager failed to start: {traceback.format_exc()}"
#         )
#         return

#     # Performance statistics
#     processed_requests = 0
#     start_time = time.time()

#     logger.warning(
#         f"[Enhanced Scheduler {dp_id}] Starting to process requests, scheduler listening to requests on {request_address}"
#     )
#     try:
#         while True:
#             # Check if there are requests
#             if await request_socket.poll(timeout=100):  # 100ms timeout
#                 try:
#                     # Receive request
#                     data = await request_socket.recv()
#                     request_data = msgpack.unpackb(data, raw=False)

#                     logger.info(
#                         f"[Enhanced Scheduler {dp_id}] Received request: {request_data.get('request_id', 'unknown')}"
#                     )

#                     # Process request
#                     await process_scheduler_request(rank, request_data)
#                     processed_requests += 1

#                 except Exception as e:
#                     logger.error(
#                         f"[Enhanced Scheduler {dp_id}] Failed to process request: {e}"
#                     )

#             # Send statistics periodically
#             current_time = time.time()
#             # Send statistics every second
#             if (current_time - start_time) >= 1.0:
#                 elapsed = current_time - start_time
#                 throughput = processed_requests / elapsed

#                 stats = {
#                     "scheduler_id": dp_config.dp_id,
#                     "running_requests": (
#                         len(Backend.ongoing_reqs)
#                         if hasattr(Backend, "ongoing_reqs")
#                         else 0
#                     ),
#                     "waiting_requests": (
#                         len(getattr(Backend.scheduler, "waiting_queue", []))
#                         if hasattr(Backend, "scheduler") and Backend.scheduler
#                         else 0
#                     ),
#                     "pending_tokens": 0,  # TODO: calculate pending tokens
#                     "throughput_tokens_per_sec": throughput,
#                     "last_update_time": current_time,
#                     "heartbeat": True,
#                 }

#                 try:
#                     stats_data = msgpack.packb(stats)
#                     await stats_socket.send(stats_data)
#                     logger.debug(
#                         f"[Enhanced Scheduler {dp_id}] throughput: {throughput:.2f}"
#                     )
#                 except Exception as e:
#                     logger.error(
#                         f"[Enhanced Scheduler {dp_id}] throughput send failed: {e}"
#                     )

#                 # Reset counter
#                 processed_requests = 0
#                 start_time = current_time

#     except KeyboardInterrupt:
#         logger.warning(f"[Enhanced Scheduler {dp_id}] Received interrupt signal")
#     except Exception as e:
#         logger.error(f"[Enhanced Scheduler {dp_id}] Service exception: {e}")
#     finally:
#         # Clean up resources
#         request_socket.close()
#         stats_socket.close()
#         context.term()
#         logger.warning(f"[Enhanced Scheduler {dp_id}] Service stopped")


# async def process_scheduler_request(rank: int, request_data: dict):
#     """Handle scheduling requests from Router"""
#     try:
#         # Build UserRequest object
#         request_id = request_data.get("request_id", gen_req_id())
#         message = request_data.get("message", [])
#         max_new_tokens = request_data.get("max_new_tokens", 50)
#         temperature = request_data.get("temperature", 1.0)
#         top_p = request_data.get("top_p", 1.0)
#         top_k = request_data.get("top_k", 50)
#         logprobs = request_data.get("logprobs", False)
#         top_logprobs = request_data.get("top_logprobs", None)

#         # Create UserRequest
#         user_request = UserRequest(
#             message=message,
#             request_id=request_id,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             logprobs=logprobs,
#             top_logprobs=top_logprobs,
#         )

#         # Create Task, honoring stop/ignore_eos semantics from request_data
#         stop_with_eos = True
#         if request_data.get("ignore_eos"):
#             stop_with_eos = False
#         elif not request_data.get("stop_with_eos"):
#             stop_with_eos = False

#         task = Task(task_id=request_id, req=user_request, stop_with_eos=stop_with_eos)

#         try:
#             from chitu.dp_token_sender import get_dp_token_manager

#             dp_id = get_global_args().dp_config.dp_id
#             token_manager = get_dp_token_manager(dp_id)
#             # ensure token manager started
#             await token_manager.start()
#             if token_manager is not None:
#                 # Wrap Task to enable token sending
#                 wrapped_task = token_manager.wrap_task(task)
#                 TaskPool.add(wrapped_task)
#             else:
#                 # If Token Manager not initialized, add the original Task directly
#                 TaskPool.add(task)

#         except Exception as e:
#             # If DP Token Manager acquisition fails, fall back to original Task
#             logger.error(
#                 f"[Enhanced Scheduler {dp_id}] Failed to get DP Token Manager: {e}"
#             )
#             TaskPool.add(task)
#             logger.warning(
#                 f"[Enhanced Scheduler {dp_id}] Fallback to original task: {request_id}"
#             )

#         logger.debug(f"[Enhanced Scheduler {dp_id}] Request handled: {request_id}")

#     except Exception as e:
#         logger.error(f"[Enhanced Scheduler {dp_id}] Failed to process request: {e}")
#         import traceback

#         logger.error(
#             f"[Enhanced Scheduler {dp_id}] Error details: {traceback.format_exc()}"
#         )


# def chitu_start():
#     Backend.state = BackendState.Running


# def chitu_terminate():
#     if torch.distributed.get_rank() == 0:
#         Backend.state = BackendState.Terminated
#         terminated_task = PackedTasksBase(
#             num_tasks=0,
#             payload_type=SerializedPackedTasksPayloadType.TerminateBackend,
#         )
#         Backend.executor.step(terminated_task)


# def chitu_is_terminated():
#     return Backend.state == BackendState.Terminated
