import gc
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

import mindspore as ms
from mindspore import Tensor, context


KERNEL_SOURCE = Path(__file__).with_name("npu_mm_reduce_scatter_base.cc")


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bfloat16:
            return value.float().cpu().numpy()
        return value.cpu().numpy()
    if value.dtype == ms.bfloat16:
        return value.astype(ms.float32).asnumpy()
    return value.asnumpy()


def _make_inputs(rank, m=32, k=256, n=24):
    rng = np.random.default_rng(100 + rank)
    x1 = rng.normal(0, 0.2, (m, k)).astype(np.float16)
    x2 = rng.normal(0, 0.2, (k, n)).astype(np.float16)
    return x1, x2


def _run_rank(rank, world_size, master_port, source_path, comm_mode, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"

    try:
        torch_npu.npu.set_device(rank)
        torch.npu.set_compile_mode(jit_compile=False)
        dist.init_process_group(backend="hccl", world_size=world_size, rank=rank)
        group = dist.distributed_c10d._get_default_group()
        hcom_name = _get_hccl_comm_name(group, rank)

        context.set_context(device_target="Ascend", device_id=rank)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        custom_ops = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_mm_reduce_scatter_base_ws{world_size}_rank{rank}_{comm_mode}",
            [source_path],
            backend="Ascend",
        ).load()

        x1, x2 = _make_inputs(rank)
        expected = torch_npu.npu_mm_reduce_scatter_base(
            torch.from_numpy(x1).npu(),
            torch.from_numpy(x2).npu(),
            hcom_name,
            world_size,
            reduce_op="sum",
            bias=None,
            comm_turn=0,
            comm_mode=comm_mode,
        )
        actual = custom_ops.npu_mm_reduce_scatter_base(
            Tensor(x1),
            Tensor(x2),
            hcom_name,
            world_size,
            "sum",
            None,
            None,
            None,
            0,
            None,
            comm_mode,
        )
        np.testing.assert_allclose(_to_numpy(actual), _to_numpy(expected), rtol=5e-2, atol=5e-2)
        result_queue.put((rank, "ok", ""))
        dist.barrier()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result_queue.put((rank, "error", repr(exc)))
        raise
    finally:
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.npu.empty_cache()
        if hasattr(ms.hal, "empty_cache"):
            ms.hal.empty_cache()


def _run_world_size_2(comm_mode):
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

    world_size = 2
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(world_size)
    processes = []
    for rank in range(world_size):
        process = ctx.Process(
            target=_run_rank,
            args=(rank, world_size, master_port, str(KERNEL_SOURCE), comm_mode, result_queue),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=240)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail(f"npu_mm_reduce_scatter_base {comm_mode} test timed out")

    results = [result_queue.get(timeout=10) for _ in range(world_size)]
    errors = [item for item in results if item[1] != "ok"]
    if errors:
        pytest.fail(f"rank failure for comm_mode={comm_mode}: {errors}")
    for process in processes:
        assert process.exitcode == 0
    assert sorted(rank for rank, _, _ in results) == [0, 1]


def test_npu_mm_reduce_scatter_base_world_size_2_aiv():
    assert hasattr(torch_npu, "npu_mm_reduce_scatter_base")
    _run_world_size_2("aiv")


def test_npu_mm_reduce_scatter_base_world_size_2_ai_cpu():
    assert hasattr(torch_npu, "npu_mm_reduce_scatter_base")
    _run_world_size_2("ai_cpu")
