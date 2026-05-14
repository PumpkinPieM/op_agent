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


KERNEL_SOURCE = Path(__file__).with_name("npu_quant_mm_reduce_scatter.cc")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _run_rank(rank, world_size, master_port, source_path, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"

    torch_npu.npu.set_device(rank)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group(backend="hccl", world_size=world_size, rank=rank)
    group = dist.distributed_c10d._get_default_group()
    try:
        hcom_name = _get_hccl_comm_name(group, rank)
    except RuntimeError as exc:
        if "Communication_Error_Bind_IP_Port" in str(exc):
            result_queue.put(("skip", str(exc)))
            dist.destroy_process_group()
            return
        raise

    context.set_context(device_target="Ascend", device_id=rank)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    custom_ops = ms.ops.CustomOpBuilder(
        f"custom_ops_npu_quant_mm_reduce_scatter_ws{world_size}_rank{rank}",
        [source_path],
        backend="Ascend",
    ).load()

    rng = np.random.default_rng(100 + rank)
    x1 = rng.normal(size=(16, 256)).astype(np.float16)
    x2 = rng.normal(size=(256, 32)).astype(np.float16)
    expected_full = None
    for expected_rank in range(world_size):
        expected_rng = np.random.default_rng(100 + expected_rank)
        expected_x1 = expected_rng.normal(size=(16, 256)).astype(np.float16)
        expected_x2 = expected_rng.normal(size=(256, 32)).astype(np.float16)
        expected_part = expected_x1.astype(np.float32) @ expected_x2.astype(np.float32)
        expected_full = expected_part if expected_full is None else expected_full + expected_part
    rows_per_rank = expected_full.shape[0] // world_size
    expected = expected_full[rank * rows_per_rank : (rank + 1) * rows_per_rank].astype(np.float16)

    actual, actual_amax = custom_ops.npu_quant_mm_reduce_scatter(
        Tensor(x1),
        Tensor(x2),
        hcom_name,
        world_size,
        "sum",
        None,
        None,
        None,
        None,
        0,
        0,
        None,
        False,
        None,
        None,
        None,
        None,
        None,
    )

    np.testing.assert_allclose(expected, actual.asnumpy(), rtol=8e-2, atol=8e-2)
    assert actual_amax.asnumpy().shape == (1,)
    result_queue.put(("ok", rank))
    dist.barrier()
    dist.destroy_process_group()


def test_npu_quant_mm_reduce_scatter_world_size_2():
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

    world_size = 2
    master_port = _find_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(world_size)
    processes = []
    for rank in range(world_size):
        process = ctx.Process(target=_run_rank, args=(rank, world_size, master_port, str(KERNEL_SOURCE), result_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=240)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("HCCL world_size=2 test timed out")
        assert process.exitcode == 0

    results = [result_queue.get(timeout=10) for _ in range(world_size)]
    skips = [message for status, message in results if status == "skip"]
    if skips:
        pytest.skip(f"HCCL communicator port is busy on this host: {skips[0]}")
    ranks = [rank for status, rank in results if status == "ok"]
    assert sorted(ranks) == [0, 1]
