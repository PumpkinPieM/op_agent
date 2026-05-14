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


KERNEL_SOURCE = Path(__file__).with_name("npu_all_gather_base_mm.cc")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _to_numpy(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _run_rank(rank, world_size, master_port, source_path, case, result_queue):
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
        f"custom_ops_npu_all_gather_base_mm_{case['id']}_ws{world_size}_rank{rank}",
        [source_path],
        backend="Ascend",
    ).load()

    rng = np.random.default_rng(1000 + rank)
    x1_np = rng.normal(size=(16, 256)).astype(np.float16)
    x2_np = rng.normal(size=(256, 32)).astype(np.float16)
    bias_np = rng.normal(size=(32,)).astype(np.float16) if case["bias"] else None

    x1_torch = torch.from_numpy(x1_np).npu()
    x2_torch = torch.from_numpy(x2_np).npu()
    bias_torch = torch.from_numpy(bias_np).npu() if bias_np is not None else None
    expected = torch_npu.npu_all_gather_base_mm(
        x1_torch,
        x2_torch,
        hcom_name,
        world_size,
        bias=bias_torch,
        x1_scale=None,
        x2_scale=None,
        gather_index=case["gather_index"],
        gather_output=case["gather_output"],
        comm_turn=0,
        output_dtype=None,
        comm_mode=case["comm_mode"],
    )

    actual = custom_ops.npu_all_gather_base_mm(
        Tensor(x1_np),
        Tensor(x2_np),
        hcom_name,
        world_size,
        Tensor(bias_np) if bias_np is not None else None,
        None,
        None,
        case["gather_index"],
        case["gather_output"],
        0,
        None,
        case["comm_mode"],
    )

    expected_np = [_to_numpy(item) for item in expected]
    actual_np = [item.asnumpy() for item in actual]
    for expected_item, actual_item in zip(expected_np, actual_np):
        assert expected_item.shape == actual_item.shape
        np.testing.assert_allclose(expected_item, actual_item, rtol=8e-2, atol=8e-2)

    result_queue.put(("ok", rank))
    dist.barrier()
    dist.destroy_process_group()


def _run_world_size_2(case):
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

    world_size = 2
    master_port = _find_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(world_size)
    processes = []
    for rank in range(world_size):
        process = ctx.Process(target=_run_rank, args=(rank, world_size, master_port, str(KERNEL_SOURCE), case, result_queue))
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


@pytest.mark.parametrize(
    "case",
    [
        {"id": "ai_cpu_gather", "comm_mode": "ai_cpu", "gather_index": 0, "gather_output": True, "bias": False},
        {"id": "ai_cpu_no_gather", "comm_mode": "ai_cpu", "gather_index": 0, "gather_output": False, "bias": True},
        {"id": "aiv_gather", "comm_mode": "aiv", "gather_index": 0, "gather_output": True, "bias": False},
    ],
)
def test_npu_all_gather_base_mm_matches_torch_npu(case):
    _run_world_size_2(case)
