import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import mindspore as ms
from mindspore import Tensor, context

KERNEL_SOURCE = Path(__file__).with_name("npu_quant_matmul_all_to_all.cc")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _torch_tensor(array, dtype=None):
    tensor = torch.tensor(array)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.npu()


def _ms_tensor(array, dtype=None):
    if dtype is None:
        return Tensor(array)
    return Tensor(array, dtype)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.float().cpu().numpy()
    if hasattr(value, "asnumpy"):
        return value.astype(ms.float32).asnumpy()
    return np.asarray(value)


def _make_case(rank):
    rng = np.random.default_rng(rank)
    m, k, n = 16, 32, 32
    x1 = rng.integers(-5, 5, size=(m, k), dtype=np.int8)
    x2 = rng.integers(-5, 5, size=(k, n), dtype=np.int8)
    x1_scale = rng.uniform(0.1, 1, size=(m,)).astype(np.float32)
    x2_scale = rng.uniform(0.1, 1, size=(n,)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32)
    return x1, x2, x1_scale, x2_scale, bias


def _run_rank(rank, world_size, master_port, source_path, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"

    torch_npu.npu.set_device(rank)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group(backend="hccl", world_size=world_size, rank=rank)
    group = dist.distributed_c10d._get_default_group()
    hcom_name = _get_hccl_comm_name(group, rank)

    context.set_context(device_target="Ascend", device_id=rank)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    custom_ops = ms.ops.CustomOpBuilder(
        f"custom_ops_npu_quant_matmul_all_to_all_ws2_rank{rank}",
        [source_path],
        backend="Ascend",
    ).load()

    x1, x2, x1_scale, x2_scale, bias = _make_case(rank)
    torch_args = (
        _torch_tensor(x1),
        _torch_tensor(x2),
        hcom_name,
        world_size,
        _torch_tensor(bias),
        _torch_tensor(x1_scale),
        _torch_tensor(x2_scale),
        None,
        None,
        None,
        3,
        2,
        0,
        [0, 0, 0],
        [-1, -2],
        -1,
        None,
        None,
        None,
        None,
        None,
        None,
        15,
    )
    ms_args = (
        _ms_tensor(x1),
        _ms_tensor(x2),
        hcom_name,
        world_size,
        _ms_tensor(bias),
        _ms_tensor(x1_scale),
        _ms_tensor(x2_scale),
        None,
        None,
        None,
        3,
        2,
        0,
        [0, 0, 0],
        [-1, -2],
        -1,
        None,
        None,
        None,
        None,
        None,
        None,
        15,
    )

    expected = torch_npu.npu_quant_matmul_all_to_all(*torch_args)
    expected_np = _to_numpy(expected)
    dist.barrier()

    actual = custom_ops.npu_quant_matmul_all_to_all(*ms_args)
    actual_np = _to_numpy(actual)
    np.testing.assert_allclose(expected_np, actual_np, rtol=1e-2, atol=1e-2, equal_nan=True)
    result_queue.put((rank, expected_np.shape, actual_np.shape, float(np.max(np.abs(expected_np - actual_np)))))
    dist.barrier()
    dist.destroy_process_group()


def test_npu_quant_matmul_all_to_all_world_size_2_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_quant_matmul_all_to_all")
    if torch.npu.device_count() < 2:
        pytest.skip("npu_quant_matmul_all_to_all world_size=2 requires at least two NPU devices")

    world_size = 2
    master_port = int(os.environ.get("MASTER_PORT", str(51000 + os.getpid() % 1000)))
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
            pytest.fail("npu_quant_matmul_all_to_all world_size=2 test timed out")
        assert process.exitcode == 0

    results = [result_queue.get(timeout=10) for _ in range(world_size)]
    assert sorted(rank for rank, _, _, _ in results) == [0, 1]
