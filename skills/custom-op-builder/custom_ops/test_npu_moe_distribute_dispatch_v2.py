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


KERNEL_SOURCE = Path(__file__).with_name("npu_moe_distribute_dispatch_v2.cc")


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def _assert_outputs_close(expected, actual):
    assert len(expected) == len(actual)
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        expected_np = _to_numpy(expected_tensor)
        actual_np = _to_numpy(actual_tensor)
        assert expected_np.shape == actual_np.shape, (
            f"output {index} shape mismatch: expected {expected_np.shape}, actual {actual_np.shape}"
        )
        assert expected_np.dtype == actual_np.dtype, (
            f"output {index} dtype mismatch: expected {expected_np.dtype}, actual {actual_np.dtype}"
        )
        np.testing.assert_array_equal(expected_np, actual_np)


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
        f"custom_ops_npu_moe_distribute_dispatch_v2_ws{world_size}_rank{rank}",
        [source_path],
        backend="Ascend",
    ).load()

    x_np = (np.arange(8 * 7168, dtype=np.float16).reshape(8, 7168) + rank * 100).astype(np.float16)
    expert_ids_np = np.tile(np.arange(4, dtype=np.int32), (8, 1))
    x = torch.from_numpy(x_np).npu()
    expert_ids = torch.from_numpy(expert_ids_np).npu()

    kwargs = dict(
        group_ep=hcom_name,
        ep_world_size=world_size,
        ep_rank_id=rank,
        moe_expert_num=4,
        scales=None,
        x_active_mask=None,
        expert_scales=None,
        elastic_info=None,
        performance_info=None,
        group_tp="",
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=0,
        quant_mode=0,
        global_bs=0,
        expert_token_nums_type=1,
        comm_alg="",
        zero_expert_num=0,
        copy_expert_num=0,
        const_expert_num=0,
        y_dtype=None,
        x_dtype=None,
        scales_dtype=None,
    )
    expected = torch_npu.npu_moe_distribute_dispatch_v2(x, expert_ids, **kwargs)
    actual = custom_ops.npu_moe_distribute_dispatch_v2(
        Tensor(x_np),
        Tensor(expert_ids_np),
        hcom_name,
        world_size,
        rank,
        4,
        None,
        None,
        None,
        None,
        None,
        "",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        "",
        0,
        0,
        0,
        None,
        None,
        None,
    )
    _assert_outputs_close(expected, actual)
    result_queue.put(rank)
    dist.barrier()
    dist.destroy_process_group()


def test_npu_moe_distribute_dispatch_v2_matches_torch_npu_world_size_2():
    if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
        pytest.skip("torch_npu on this host does not expose npu_moe_distribute_dispatch_v2")
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

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
            pytest.fail("HCCL world_size=2 dispatch test timed out")
        assert process.exitcode == 0

    ranks = [result_queue.get(timeout=10) for _ in range(world_size)]
    assert sorted(ranks) == [0, 1]
