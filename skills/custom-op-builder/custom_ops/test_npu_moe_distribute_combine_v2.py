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


KERNEL_SOURCE = Path(__file__).with_name("npu_moe_distribute_combine_v2.cc")


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
    hcom_name = _get_hccl_comm_name(group, rank)

    context.set_context(device_target="Ascend", device_id=rank)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    custom_ops = ms.ops.CustomOpBuilder(
        f"custom_ops_npu_moe_distribute_combine_v2_ws{world_size}_rank{rank}",
        [source_path],
        backend="Ascend",
    ).load()

    x_np = (np.arange(8 * 7168, dtype=np.float16).reshape(8, 7168) + rank * 100).astype(np.float16)
    expert_ids_np = np.tile(np.arange(4, dtype=np.int32), (8, 1))
    expert_scales_np = np.ones((8, 4), dtype=np.float32)
    x = torch.from_numpy(x_np).npu()
    expert_ids = torch.from_numpy(expert_ids_np).npu()
    expert_scales = torch.from_numpy(expert_scales_np).npu()

    dispatch = torch_npu.npu_moe_distribute_dispatch_v2(
        x,
        expert_ids,
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
    expand_x, _, assist_info_for_combine, _, ep_send_counts, tp_send_counts, expand_scales = dispatch
    expected = torch_npu.npu_moe_distribute_combine_v2(
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_counts,
        expert_scales,
        hcom_name,
        world_size,
        rank,
        4,
        tp_send_counts=None,
        x_active_mask=None,
        expand_scales=expand_scales,
        shared_expert_x=None,
        elastic_info=None,
        ori_x=None,
        const_expert_alpha_1=None,
        const_expert_alpha_2=None,
        const_expert_v=None,
        performance_info=None,
        group_tp="",
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=0,
        global_bs=0,
        comm_quant_mode=0,
        comm_alg="",
        zero_expert_num=0,
        copy_expert_num=0,
        const_expert_num=0,
    )
    actual = custom_ops.npu_moe_distribute_combine_v2(
        Tensor(expand_x.cpu().numpy()),
        Tensor(expert_ids_np),
        Tensor(assist_info_for_combine.cpu().numpy()),
        Tensor(ep_send_counts.cpu().numpy()),
        Tensor(expert_scales_np),
        hcom_name,
        world_size,
        rank,
        4,
        None,
        None,
        Tensor(expand_scales.cpu().numpy()),
        None,
        None,
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
        "",
        0,
        0,
        0,
    )
    np.testing.assert_allclose(expected.cpu().numpy(), actual.asnumpy(), rtol=0, atol=0)
    result_queue.put(rank)
    dist.barrier()
    dist.destroy_process_group()


def test_npu_moe_distribute_combine_v2_matches_torch_npu_world_size_2():
    if not hasattr(torch_npu, "npu_moe_distribute_combine_v2"):
        pytest.skip("torch_npu on this host does not expose npu_moe_distribute_combine_v2")
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

    world_size = 2
    master_port = int(os.environ.get("MASTER_PORT", str(52000 + os.getpid() % 1000)))
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
            pytest.fail("HCCL world_size=2 combine test timed out")
        assert process.exitcode == 0

    ranks = [result_queue.get(timeout=10) for _ in range(world_size)]
    assert sorted(ranks) == [0, 1]
