import os
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

import mindspore as ms
from mindspore import Tensor, context


KERNEL_SOURCE = Path(__file__).with_name("npu_all_to_all_quant_matmul.cc")
TORCH_FLOAT16 = 5
BASE_DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _make_inputs(rank, case):
    rng = np.random.default_rng(20260518 + rank * 31 + case["seed"])
    x1 = rng.normal(0, 0.2, case["x1_shape"]).astype(np.float16)
    x2 = rng.integers(-8, 8, case["x2_shape"], dtype=np.int8)
    x2_scale = rng.uniform(0.01, 0.05, (case["x2_shape"][1],)).astype(np.float32)
    bias = None
    if case["with_bias"]:
        bias = rng.normal(0, 0.2, (case["x2_shape"][1],)).astype(np.float32)
    return x1, x2, x2_scale, bias


def _to_np(value):
    if value is None:
        return None
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _to_ms_np(value):
    if value is None:
        return None
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_outputs_close(expected, actual, check_alltoall):
    assert len(actual) == 2
    expected_out = _to_np(expected[0])
    actual_out = _to_ms_np(actual[0])
    assert actual_out.shape == expected_out.shape
    np.testing.assert_allclose(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    if check_alltoall:
        expected_comm = _to_np(expected[1])
        actual_comm = _to_ms_np(actual[1])
        assert actual_comm.shape == expected_comm.shape
        np.testing.assert_allclose(actual_comm, expected_comm, rtol=1e-2, atol=1e-2)


def npu_all_to_all_quant_matmul(
    custom_ops,
    x1,
    x2,
    hcom,
    world_size,
    all2all_out_flag=True,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    common_scale=None,
    x1_offset=None,
    x2_offset=None,
    x1_quant_mode=None,
    x2_quant_mode=None,
    common_quant_mode=None,
    group_sizes=None,
    all2all_axes=None,
    comm_quant_dtype=None,
    x1_quant_dtype=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    output_scale_dtype=None,
    comm_scale_dtype=None,
    y_dtype=None,
):
    result = custom_ops.npu_all_to_all_quant_matmul(
        x1,
        x2,
        hcom,
        world_size,
        all2all_out_flag,
        bias,
        x1_scale,
        x2_scale,
        common_scale,
        x1_offset,
        x2_offset,
        x1_quant_mode,
        x2_quant_mode,
        common_quant_mode,
        group_sizes,
        all2all_axes,
        comm_quant_dtype,
        x1_quant_dtype,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        output_scale_dtype,
        comm_scale_dtype,
        y_dtype,
    )
    if not all2all_out_flag:
        return (result[0], None)
    return result


def _run_rank(rank, world_size, master_port, cases, source_path, result_queue):
    process_group_initialized = False
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["HCCL_IF_BASE_PORT"] = str(master_port + 100)
        os.environ["HCCL_WHITELIST_DISABLE"] = "1"

        device_id = BASE_DEVICE_ID + rank
        torch_npu.npu.set_device(device_id)
        torch.npu.set_compile_mode(jit_compile=False)
        dist.init_process_group(backend="hccl", world_size=world_size, rank=rank)
        process_group_initialized = True
        group = dist.distributed_c10d._get_default_group()
        hcom = _get_hccl_comm_name(group, rank)

        context.set_context(device_target="Ascend", device_id=device_id)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        custom_ops = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_all_to_all_quant_matmul_ws{world_size}_rank{rank}",
            [source_path],
            backend="Ascend",
        ).load()

        for case in cases:
            x1_np, x2_np, x2_scale_np, bias_np = _make_inputs(rank, case)
            x1_t = torch.from_numpy(x1_np).npu()
            x2_t = torch.from_numpy(x2_np).npu()
            x2_scale_t = torch.from_numpy(x2_scale_np).npu()
            bias_t = torch.from_numpy(bias_np).npu() if bias_np is not None else None
            expected = torch_npu.npu_all_to_all_quant_matmul(
                x1_t,
                x2_t,
                hcom,
                world_size,
                all2all_out_flag=case["out_flag"],
                bias=bias_t,
                x1_scale=None,
                x2_scale=x2_scale_t,
                common_scale=None,
                x1_offset=None,
                x2_offset=None,
                x1_quant_mode=7,
                x2_quant_mode=2,
                common_quant_mode=0,
                group_sizes=None,
                all2all_axes=case["axes"],
                comm_quant_dtype=None,
                x1_quant_dtype=None,
                x1_dtype=None,
                x2_dtype=None,
                x1_scale_dtype=None,
                x2_scale_dtype=None,
                output_scale_dtype=None,
                comm_scale_dtype=None,
                y_dtype=TORCH_FLOAT16,
            )

            actual = npu_all_to_all_quant_matmul(
                custom_ops,
                Tensor(x1_np),
                Tensor(x2_np),
                hcom,
                world_size,
                case["out_flag"],
                Tensor(bias_np) if bias_np is not None else None,
                None,
                Tensor(x2_scale_np),
                None,
                None,
                None,
                7,
                2,
                0,
                None,
                case["axes"],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                TORCH_FLOAT16,
            )
            _assert_outputs_close(expected, actual, case["out_flag"])
        result_queue.put(("ok", rank, None))
        dist.barrier()
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if process_group_initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                pass


CASES = [
    {
        "id": "dynamic_a16w8_bias_axes_out",
        "seed": 1,
        "x1_shape": (16, 32),
        "x2_shape": (64, 24),
        "with_bias": True,
        "axes": [-2, -1],
        "out_flag": True,
    },
    {
        "id": "dynamic_a16w8_bias_default_axes_no_out",
        "seed": 2,
        "x1_shape": (8, 16),
        "x2_shape": (32, 8),
        "with_bias": True,
        "axes": None,
        "out_flag": False,
    },
]


def test_npu_all_to_all_quant_matmul_matches_torch_npu():
    if not hasattr(torch_npu, "npu_all_to_all_quant_matmul"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    if torch.npu.device_count() < BASE_DEVICE_ID + 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

    world_size = 2
    master_port = 31000 + (os.getpid() % 10000)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(world_size)
    processes = []
    for rank in range(world_size):
        process = ctx.Process(target=_run_rank, args=(rank, world_size, master_port, CASES, str(KERNEL_SOURCE), result_queue))
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
    failures = [item for item in results if item[0] != "ok"]
    assert not failures, "\n".join(f"rank {rank} failed:\n{error}" for _, rank, error in failures)
