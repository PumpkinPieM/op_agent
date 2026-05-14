import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
mp = pytest.importorskip("torch.multiprocessing")
torch_npu = pytest.importorskip("torch_npu")

import mindspore as ms
from mindspore import Tensor, context


KERNEL_SOURCE = Path(__file__).with_name("_npu_distribute_barrier.cc")


class HostCapabilitySkip(RuntimeError):
    pass


def _is_host_capability_error(exc):
    message = str(exc).lower()
    return (
        "nnopexecutor != nullptr failed" in message
        or "not support" in message
        or "unsupported" in message
        or "do not support" in message
    )


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _torch_tensor(array):
    return torch.from_numpy(np.array(array, copy=True)).npu()


def _ms_tensor(array):
    return Tensor(np.array(array, copy=True))


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_same_tensor(expected, actual, x_ref):
    expected_np = _np_from_torch(expected)
    actual_np = _np_from_ms(actual)
    np.testing.assert_array_equal(expected_np, np.array(x_ref, copy=True))
    np.testing.assert_array_equal(actual_np, expected_np)


def _run_case(custom_ops, case, torch_group_name, ms_group_name, world_size, rank):
    x_ref = case["x_ref"](rank)
    time_out = case.get("time_out", lambda _rank: None)(rank)
    elastic_info = case.get("elastic_info", lambda _rank: None)(rank)

    try:
        torch_expected = torch_npu._npu_distribute_barrier(
            x_ref=_torch_tensor(x_ref),
            group=torch_group_name,
            world_size=world_size,
            time_out=None if time_out is None else _torch_tensor(time_out),
            elastic_info=None if elastic_info is None else _torch_tensor(elastic_info),
        )
    except RuntimeError as exc:
        if _is_host_capability_error(exc):
            raise HostCapabilitySkip(f"torch_npu distribute barrier is not supported on this host: {exc}") from exc
        raise
    actual = custom_ops._npu_distribute_barrier(
        _ms_tensor(x_ref),
        ms_group_name,
        world_size,
        None if time_out is None else _ms_tensor(time_out),
        None if elastic_info is None else _ms_tensor(elastic_info),
    )
    _assert_same_tensor(torch_expected, actual, x_ref)


CASES = [
    {
        "id": "int32_default_optional",
        "x_ref": lambda rank: np.array([rank + 1], dtype=np.int32),
    },
    {
        "id": "float32_default_optional",
        "x_ref": lambda rank: np.array([rank + 1.0, rank + 2.0], dtype=np.float32),
    },
    {
        "id": "int32_with_timeout",
        "x_ref": lambda rank: np.array([rank + 3], dtype=np.int32),
        "time_out": lambda _rank: np.array([100000], dtype=np.int32),
    },
]


def _run_rank(rank, world_size, master_port, source_path, case_ids, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"

    try:
        torch_npu.npu.set_device(rank)
        torch.npu.set_compile_mode(jit_compile=False)
        dist.init_process_group(backend="hccl", world_size=world_size, rank=rank)
        ranks = list(range(world_size))
        torch_group = dist.new_group(backend="hccl", ranks=ranks)
        ms_group = dist.new_group(backend="hccl", ranks=ranks)
        torch_group_name = _get_hccl_comm_name(torch_group, rank)
        ms_group_name = _get_hccl_comm_name(ms_group, rank)

        context.set_context(device_target="Ascend", device_id=rank)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        custom_ops = ms.ops.CustomOpBuilder(
            f"custom_ops__npu_distribute_barrier_test_v2_ws{world_size}_rank{rank}",
            [source_path],
            backend="Ascend",
        ).load()

        selected_cases = [case for case in CASES if case["id"] in case_ids]
        for case in selected_cases:
            _run_case(custom_ops, case, torch_group_name, ms_group_name, world_size, rank)
            dist.barrier()

        result_queue.put(("ok", rank, None))
    except HostCapabilitySkip as exc:
        result_queue.put(("skip", rank, str(exc)))
    except Exception as exc:
        result_queue.put(("error", rank, repr(exc)))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_npu_distribute_barrier_hccl_world_size_2(case):
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

    world_size = 2
    master_port = int(os.environ.get("MASTER_PORT", str(51000 + os.getpid() % 1000)))
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(world_size)
    processes = []

    for rank in range(world_size):
        process = ctx.Process(
            target=_run_rank,
            args=(rank, world_size, master_port, str(KERNEL_SOURCE), [case["id"]], result_queue),
        )
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
    skips = [item for item in results if item[0] == "skip"]
    if skips:
        pytest.skip(skips[0][2])
    errors = [item for item in results if item[0] != "ok"]
    assert not errors, errors
    assert sorted(item[1] for item in results) == [0, 1]
