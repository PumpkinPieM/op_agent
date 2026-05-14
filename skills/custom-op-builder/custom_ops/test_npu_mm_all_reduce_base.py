import gc
import os
import socket
import traceback
from pathlib import Path

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
dist = pytest.importorskip("torch.distributed")
mp = pytest.importorskip("torch.multiprocessing")

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_mm_all_reduce_base.cc")
_CUSTOM_OPS = None


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_mm_all_reduce_base_test_{os.getpid()}",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def _np_from_torch(value):
    return value.float().cpu().numpy() if value.dtype == torch.bfloat16 else value.cpu().numpy()


def _np_from_ms(value):
    return value.astype(ms.float32).asnumpy() if value.dtype == ms.bfloat16 else value.asnumpy()


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def test_npu_mm_all_reduce_base_builder_loads():
    assert hasattr(_ops(), "npu_mm_all_reduce_base")


def _run_rank(rank, world_size, master_port, source_path, result_queue):
    process_group_initialized = False
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["HCCL_IF_BASE_PORT"] = str(master_port + 100)
        os.environ["HCCL_WHITELIST_DISABLE"] = "1"

        device_id = DEVICE_ID + rank
        torch_npu.npu.set_device(device_id)
        torch.npu.set_compile_mode(jit_compile=False)
        dist.init_process_group(backend="hccl", world_size=world_size, rank=rank)
        process_group_initialized = True
        group = dist.distributed_c10d._get_default_group()
        try:
            hcom = _get_hccl_comm_name(group, rank)
        except RuntimeError as exc:
            if "Communication_Error_Bind_IP_Port" in str(exc):
                result_queue.put(("skip", rank, str(exc)))
                return
            raise

        context.set_context(device_target="Ascend", device_id=device_id)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        custom_ops = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_mm_all_reduce_base_ws{world_size}_rank{rank}_{os.getpid()}",
            [source_path],
            backend="Ascend",
        ).load()

        cases = [
            {"dtype": "fp16", "with_bias": False, "with_x3": False, "seed": 31},
            {"dtype": "fp16", "with_bias": True, "with_x3": False, "seed": 37},
            {"dtype": "bf16", "with_bias": True, "with_x3": True, "seed": 41},
        ]
        for case in cases:
            rng = np.random.default_rng(case["seed"] + rank)
            x1_np = rng.normal(size=(2, 8)).astype(np.float32)
            x2_np = rng.normal(size=(8, 4)).astype(np.float32)
            bias_np = rng.normal(size=(4,)).astype(np.float32) if case["with_bias"] else None
            x3_np = rng.normal(size=(2, 4)).astype(np.float32) if case["with_x3"] else None

            torch_dtype = torch.bfloat16 if case["dtype"] == "bf16" else torch.float16
            ms_dtype = ms.bfloat16 if case["dtype"] == "bf16" else ms.float16
            x1_t = torch.from_numpy(x1_np).npu().to(torch_dtype)
            x2_t = torch.from_numpy(x2_np).npu().to(torch_dtype)
            torch_kwargs = {"reduce_op": "sum", "comm_turn": 0}
            ms_args = {"reduce_op": "sum", "comm_turn": 0}
            if bias_np is not None:
                torch_kwargs["bias"] = torch.from_numpy(bias_np).npu().to(torch_dtype)
                ms_args["bias"] = Tensor(bias_np).astype(ms_dtype)
            if x3_np is not None:
                torch_kwargs["x3"] = torch.from_numpy(x3_np).npu().to(torch_dtype)
                ms_args["x3"] = Tensor(x3_np).astype(ms_dtype)

            expected = torch_npu.npu_mm_all_reduce_base(x1_t, x2_t, hcom, **torch_kwargs)
            actual = custom_ops.npu_mm_all_reduce_base(
                Tensor(x1_np).astype(ms_dtype),
                Tensor(x2_np).astype(ms_dtype),
                hcom,
                ms_args.get("reduce_op"),
                ms_args.get("bias"),
                None,
                None,
                ms_args.get("x3"),
                None,
                None,
                None,
                None,
                0,
                ms_args["comm_turn"],
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            )
            expected_np = _np_from_torch(expected)
            actual_np = _np_from_ms(actual)
            assert expected_np.shape == actual_np.shape == (2, 4)
            np.testing.assert_allclose(expected_np, actual_np, rtol=1e-2, atol=1e-2)

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


def test_npu_mm_all_reduce_base_matches_torch_npu_hccl_group():
    if not hasattr(torch_npu, "npu_mm_all_reduce_base"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    if torch.npu.device_count() < DEVICE_ID + 2:
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
        process.join(timeout=300)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("HCCL world_size=2 test timed out")
        assert process.exitcode == 0

    results = [result_queue.get(timeout=10) for _ in range(world_size)]
    skips = [message for status, _, message in results if status == "skip"]
    if skips:
        pytest.skip(f"HCCL communicator port is busy on this host: {skips[0]}")
    failures = [item for item in results if item[0] != "ok"]
    assert not failures, "\n".join(f"rank {rank} failed:\n{error}" for _, rank, error in failures)
