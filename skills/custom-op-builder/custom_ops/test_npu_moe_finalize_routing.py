import gc
import os
from pathlib import Path

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_moe_finalize_routing.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_moe_finalize_routing_test", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def _torch_tensor(arr):
    return torch.from_numpy(np.array(arr, copy=True)).npu()


def _ms_tensor(arr):
    return Tensor(np.array(arr, copy=True))


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_close(expected, actual, rtol=1e-4, atol=1e-4):
    expected_np = _np_from_torch(expected)
    actual_np = _np_from_ms(actual)
    assert expected_np.shape == actual_np.shape
    np.testing.assert_allclose(expected_np, actual_np, rtol=rtol, atol=atol)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row,
                             expert_for_source_row, drop_pad_mode):
    return _ops().npu_moe_finalize_routing(
        expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row,
        drop_pad_mode
    )


def _dropless_case():
    rows, top_k, hidden, expert_num = 5, 4, 5, 8
    expanded_permuted_rows = (np.arange(rows * top_k * hidden, dtype=np.float32).reshape(rows * top_k, hidden) / 10.0)
    skip1 = np.linspace(-1.0, 1.0, rows * hidden, dtype=np.float32).reshape(rows, hidden)
    skip2 = np.linspace(0.5, -0.5, rows * hidden, dtype=np.float32).reshape(rows, hidden)
    bias = np.linspace(-0.2, 0.3, expert_num * hidden, dtype=np.float32).reshape(expert_num, hidden)
    scales = np.full((rows, top_k), 0.25, dtype=np.float32)
    expanded_src_to_dst_row = np.array([2, 1, 4, 3, 0, 5, 8, 7, 6, 9, 12, 11, 10, 13, 16, 15, 14, 17, 18, 19],
                                       dtype=np.int32)
    expert_for_source_row = np.arange(rows * top_k, dtype=np.int32).reshape(rows, top_k) % expert_num
    return expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row, 0


def _drop_pad_case():
    rows, top_k, hidden, expert_num, capacity = 6, 1, 4, 4, 3
    expanded_permuted_rows = np.linspace(-1.0, 1.0, expert_num * capacity * hidden, dtype=np.float32).reshape(
        expert_num, capacity, hidden
    )
    skip1 = np.ones((rows, hidden), dtype=np.float32)
    skip2 = np.full((rows, hidden), -0.25, dtype=np.float32)
    bias = np.linspace(0.1, 0.4, expert_num * hidden, dtype=np.float32).reshape(expert_num, hidden)
    scales = None
    expanded_src_to_dst_row = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    expert_for_source_row = np.array([[0], [1], [2], [3], [0], [1]], dtype=np.int32)
    return expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row, 1


@pytest.mark.parametrize("case_fn", [_dropless_case, _drop_pad_case], ids=["dropless_scales", "drop_pad_no_scales"])
def test_npu_moe_finalize_routing_matches_torch_npu(case_fn):
    data = case_fn()
    torch_args = tuple(None if value is None else _torch_tensor(value) for value in data[:-1]) + (data[-1],)
    ms_args = tuple(None if value is None else _ms_tensor(value) for value in data[:-1]) + (data[-1],)

    expected = torch_npu.npu_moe_finalize_routing(*torch_args)
    actual = npu_moe_finalize_routing(*ms_args)
    _assert_close(expected, actual)
