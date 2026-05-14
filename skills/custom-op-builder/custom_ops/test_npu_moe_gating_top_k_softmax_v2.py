import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_moe_gating_top_k_softmax_v2.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_moe_gating_top_k_softmax_v2_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_moe_gating_top_k_softmax_v2(x, k=1, finished=None, renorm=0, output_softmax=False):
    return _custom_ops.npu_moe_gating_top_k_softmax_v2(x, k, finished, renorm, output_softmax)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _softmax(x):
    max_value = np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x - max_value)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _cpu_reference(x, k, finished, renorm, output_softmax):
    num_experts = x.shape[-1]
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, num_experts)
    finished_1d = None if finished is None else finished.reshape(-1)

    full_softmax = _softmax(x_2d.astype(np.float32))
    if renorm == 1:
        indices = np.argsort(-x_2d, axis=-1, kind="stable")[:, :k].astype(np.int32)
        values = np.take_along_axis(x_2d, indices, axis=-1)
        y = _softmax(values.astype(np.float32)).astype(x.dtype)
    else:
        indices = np.argsort(-full_softmax, axis=-1, kind="stable")[:, :k].astype(np.int32)
        y = np.take_along_axis(full_softmax, indices, axis=-1).astype(x.dtype)

    if finished_1d is not None:
        indices = np.where(finished_1d.reshape(-1, 1), num_experts, indices)

    y = y.reshape(*leading_shape, k)
    indices = indices.reshape(*leading_shape, k)
    if renorm == 0 and output_softmax:
        softmax_result = full_softmax.reshape(*leading_shape, num_experts).astype(np.float32)
    else:
        softmax_result = np.empty((0,), dtype=np.float32)
    return y, indices, softmax_result


def _ms_tensor(array):
    return None if array is None else Tensor(array)


@pytest.mark.parametrize(
    "shape,k,renorm,output_softmax,with_finished,dtype",
    [
        ((4, 8), 2, 0, True, True, np.float16),
        ((4, 8), 3, 0, False, False, np.float32),
        ((2, 3, 6), 2, 1, True, True, np.float16),
        ((3, 5), 1, 0, True, False, np.float16),
    ],
)
def test_npu_moe_gating_top_k_softmax_v2_matches_cpu(shape, k, renorm, output_softmax, with_finished, dtype):
    if not hasattr(torch_npu, "npu_moe_gating_top_k_softmax_v2"):
        pytest.skip("torch_npu on this host does not expose npu_moe_gating_top_k_softmax_v2")
    rng = np.random.default_rng(2026 + len(shape) + k + renorm)
    x = rng.normal(size=shape).astype(dtype)
    finished = None
    if with_finished:
        finished = np.zeros(shape[:-1], dtype=np.bool_)
        finished.reshape(-1)[::2] = True

    expected = _cpu_reference(x, k, finished, renorm, output_softmax)
    actual = npu_moe_gating_top_k_softmax_v2(_ms_tensor(x), k, _ms_tensor(finished), renorm, output_softmax)

    actual_np = [item.asnumpy() for item in actual]
    np.testing.assert_allclose(expected[0], actual_np[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(expected[1], actual_np[1])
    np.testing.assert_allclose(expected[2], actual_np[2], rtol=2e-3, atol=2e-3)
