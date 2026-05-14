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
KERNEL_SOURCE = Path(__file__).with_name("npu_kronecker_quant.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_kronecker_quant")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_kronecker_quant_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_kronecker_quant(x, kronecker_p1, kronecker_p2, clip_ratio=None, dst_dtype=None):
    return _custom_ops.npu_kronecker_quant(x, kronecker_p1, kronecker_p2, clip_ratio, dst_dtype)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(array, dtype):
    tensor = torch.from_numpy(array).npu()
    return tensor.to(torch.bfloat16) if dtype == "bf16" else tensor


def _ms_tensor(array, dtype):
    tensor = Tensor(array)
    return tensor.astype(ms.bfloat16) if dtype == "bf16" else tensor


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.float().cpu().numpy() if value.dtype == torch.bfloat16 else value.cpu().numpy()
    if value.dtype == ms.bfloat16:
        return value.astype(ms.float32).asnumpy()
    return value.asnumpy()


def _assert_outputs_close(expected, actual):
    assert len(expected) == len(actual) == 2
    for expected_tensor, actual_tensor in zip(expected, actual):
        expected_np = _to_numpy(expected_tensor)
        actual_np = _to_numpy(actual_tensor)
        assert expected_np.shape == actual_np.shape
        np.testing.assert_allclose(expected_np, actual_np, rtol=0, atol=0)


def _case(dtype):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(2, 3, 16)).astype(np.float16)
    p1 = np.eye(3, dtype=np.float16)
    p2 = np.eye(16, dtype=np.float16)
    return (
        (_torch_tensor(x, dtype), _torch_tensor(p1, dtype), _torch_tensor(p2, dtype)),
        (_ms_tensor(x, dtype), _ms_tensor(p1, dtype), _ms_tensor(p2, dtype)),
    )


@pytest.mark.skipif(not HAS_TORCH_NPU_INTERFACE, reason="torch_npu does not expose this interface")
@pytest.mark.parametrize("dtype,clip_ratio", [("fp16", None), ("fp16", 0.75), ("bf16", 1.0)])
def test_npu_kronecker_quant_against_torch_npu(dtype, clip_ratio):
    torch_args, ms_args = _case(dtype)
    expected = torch_npu.npu_kronecker_quant(*torch_args, clip_ratio)
    actual = npu_kronecker_quant(*ms_args, clip_ratio, None)
    _assert_outputs_close(expected, actual)


def test_npu_kronecker_quant_output_metadata():
    _, ms_args = _case("fp16")
    out, quant_scale = npu_kronecker_quant(*ms_args, None, None)
    assert out.asnumpy().shape == (2, 3, 2)
    assert quant_scale.asnumpy().shape == (2,)
