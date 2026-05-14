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
KERNEL_SOURCE = Path(__file__).with_name("npu_hans_encode.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_hans_encode")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_hans_encode_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_hans_encode(x, pdf, statistic=False, reshuff=False):
    return _custom_ops.npu_hans_encode(x, pdf, statistic, reshuff)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(array, dtype):
    tensor = torch.from_numpy(array).npu()
    if dtype == "bf16":
        return tensor.to(torch.bfloat16)
    if dtype == "fp16":
        return tensor.to(torch.float16)
    return tensor.to(torch.float32)


def _ms_tensor(array, dtype):
    tensor = Tensor(array)
    if dtype == "bf16":
        return tensor.astype(ms.bfloat16)
    if dtype == "fp16":
        return tensor.astype(ms.float16)
    return tensor.astype(ms.float32)


def _np_from_torch(tensor):
    return tensor.float().cpu().numpy() if tensor.dtype == torch.bfloat16 else tensor.cpu().numpy()


def _np_from_ms(tensor):
    return tensor.astype(ms.float32).asnumpy() if tensor.dtype == ms.bfloat16 else tensor.asnumpy()


def _output_tensors_for_torch(input_tensor, reshuff):
    input_numel = input_tensor.numel()
    element_size = input_tensor.element_size()
    mantissa_numel = input_numel * (element_size - 1) // element_size
    compressed_bound_bytes = input_numel + input_numel // 64 + 8448 * 64
    compressed_bound_numel = (compressed_bound_bytes + element_size - 1) // element_size
    var_numel = input_numel if reshuff else compressed_bound_numel
    pdf = torch.zeros(256, dtype=torch.int32, device=input_tensor.device)
    mantissa = torch.zeros(mantissa_numel, dtype=input_tensor.dtype, device=input_tensor.device)
    fixed = torch.zeros(compressed_bound_numel, dtype=input_tensor.dtype, device=input_tensor.device)
    var = torch.zeros(var_numel, dtype=input_tensor.dtype, device=input_tensor.device)
    return pdf, mantissa, fixed, var


def _assert_outputs_close(expected, actual, reshuff=False):
    assert len(expected) == len(actual) == 4
    compare_count = 3 if reshuff else 4
    for expected_tensor, actual_tensor in zip(expected[:compare_count], actual[:compare_count]):
        expected_np = _np_from_torch(expected_tensor)
        actual_np = _np_from_ms(actual_tensor)
        assert expected_np.shape == actual_np.shape
        if expected_np.dtype.kind in "iu":
            np.testing.assert_array_equal(expected_np, actual_np)
        else:
            np.testing.assert_allclose(expected_np, actual_np, rtol=0, atol=0)


@pytest.mark.skipif(not HAS_TORCH_NPU_INTERFACE, reason="torch_npu does not expose this interface")
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("statistic,reshuff", [(True, False), (True, True), (False, False)])
def test_npu_hans_encode_matches_torch_npu(dtype, statistic, reshuff):
    rng = np.random.default_rng(19)
    x_np = rng.normal(size=(32768,)).astype(np.float32)
    torch_x = _torch_tensor(x_np, dtype)
    torch_outputs = _output_tensors_for_torch(torch_x, reshuff)
    expected = torch_npu.npu_hans_encode(torch_x, statistic, reshuff, out=torch_outputs)
    actual = npu_hans_encode(_ms_tensor(x_np, dtype), Tensor(np.zeros((256,), dtype=np.int32)), statistic, reshuff)
    _assert_outputs_close(expected, actual, reshuff)


def test_npu_hans_encode_output_metadata():
    x = np.zeros((32768,), dtype=np.float32)
    pdf, mantissa, fixed, var = npu_hans_encode(Tensor(x), Tensor(np.zeros((256,), dtype=np.int32)), True, False)
    assert pdf.asnumpy().shape == (256,)
    assert pdf.asnumpy().dtype == np.int32
    assert mantissa.asnumpy().shape == (24576,)
    assert fixed.asnumpy().shape == var.asnumpy().shape
