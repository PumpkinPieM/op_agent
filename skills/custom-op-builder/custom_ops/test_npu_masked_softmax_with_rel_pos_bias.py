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
KERNEL_SOURCE = Path(__file__).with_name("npu_masked_softmax_with_rel_pos_bias.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_masked_softmax_with_rel_pos_bias")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_masked_softmax_with_rel_pos_bias_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0):
    return _custom_ops.npu_masked_softmax_with_rel_pos_bias(
        x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(array):
    return torch.from_numpy(array).npu()


def _ms_tensor(array):
    return Tensor(array)


def _numpy_from_ms(tensor):
    if tensor.dtype == ms.bfloat16:
        return tensor.astype(ms.float32).asnumpy()
    return tensor.asnumpy()


def _assert_close(expected, actual, rtol=5e-3, atol=5e-3):
    expected_np = expected.float().cpu().numpy() if expected.dtype == torch.bfloat16 else expected.cpu().numpy()
    actual_np = _numpy_from_ms(actual)
    assert expected_np.shape == actual_np.shape
    np.testing.assert_allclose(expected_np, actual_np, rtol=rtol, atol=atol)


@pytest.mark.skipif(not HAS_TORCH_NPU_INTERFACE, reason="torch_npu does not expose this interface")
@pytest.mark.parametrize(
    "shape,mask_shape,bias_shape,scale_value,inner_precision_mode,dtype",
    [
        ((1, 2, 3, 4, 32), (1, 2, 1, 4, 32), (1, 1, 3, 4, 32), 1.0, 0, np.float32),
        ((2, 3, 4, 32), (2, 1, 4, 32), (1, 3, 4, 32), 0.5, 0, np.float16),
        ((1, 2, 2, 4, 32), (2, 4, 32), (2, 4, 32), 0.25, 1, np.float16),
    ],
)
def test_npu_masked_softmax_with_rel_pos_bias_against_torch_npu(
    shape, mask_shape, bias_shape, scale_value, inner_precision_mode, dtype
):
    rng = np.random.default_rng(7)
    x = rng.normal(size=shape).astype(dtype)
    atten_mask = rng.normal(size=mask_shape).astype(dtype)
    relative_pos_bias = rng.normal(size=bias_shape).astype(dtype)

    expected = torch_npu.npu_masked_softmax_with_rel_pos_bias(
        _torch_tensor(x), _torch_tensor(atten_mask), _torch_tensor(relative_pos_bias),
        scale_value, inner_precision_mode
    )
    actual = npu_masked_softmax_with_rel_pos_bias(
        _ms_tensor(x), _ms_tensor(atten_mask), _ms_tensor(relative_pos_bias),
        scale_value, inner_precision_mode
    )
    _assert_close(expected, actual)


def test_npu_masked_softmax_with_rel_pos_bias_matches_numpy_formula():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1, 2, 3, 4, 32)).astype(np.float32)
    atten_mask = rng.normal(size=(1, 2, 1, 4, 32)).astype(np.float32)
    relative_pos_bias = rng.normal(size=(1, 1, 3, 4, 32)).astype(np.float32)
    logits = x + atten_mask + relative_pos_bias
    logits -= logits.max(axis=-1, keepdims=True)
    expected = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    actual = npu_masked_softmax_with_rel_pos_bias(Tensor(x), Tensor(atten_mask), Tensor(relative_pos_bias), 1.0, 0)
    np.testing.assert_allclose(expected, _numpy_from_ms(actual), rtol=5e-3, atol=5e-3)
