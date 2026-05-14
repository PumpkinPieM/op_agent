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
KERNEL_SOURCE = Path(__file__).with_name("npu_hans_decode.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_hans_decode_test_{os.getpid()}",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def npu_hans_decode(mantissa, fixed, var, pdf, reshuff=False):
    return _ops().npu_hans_decode(mantissa, fixed, var, pdf, reshuff)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(array, dtype):
    tensor = torch.from_numpy(np.array(array, copy=True)).npu()
    if dtype == "bf16":
        return tensor.to(torch.bfloat16)
    if dtype == "fp16":
        return tensor.to(torch.float16)
    return tensor.to(torch.float32)


def _ms_from_torch(tensor):
    array = tensor.float().cpu().numpy() if tensor.dtype == torch.bfloat16 else tensor.cpu().numpy()
    out = Tensor(array)
    if tensor.dtype == torch.bfloat16:
        return out.astype(ms.bfloat16)
    if tensor.dtype == torch.float16:
        return out.astype(ms.float16)
    return out.astype(ms.float32)


def _np_from_torch(tensor):
    return tensor.float().cpu().numpy() if tensor.dtype == torch.bfloat16 else tensor.cpu().numpy()


def _np_from_ms(tensor):
    return tensor.astype(ms.float32).asnumpy() if tensor.dtype == ms.bfloat16 else tensor.asnumpy()


def _encode_inputs(input_tensor, statistic, reshuff):
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
    return torch_npu.npu_hans_encode(input_tensor, statistic, reshuff, out=(pdf, mantissa, fixed, var))


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("reshuff", [False, True])
def test_npu_hans_decode_roundtrip_matches_torch_npu(dtype, reshuff):
    rng = np.random.default_rng(23)
    x_np = rng.normal(size=(32768,)).astype(np.float32)
    torch_x = _torch_tensor(x_np, dtype)
    pdf, mantissa, fixed, var = _encode_inputs(torch_x, True, reshuff)
    recover = torch.zeros_like(torch_x)
    expected = torch_npu.npu_hans_decode(mantissa, fixed, var, pdf, reshuff, out=recover)
    actual = npu_hans_decode(_ms_from_torch(mantissa), _ms_from_torch(fixed), _ms_from_torch(var),
                             Tensor(pdf.cpu().numpy()), reshuff)

    expected_np = _np_from_torch(expected)
    actual_np = _np_from_ms(actual)
    assert expected_np.shape == actual_np.shape == (32768,)
    np.testing.assert_allclose(expected_np, actual_np, rtol=0, atol=0)


def test_npu_hans_decode_builder_loads():
    assert hasattr(_ops(), "npu_hans_decode")
