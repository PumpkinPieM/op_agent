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
KERNEL_SOURCE = Path(__file__).with_name("npu_quant_matmul_reduce_sum.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = True
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_quant_matmul_reduce_sum_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
def npu_quant_matmul_reduce_sum(*args):
    return _custom_ops.npu_quant_matmul_reduce_sum(*args)
@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
def _pta(x):
    if x is None:
        return None
    t = torch.from_numpy(x).npu()
    return t

def _ms(x):
    if x is None:
        return None
    return Tensor(x)

def _np(x):
    if isinstance(x, (tuple, list)):
        return [_np(v) for v in x]
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return x.float().cpu().numpy()
        return x.cpu().numpy()
    if hasattr(x, "asnumpy"):
        if x.dtype == ms.bfloat16:
            return x.astype(ms.float32).asnumpy()
        return x.asnumpy()
    return np.asarray(x)

def _assert_close(expected, actual, rtol=1e-2, atol=1e-2):
    e=_np(expected); a=_np(actual)
    if not isinstance(e, list): e=[e]
    if not isinstance(a, list): a=[a]
    assert len(e)==len(a)
    for ev,av in zip(e,a):
        assert ev.shape == av.shape
        np.testing.assert_allclose(ev, av, rtol=rtol, atol=atol, equal_nan=True)
def _case():
    rng = np.random.default_rng(0)
    batch, m, k, n = 2, 128, 256, 128
    x1 = rng.integers(-5, 5, size=(batch, m, k), dtype=np.int8)
    x2 = rng.integers(-5, 5, size=(batch, k, n), dtype=np.int8)
    x1_scale = rng.random(size=(batch, m), dtype=np.float32)
    x2_scale = rng.random(size=(n,), dtype=np.float32)

    torch_x2 = torch_npu.npu_format_cast(_pta(x2).contiguous(), 29)
    ms_x2 = ms.ops.auto_generate.format_cast(_ms(x2), 29)
    return (
        (_pta(x1), torch_x2, _pta(x1_scale), _pta(x2_scale).to(torch.bfloat16)),
        (_ms(x1), ms_x2, _ms(x1_scale), _ms(x2_scale).astype(ms.bfloat16)),
    )
def _torch_reference(torch_args):
    required_count = 2
    keyword_names = ['x1_scale', 'x2_scale']
    kwargs = {name: value for name, value in zip(keyword_names, torch_args[required_count:]) if value is not None}

    dtype_map = {5: torch.float16, 6: torch.float32, 27: torch.bfloat16}
    for key, value in list(kwargs.items()):
        if key.endswith("dtype") and isinstance(value, int) and value in dtype_map:
            kwargs[key] = dtype_map[value]
    return torch_npu.npu_quant_matmul_reduce_sum(*torch_args[:required_count], **kwargs)

def test_npu_quant_matmul_reduce_sum_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_quant_matmul_reduce_sum")
    torch_args, ms_args = _case()
    expected = _torch_reference(torch_args)
    actual = npu_quant_matmul_reduce_sum(*ms_args)
    _assert_close(expected, actual)
