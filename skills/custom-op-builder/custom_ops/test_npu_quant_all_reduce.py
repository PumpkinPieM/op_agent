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
KERNEL_SOURCE = Path(__file__).with_name("npu_quant_all_reduce.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_quant_all_reduce_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
def npu_quant_all_reduce(*args):
    return _custom_ops.npu_quant_all_reduce(*args)
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
    x = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float16)
    scales = np.ones((4,), dtype=np.float16)
    hcom = ""
    world_size = 1
    reduce_op_opt = None
    output_dtype_opt = None
    x_dtype_opt = None
    scales_dtype_opt = None
    return (_pta(x), _pta(scales), hcom, world_size, reduce_op_opt, output_dtype_opt, x_dtype_opt, scales_dtype_opt), (_ms(x), _ms(scales), hcom, world_size, reduce_op_opt, output_dtype_opt, x_dtype_opt, scales_dtype_opt)
def _torch_reference(torch_args):
    required_count = 4
    keyword_names = ['reduce_op', 'output_dtype', 'x_dtype', 'scales_dtype']
    kwargs = {name: value for name, value in zip(keyword_names, torch_args[required_count:]) if value is not None}

    dtype_map = {5: torch.float16, 6: torch.float32, 27: torch.bfloat16}
    for key, value in list(kwargs.items()):
        if key.endswith("dtype") and isinstance(value, int) and value in dtype_map:
            kwargs[key] = dtype_map[value]
    return torch_npu.npu_quant_all_reduce(*torch_args[:required_count], **kwargs)

def test_npu_quant_all_reduce_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_quant_all_reduce")
    torch_args, ms_args = _case()
    expected = _torch_reference(torch_args)
    actual = npu_quant_all_reduce(*ms_args)
    _assert_close(expected, actual)
