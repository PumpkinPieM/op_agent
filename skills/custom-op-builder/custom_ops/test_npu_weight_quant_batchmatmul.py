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
KERNEL_SOURCE = Path(__file__).with_name("npu_weight_quant_batchmatmul.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_weight_quant_batchmatmul_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
def npu_weight_quant_batchmatmul(*args):
    return _custom_ops.npu_weight_quant_batchmatmul(*args)
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
    antiquant_scale = np.ones((1, 3), dtype=np.float16)
    weight = np.random.default_rng(1).integers(-2, 2, size=(4, 3), dtype=np.int8)
    antiquant_offset_opt = None
    quant_scale_opt = None
    quant_offset_opt = None
    bias_opt = None
    antiquant_group_size = 0
    inner_precise = 0
    weight_dtype_opt = None
    return (_pta(x), _pta(weight), _pta(antiquant_scale), _pta(antiquant_offset_opt), _pta(quant_scale_opt), _pta(quant_offset_opt), _pta(bias_opt), antiquant_group_size, inner_precise, weight_dtype_opt), (_ms(x), _ms(weight), _ms(antiquant_scale), _ms(antiquant_offset_opt), _ms(quant_scale_opt), _ms(quant_offset_opt), _ms(bias_opt), antiquant_group_size, inner_precise, weight_dtype_opt)
def _torch_reference(torch_args):
    required_count = 10
    keyword_names = []
    kwargs = {name: value for name, value in zip(keyword_names, torch_args[required_count:]) if value is not None}

    dtype_map = {5: torch.float16, 6: torch.float32, 27: torch.bfloat16}
    for key, value in list(kwargs.items()):
        if key.endswith("dtype") and isinstance(value, int) and value in dtype_map:
            kwargs[key] = dtype_map[value]
    return torch_npu.npu_weight_quant_batchmatmul(*torch_args[:required_count], **kwargs)

def test_npu_weight_quant_batchmatmul_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_weight_quant_batchmatmul")
    torch_args, ms_args = _case()
    expected = _torch_reference(torch_args)
    actual = npu_weight_quant_batchmatmul(*ms_args)
    _assert_close(expected, actual)
