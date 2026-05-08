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
KERNEL_SOURCE = Path(__file__).with_name("npu_rope_quant_kvcache.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_rope_quant_kvcache_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
def npu_rope_quant_kvcache(*args):
    return _custom_ops.npu_rope_quant_kvcache(*args)
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
    sin = np.ones((4,), dtype=np.float16)
    k_cache = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    scale_k = np.ones((4,), dtype=np.float16)
    scale_v = np.ones((4,), dtype=np.float16)
    v_cache = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    x = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float16)
    indices = np.zeros((2,), dtype=np.int32)
    cos = np.ones((4,), dtype=np.float16)
    size_splits = [2, 1, 1]
    offset_k_opt = None
    offset_v_opt = None
    quant_mode = 0
    input_layout_opt = None
    kv_output = False
    cache_mode_opt = None
    return (_pta(x), _pta(cos), _pta(sin), _pta(k_cache), _pta(v_cache), _pta(indices), _pta(scale_k), _pta(scale_v), size_splits, _pta(offset_k_opt), _pta(offset_v_opt), quant_mode, input_layout_opt, kv_output, cache_mode_opt), (_ms(x), _ms(cos), _ms(sin), _ms(k_cache), _ms(v_cache), _ms(indices), _ms(scale_k), _ms(scale_v), size_splits, _ms(offset_k_opt), _ms(offset_v_opt), quant_mode, input_layout_opt, kv_output, cache_mode_opt)
def _torch_reference(torch_args):
    required_count = 9
    keyword_names = ['offset_k', 'offset_v', 'quant_mode', 'input_layout', 'kv_output', 'cache_mode']
    kwargs = {name: value for name, value in zip(keyword_names, torch_args[required_count:]) if value is not None}

    dtype_map = {5: torch.float16, 6: torch.float32, 27: torch.bfloat16}
    for key, value in list(kwargs.items()):
        if key.endswith("dtype") and isinstance(value, int) and value in dtype_map:
            kwargs[key] = dtype_map[value]
    return torch_npu.npu_rope_quant_kvcache(*torch_args[:required_count], **kwargs)

def test_npu_rope_quant_kvcache_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_rope_quant_kvcache")
    torch_args, ms_args = _case()
    expected = _torch_reference(torch_args)
    actual = npu_rope_quant_kvcache(*ms_args)
    _assert_close(expected, actual)
