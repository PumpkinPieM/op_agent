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
KERNEL_SOURCE = Path(__file__).with_name("npu_qkv_rms_norm_rope_cache.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_qkv_rms_norm_rope_cache_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
def npu_qkv_rms_norm_rope_cache(*args):
    return _custom_ops.npu_qkv_rms_norm_rope_cache(*args)
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
    q_gamma = np.ones((4,), dtype=np.float16)
    index = np.zeros((2,), dtype=np.int32)
    k_cache = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    k_gamma = np.ones((4,), dtype=np.float16)
    q_out = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    v_cache = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    qkv = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float16)
    cos = np.ones((4,), dtype=np.float16)
    qkv_size = [1, 2, 2, 2]
    head_nums = [1, 1, 1]
    k_scale_opt = None
    v_scale_opt = None
    k_offset_opt = None
    v_offset_opt = None
    epsilon = 1.0 if "scale" in "epsilon" else 1e-5
    cache_mode_opt = None
    is_output_qkv = False
    return (_pta(qkv), _pta(q_gamma), _pta(k_gamma), _pta(cos), _pta(sin), _pta(index), _pta(q_out), _pta(k_cache), _pta(v_cache), qkv_size, head_nums, _pta(k_scale_opt), _pta(v_scale_opt), _pta(k_offset_opt), _pta(v_offset_opt), epsilon, cache_mode_opt, is_output_qkv), (_ms(qkv), _ms(q_gamma), _ms(k_gamma), _ms(cos), _ms(sin), _ms(index), _ms(q_out), _ms(k_cache), _ms(v_cache), qkv_size, head_nums, _ms(k_scale_opt), _ms(v_scale_opt), _ms(k_offset_opt), _ms(v_offset_opt), epsilon, cache_mode_opt, is_output_qkv)
def _torch_reference(torch_args):
    required_count = 11
    keyword_names = ['k_scale', 'v_scale', 'k_offset', 'v_offset', 'epsilon', 'cache_mode', 'is_output_qkv']
    kwargs = {name: value for name, value in zip(keyword_names, torch_args[required_count:]) if value is not None}

    dtype_map = {5: torch.float16, 6: torch.float32, 27: torch.bfloat16}
    for key, value in list(kwargs.items()):
        if key.endswith("dtype") and isinstance(value, int) and value in dtype_map:
            kwargs[key] = dtype_map[value]
    return torch_npu.npu_qkv_rms_norm_rope_cache(*torch_args[:required_count], **kwargs)

def test_npu_qkv_rms_norm_rope_cache_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_qkv_rms_norm_rope_cache")
    torch_args, ms_args = _case()
    expected = _torch_reference(torch_args)
    actual = npu_qkv_rms_norm_rope_cache(*ms_args)
    _assert_close(expected, actual)
