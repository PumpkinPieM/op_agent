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
KERNEL_SOURCE = Path(__file__).with_name("npu_moe_distribute_dispatch_v2.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_moe_distribute_dispatch_v2_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
def npu_moe_distribute_dispatch_v2(*args):
    return _custom_ops.npu_moe_distribute_dispatch_v2(*args)
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
    expert_ids = np.zeros((2,), dtype=np.int32)
    group_ep = ""
    ep_world_size = 1
    ep_rank_id = 0
    moe_expert_num = 1
    scales_opt = None
    x_active_mask_opt = None
    expert_scales_opt = None
    elastic_info_opt = None
    performance_info_opt = None
    group_tp = ""
    tp_world_size = 0
    tp_rank_id = 0
    expert_shard_type = 0
    shared_expert_num = 0
    shared_expert_rank_num = 0
    quant_mode = 0
    global_bs = 0
    expert_token_nums_type = 0
    comm_alg = ""
    zero_expert_num = 0
    copy_expert_num = 0
    const_expert_num = 0
    y_dtype_opt = None
    x_dtype_opt = None
    scales_dtype_opt = None
    return (_pta(x), _pta(expert_ids), group_ep, ep_world_size, ep_rank_id, moe_expert_num, _pta(scales_opt), _pta(x_active_mask_opt), _pta(expert_scales_opt), _pta(elastic_info_opt), _pta(performance_info_opt), group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_alg, zero_expert_num, copy_expert_num, const_expert_num, y_dtype_opt, x_dtype_opt, scales_dtype_opt), (_ms(x), _ms(expert_ids), group_ep, ep_world_size, ep_rank_id, moe_expert_num, _ms(scales_opt), _ms(x_active_mask_opt), _ms(expert_scales_opt), _ms(elastic_info_opt), _ms(performance_info_opt), group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_alg, zero_expert_num, copy_expert_num, const_expert_num, y_dtype_opt, x_dtype_opt, scales_dtype_opt)
def _torch_reference(torch_args):
    required_count = 6
    keyword_names = ['scales', 'x_active_mask', 'expert_scales', 'elastic_info', 'performance_info', 'group_tp', 'tp_world_size', 'tp_rank_id', 'expert_shard_type', 'shared_expert_num', 'shared_expert_rank_num', 'quant_mode', 'global_bs', 'expert_token_nums_type', 'comm_alg', 'zero_expert_num', 'copy_expert_num', 'const_expert_num', 'y_dtype', 'x_dtype', 'scales_dtype']
    kwargs = {name: value for name, value in zip(keyword_names, torch_args[required_count:]) if value is not None}

    dtype_map = {5: torch.float16, 6: torch.float32, 27: torch.bfloat16}
    for key, value in list(kwargs.items()):
        if key.endswith("dtype") and isinstance(value, int) and value in dtype_map:
            kwargs[key] = dtype_map[value]
    return torch_npu.npu_moe_distribute_dispatch_v2(*torch_args[:required_count], **kwargs)

def test_npu_moe_distribute_dispatch_v2_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
    torch_args, ms_args = _case()
    expected = _torch_reference(torch_args)
    actual = npu_moe_distribute_dispatch_v2(*ms_args)
    _assert_close(expected, actual)
