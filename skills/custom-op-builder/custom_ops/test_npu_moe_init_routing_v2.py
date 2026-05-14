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
KERNEL_SOURCE = Path(__file__).with_name("npu_moe_init_routing_v2.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_moe_init_routing_v2_test", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def _torch_tensor(arr, dtype=None):
    t = torch.from_numpy(np.array(arr, copy=True)).npu()
    return t if dtype is None else t.to(dtype)


def _ms_tensor(arr, dtype=None):
    t = Tensor(np.array(arr, copy=True))
    return t if dtype is None else t.astype(dtype)


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_close(expected, actual, rtol=1e-3, atol=1e-3):
    assert len(expected) == len(actual) == 4
    for exp, act in zip(expected, actual):
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        if exp_np.dtype.kind in "iu":
            np.testing.assert_array_equal(exp_np, act_np)
        else:
            np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def npu_moe_init_routing_v2(x, expert_idx, scale, offset, active_num, expert_capacity, expert_num, drop_pad_mode,
                            expert_tokens_num_type, expert_tokens_num_flag, quant_mode, active_expert_range,
                            row_idx_type, x_dtype):
    return _ops().npu_moe_init_routing_v2(
        x, expert_idx, scale, offset, active_num, expert_capacity, expert_num, drop_pad_mode,
        expert_tokens_num_type, expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type, x_dtype
    )


CASES = [
    {
        "id": "dropless_count_with_scale",
        "dtype_torch": torch.float16,
        "dtype_ms": ms.float16,
        "scale": True,
        "expert_tokens_num_type": 1,
        "expert_tokens_num_flag": True,
        "active_expert_range": [0, 4],
        "row_idx_type": 0,
    },
    {
        "id": "dropless_cumsum_row_idx_type_1",
        "dtype_torch": torch.float32,
        "dtype_ms": ms.float32,
        "scale": True,
        "expert_tokens_num_type": 0,
        "expert_tokens_num_flag": True,
        "active_expert_range": [0, 4],
        "row_idx_type": 1,
    },
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_npu_moe_init_routing_v2_matches_torch_npu(case):
    x = np.array(
        [[0.5, -0.1, 0.2, 0.7],
         [1.2, 0.3, -0.4, 0.8],
         [-0.6, 0.9, 0.1, -0.2],
         [0.4, -0.8, 1.1, 0.0]],
        dtype=np.float32,
    )
    expert_idx = np.array([[0, 2], [1, 3], [2, 0], [3, 1]], dtype=np.int32)
    scale = np.array([1.0, 0.5, 1.5, 0.75], dtype=np.float32) if case["scale"] else None
    active_num = x.shape[0] * expert_idx.shape[1]
    expert_num = 4

    torch_x = _torch_tensor(x, case["dtype_torch"])
    torch_expert_idx = _torch_tensor(expert_idx)
    torch_scale = None if scale is None else _torch_tensor(scale)
    ms_x = _ms_tensor(x, case["dtype_ms"])
    ms_expert_idx = _ms_tensor(expert_idx)
    ms_scale = None if scale is None else _ms_tensor(scale)

    expected = torch_npu.npu_moe_init_routing_v2(
        torch_x, torch_expert_idx, scale=torch_scale, offset=None, active_num=active_num, expert_capacity=-1,
        expert_num=expert_num, drop_pad_mode=0, expert_tokens_num_type=case["expert_tokens_num_type"],
        expert_tokens_num_flag=case["expert_tokens_num_flag"], quant_mode=-1,
        active_expert_range=case["active_expert_range"], row_idx_type=case["row_idx_type"]
    )
    actual = npu_moe_init_routing_v2(
        ms_x, ms_expert_idx, ms_scale, None, active_num, -1, expert_num, 0, case["expert_tokens_num_type"],
        case["expert_tokens_num_flag"], -1, case["active_expert_range"], case["row_idx_type"], None
    )
    _assert_close(expected, actual)
