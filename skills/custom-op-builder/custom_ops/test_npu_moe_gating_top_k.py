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
KERNEL_SOURCE = Path(__file__).with_name("npu_moe_gating_top_k.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        if not hasattr(torch_npu, "npu_moe_gating_top_k"):
            pytest.skip("torch_npu on this host does not expose npu_moe_gating_top_k")
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_moe_gating_top_k_test", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def _torch_tensor(arr, dtype):
    t = torch.from_numpy(np.array(arr, copy=True)).npu()
    return t.to(dtype)


def _ms_tensor(arr, dtype):
    t = Tensor(np.array(arr, copy=True))
    return t.astype(dtype)


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_outputs_close(expected, actual, compare_out, rtol=1e-3, atol=1e-3):
    assert len(expected) == len(actual) == 3
    compare_count = 3 if compare_out else 2
    for exp, act in zip(expected[:compare_count], actual[:compare_count]):
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        if exp_np.dtype.kind in "iu":
            np.testing.assert_array_equal(exp_np, act_np)
        else:
            np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol)
    assert _np_from_ms(actual[2]).shape == _np_from_torch(expected[2]).shape


def npu_moe_gating_top_k(x, k, bias, k_group, group_count, group_select_mode, renorm, norm_type, out_flag,
                         routed_scaling_factor, eps):
    return _ops().npu_moe_gating_top_k(
        x, k, bias, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


CASES = [
    {
        "id": "fp32_softmax_with_bias",
        "dtype_torch": torch.float32,
        "dtype_ms": ms.float32,
        "bias": True,
        "k": 2,
        "k_group": 1,
        "group_count": 1,
        "group_select_mode": 0,
        "renorm": 0,
        "norm_type": 0,
        "out_flag": True,
        "scale": 1.0,
    },
    {
        "id": "fp16_sigmoid_grouped_no_bias",
        "dtype_torch": torch.float16,
        "dtype_ms": ms.float16,
        "bias": False,
        "k": 2,
        "k_group": 2,
        "group_count": 4,
        "group_select_mode": 1,
        "renorm": 0,
        "norm_type": 1,
        "out_flag": False,
        "scale": 0.5,
    },
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_npu_moe_gating_top_k_matches_torch_npu(case):
    x = np.array(
        [[0.1, -0.2, 0.3, 1.1, -1.2, 0.7, 0.0, 0.5],
         [1.0, 0.4, -0.5, 0.2, -0.7, 1.4, 0.6, -0.1],
         [-0.8, 0.9, 0.2, -1.1, 0.4, 0.3, -0.6, 1.2]],
        dtype=np.float32,
    )
    bias_np = np.linspace(-0.3, 0.4, x.shape[1], dtype=np.float32) if case["bias"] else None
    torch_x = _torch_tensor(x, case["dtype_torch"])
    torch_bias = None if bias_np is None else _torch_tensor(bias_np, case["dtype_torch"])
    ms_x = _ms_tensor(x, case["dtype_ms"])
    ms_bias = None if bias_np is None else _ms_tensor(bias_np, case["dtype_ms"])

    expected = torch_npu.npu_moe_gating_top_k(
        torch_x, case["k"], bias=torch_bias, k_group=case["k_group"], group_count=case["group_count"],
        group_select_mode=case["group_select_mode"], renorm=case["renorm"], norm_type=case["norm_type"],
        out_flag=case["out_flag"], routed_scaling_factor=case["scale"], eps=1e-20
    )
    actual = npu_moe_gating_top_k(
        ms_x, case["k"], ms_bias, case["k_group"], case["group_count"], case["group_select_mode"],
        case["renorm"], case["norm_type"], case["out_flag"], case["scale"], 1e-20
    )
    _assert_outputs_close(expected, actual, compare_out=case["out_flag"])
