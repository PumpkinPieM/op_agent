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
KERNEL_SOURCE = Path(__file__).with_name("npu_grouped_matmul_swiglu_quant.cc")
_CUSTOM_OPS = None

torch_npu.npu.config.allow_internal_format = True


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        module_name = f"custom_ops_npu_grouped_matmul_swiglu_quant_test_{os.getpid()}"
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(module_name, [str(KERNEL_SOURCE)], backend="Ascend").load()
    return _CUSTOM_OPS


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _to_torch(array):
    return torch.from_numpy(array).npu()


def _to_ms(array):
    return Tensor(array)


def _to_ms_nz(array):
    return ms.ops.auto_generate.format_cast(_to_ms(array), 29)


def _np_from_torch(value):
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    return value.asnumpy()


def _make_case(m=4, k=16, n=32, expert=2, counts=(2, 2)):
    rng = np.random.default_rng(20260516 + m + k + n + expert)
    x = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
    weight = rng.integers(-8, 8, size=(expert, k, n), dtype=np.int8)
    weight_scale = rng.uniform(0.001, 0.01, size=(expert, n)).astype(np.float32)
    x_scale = rng.uniform(0.001, 0.01, size=(m,)).astype(np.float32)
    group_list = np.cumsum(np.asarray(counts, dtype=np.int64))
    return x, weight, group_list, weight_scale, x_scale


def npu_grouped_matmul_swiglu_quant(x, weight, group_list, weight_scale, x_scale, bias=None, offset=None):
    return _ops().npu_grouped_matmul_swiglu_quant(x, weight, group_list, weight_scale, x_scale, bias, offset)


def test_custom_op_builds():
    assert hasattr(_ops(), "npu_grouped_matmul_swiglu_quant")


def _run_case(case):
    if not hasattr(torch_npu, "npu_grouped_matmul_swiglu_quant"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    x, weight, group_list, weight_scale, x_scale = case
    torch_weight = torch_npu.npu_format_cast(_to_torch(weight).contiguous(), 29)
    expected = torch_npu.npu_grouped_matmul_swiglu_quant(
        _to_torch(x),
        torch_weight,
        _to_torch(group_list),
        _to_torch(weight_scale),
        _to_torch(x_scale),
        bias=None,
        offset=None,
    )
    actual = npu_grouped_matmul_swiglu_quant(
        _to_ms(x),
        _to_ms_nz(weight),
        _to_ms(group_list),
        _to_ms(weight_scale),
        _to_ms(x_scale),
        None,
        None,
    )
    actual_np = [_np_from_ms(item) for item in actual]
    expected_np = [_np_from_torch(item) for item in expected]
    assert [item.shape for item in actual_np] == [item.shape for item in expected_np]
    assert actual_np[0].dtype == np.int8
    assert actual_np[1].dtype == np.float32
    assert actual_np[2].dtype == np.float32
    # outputOffset is documented as a reserved/unsupported output for this ACLNN interface,
    # so only the two defined outputs are valid equality oracles.
    for exp, act in zip(expected_np[:2], actual_np[:2]):
        np.testing.assert_array_equal(exp, act)


@pytest.mark.parametrize(
    "case",
    [
        _make_case(m=4, k=16, n=32, expert=2, counts=(2, 2)),
        _make_case(m=6, k=32, n=64, expert=3, counts=(1, 3, 2)),
    ],
)
def test_npu_grouped_matmul_swiglu_quant_matches_torch_npu(case):
    _run_case(case)
