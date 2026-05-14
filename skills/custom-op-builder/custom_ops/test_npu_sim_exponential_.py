import gc
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_sim_exponential_.cc")


torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_sim_exponential__test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_sim_exponential_(x, lambd=1.0):
    return _custom_ops.npu_sim_exponential_(x, lambd)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shape,lambd", [((256,), 0.5), ((64, 16), 1.0), ((8, 8, 8), 2.0)])
def test_npu_sim_exponential_distribution_properties(dtype, shape, lambd):
    out = npu_sim_exponential_(Tensor(np.empty(shape, dtype=dtype)), lambd).asnumpy()
    assert out.shape == shape
    assert out.dtype == dtype
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)
    mean = float(out.astype(np.float32).mean())
    assert 0.15 / lambd < mean < 6.0 / lambd


def test_npu_sim_exponential_fixed_seed_is_repeatable():
    x0 = Tensor(np.empty((128,), dtype=np.float32))
    x1 = Tensor(np.empty((128,), dtype=np.float32))
    out0 = npu_sim_exponential_(x0, 1.5).asnumpy()
    out1 = npu_sim_exponential_(x1, 1.5).asnumpy()
    np.testing.assert_array_equal(out0, out1)


def test_npu_sim_exponential_inf_lambda_zero_fills():
    out = npu_sim_exponential_(Tensor(np.ones((16,), dtype=np.float32)), math.inf).asnumpy()
    np.testing.assert_array_equal(out, np.zeros((16,), dtype=np.float32))


@pytest.mark.parametrize("lambd", [0.0, -1.0])
def test_npu_sim_exponential_rejects_non_positive_lambda(lambd):
    with pytest.raises(RuntimeError):
        npu_sim_exponential_(Tensor(np.empty((4,), dtype=np.float32)), lambd).asnumpy()
