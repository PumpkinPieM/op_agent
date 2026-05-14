import gc
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_trans_quant_param.cc")


torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_trans_quant_param_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_trans_quant_param(scale, offset=None, round_mode=None):
    return _custom_ops.npu_trans_quant_param(scale, offset, round_mode)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(arr):
    return torch.from_numpy(np.array(arr, copy=True)).npu()


def _ms_tensor(arr):
    return Tensor(np.array(arr, copy=True))


@pytest.mark.parametrize(
    "scale,offset,round_mode",
    [
        (np.array([0.5], np.float32), None, None),
        (np.linspace(0.125, 1.25, 8, dtype=np.float32), np.linspace(-3.0, 4.0, 8, dtype=np.float32), 0),
        (np.array([0.125], dtype=np.float32), np.linspace(-260.0, 260.0, 8, dtype=np.float32), 1),
        (
            np.linspace(0.125, 1.25, 8, dtype=np.float32).reshape(1, 8),
            np.linspace(-3.0, 4.0, 8, dtype=np.float32).reshape(1, 8),
            1,
        ),
    ],
)
def test_npu_trans_quant_param_matches_torch_npu(scale, offset, round_mode):
    expected = torch_npu.npu_trans_quant_param(
        _torch_tensor(scale), None if offset is None else _torch_tensor(offset), round_mode
    )
    actual = npu_trans_quant_param(_ms_tensor(scale), None if offset is None else _ms_tensor(offset), round_mode)
    np.testing.assert_array_equal(expected.cpu().numpy(), actual.asnumpy())


def test_npu_trans_quant_param_rejects_invalid_round_mode():
    scale = np.ones((4,), np.float32)
    with pytest.raises(RuntimeError):
        npu_trans_quant_param(_ms_tensor(scale), None, 2).asnumpy()
