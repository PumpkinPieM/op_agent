import gc
import os
from pathlib import Path

import numpy as np
import pytest
import mindspore as ms
from mindspore import context

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_dropout_gen_mask.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=True)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_dropout_gen_mask_test", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def npu_dropout_gen_mask(size, p, dtype=None, layout=None, device=None, pin_memory=None):
    return _ops().npu_dropout_gen_mask(size, p, dtype, layout, device, pin_memory)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _mask_length(size):
    return ((int(np.prod(size)) + 127) // 128) * 16


def test_npu_dropout_gen_mask_shape_and_dtype():
    size = [17, 19]
    out = npu_dropout_gen_mask(size, 0.4)
    out_np = out.asnumpy()
    assert out_np.dtype == np.uint8
    assert out_np.shape == (_mask_length(size),)


@pytest.mark.parametrize("size", [[2, 8], [32, 16], [1, 129]])
def test_npu_dropout_gen_mask_p0_is_full_mask(size):
    out = npu_dropout_gen_mask(size, 0.0).asnumpy()
    expected = np.full(_mask_length(size), 255, dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)
