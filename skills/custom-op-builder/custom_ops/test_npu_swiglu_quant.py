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
KERNEL_SOURCE = Path(__file__).with_name("npu_swiglu_quant.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_swiglu_quant_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_swiglu_quant(
    x, *, smooth_scales=None, offsets=None, group_index=None, activate_left=False, quant_mode=0, group_list_type=0, dst_type=None
):
    return _custom_ops.npu_swiglu_quant(
        x, smooth_scales, offsets, group_index, activate_left, quant_mode, group_list_type, dst_type
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("quant_mode", [0, 1])
@pytest.mark.parametrize("activate_left", [False, True])
def test_npu_swiglu_quant_matches_torch_npu(quant_mode, activate_left):
    rng = np.random.default_rng(43)
    x = rng.normal(size=(4, 16)).astype(np.float16)
    smooth_scales = rng.uniform(0.5, 1.5, size=(1, 8)).astype(np.float32)
    offsets = rng.uniform(-0.25, 0.25, size=(1, 8)).astype(np.float32) if quant_mode == 0 else None
    expected = torch_npu.npu_swiglu_quant(
        torch.from_numpy(x).npu(),
        smooth_scales=torch.from_numpy(smooth_scales).npu(),
        offsets=None if offsets is None else torch.from_numpy(offsets).npu(),
        activate_left=activate_left,
        quant_mode=quant_mode,
    )
    actual = npu_swiglu_quant(
        Tensor(x),
        smooth_scales=Tensor(smooth_scales),
        offsets=None if offsets is None else Tensor(offsets),
        activate_left=activate_left,
        quant_mode=quant_mode,
    )
    np.testing.assert_array_equal(actual[0].asnumpy(), expected[0].cpu().numpy())
    np.testing.assert_allclose(actual[1].asnumpy(), expected[1].cpu().numpy(), rtol=1e-3, atol=1e-3)
