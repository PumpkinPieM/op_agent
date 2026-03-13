#!/usr/bin/env python3
"""
MindSpore build verification script.

Runs a layered check after compilation to confirm the build is usable.
Exits 0 if all applicable tests pass, non-zero otherwise.

Usage:
    python scripts/verify_build.py                    # auto-detect device
    python scripts/verify_build.py --device Ascend    # force Ascend
    python scripts/verify_build.py --device CPU       # CPU-only check
    python scripts/verify_build.py --build-dir /path/to/build/package
"""

import argparse
import sys
import os


def _setup_path(build_dir):
    if build_dir and os.path.isdir(build_dir):
        sys.path.insert(0, build_dir)
        lib_dir = os.path.join(build_dir, "mindspore", "lib")
        if os.path.isdir(lib_dir):
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld}" if ld else lib_dir


def test_import():
    """Level 1: Can we import mindspore at all?"""
    import mindspore as ms
    ver = ms.__version__
    print(f"  MindSpore version: {ver}")
    return True


def test_cpu_tensor():
    """Level 2: Basic tensor ops on CPU."""
    import mindspore as ms
    import numpy as np
    x = ms.Tensor(np.ones([2, 3], dtype=np.float32))
    y = ms.Tensor(np.ones([2, 3], dtype=np.float32))
    z = x + y
    assert z.shape == (2, 3), f"unexpected shape {z.shape}"
    assert float(z[0, 0].asnumpy()) == 2.0
    print(f"  Tensor add OK, shape={z.shape}")
    return True


def test_device_target(device):
    """Level 3: Set device target (Ascend/GPU)."""
    import mindspore as ms
    ms.set_context(device_target=device)
    actual = ms.get_context("device_target")
    assert actual == device, f"expected {device}, got {actual}"
    print(f"  Device target set to {actual}")
    return True


def test_simple_network():
    """Level 4: Forward pass through a tiny network."""
    import mindspore as ms
    import mindspore.nn as nn
    import numpy as np

    class TinyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.fc = nn.Dense(10, 5)
            self.relu = nn.ReLU()

        def construct(self, x):
            return self.relu(self.fc(x))

    net = TinyNet()
    x = ms.Tensor(np.random.randn(2, 10).astype(np.float32))
    out = net(x)
    assert out.shape == (2, 5), f"unexpected shape {out.shape}"
    print(f"  Network forward OK, output shape={out.shape}")
    return True


def _detect_device():
    """Return 'Ascend' if /dev/davinci* exists, 'GPU' if nvidia-smi works, else 'CPU'."""
    import glob
    import subprocess
    if glob.glob("/dev/davinci*"):
        return "Ascend"
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return "GPU"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return "CPU"


def main():
    parser = argparse.ArgumentParser(description="Verify MindSpore build")
    parser.add_argument("--device", choices=["Ascend", "GPU", "CPU"], default=None,
                        help="Target device (auto-detected if omitted)")
    parser.add_argument("--build-dir", default=None,
                        help="Path to build/package dir (prepended to sys.path)")
    args = parser.parse_args()

    _setup_path(args.build_dir)

    device = args.device or _detect_device()
    npu_or_gpu = device in ("Ascend", "GPU")

    tests = [
        ("Import", test_import),
        ("CPU tensor ops", test_cpu_tensor),
    ]
    if npu_or_gpu:
        tests.append((f"Set device={device}", lambda: test_device_target(device)))
    tests.append(("Simple network", test_simple_network))

    results = []
    for name, fn in tests:
        print(f"[{name}]")
        try:
            passed = fn()
            results.append((name, passed))
            print(f"  -> PASS")
        except Exception as e:
            results.append((name, False))
            print(f"  -> FAIL: {e}")

    print("\n--- Summary ---")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, r in results:
        print(f"  {'PASS' if r else 'FAIL'}: {name}")
    print(f"\n{passed}/{total} passed (device={device})")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
