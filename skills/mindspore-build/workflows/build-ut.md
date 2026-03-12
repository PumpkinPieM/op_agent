# Build MindSpore C++ Unit Tests

> **Status: VERIFIED.** Build confirmed on CI (x86 CentOS, GCC 7.3.0, 8851 tests) and
> on a developer machine (aarch64 openEuler, GCC 10.3.1, CANN 8.5.0 via conda).

Self-contained guide. No other files needed.

## Key Difference from Production Build

UT build (`-t on`) and device build (`-e`) are **mutually exclusive**.
The build script enforces this: if `-t` is set to `on` or `ut`, `DEVICE` is cleared.

UT compilation builds test binaries that link against MindSpore core libraries,
but does NOT target a specific device backend.

> **Note**: CI uses `-t on` (not `-t ut`). Both are equivalent — the build script
> treats `on` and `ut` identically: `RUN_TESTCASES=on`, `ENABLE_HIDDEN=off`.

## Prerequisites

CANN **must** be sourced before building — CMake fails at `cmake/ascend_variables.cmake`
if CANN env vars are missing. UT tests themselves run on CPU. CI environment reference:

| Software | CI Version | Install (CentOS/openEuler) |
|----------|-----------|---------------------------|
| Python | 3.9.0 | `conda create -n <user>_py39 python=3.9 -y` |
| GCC | 7.3.0 | `/usr/local/gcc/gcc730/` or `yum install devtoolset-7` |
| CMake | 3.22+ | `/usr/local/cmake/` or download from cmake.org |
| git, tcl, patch | — | `yum install git tcl patch -y` |
| CANN | 8.5.0 | Ascend driver + toolkit linked at `/usr/local/Ascend` |
| ccache | recommended | `/usr/local/ccache/bin/` in PATH |

Quick check:
```bash
gcc --version    # GCC 7.3+
cmake --version  # CMake 3.22+
python3 --version
ls /usr/local/Ascend/cann/  # CANN present
```

## Build Command

```bash
cd <mindspore-repo-root>
bash build.sh -t on -j48
```

CI uses `-t on -j48`. The actual CMake flags generated:
```
-DENABLE_TESTCASES=ON -DENABLE_DUMP_PROTO=ON -DENABLE_DUMP_IR=on
-DENABLE_PYTHON=on -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_AKG=ON
-DENABLE_DEBUGGER=ON -DENABLE_HIDDEN=OFF -DENABLE_FAST_HASH_TABLE=ON
```

### Parameter Reference

| Param | Meaning | Values | Default |
|-------|---------|--------|---------|
| `-t` | Build test binaries | `on`/`ut` (equivalent), `off` | `off` |
| `-j` | Parallel threads | integer | `8` |
| `-i` | Incremental build | flag | off |
| `-d` | Debug mode (for gdb) | flag | off |
| `-S on` | Gitee mirror for ~29/38 deps; 7 deps still need GitHub; disables `$MSLIBS_SERVER` | `on`/`off` | `off` |

### What `-t on` Does

1. Sets `RUN_TESTCASES=on` and `ENABLE_HIDDEN=off` (exports all symbols for testing)
2. Clears `DEVICE` — cannot combine with `-e`
3. Passes `-DENABLE_TESTCASES=ON` to CMake
4. Unpacks `ms_kernels_internal.tar.gz` and `ms_kernels_dependency.tar.gz` for internal kernel UT
5. Initializes `dvm` submodule (`git submodule update --init`)
6. CMake includes `tests/ut/cpp/CMakeLists.txt` which builds multiple test executables

### UT Test Executables

The build produces 13 separate test executables (confirmed from CI + `runtest.sh`):

| Executable | What it tests |
|-----------|---------------|
| `ut_CORE_tests` | IR, abstract types, op definitions, infer logic |
| `ut_FRONTEND_tests` | Frontend passes, parallel strategies |
| `ut_BACKEND_tests` | Backend compilation, graph scheduling |
| `ut_OLD_BACKEND_tests` | Legacy backend tests |
| `ut_PYNATIVE_tests` | PyNative mode execution |
| `ut_MINDDATA0_tests` | Data processing pipeline (part 1) |
| `ut_MINDDATA1_tests` | Data processing pipeline (part 2) |
| `ut_SYMBOL_ENGINE_tests` | Symbolic shape engine |
| `ut_CCSRC_tests` | Core C++ source tests |
| `ut_GRAPH_KERNEL_tests` | Graph kernel fusion |
| `ut_INTERNAL_KERNEL_tests` | Internal kernel tests |
| `ut_PS_tests` | Parameter server tests |
| `ut_OTHERS_tests` | Miscellaneous tests |
| `ut_ERROR_HANDLE_tests` | Error handling tests |

CI runs ~8851 tests across these executables.

## Common Recipes

### Full UT build
```bash
bash build.sh -t on -j48
```

### Incremental UT build (after first build)
```bash
bash build.sh -t on -j48 -i
```

### Debug UT build (for gdb/lldb debugging)
```bash
bash build.sh -t on -j48 -d
```

## Running Tests After Build

Use the provided `runtest.sh` (recommended) or run executables directly.

### Using runtest.sh (CI method)
```bash
# Run ALL C++ UT tests
cd <repo>
bash tests/ut/cpp/runtest.sh

# Run tests matching a filter across all executables
bash tests/ut/cpp/runtest.sh "*YourTestName*"
```

`runtest.sh` automatically:
- Sets `LD_LIBRARY_PATH` (gtest libs, mindspore python libs)
- Sets `PYTHONPATH` (test inputs, mindspore python)
- Copies test data (`tests/ut/data/`)
- Generates album test JSON
- Runs all 13 executables sequentially

### Running individual executables
```bash
cd build/mindspore/tests/ut/cpp

# Set required paths first
export LD_LIBRARY_PATH=${BUILD_PATH}/mindspore/googletest/googlemock/gtest:${PROJECT_PATH}/mindspore/python/mindspore:${PROJECT_PATH}/mindspore/python/mindspore/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${PROJECT_PATH}/tests/ut/cpp/python_input:${PROJECT_PATH}/mindspore/python:$PYTHONPATH
export GLOG_v=2

# Run one executable
./ut_CORE_tests --gtest_filter="*YourTestName*"
```

## Python UT (No Build Needed)

Python unit tests under `tests/ut/python/` do NOT require UT compilation.
They run against the installed MindSpore package:

```bash
pytest tests/ut/python/ops/test_your_op.py -v
```

## Python ST (Needs Production Build, Not UT Build)

System tests (`tests/st/`) need a production build for the target device:
- Ascend ST → build with `-e ascend`
- GPU ST → build with `-e gpu`
- CPU ST → build with `-e cpu`

ST tests are NOT compiled — they are Python scripts that import mindspore.

## Environment Setup

UT build shares the same environment as Ascend production build. Use a pre-configured
`env_ms.sh` script or set up manually. The script must configure:

- Conda activation + CANN `set_env.sh` sourcing
- `ASCEND_CUSTOM_PATH`, `CC`, `CXX`, `LIBRARY_PATH`, `LD_LIBRARY_PATH`
- `BUILD_PATH=<repo>/build` (required by `runtest.sh`)
- `PYTHONPATH=<repo>/build/package` (for Python UT that imports mindspore)
- ccache + `MSLIBS_CACHE_PATH` (optional but recommended)

See `build-ascend.md` "Environment Setup" section for the full setup.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Can not find CANN installation path` | CANN env vars not set | `source <conda_prefix>/Ascend/cann/set_env.sh` + `export ASCEND_CUSTOM_PATH=...` before build |
| `tar: Cannot change ownership` (isl, mkl_dnn, ms_kernels_internal, etc.) | Sandbox/container restricts `chown` | Add `--no-same-owner` to tar commands in `cmake/utils.cmake`, `cmake/external_libs/*.cmake`, and `scripts/build/check_and_build_ms_kernels_internal.sh` |
| `No CMAKE_C_COMPILER could be found` | ccache PATH wrapper dir missing | Explicitly `export CC=/usr/bin/gcc CXX=/usr/bin/g++` |
| `undefined reference to vtable for AddLayernormFusion` | `ms_kernels_internal.tar.gz` is git-lfs pointer, not real archive | `git lfs install && git lfs pull`, then rebuild. Without git-lfs, `ENABLE_INTERNAL_KERNELS` is off and derived class vtables are missing |
| `./ut_INTERNAL_KERNEL_tests: No such file or directory` | `ENABLE_INTERNAL_KERNELS` not set because git-lfs files missing | Same as above: `git lfs install && git lfs pull`, rebuild. The binary is only produced when `ms_kernels_internal.tar.gz` is a real archive |
| `cannot combine -t and -e` | Used both flags | Remove `-e`, UT build doesn't need it |
| `cannot find -latomic` | libatomic not in linker path | `export LIBRARY_PATH=$(python -c "import sys;print(sys.prefix)")/lib:$LIBRARY_PATH` |
| Linker errors in test binary | Missing symbols | Ensure full build (not incremental) first time |
| `gtest not found` | Dependencies not downloaded | Add `-S on` or check network |
| Test binary crashes | Debug build helps | Rebuild with `-d`, run under gdb |
| Python UT import error | MindSpore not installed | Do a production build first, set PYTHONPATH |
| `ms_kernels_internal.tar.gz` missing | Internal kernel archive not present | `git lfs pull`; without it, internal kernel UT tests are skipped |
