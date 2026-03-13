#!/bin/bash
# [EXPERIMENTAL] MindSpore Docker Build Setup Script
# Status: Not yet validated end-to-end. Review before running.
# For: openEuler 22.03, aarch64, 910B3, Docker available
#
# Usage:
#   bash setup_build.sh           # Phase 1: probe existing image
#   bash setup_build.sh phase2    # Phase 2: create container
#   bash setup_build.sh phase3    # Phase 3: clone source + install deps
#
# Follows the same conventions as other containers on this machine:
#   - Work dir: /home/<your_name> (mounted into container as /home)
#   - Container name: <your_name>_ms
#   - Image: mindspore2.8:8.5.0

set -euo pipefail

SEP="================================================================"

# ── Config (edit these to match your setup) ──────────────────────────
HOST_HOME="/home/<your_name>"       # Edit: your home dir on host
CONTAINER_NAME="<your_name>_ms"
MS_IMAGE="mindspore2.8:8.5.0"
NPU_DEVICE="0"                     # which davinci device (0-7), use 1 to not hog all
# ─────────────────────────────────────────────────────────────────────

echo "$SEP"
echo "MindSpore Build Setup"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config: image=$MS_IMAGE  container=$CONTAINER_NAME  home=$HOST_HOME"
echo "$SEP"

# ── Phase 1: Probe existing image ───────────────────────────────────

echo ""
echo "[Phase 1] Probing Docker image: $MS_IMAGE"
echo "$SEP"

if ! docker image inspect "$MS_IMAGE" &>/dev/null; then
    echo "FATAL: Image '$MS_IMAGE' not found."
    echo "Available images:"
    docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}'
    exit 1
fi

echo "--- CANN ---"
docker run --rm "$MS_IMAGE" bash -c '
    for d in /usr/local/Ascend/ascend-toolkit/latest \
             /usr/local/Ascend/ascend-toolkit \
             /usr/local/Ascend/cann; do
        [ -d "$d" ] && echo "  Found: $d" && break
    done
    for f in /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
             /usr/local/Ascend/version.info; do
        [ -f "$f" ] && echo "  Version: $(head -1 $f)" && break
    done
    FOUND_ENV=0
    for p in /usr/local/Ascend/ascend-toolkit/set_env.sh \
             /usr/local/Ascend/cann/set_env.sh; do
        [ -f "$p" ] && echo "  set_env.sh: $p" && FOUND_ENV=1 && break
    done
    [ "$FOUND_ENV" -eq 0 ] && echo "  set_env.sh: NOT FOUND"
' 2>/dev/null || echo "  Failed to probe CANN"

echo ""
echo "--- Build tools ---"
docker run --rm "$MS_IMAGE" bash -c '
    echo "  GCC:     $(gcc --version 2>&1 | head -1 || echo NOT FOUND)"
    echo "  G++:     $(g++ --version 2>&1 | head -1 || echo NOT FOUND)"
    echo "  CMake:   $(cmake --version 2>&1 | head -1 || echo NOT FOUND)"
    echo "  Git:     $(git --version 2>&1 || echo NOT FOUND)"
    echo "  git-lfs: $(git-lfs version 2>&1 | head -1 || echo NOT FOUND)"
    echo "  Python:  $(python3 --version 2>&1 || python --version 2>&1 || echo NOT FOUND)"
    echo "  pip:     $(pip --version 2>&1 | head -1 || echo NOT FOUND)"
' 2>/dev/null || echo "  Failed to probe tools"

echo ""
echo "--- Key Python packages ---"
docker run --rm "$MS_IMAGE" bash -c '
    pip list 2>/dev/null | grep -iE "numpy|pyyaml|wheel|setuptools|sympy|protobuf|scipy|jinja2|mindspore" | sed "s/^/  /"
' 2>/dev/null || echo "  Failed to probe packages"

echo ""
echo "--- MindSpore pre-installed? ---"
docker run --rm "$MS_IMAGE" bash -c '
    python3 -c "import mindspore; print(\"  Version:\", mindspore.__version__)" 2>/dev/null \
        || python -c "import mindspore; print(\"  Version:\", mindspore.__version__)" 2>/dev/null \
        || echo "  Not installed"
' 2>/dev/null || echo "  Check failed"

echo ""
echo "$SEP"
echo "[Phase 1 Complete]"
echo "Review output above. If CANN + GCC + CMake are present, run:"
echo "  bash setup_build.sh phase2"
echo "$SEP"

if [ "${1:-}" != "phase2" ] && [ "${1:-}" != "phase3" ]; then
    exit 0
fi

# ── Phase 2: Create build container ─────────────────────────────────

echo ""
echo "[Phase 2] Creating container: $CONTAINER_NAME"
echo "$SEP"

# Create host directory
if [ ! -d "$HOST_HOME" ]; then
    echo "Creating $HOST_HOME ..."
    mkdir -p "$HOST_HOME"
else
    echo "$HOST_HOME already exists."
fi

# Check existing container
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "Container '$CONTAINER_NAME' already exists."
    STATE=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)
    if [ "$STATE" = "true" ]; then
        echo "  Status: running"
        echo "  Enter it with: docker exec -it $CONTAINER_NAME bash"
    else
        echo "  Status: stopped. Starting..."
        docker start "$CONTAINER_NAME"
        echo "  Started. Enter with: docker exec -it $CONTAINER_NAME bash"
    fi
else
    echo "Creating new container..."
    echo "  Image:   $MS_IMAGE"
    echo "  Mount:   $HOST_HOME -> /home"
    echo "  Device:  davinci${NPU_DEVICE}"

    docker run -dit --name "$CONTAINER_NAME" \
        --ipc=host \
        --device="/dev/davinci${NPU_DEVICE}" \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /usr/local/dcmi:/usr/local/dcmi:ro \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v /etc/hccn.conf:/etc/hccn.conf:ro \
        -v /etc/localtime:/etc/localtime:ro \
        -v /var/log/npu:/usr/slog \
        -v "$HOST_HOME":/home \
        "$MS_IMAGE" \
        /bin/bash

    echo "Container '$CONTAINER_NAME' created and running."
fi

echo ""
echo "Verifying inside container..."
docker exec "$CONTAINER_NAME" bash -c '
    echo "  npu-smi:"
    npu-smi info 2>&1 | head -12 | sed "s/^/    /"
    echo ""
    for p in /usr/local/Ascend/ascend-toolkit/set_env.sh \
             /usr/local/Ascend/cann/set_env.sh; do
        if [ -f "$p" ]; then
            source "$p" 2>/dev/null
            echo "  CANN sourced: $p"
            break
        fi
    done
    echo "  gcc: $(gcc --version 2>&1 | head -1)"
    echo "  cmake: $(cmake --version 2>&1 | head -1)"
    echo "  python: $(python3 --version 2>&1 || python --version 2>&1)"
' 2>/dev/null

echo ""
echo "$SEP"
echo "[Phase 2 Complete]"
echo ""
echo "Enter container:  docker exec -it $CONTAINER_NAME bash"
echo "Next step:        bash setup_build.sh phase3"
echo "$SEP"

if [ "${1:-}" != "phase3" ]; then
    exit 0
fi

# ── Phase 3: Clone source and install deps ───────────────────────────

echo ""
echo "[Phase 3] Clone MindSpore + install build deps"
echo "$SEP"

docker exec "$CONTAINER_NAME" bash -c '
    cd /home

    if [ -d "mindspore/.git" ]; then
        echo "MindSpore repo already exists at /home/mindspore"
        cd mindspore
        echo "  Branch: $(git branch --show-current 2>/dev/null || echo detached)"
        echo "  Commit: $(git log --oneline -1 2>/dev/null)"
    else
        echo "Cloning MindSpore from atomgit.com ..."
        git clone https://atomgit.com/mindspore/mindspore.git
        cd mindspore
        echo "Initializing submodules..."
        git submodule update --init
        echo "Clone complete."
    fi
'

echo ""
echo "Installing Python build dependencies..."
docker exec "$CONTAINER_NAME" bash -c '
    pip install wheel setuptools pyyaml "numpy>=1.20.0,<2.0.0" \
        -i https://repo.huaweicloud.com/repository/pypi/simple/ 2>&1 | tail -5
    pip uninstall te topi hccl -y 2>/dev/null || true
    pip install sympy protobuf attrs cloudpickle decorator \
        ml-dtypes psutil scipy tornado jinja2 \
        -i https://repo.huaweicloud.com/repository/pypi/simple/ 2>&1 | tail -5
    echo ""
    echo "Key packages:"
    pip list 2>/dev/null | grep -iE "numpy|pyyaml|wheel|setuptools" | sed "s/^/  /"
'

echo ""
echo "$SEP"
echo "[Phase 3 Complete]"
echo ""
echo "Everything is ready. To build:"
echo ""
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "  # Inside container:"
echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
echo "  cd /home/mindspore"
echo "  bash build.sh -e ascend -V 910b -j128"
echo ""
echo "Build output will be at:"
echo "  Container: /home/mindspore/output/"
echo "  Host:      $HOST_HOME/mindspore/output/"
echo "$SEP"
