# Docker Build Environment for Ascend (version isolation)

> **Status: NOT YET VERIFIED.** This workflow has not been successfully tested end-to-end.
> Use with caution and report issues.

Self-contained guide. No other files needed.

> **Most users don't need this.** For normal Ascend builds (including shared machines),
> use `server-init-ascend.md` or `build-ascend.md` directly — install system packages
> on the host, use conda with `<user>_py<ver>` naming, and `env.sh` for session-only
> environment setup. That approach is simpler and doesn't require Docker.

## When to Use

- You need a **different toolchain version** than what's on the host (e.g. GCC 9 when host has GCC 7)
- You need to test against **multiple CANN versions** on the same machine
- You want a fully reproducible build image that can be shared across machines

## How It Works

```
Host machine (shared, don't touch)
├── NPU driver + firmware (pre-installed by admin)
├── CANN toolkit at /usr/local/Ascend/
├── /dev/davinci* devices
│
└── Your Docker container (isolated, yours only)
    ├── GCC, CMake, git, git-lfs (installed in container)
    ├── Conda + Python environment
    ├── MindSpore source (mounted from host)
    └── Build output (persisted via volume)
```

Key: NPU driver and CANN live on the host. The container mounts them read-only.
You install build tools inside the container without affecting anyone else.

## Prerequisites (on host)

1. **Docker installed** (18.03+): `docker --version`
2. **NPU accessible**: `npu-smi info`
3. **CANN installed on host**: `ls /usr/local/Ascend/ascend-toolkit/latest/`
4. **Your user in docker group**: `groups` should show `docker`
   (if not: ask admin to run `sudo usermod -aG docker $USER`)
5. **MindSpore source cloned** somewhere on host

## Step 0: Gather Host Info

```bash
uname -m                    # x86_64 or aarch64 (determines base image)
npu-smi info                # NPU model
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null  # CANN version
ls /dev/davinci*            # available NPU devices
docker --version
free -h && nproc            # RAM and CPU cores
```

## Step 1: Create Project Directory on Host

```bash
mkdir -p ~/ms_docker_build
cd ~/ms_docker_build
```

If you haven't cloned MindSpore yet:
```bash
git clone https://atomgit.com/mindspore/mindspore.git
cd mindspore && git submodule update --init && cd ..
```

## Step 2: Create Dockerfile

```bash
cat > ~/ms_docker_build/Dockerfile << 'DOCKEREOF'
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-7 g++-7 \
    git tcl patch libnuma-dev flex \
    curl wget ca-certificates \
    openssh-client \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70 \
    && rm -rf /var/lib/apt/lists/*
# ^ update-alternatives is acceptable inside a container (isolated from host)

# git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.22.3
ARG ARCH_SUFFIX=x86_64
RUN curl -O https://cmake.org/files/v3.22/cmake-3.22.3-linux-${ARCH_SUFFIX}.sh \
    && mkdir -p /usr/local/cmake \
    && bash cmake-3.22.3-linux-${ARCH_SUFFIX}.sh --prefix=/usr/local/cmake --exclude-subdir \
    && rm cmake-3.22.3-linux-*.sh
ENV PATH="/usr/local/cmake/bin:${PATH}"

# Miniconda
RUN curl -o /tmp/miniconda.sh \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_25.7.0-2-Linux-$(uname -m).sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Python environment
RUN conda create -c conda-forge -n ms python=3.9.11 -y
SHELL ["conda", "run", "-n", "ms", "/bin/bash", "-c"]

RUN pip install wheel setuptools pyyaml "numpy>=1.20.0,<2.0.0" \
    && pip uninstall te topi hccl -y 2>/dev/null; \
    pip install sympy protobuf attrs cloudpickle decorator \
    ml-dtypes psutil scipy tornado jinja2

# Default env vars
ENV GLOG_v=2
ENV ASCEND_CUSTOM_PATH=/usr/local/Ascend

WORKDIR /workspace/mindspore

# Entrypoint: activate conda + source CANN
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate ms\n\
for p in /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/cann/set_env.sh; do\n\
    [ -f "$p" ] && source "$p" && break\n\
done\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
DOCKEREOF
```

### For aarch64 machines

Change the `ARCH_SUFFIX` build arg:
```bash
docker build --build-arg ARCH_SUFFIX=aarch64 -t ms-build .
```

## Step 3: Build the Docker Image

```bash
cd ~/ms_docker_build

# x86_64 (default)
docker build -t ms-build .

# aarch64
# docker build --build-arg ARCH_SUFFIX=aarch64 -t ms-build .
```

This takes 5-10 minutes on first build. The image is cached after that.

## Step 4: Run the Container

### Determine which NPU devices to mount

```bash
# See available devices
ls /dev/davinci*
```

### Launch container

```bash
docker run -it --name ms-dev \
    --ipc=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v ~/ms_docker_build/mindspore:/workspace/mindspore \
    ms-build
```

Key points:
- **`-v .../mindspore:/workspace/mindspore`**: mounts your source code into the container.
  Edits on host are immediately visible inside.
- **`--device`**: only mount the NPU devices you need (don't hog all 8 if you only need 1).
  Use `davinci0` for single-device testing.
- **`:ro`**: read-only mounts for host system files (driver, CANN, tools).
- **`--name ms-dev`**: name the container so you can re-enter it later.

### If CANN is at a different path

Some machines have CANN under `/usr/local/Ascend/cann/` instead of `ascend-toolkit`:
```bash
# Add this mount instead of (or in addition to) the ascend-toolkit mount:
-v /usr/local/Ascend/cann:/usr/local/Ascend/cann:ro
```

## Step 5: Verify Inside Container

Once inside the container:

```bash
# Environment should be auto-configured by entrypoint
gcc --version          # GCC 7
cmake --version        # 3.22.3+
python --version       # 3.9.11
npu-smi info           # NPU visible from container
echo $ASCEND_HOME      # set by CANN set_env.sh
```

## Step 6: Build MindSpore

```bash
# Already inside container, already in /workspace/mindspore
bash build.sh -e ascend -V 910b -j64
```

### Install and verify

```bash
pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

## Daily Workflow

### Re-enter an existing container

```bash
# If container is stopped
docker start ms-dev
docker exec -it ms-dev /bin/bash

# Inside container, environment is auto-configured
cd /workspace/mindspore
bash build.sh -e ascend -V 910b -j64 -i    # incremental build
```

### Start a fresh container (if needed)

```bash
docker rm ms-dev    # remove old container
# Re-run the docker run command from Step 4
```

### Copy build artifacts out (if not using volume mount)

```bash
docker cp ms-dev:/workspace/mindspore/output/mindspore-*.whl ./
```

## env.sh for Docker Workflow

If you want a one-command entry point on the host side, create `env.sh`:

```bash
cat > ~/ms_docker_build/env.sh << 'ENVEOF'
#!/bin/bash
# Start or re-enter the MindSpore Docker build environment

CONTAINER_NAME="ms-dev"

if docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "Attaching to running container '$CONTAINER_NAME' ..."
    docker exec -it "$CONTAINER_NAME" /bin/bash
elif docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "Starting stopped container '$CONTAINER_NAME' ..."
    docker start "$CONTAINER_NAME"
    docker exec -it "$CONTAINER_NAME" /bin/bash
else
    echo "Container '$CONTAINER_NAME' not found. Create it first with:"
    echo "  docker run -it --name ms-dev ... (see docker-build-ascend.md Step 4)"
fi
ENVEOF

chmod +x ~/ms_docker_build/env.sh
```

Usage:
```bash
~/ms_docker_build/env.sh    # enters the container, environment ready
```

## Thread Count Guideline

Same as bare-metal — each thread uses 2-4 GB RAM:

| Machine RAM | Recommended -j |
|-------------|----------------|
| 64 GB | 16-32 |
| 128 GB | 32-64 |
| 256 GB+ | 64-128 |

Docker does not limit RAM by default. If the admin set `--memory` limits on your
container, adjust `-j` accordingly.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `docker: permission denied` | User not in docker group | Ask admin: `sudo usermod -aG docker $USER`, then re-login |
| `npu-smi info` fails in container | Missing device mounts | Check `--device` flags match available `/dev/davinci*` |
| `CANN set_env.sh not found` in container | CANN not mounted | Add `-v /usr/local/Ascend/ascend-toolkit:...` or `cann:...` |
| Build OOM (Killed) | Too many -j threads | Reduce -j; check if admin set `--memory` limit |
| Slow I/O during build | Volume mount overhead | Normal for Docker; use `--tmpfs /tmp` for temp files |
| `cannot open shared object` at runtime | Driver version mismatch | Host driver must match CANN version |
| Container lost after reboot | Container was not named | Always use `--name ms-dev`; use `docker start` to resume |
| NPU occupied by another user's container | Device conflict | Use a different `--device=/dev/davinciN` |
| `git submodule update` fails in container | git-lfs or network issue | Run on host first, then mount the complete repo |
