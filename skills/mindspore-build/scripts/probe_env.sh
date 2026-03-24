#!/bin/bash
# MindSpore Build Environment Probe
# Usage: bash probe_env.sh
# This script is READ-ONLY — it does not modify anything on the system.

set -u

SEP="================================================================"

echo "$SEP"
echo "MindSpore Build Environment Probe"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "$SEP"

# ── OS ───────────────────────────────────────────────────────────────
echo ""
echo "[1/10] OS Info"
echo "---"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  Name:    ${PRETTY_NAME:-$NAME}"
    echo "  ID:      ${ID:-unknown}"
    echo "  Version: ${VERSION_ID:-unknown}"
else
    echo "  /etc/os-release not found"
fi
echo "  Kernel:  $(uname -r)"
echo "  Arch:    $(uname -m)"

# ── Hardware ─────────────────────────────────────────────────────────
echo ""
echo "[2/10] Hardware"
echo "---"
echo "  CPU cores: $(nproc 2>/dev/null || echo 'unknown')"
MEM_TOTAL=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
echo "  RAM (GB):  ${MEM_TOTAL:-unknown}"
echo "  Disk free (home):"
df -h "$HOME" 2>/dev/null | tail -1 | awk '{printf "    %s total, %s used, %s free (%s)\n", $2, $3, $4, $5}'

# ── NPU ──────────────────────────────────────────────────────────────
echo ""
echo "[3/10] NPU (Ascend)"
echo "---"
if command -v npu-smi &>/dev/null; then
    echo "  npu-smi: $(which npu-smi)"
    npu-smi info 2>&1 | head -30 | sed 's/^/  /'
else
    echo "  npu-smi: NOT FOUND"
fi

# ── CANN ─────────────────────────────────────────────────────────────
echo ""
echo "[4/10] CANN (Ascend Toolkit)"
echo "---"
CANN_FOUND=0
for dir in /usr/local/Ascend/ascend-toolkit/latest \
           /usr/local/Ascend/ascend-toolkit \
           /usr/local/Ascend/cann; do
    if [ -d "$dir" ]; then
        echo "  Found: $dir"
        CANN_FOUND=1
        # version
        for vf in "$dir/version.cfg" "$dir/version.info" \
                  /usr/local/Ascend/version.info; do
            if [ -f "$vf" ]; then
                echo "  Version file ($vf):"
                head -3 "$vf" | sed 's/^/    /'
                break
            fi
        done
        break
    fi
done
if [ "$CANN_FOUND" -eq 0 ]; then
    echo "  CANN: NOT FOUND under /usr/local/Ascend/"
    ls /usr/local/Ascend/ 2>/dev/null | sed 's/^/    /' || echo "    /usr/local/Ascend/ does not exist"
fi

# CANN set_env.sh
CANN_ENV=""
for p in /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/cann/set_env.sh; do
    if [ -f "$p" ]; then
        CANN_ENV="$p"
        break
    fi
done
echo "  set_env.sh: ${CANN_ENV:-NOT FOUND}"

# NPU devices
echo "  NPU devices:"
ls /dev/davinci* 2>/dev/null | sed 's/^/    /' || echo "    none found"

# ── Docker ───────────────────────────────────────────────────────────
echo ""
echo "[5/10] Docker"
echo "---"
if command -v docker &>/dev/null; then
    echo "  docker: $(docker --version 2>&1)"
    if docker info &>/dev/null 2>&1; then
        echo "  permission: OK (can run docker commands)"
    else
        echo "  permission: DENIED (user not in docker group?)"
        echo "    groups: $(groups)"
    fi
    echo "  running containers: $(docker ps -q 2>/dev/null | wc -l)"
    echo "  images:"
    docker images --format '    {{.Repository}}:{{.Tag}}  {{.Size}}' 2>/dev/null | head -10
else
    echo "  docker: NOT FOUND"
fi

# ── GCC ──────────────────────────────────────────────────────────────
echo ""
echo "[6/10] GCC"
echo "---"
if command -v gcc &>/dev/null; then
    echo "  gcc: $(gcc --version 2>&1 | head -1)"
else
    echo "  gcc: NOT FOUND"
fi
if command -v g++ &>/dev/null; then
    echo "  g++: $(g++ --version 2>&1 | head -1)"
else
    echo "  g++: NOT FOUND"
fi

# ── CMake ────────────────────────────────────────────────────────────
echo ""
echo "[7/10] CMake"
echo "---"
if command -v cmake &>/dev/null; then
    echo "  cmake: $(cmake --version 2>&1 | head -1)"
else
    echo "  cmake: NOT FOUND"
fi

# ── Python / Conda ───────────────────────────────────────────────────
echo ""
echo "[8/10] Python / Conda"
echo "---"
if command -v conda &>/dev/null; then
    echo "  conda: $(conda --version 2>&1)"
    echo "  conda envs:"
    conda env list 2>/dev/null | grep -v '^#' | grep -v '^$' | sed 's/^/    /'
else
    echo "  conda: NOT FOUND"
fi
if command -v python3 &>/dev/null; then
    echo "  python3: $(python3 --version 2>&1)"
    echo "  python3 path: $(which python3)"
elif command -v python &>/dev/null; then
    echo "  python: $(python --version 2>&1)"
    echo "  python path: $(which python)"
else
    echo "  python: NOT FOUND"
fi

# ── Git ──────────────────────────────────────────────────────────────
echo ""
echo "[9/10] Git"
echo "---"
if command -v git &>/dev/null; then
    echo "  git: $(git --version 2>&1)"
else
    echo "  git: NOT FOUND"
fi
if command -v git-lfs &>/dev/null; then
    echo "  git-lfs: $(git-lfs version 2>&1 | head -1)"
else
    echo "  git-lfs: NOT FOUND"
fi

# ── Network ──────────────────────────────────────────────────────────
echo ""
echo "[10/10] Network"
echo "---"
echo -n "  github.com: "
if curl -sI --connect-timeout 5 https://github.com >/dev/null 2>&1; then
    echo "reachable"
else
    echo "UNREACHABLE (may need -S on for Gitee mirrors)"
fi
echo -n "  atomgit.com (GitCode): "
if curl -sI --connect-timeout 5 https://atomgit.com >/dev/null 2>&1; then
    echo "reachable"
else
    echo "UNREACHABLE"
fi
echo -n "  pypi (huaweicloud): "
if curl -sI --connect-timeout 5 https://repo.huaweicloud.com >/dev/null 2>&1; then
    echo "reachable"
else
    echo "UNREACHABLE"
fi
echo -n "  swr.cn-south-1 (docker registry): "
if curl -sI --connect-timeout 5 https://swr.cn-south-1.myhuaweicloud.com >/dev/null 2>&1; then
    echo "reachable"
else
    echo "UNREACHABLE"
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "$SEP"
echo "SUMMARY"
echo "$SEP"

ISSUES=0

echo -n "  NPU:    "
if command -v npu-smi &>/dev/null; then echo "OK"; else echo "MISSING"; ISSUES=$((ISSUES+1)); fi

echo -n "  CANN:   "
if [ "$CANN_FOUND" -eq 1 ]; then echo "OK"; else echo "MISSING"; ISSUES=$((ISSUES+1)); fi

echo -n "  Docker: "
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    echo "OK"
else
    echo "MISSING or NO PERMISSION"
    ISSUES=$((ISSUES+1))
fi

echo -n "  GCC:    "
if command -v gcc &>/dev/null; then echo "OK ($(gcc -dumpversion 2>/dev/null))"; else echo "MISSING (install needed)"; fi

echo -n "  CMake:  "
if command -v cmake &>/dev/null; then echo "OK"; else echo "MISSING (install needed)"; fi

echo -n "  Git:    "
if command -v git &>/dev/null; then echo "OK"; else echo "MISSING"; ISSUES=$((ISSUES+1)); fi

echo ""
if [ "$ISSUES" -eq 0 ]; then
    echo "  >>> All critical dependencies found. Ready to proceed."
else
    echo "  >>> $ISSUES critical issue(s) found. See details above."
fi

echo ""
echo "$SEP"
echo "Probe complete. Copy ALL output above and send it back."
echo "$SEP"
