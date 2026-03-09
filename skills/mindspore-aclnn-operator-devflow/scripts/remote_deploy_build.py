"""
Remote Deploy & Build Script for MindSpore ACLNN Operator Development.

Pushes git-committed changes to a remote Ascend server, extracts them,
and runs the build command.  Build stdout/stderr are captured to a local
log file for the agent to analyse.

Server connection info is read from servers.json (next to this script's
parent directory).  Use --device to pick which server to use.

Usage:
    python scripts/remote_deploy_build.py ^
        --local-repo D:\\open_source\\mindspore ^
        --log-file D:\\open_source\\build_output.log

    python scripts/remote_deploy_build.py ^
        --device 910b ^
        --local-repo D:\\open_source\\mindspore ^
        --log-file D:\\open_source\\build_output.log

Exit codes:
    0   build succeeded
    1   build failed (check --log-file for errors)
    2   deploy/upload/extract failed (infra issue, not a code error)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
SERVERS_JSON = os.path.join(SKILL_DIR, "servers.json")


# ── helpers ──────────────────────────────────────────────────────────────

def _load_server_config(device: str = None) -> dict:
    """Load server config from servers.json by device type."""
    with open(SERVERS_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if device is None:
        device = cfg.get("default", "910b")
    servers = cfg.get("servers", {})
    if device not in servers:
        available = ", ".join(servers.keys())
        print(f"FATAL: device '{device}' not found in servers.json. "
              f"Available: {available}", file=sys.stderr)
        sys.exit(2)
    server = servers[device]
    print(f"  Using server [{device}]: {server['host']} "
          f"({server.get('description', '')})")
    return server


def _make_askpass(password: str) -> str:
    """Create a .bat helper that echoes the password (Windows SSH_ASKPASS)."""
    bat = os.path.join(tempfile.gettempdir(), "ms_askpass.bat")
    with open(bat, "w") as f:
        f.write(f"@echo off\necho {password}\n")
    return bat


def _ssh_env(askpass_bat: str) -> dict:
    env = os.environ.copy()
    env["SSH_ASKPASS"] = askpass_bat
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = ":0"
    return env


def _run(cmd, env, desc, timeout=7200):
    print(f"[STEP] {desc}")
    result = subprocess.run(
        cmd, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
    )
    return result


# ── main pipeline ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deploy & build on remote Ascend server")
    ap.add_argument("--device", default=None,
                    help="Device type key in servers.json (e.g. 910b, 910a, gpu). "
                         "Defaults to the 'default' field in servers.json.")
    ap.add_argument("--local-repo", required=True, help="Local git repo root")
    ap.add_argument("--log-file", required=True, help="Local path to write build log")
    ap.add_argument("--source", default="HEAD",
                    help="Git ref for changes (default HEAD = latest commit)")
    ap.add_argument("--remote-dir", default=None,
                    help="Override remote_dir from servers.json")
    ap.add_argument("--build-cmd", default=None,
                    help="Override build_cmd from servers.json")
    args = ap.parse_args()

    server = _load_server_config(args.device)
    host = server["host"]
    user = server["user"]
    password = server["password"]
    remote_dir = args.remote_dir or server["remote_dir"]
    build_cmd = args.build_cmd or server["build_cmd"]

    askpass = _make_askpass(password)
    env = _ssh_env(askpass)
    target = f"{user}@{host}"

    # ── 1. Identify changed files ────────────────────────────────────────
    print("=" * 60)
    print("[1/5] Identifying changed files …")
    r = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", args.source],
        capture_output=True, text=True, cwd=args.local_repo,
    )
    if r.returncode != 0:
        print(f"FATAL: git diff-tree failed: {r.stderr}", file=sys.stderr)
        return 2
    files = [f.strip() for f in r.stdout.strip().splitlines() if f.strip()]
    if not files:
        print("No files changed in the specified commit.")
        return 0
    print(f"  {len(files)} file(s) changed:")
    for f in files:
        print(f"    • {f}")

    # ── 2. Archive ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/5] Creating archive …")
    tar_path = os.path.join(tempfile.gettempdir(), "ms_deploy.tar.gz")
    r = subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", tar_path, args.source, "--"] + files,
        capture_output=True, text=True, cwd=args.local_repo,
    )
    if r.returncode != 0:
        print(f"FATAL: git archive failed: {r.stderr}", file=sys.stderr)
        return 2
    size_kb = os.path.getsize(tar_path) / 1024
    print(f"  Archive: {tar_path} ({size_kb:.1f} KB)")

    # ── 3. Upload ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/5] Uploading to server …")
    r = _run(
        ["scp", "-o", "StrictHostKeyChecking=no", tar_path,
         f"{target}:/tmp/ms_deploy.tar.gz"],
        env, "SCP upload", timeout=120,
    )
    if r.returncode != 0:
        print(f"FATAL: SCP failed: {r.stderr}", file=sys.stderr)
        return 2
    print("  Upload OK")

    # ── 4. Extract ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[4/5] Extracting on server …")
    extract = (
        f"cd {remote_dir} && "
        f"tar xzf /tmp/ms_deploy.tar.gz && "
        f"rm -f /tmp/ms_deploy.tar.gz && "
        f"echo __EXTRACT_OK__"
    )
    r = _run(
        ["ssh", "-o", "StrictHostKeyChecking=no", target, extract],
        env, "SSH extract", timeout=60,
    )
    if r.returncode != 0 or "__EXTRACT_OK__" not in r.stdout:
        print(f"FATAL: extract failed: {r.stderr}", file=sys.stderr)
        return 2
    print("  Extract OK")

    # ── 5. Build ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[5/5] Building: {build_cmd}")
    print("  (this may take a while …)")
    build = f"cd {remote_dir} && {build_cmd} 2>&1"
    r = _run(
        ["ssh", "-o", "StrictHostKeyChecking=no", target, build],
        env, "SSH build", timeout=7200,
    )

    # Write full build log
    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write(r.stdout or "")
        if r.stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(r.stderr)
    print(f"  Build log written to: {args.log_file}")

    # ── Cleanup ──────────────────────────────────────────────────────────
    try:
        os.remove(askpass)
    except OSError:
        pass
    try:
        os.remove(tar_path)
    except OSError:
        pass

    if r.returncode == 0:
        print("\n" + "=" * 60)
        print("BUILD SUCCEEDED")
        return 0
    else:
        print("\n" + "=" * 60)
        print(f"BUILD FAILED (exit code {r.returncode})")
        # Print last 40 lines as quick summary
        lines = (r.stdout or "").splitlines()
        tail = lines[-40:] if len(lines) > 40 else lines
        print("\n--- last 40 lines of build output ---")
        print("\n".join(tail))
        return 1


if __name__ == "__main__":
    sys.exit(main())
