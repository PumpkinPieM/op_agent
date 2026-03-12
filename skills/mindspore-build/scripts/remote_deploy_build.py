"""
Remote Deploy & Build for MindSpore.

Pushes git-committed changes to a remote build server, extracts them,
and runs the build command. Build output is captured to a local log file.

Server connection info is read from servers.json (skill root directory).

Authentication priority (highest to lowest):
    1. SSH Key   - no password needed, uses ~/.ssh/id_rsa
    2. Env var   - password from MS_SSH_PASS_<DEVICE> environment variable
    3. JSON file - plaintext password in servers.json (not recommended)

Usage:
    # Deploy only committed changes
    python scripts/remote_deploy_build.py \
        --local-repo /path/to/mindspore \
        --log-file /path/to/build_output.log

    # Include uncommitted changes (staged + unstaged)
    python scripts/remote_deploy_build.py \
        --local-repo /path/to/mindspore \
        --log-file /path/to/build_output.log \
        --include-uncommitted

    # Specify server
    python scripts/remote_deploy_build.py \
        --device 910b \
        --local-repo /path/to/mindspore \
        --log-file /path/to/build_output.log

Exit codes:
    0   build succeeded
    1   build failed (check --log-file for errors)
    2   deploy/network/infrastructure failure
    3   uncommitted changes detected but not included (use --include-uncommitted)
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
SERVERS_JSON = os.path.join(SKILL_DIR, "servers.json")

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]


# ── Server config & credentials ─────────────────────────────────────────

def _load_server_config(device=None):
    if not os.path.exists(SERVERS_JSON):
        print(
            f"FATAL: {SERVERS_JSON} not found.\n"
            "Copy servers.example.json to servers.json and fill in your server info.",
            file=sys.stderr,
        )
        sys.exit(2)
    with open(SERVERS_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if device is None:
        device = cfg.get("default", "910b")
    servers = cfg.get("servers", {})
    if device not in servers:
        available = ", ".join(servers.keys())
        print(
            f"FATAL: device '{device}' not found in servers.json. "
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(2)
    server = servers[device]
    print(f"  Using server [{device}]: {server['host']} "
          f"({server.get('description', '')})")
    return server, device


def _resolve_auth(server, device_key):
    """Resolve authentication method. Returns dict with 'method' and optional 'password'."""
    method = server.get("auth_method", "env")

    if method == "ssh_key":
        return {"method": "ssh_key"}

    env_var = f"MS_SSH_PASS_{device_key.upper()}"

    if method == "env":
        password = os.environ.get(env_var)
        if password:
            return {"method": "password", "password": password}
        if "password" in server and server["password"]:
            print(
                f"  WARNING: No ${env_var} found, falling back to plaintext "
                f"password in servers.json. Consider using ssh_key or setting "
                f"the environment variable.",
                file=sys.stderr,
            )
            return {"method": "password", "password": server["password"]}
        print(
            f"FATAL: auth_method is 'env' but ${env_var} is not set and no "
            f"fallback password in servers.json.\n"
            f"Options:\n"
            f"  1. Set auth_method to 'ssh_key' and configure SSH keys\n"
            f"  2. Export {env_var}=<your_password>\n"
            f"  3. Add 'password' field to servers.json (not recommended)",
            file=sys.stderr,
        )
        sys.exit(2)

    if method == "password":
        if "password" not in server or not server["password"]:
            print(
                "FATAL: auth_method is 'password' but no password in servers.json.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            "  WARNING: Using plaintext password from servers.json. "
            "Consider ssh_key or environment variable instead.",
            file=sys.stderr,
        )
        return {"method": "password", "password": server["password"]}

    print(f"FATAL: unknown auth_method '{method}'. "
          f"Use 'ssh_key', 'env', or 'password'.", file=sys.stderr)
    sys.exit(2)


# ── SSH/SCP command builders ────────────────────────────────────────────

def _has_sshpass():
    return shutil.which("sshpass") is not None


def _make_askpass(password):
    """Fallback: create a temporary script for SSH_ASKPASS."""
    if sys.platform == "win32":
        path = os.path.join(tempfile.gettempdir(), "ms_askpass.bat")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"@echo off\necho {password}\n")
    else:
        path = os.path.join(tempfile.gettempdir(), "ms_askpass.sh")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
        with os.fdopen(fd, "w") as f:
            f.write(f"#!/bin/sh\necho '{password}'\n")
    return path


def _build_ssh_cmd(base_cmd, auth, target, remote_cmd=None):
    """Build an ssh or scp command with the appropriate auth mechanism."""
    if auth["method"] == "ssh_key":
        cmd = [base_cmd] + SSH_OPTS
        if base_cmd == "ssh":
            cmd += [target, remote_cmd]
        return cmd, os.environ.copy(), None

    password = auth["password"]

    if _has_sshpass():
        cmd = ["sshpass", "-p", password, base_cmd] + SSH_OPTS
        if base_cmd == "ssh":
            cmd += [target, remote_cmd]
        return cmd, os.environ.copy(), None

    askpass_path = _make_askpass(password)
    env = os.environ.copy()
    env["SSH_ASKPASS"] = askpass_path
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = ":0"
    cmd = [base_cmd] + SSH_OPTS
    if base_cmd == "ssh":
        cmd += [target, remote_cmd]
    return cmd, env, askpass_path


def _build_scp_cmd(auth, src, dst):
    """Build an scp command."""
    if auth["method"] == "ssh_key":
        return ["scp"] + SSH_OPTS + [src, dst], os.environ.copy(), None

    password = auth["password"]

    if _has_sshpass():
        return (["sshpass", "-p", password, "scp"] + SSH_OPTS + [src, dst],
                os.environ.copy(), None)

    askpass_path = _make_askpass(password)
    env = os.environ.copy()
    env["SSH_ASKPASS"] = askpass_path
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = ":0"
    return ["scp"] + SSH_OPTS + [src, dst], env, askpass_path


def _run(cmd, env, desc, timeout=7200):
    print(f"[STEP] {desc}")
    return subprocess.run(
        cmd, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
    )


def _get_uncommitted_files(repo_path):
    """Get list of uncommitted files (staged + unstaged, excluding deleted)."""
    files = set()
    
    # Staged changes
    r = subprocess.run(
        ["git", "diff", "--name-only", "--cached"],
        capture_output=True, text=True, cwd=repo_path,
    )
    if r.returncode == 0:
        for f in r.stdout.strip().splitlines():
            if f.strip():
                files.add(f.strip())
    
    # Unstaged changes
    r = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True, text=True, cwd=repo_path,
    )
    if r.returncode == 0:
        for f in r.stdout.strip().splitlines():
            if f.strip():
                files.add(f.strip())
    
    # Untracked files
    r = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, cwd=repo_path,
    )
    if r.returncode == 0:
        for f in r.stdout.strip().splitlines():
            if f.strip():
                files.add(f.strip())
    
    # Filter out deleted files
    existing_files = []
    for f in files:
        full_path = os.path.join(repo_path, f)
        if os.path.exists(full_path):
            existing_files.append(f)
    
    return existing_files


def _check_uncommitted_changes(repo_path):
    """Check if there are uncommitted changes. Returns (has_changes, file_list)."""
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=repo_path,
    )
    if r.returncode != 0:
        return False, []
    
    has_changes = bool(r.stdout.strip())
    if not has_changes:
        return False, []
    
    files = _get_uncommitted_files(repo_path)
    return True, files


def _create_archive_with_uncommitted(repo_path, committed_files, uncommitted_files, tar_path):
    """Create a tar.gz archive containing both committed and uncommitted files."""
    with tarfile.open(tar_path, "w:gz") as tar:
        # Add committed files from git archive
        r = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD", "--"] + committed_files,
            capture_output=True, cwd=repo_path,
        )
        if r.returncode != 0:
            return False, "git archive failed"
        
        # Extract committed files from the archive and add to our tar
        committed_tar = tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:")
        for member in committed_tar.getmembers():
            tar.addfile(member, committed_tar.extractfile(member))
        committed_tar.close()
        
        # Add uncommitted files directly
        for f in uncommitted_files:
            full_path = os.path.join(repo_path, f)
            if os.path.exists(full_path):
                tar.add(full_path, arcname=f)
    
    return True, None


# ── Main pipeline ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deploy & build on remote server")
    ap.add_argument("--device", default=None,
                    help="Device key in servers.json (e.g. 910b, 910a, gpu)")
    ap.add_argument("--local-repo", required=True, help="Local git repo root")
    ap.add_argument("--log-file", required=True, help="Path to write build log")
    ap.add_argument("--source", default="HEAD",
                    help="Git ref for changes (default: HEAD)")
    ap.add_argument("--remote-dir", default=None,
                    help="Override remote_dir from servers.json")
    ap.add_argument("--build-cmd", default=None,
                    help="Override build_cmd from servers.json")
    ap.add_argument("--include-uncommitted", action="store_true",
                    help="Include uncommitted (staged + unstaged) changes in deployment")
    ap.add_argument("--check-uncommitted", action="store_true",
                    help="Only check for uncommitted changes, don't build. Exit 3 if found.")
    args = ap.parse_args()

    server, device_key = _load_server_config(args.device)
    host = server["host"]
    user = server["user"]
    remote_dir = args.remote_dir or server["remote_dir"]
    build_cmd = args.build_cmd or server["build_cmd"]
    target = f"{user}@{host}"

    auth = _resolve_auth(server, device_key)
    print(f"  Auth method: {auth['method']}"
          + (" (sshpass)" if auth["method"] == "password" and _has_sshpass()
             else " (SSH_ASKPASS fallback)" if auth["method"] == "password"
             else ""))

    temp_files = []

    try:
        # 0. Check for uncommitted changes
        has_uncommitted, uncommitted_files = _check_uncommitted_changes(args.local_repo)
        
        if args.check_uncommitted:
            if has_uncommitted:
                print("=" * 60)
                print("UNCOMMITTED CHANGES DETECTED")
                print(f"  {len(uncommitted_files)} file(s) with uncommitted changes:")
                for f in uncommitted_files[:20]:
                    print(f"    - {f}")
                if len(uncommitted_files) > 20:
                    print(f"    ... and {len(uncommitted_files) - 20} more")
                print("\nOptions:")
                print("  1. Commit changes first: git add -A && git commit -m 'WIP'")
                print("  2. Include uncommitted: add --include-uncommitted flag")
                return 3
            else:
                print("No uncommitted changes detected.")
                return 0
        
        if has_uncommitted and not args.include_uncommitted:
            print("=" * 60)
            print("WARNING: Uncommitted changes detected!")
            print(f"  {len(uncommitted_files)} file(s) will NOT be deployed:")
            for f in uncommitted_files[:10]:
                print(f"    - {f}")
            if len(uncommitted_files) > 10:
                print(f"    ... and {len(uncommitted_files) - 10} more")
            print("\nTo include these changes, use --include-uncommitted flag.")
            print("Continuing with committed changes only...\n")

        # 1. Identify changed files (committed)
        print("=" * 60)
        print("[1/5] Identifying changed files ...")
        r = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--name-only",
             args.source],
            capture_output=True, text=True, cwd=args.local_repo,
        )
        if r.returncode != 0:
            print(f"FATAL: git diff-tree failed: {r.stderr}", file=sys.stderr)
            return 2
        committed_files = [f.strip() for f in r.stdout.strip().splitlines() if f.strip()]
        
        # Combine committed and uncommitted files
        all_files = set(committed_files)
        if args.include_uncommitted and has_uncommitted:
            all_files.update(uncommitted_files)
        
        all_files = sorted(list(all_files))
        
        if not all_files:
            print("No files to deploy.")
            return 0
        
        print(f"  {len(all_files)} file(s) to deploy:")
        print(f"    - {len(committed_files)} from commits")
        if args.include_uncommitted and has_uncommitted:
            print(f"    - {len(uncommitted_files)} uncommitted")
        for f in all_files[:20]:
            print(f"    - {f}")
        if len(all_files) > 20:
            print(f"    ... and {len(all_files) - 20} more")

        # 2. Archive files
        print("\n" + "=" * 60)
        print("[2/5] Creating archive ...")
        tar_path = os.path.join(tempfile.gettempdir(), "ms_deploy.tar.gz")
        temp_files.append(tar_path)
        
        if args.include_uncommitted and has_uncommitted:
            # Create archive with both committed and uncommitted files
            success, error = _create_archive_with_uncommitted(
                args.local_repo, committed_files, uncommitted_files, tar_path)
            if not success:
                print(f"FATAL: Failed to create archive: {error}", file=sys.stderr)
                return 2
        else:
            # Use git archive for committed files only
            r = subprocess.run(
                ["git", "archive", "--format=tar.gz", "-o", tar_path,
                 args.source, "--"] + committed_files,
                capture_output=True, text=True, cwd=args.local_repo,
            )
            if r.returncode != 0:
                print(f"FATAL: git archive failed: {r.stderr}", file=sys.stderr)
                return 2
        
        size_kb = os.path.getsize(tar_path) / 1024
        print(f"  Archive: {tar_path} ({size_kb:.1f} KB)")

        # 3. Upload via SCP
        print("\n" + "=" * 60)
        print("[3/5] Uploading to server ...")
        scp_cmd, scp_env, askpass = _build_scp_cmd(
            auth, tar_path, f"{target}:/tmp/ms_deploy.tar.gz")
        if askpass:
            temp_files.append(askpass)
        r = _run(scp_cmd, scp_env, "SCP upload", timeout=120)
        if r.returncode != 0:
            print(f"FATAL: SCP failed: {r.stderr}", file=sys.stderr)
            return 2
        print("  Upload OK")

        # 4. Extract on server
        print("\n" + "=" * 60)
        print("[4/5] Extracting on server ...")
        extract_remote = (
            f"cd {remote_dir} && "
            f"tar xzf /tmp/ms_deploy.tar.gz && "
            f"rm -f /tmp/ms_deploy.tar.gz && "
            f"echo __EXTRACT_OK__"
        )
        ssh_cmd, ssh_env, askpass = _build_ssh_cmd(
            "ssh", auth, target, extract_remote)
        if askpass:
            temp_files.append(askpass)
        r = _run(ssh_cmd, ssh_env, "SSH extract", timeout=60)
        if r.returncode != 0 or "__EXTRACT_OK__" not in r.stdout:
            print(f"FATAL: extract failed: {r.stderr}", file=sys.stderr)
            return 2
        print("  Extract OK")

        # 5. Build
        print("\n" + "=" * 60)
        print(f"[5/5] Building: {build_cmd}")
        print("  (this may take a while ...)")
        ssh_cmd, ssh_env, askpass = _build_ssh_cmd(
            "ssh", auth, target, f"cd {remote_dir} && {build_cmd} 2>&1")
        if askpass:
            temp_files.append(askpass)
        r = _run(ssh_cmd, ssh_env, "SSH build", timeout=7200)

        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(r.stdout or "")
            if r.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(r.stderr)
        print(f"  Build log written to: {args.log_file}")

        if r.returncode == 0:
            print("\n" + "=" * 60)
            print("BUILD SUCCEEDED")
            return 0

        print("\n" + "=" * 60)
        print(f"BUILD FAILED (exit code {r.returncode})")
        lines = (r.stdout or "").splitlines()
        tail = lines[-40:] if len(lines) > 40 else lines
        print("\n--- last 40 lines ---")
        print("\n".join(tail))
        return 1

    finally:
        for path in temp_files:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
