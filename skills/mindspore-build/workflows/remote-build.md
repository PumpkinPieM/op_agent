# Remote Build (deploy to server and compile)

> **Status: NOT YET VERIFIED.** The `--include-uncommitted` feature has not been tested
> end-to-end. Basic remote build flow is designed but awaits validation.

Self-contained guide. No other files needed.

## When to Use

- You develop on a local machine (Windows/Mac/Linux) but need to compile on a remote
  Ascend/GPU server
- Another skill (e.g. aclnn) needs to compile after code changes
- You want automated deploy -> build -> error-fix -> retry cycle

## Prerequisites

1. **Code changes**: the script can deploy either:
   - **Committed changes only** (default): extracts files from the latest git commit
   - **Include uncommitted**: use `--include-uncommitted` flag to include staged + unstaged changes
2. **`servers.json` exists** in the skill root directory with your server info.
   Copy `servers.example.json` to `servers.json` and fill in real values.
3. **SSH access** to the target server (the script uses `scp` + `ssh`)
4. **Remote repo already cloned**: the `remote_dir` in `servers.json` must point to
   an existing MindSpore source tree on the server

## Authentication

Three methods are supported, in priority order:

| Priority | Method | `auth_method` | How it works |
|----------|--------|---------------|--------------|
| 1 (recommended) | SSH Key | `"ssh_key"` | Uses `~/.ssh/id_rsa` (or agent). No password needed. |
| 2 | Environment variable | `"env"` (default) | Reads password from `MS_SSH_PASS_<DEVICE>` env var. Falls back to JSON `password` field if env var is not set. |
| 3 (not recommended) | JSON plaintext | `"password"` | Reads password directly from `servers.json`. Prints a warning on every use. |

For password-based methods, the script prefers `sshpass` if installed (cleaner, no temp files).
If `sshpass` is not available, it falls back to `SSH_ASKPASS` with a temporary helper script
that is deleted immediately after use.

### Option 1: SSH Key (recommended)

```bash
# Generate key pair (if you don't have one)
ssh-keygen -t ed25519

# Copy public key to server
ssh-copy-id user@192.168.x.xx
```

`servers.json`:
```json
{
  "servers": {
    "910b": {
      "host": "192.168.x.xx",
      "user": "your_user",
      "auth_method": "ssh_key",
      "remote_dir": "/home/you/mindspore",
      "build_cmd": "bash build.sh -e ascend -V 910b -j128",
      "description": "Ascend 910B dev server"
    }
  },
  "default": "910b"
}
```

### Option 2: Environment Variable

Set the password in your shell profile or before running the script:

```bash
# Bash / Zsh
export MS_SSH_PASS_910B="your_password"

# PowerShell
$env:MS_SSH_PASS_910B = "your_password"
```

The env var name is `MS_SSH_PASS_` + device key in uppercase (e.g. `910B`, `910A`, `GPU`).

`servers.json`:
```json
{
  "servers": {
    "910b": {
      "host": "192.168.x.xx",
      "user": "your_user",
      "auth_method": "env",
      "remote_dir": "/home/you/mindspore",
      "build_cmd": "bash build.sh -e ascend -V 910b -j128",
      "description": "Ascend 910B dev server"
    }
  },
  "default": "910b"
}
```

### Option 3: JSON Plaintext (not recommended)

```json
{
  "servers": {
    "910b": {
      "host": "192.168.x.xx",
      "user": "your_user",
      "auth_method": "password",
      "password": "your_password",
      "remote_dir": "/home/you/mindspore",
      "build_cmd": "bash build.sh -e ascend -V 910b -j128",
      "description": "Ascend 910B dev server"
    }
  },
  "default": "910b"
}
```

> **Security**: `servers.json` may contain passwords. Add it to `.gitignore`.
> The example file `servers.example.json` uses placeholder values and is safe to commit.

### Installing sshpass (optional, improves password auth)

```bash
# Ubuntu / Debian
sudo apt install sshpass

# CentOS / RHEL
sudo yum install sshpass

# macOS
brew install hudochenkov/sshpass/sshpass
```

## Usage

### Basic (uses default server from servers.json)

```bash
python <skill_dir>/scripts/remote_deploy_build.py \
    --local-repo /path/to/mindspore \
    --log-file /path/to/build_output.log
```

### Specify a different server

```bash
python <skill_dir>/scripts/remote_deploy_build.py \
    --device 910a \
    --local-repo /path/to/mindspore \
    --log-file /path/to/build_output.log
```

### Override build command

```bash
python <skill_dir>/scripts/remote_deploy_build.py \
    --local-repo /path/to/mindspore \
    --log-file /path/to/build_output.log \
    --build-cmd "bash build.sh -e ascend -V 910b -j64 -i"
```

### Parameters

| Param | Required | Description |
|-------|----------|-------------|
| `--local-repo` | Yes | Path to local MindSpore git repo |
| `--log-file` | Yes | Where to write build output log |
| `--device` | No | Server key in servers.json (default: `"default"` field) |
| `--source` | No | Git ref to extract changes from (default: `HEAD`) |
| `--remote-dir` | No | Override remote directory from servers.json |
| `--build-cmd` | No | Override build command from servers.json |
| `--include-uncommitted` | No | Include staged + unstaged changes in deployment |
| `--check-uncommitted` | No | Only check for uncommitted changes, exit 3 if found |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Build succeeded |
| 1 | Build failed (check `--log-file` for errors) |
| 2 | Infrastructure failure (network, SSH, SCP, extract) |
| 3 | Uncommitted changes detected (use `--include-uncommitted` to include) |

## Uncommitted Changes

By default, only committed changes are deployed. To include uncommitted work:

```bash
python <skill_dir>/scripts/remote_deploy_build.py \
    --local-repo /path/to/mindspore \
    --log-file /path/to/build_output.log \
    --include-uncommitted
```

This deploys staged, unstaged, and untracked files alongside the latest commit's changes.

## What the Script Does

```
[1/5] git diff-tree  -> list changed files in latest commit
[2/5] git archive   -> pack only changed files into tar.gz
[3/5] scp           -> upload tar.gz to server /tmp/
[4/5] ssh           -> extract tar.gz into remote_dir
[5/5] ssh           -> run build_cmd, capture output to log file
```

Only changed files are transferred -- not the entire repo.

## AI Auto-Retry Workflow (when called by a skill)

When this script is invoked by an AI agent (e.g. aclnn skill Step 8):

1. Run the script. Check exit code.
2. If exit code = 0 -> success, continue to next step
3. If exit code = 1 -> read `build_output.log`, find errors, fix code, commit, retry
4. If exit code = 2 -> infrastructure issue, report to user, stop
5. **Max 3 retries**. After 3 failures, stop and show the last error to user.

Each retry cycle:
- Read log -> extract error file/line/message
- Fix the code with minimal changes
- `git add` + `git commit` the fix
- Re-run the script

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `servers.json not found` | Not created yet | Copy `servers.example.json` -> `servers.json` |
| `SCP failed` | Network/firewall/wrong credentials | Verify host, user, auth config; test `ssh user@host` manually |
| `extract failed` | remote_dir doesn't exist | Create the directory or clone the repo on server |
| Exit code 1, log shows compile errors | Code issue | Read log, fix code, commit, retry |
| Exit code 1, log shows missing deps | Server environment | SSH into server and install deps manually |
| `No files changed` | Nothing committed | `git add` + `git commit` your changes first |
| `MS_SSH_PASS_xxx is not set` | Env var not exported | Export the variable or switch to ssh_key |
| `sshpass: command not found` | sshpass not installed | Install it or rely on SSH_ASKPASS fallback (automatic) |
