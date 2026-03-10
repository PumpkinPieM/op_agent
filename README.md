# Op-Agent

## Workspace policy (minimal)

- `workspace/` is local-only.
- This repository does **not** use git submodules for `workspace/*`.
- Users clone dependencies locally by themselves.

## Quick setup

```bash
./scripts/init_workspace.sh

git clone https://gitcode.com/mindspore/mindspore.git workspace/mindspore
git clone https://gitcode.com/Ascend/op-plugin.git workspace/op-plugin
```

## Optional local symlink setup (your machine)

```bash
ln -sfn /localpath/to/mindspore workspace/mindspore
ln -sfn /localpath/to/op-plugin workspace/op-plugin
```
