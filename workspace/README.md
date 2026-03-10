# workspace (local-only)

This directory is intentionally not managed by git submodules.

Clone dependencies locally as needed:

```bash
git clone https://gitcode.com/mindspore/mindspore.git workspace/mindspore
git clone https://gitcode.com/Ascend/op-plugin.git workspace/op-plugin
git clone https://github.com/Fzilan/aclnn-dashboard.git workspace/aclnn-dashboard
```

Anything under `workspace/*` is ignored by git in this repository.
