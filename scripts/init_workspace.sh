#!/bin/bash
# Create local workspace folders. No git submodule behavior.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$REPO_DIR/workspace"

mkdir -p "$WORKSPACE_DIR"/mindspore
mkdir -p "$WORKSPACE_DIR"/op-plugin
mkdir -p "$WORKSPACE_DIR"/aclnn-dashboard

echo "Workspace directories are ready:"
echo "  $WORKSPACE_DIR/mindspore"
echo "  $WORKSPACE_DIR/op-plugin"
echo "  $WORKSPACE_DIR/aclnn-dashboard"
echo ""
echo "Clone repositories manually, for example:"
echo "  git clone https://gitcode.com/mindspore/mindspore.git $WORKSPACE_DIR/mindspore"
echo "  git clone https://gitcode.com/Ascend/op-plugin.git $WORKSPACE_DIR/op-plugin"
echo "  git clone https://github.com/Fzilan/aclnn-dashboard.git $WORKSPACE_DIR/aclnn-dashboard"
