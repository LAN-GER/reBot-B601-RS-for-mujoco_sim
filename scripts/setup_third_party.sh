#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDK_DIR="${REPO_ROOT}/third_party/reBotArm_control_py"
SDK_URL="https://github.com/vectorBH6/reBotArm_control_py.git"

if [ -d "${SDK_DIR}/.git" ]; then
    echo "[setup_third_party] SDK already exists at ${SDK_DIR}, pulling latest..."
    cd "${SDK_DIR}"
    git pull
else
    echo "[setup_third_party] Cloning SDK from ${SDK_URL} ..."
    mkdir -p "${REPO_ROOT}/third_party"
    git clone "${SDK_URL}" "${SDK_DIR}"
fi

echo "[setup_third_party] SDK ready at ${SDK_DIR}"
