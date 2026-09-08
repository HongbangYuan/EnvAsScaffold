#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install -e './agent_system/environments/env_package/sciworld/ScienceWorld'
"$PYTHON_BIN" -m pip install -e './agent_system/environments/env_package/bfcl/berkeley_function_call_leaderboard'
"$PYTHON_BIN" -m pip install -e '.[vllm]'
# The FSDP worker selects FlashAttention 2. Install after torch/vLLM.
"$PYTHON_BIN" -m pip install flash-attn --no-build-isolation
