#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK="${1:-sciworld}"
if (( $# > 0 )); then shift; fi
exec bash "$REPO_ROOT/scripts/train.sh" "$BENCHMARK" grpo standard "$@" \
  trainer.val_only=True trainer.val_before_train=True trainer.resume_mode=disable
