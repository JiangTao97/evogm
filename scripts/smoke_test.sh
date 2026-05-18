#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/smoke_test.sh gpu|npu [hydra overrides...]"
    exit 2
fi

BACKEND="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

MODEL_ROOT="${EVOGM_MODEL_DIR:-models/qwen25-1.5b-lora-experts}"
if [[ ! -f "$MODEL_ROOT/base/config.json" ]]; then
    echo "ERROR: model weights not found under $MODEL_ROOT"
    echo "Download TaoJiangCN/qwen2.5-1.5b-tulu-v2-lora-experts or set EVOGM_MODEL_DIR."
    exit 2
fi

SMOKE_DATA="$(mktemp -d "${TMPDIR:-/tmp}/evogm-smoke-data.XXXXXX")"
trap 'rm -rf "$SMOKE_DATA"' EXIT

python - "$PROJECT_ROOT/data/swarm_eval" "$SMOKE_DATA" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

for path in source.glob("*.json"):
    payload = json.loads(path.read_text())
    slim = {}
    for key, value in payload.items():
        if isinstance(value, list):
            slim[key] = value[:1]
        else:
            slim[key] = value
    (dest / path.name).write_text(json.dumps(slim, ensure_ascii=False, indent=2))
PY

COMMON_OVERRIDES=(
    method.n_rounds=1
    method.top_k=3
    method.population_size=3
    method.max_iter=1
    method.generator_epochs=1
    method.num_gpus=1
    method.use_wandb=false
    taskpool.batch_size=1
    taskpool.data_dir="$SMOKE_DATA"
    print_config=false
    save_report=outputs/smoke_${BACKEND}/report.json
    log_dir=outputs/smoke_${BACKEND}
)

case "$BACKEND" in
    gpu)
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            export CUDA_VISIBLE_DEVICES=0
        fi
        python -m fusion_bench.scripts.cli --config-name qwen25_evogm "${COMMON_OVERRIDES[@]}" "$@"
        ;;
    npu)
        if ! command -v npu-smi >/dev/null 2>&1; then
            echo "ERROR: npu-smi not found. Install Ascend driver/CANN first."
            exit 2
        fi
        python - <<'PY'
import torch
import torch_npu  # noqa: F401
assert torch.npu.is_available(), "NPU is not available"
print(f"NPU count: {torch.npu.device_count()}")
PY
        if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
            export ASCEND_RT_VISIBLE_DEVICES=0
        fi
        python -m fusion_bench.scripts.cli --config-name qwen25_evogm_npu "${COMMON_OVERRIDES[@]}" "$@"
        ;;
    *)
        echo "Usage: bash scripts/smoke_test.sh gpu|npu [hydra overrides...]"
        exit 2
        ;;
esac
