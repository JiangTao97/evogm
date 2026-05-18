#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

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
    ASCEND_RT_VISIBLE_DEVICES="$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(",".join(str(i) for i in range(torch.npu.device_count())))
PY
)"
fi
export ASCEND_RT_VISIBLE_DEVICES

IFS=',' read -ra NPU_ARRAY <<< "$ASCEND_RT_VISIBLE_DEVICES"
NUM_NPUS="${#NPU_ARRAY[@]}"
echo "Using ${NUM_NPUS} NPU(s): ${ASCEND_RT_VISIBLE_DEVICES}"

python -m fusion_bench.scripts.cli --config-name qwen25_evogm_npu method.evaluation_mode=single_task method.num_gpus="${NUM_NPUS}" "$@"
