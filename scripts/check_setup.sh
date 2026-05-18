#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODE="${1:-gpu}"
if [[ "$MODE" != "gpu" && "$MODE" != "npu" ]]; then
  echo "usage: bash scripts/check_setup.sh [gpu|npu]" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

echo "[check] Python imports ($MODE)"
CHECK_SETUP_MODE="$MODE" python - <<'PY'
import importlib.metadata as metadata
import os

from fusion_bench.method import AlgorithmFactory
from fusion_bench.modelpool import ModelPoolFactory
from fusion_bench.taskpool import TaskPoolFactory
from fusion_bench.tasks.qwen_eval import QwenEvaluationTask

assert "evogm" in AlgorithmFactory.available_algorithms()
assert "evogm_npu" in AlgorithmFactory.available_algorithms()
assert "AutoModelForCausalLMPool" in ModelPoolFactory.available_modelpools()
assert "QwenEvaluationTaskPool" in TaskPoolFactory.available_taskpools()
print("  imports ok")

packages = [
    "torch",
    "torch-npu",
    "transformers",
    "peft",
    "accelerate",
    "lightning",
    "hydra-core",
    "omegaconf",
    "numpy",
    "scikit-learn",
    "scipy",
    "pandas",
    "wandb",
]
for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        if package == "torch-npu":
            if os.environ["CHECK_SETUP_MODE"] == "gpu":
                print("  torch-npu not installed (ok for GPU-only setup)")
                continue
            raise SystemExit("torch-npu is required for NPU setup")
        raise
    print(f"  {package}=={version}")

if os.environ["CHECK_SETUP_MODE"] == "npu":
    import torch
    import torch_npu  # noqa: F401

    if not hasattr(torch, "npu"):
        raise SystemExit("torch has no npu backend; check torch/torch-npu installation")
    if not torch.npu.is_available():
        raise SystemExit("torch.npu is not available; check Ascend driver/CANN/runtime")
    print(f"  npu device count: {torch.npu.device_count()}")
PY

echo "[check] Hydra dry-runs"
python -m fusion_bench.scripts.cli --config-name qwen25_evogm dry_run=true print_config=false log_dir=outputs/check_setup/gpu
python -m fusion_bench.scripts.cli --config-name qwen25_evogm_npu dry_run=true print_config=false log_dir=outputs/check_setup/npu

echo "[check] Task data"
python - <<'PY'
import json
from pathlib import Path

root = Path("data/swarm_eval")
required = [
    "mmlu",
    "mmlu_pro",
    "hellaswag",
    "knowledge_crosswords",
    "gsm8k",
    "nlgraph",
    "truthfulqa",
]

for name in required:
    path = root / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing dataset file: {path}")
    payload = json.loads(path.read_text())
    for split in ("dev", "test"):
        if split not in payload or not isinstance(payload[split], list) or not payload[split]:
            raise SystemExit(f"{path} is missing a non-empty {split} split")
print("  data ok")
PY

echo "[check] Model layout"
python - <<'PY'
import os
from pathlib import Path

model_root = Path(os.environ.get("EVOGM_MODEL_DIR", "models/qwen25-1.5b-lora-experts"))
expected_experts = [
    "tulu_code_alpaca",
    "tulu_cot",
    "tulu_flan_v2",
    "tulu_gpt4_alpaca",
    "tulu_lima",
    "tulu_oasst1",
    "tulu_open_orca",
    "tulu_science",
    "tulu_sharegpt",
    "tulu_wizardlm",
]

if not model_root.exists():
    print(f"  model root not found yet: {model_root}")
    print("  download TaoJiangCN/qwen2.5-1.5b-tulu-v2-lora-experts or set EVOGM_MODEL_DIR")
else:
    missing = []
    if not (model_root / "base" / "config.json").exists():
        missing.append("base/config.json")
    for expert in expected_experts:
        if not (model_root / "experts" / expert / "adapter_config.json").exists():
            missing.append(f"experts/{expert}/adapter_config.json")
    if missing:
        print("  model root exists, but expected files are missing:")
        for item in missing:
            print(f"    - {item}")
        raise SystemExit(1)
    print(f"  model layout ok: {model_root}")
PY

echo "[check] Setup check complete"
