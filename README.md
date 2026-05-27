
<h1 align="center">
🌟 EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization 🌟
</h1>

**Accepted at ICML 2026**

<div align="center">
  <img src="assets/evogm-overview.png" alt="EvoGM Overview" width="100%">
</div>

*EvoGM learns generative proposals for LoRA expert merging coefficients, using winner-loser search history to guide evolutionary model merging.*

## Overview

EvoGM addresses evolutionary model merging as a learnable search problem. Instead of relying on hand-crafted mutation or crossover operators, it treats validation performance as feedback and learns where high-quality merging coefficients are likely to lie.

The core idea is to build winner-loser pairs from historical search trajectories and train a dual-generator to transform weak coefficient candidates into stronger ones while preserving diversity through cycle consistency. The learned generator is then embedded back into an evolutionary loop, so each round can sample better candidates from the observed performance landscape rather than searching blindly.

For LLM merging, these candidates are LoRA expert merging coefficients. EvoGM evaluates them on downstream tasks, selects elite merges, and periodically refreshes the expert basis, allowing both the search strategy and the merged model pool to improve together.

## Release Contents

This repository is the clean release package for EvoGM, focused on Qwen2.5-1.5B LoRA expert merging. It contains the main EvoGM code path, GPU and Huawei Ascend NPU entrypoints, and the 8-task Qwen2.5 experiment setup.

The released experiment supports:

- Multi-task EvoGM over all 8 tasks.
- Single-task EvoGM over each task independently.
- Optional single-task subsets, for example `method.target_tasks=[gsm8k]`.

The included tasks are `mmlu`, `mmlu_pro`, `hellaswag`, `knowledge_crosswords`, `gsm8k`, `nlgraph`, `truthfulqa`, and `mmlu_abstain`. `mmlu_abstain` reuses the bundled `mmlu.json` file with the AbstainQA evaluator.

## Environment

Use Python 3.10. A fresh environment is recommended. The requirement files are pinned to the core packages used by the released experiments rather than a minimal import-only set.

GPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
pip install -e .
```

NPU:

```bash
conda create -n evogm_npu python=3.10 -y
conda activate evogm_npu
pip install -U pip
# Install torch and torch_npu for your CANN version first, then:
pip install -r requirements-npu.txt
pip install -e .
```

NPU runs require a working Ascend driver, CANN runtime, `npu-smi`, and a `torch_npu` build matching your PyTorch/CANN stack.

The NPU experiments were checked against the project environment named `evogm_npu` on Ascend machines with CANN 8.1.RC2, PyTorch 2.5.1, and `torch-npu` 2.5.1. If your cluster uses another CANN release, install the matching PyTorch/`torch_npu` pair first and then install the pinned Python packages from `requirements-npu.txt`. Do not let a generic PyPI torch wheel replace the Ascend-compatible build.

## Model Weights

Model weights are not included in this repository. Download the Qwen2.5-1.5B base model plus the 10 released Tulu v2 LoRA expert adapters from:

```text
https://huggingface.co/TaoJiangCN/qwen2.5-1.5b-tulu-v2-lora-experts
```

With `huggingface-hub` installed, one direct way to fetch the weights is:

```bash
huggingface-cli download TaoJiangCN/qwen2.5-1.5b-tulu-v2-lora-experts \
  --local-dir models/qwen25-1.5b-lora-experts
```

The downloaded files should be arranged like this:

```text
models/qwen25-1.5b-lora-experts/
  base/
    config.json
    model.safetensors or model.safetensors.index.json
    tokenizer files...
  experts/
    tulu_code_alpaca/adapter_config.json
    tulu_cot/adapter_config.json
    tulu_flan_v2/adapter_config.json
    tulu_gpt4_alpaca/adapter_config.json
    tulu_lima/adapter_config.json
    tulu_oasst1/adapter_config.json
    tulu_open_orca/adapter_config.json
    tulu_science/adapter_config.json
    tulu_sharegpt/adapter_config.json
    tulu_wizardlm/adapter_config.json
```

The default configs read from `models/qwen25-1.5b-lora-experts`. You can override the location:

```bash
export EVOGM_MODEL_DIR=/path/to/qwen25-1.5b-lora-experts
```

You can also override the dataset directory:

```bash
export EVOGM_DATA_DIR=/path/to/swarm_eval
```

## Setup Check

Run the setup check before launching experiments:

```bash
bash scripts/check_setup.sh
```

For NPU machines, use the stricter NPU check:

```bash
bash scripts/check_setup.sh npu
```

This validates imports, key package versions, Hydra config composition, bundled task data, and prints model layout guidance. In NPU mode it also verifies that `torch_npu` is installed and `torch.npu` is available. Missing model weights are reported clearly because weights are expected to be downloaded separately.

## Example

After model weights are in place, run a minimal smoke test. It creates a temporary one-example dataset from the bundled JSON files and uses tiny EvoGM search settings.

GPU:

```bash
bash scripts/smoke_test.sh gpu
```

NPU:

```bash
bash scripts/smoke_test.sh npu
```

## Full Experiments

GPU multi-task:

```bash
bash scripts/run_gpu_multi.sh
```

GPU single-task over all 8 task entries:

```bash
bash scripts/run_gpu_single.sh
```

GPU single-task for one task:

```bash
bash scripts/run_gpu_single.sh 'method.target_tasks=[gsm8k]'
```

NPU multi-task:

```bash
bash scripts/run_npu_multi.sh
```

NPU single-task:

```bash
bash scripts/run_npu_single.sh
```

NPU single-task for one task:

```bash
bash scripts/run_npu_single.sh 'method.target_tasks=[gsm8k]'
```

You can override device visibility in the usual way:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_gpu_multi.sh
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_npu_multi.sh
```

## Relationship to EvoGO

EvoGM is inspired by [EvoGO](https://github.com/EMI-Group/evogo), a closely related upstream line on evolutionary generative optimization in [EvoX](https://github.com/EMI-Group/evox). EvoGO studies the general black-box optimization setting, while EvoGM adapts the same spirit of learned evolutionary proposal generation to LLM expert merging.