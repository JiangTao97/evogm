# EvoGM Qwen2.5-1.5B

This repository is a clean release package for EvoGM, focused on Qwen2.5-1.5B LoRA expert merging.
It contains only the main EvoGM code path, GPU and Huawei Ascend NPU entrypoints, and the 8-task Qwen2.5 experiment setup.

EvoGM learns a generative search process over LoRA expert merging coefficients. The released experiment supports:

- Multi-task EvoGM over all 8 tasks.
- Single-task EvoGM over each task independently.
- Optional single-task subsets, for example `method.target_tasks=[gsm8k]`.

The included tasks are `mmlu`, `mmlu_pro`, `hellaswag`, `knowledge_crosswords`, `gsm8k`, `nlgraph`, `truthfulqa`, and `mmlu_abstain`. `mmlu_abstain` reuses the bundled `mmlu.json` file with the AbstainQA evaluator.

## Repository Layout

```text
fusion_bench/          EvoGM, Qwen model/task pools, and CLI
config/                Hydra configs for GPU and NPU experiments
data/swarm_eval/       8-task experiment JSON files
scripts/               Public run and setup scripts
outputs/               Runtime outputs, ignored by git
models/                Local model weights, ignored by git
```

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

### Dependency Scope

The requirement files are intentionally scoped to this clean EvoGM Qwen2.5 release package. The internal `evogm_npu` environment used on research servers contains many extra packages for unrelated baselines, broader evaluation suites, server tooling, and pruned experiments, such as `evalscope`, `lm_eval`, `opencompass`, `vllm`, and vision/evaluation utilities. Those are not required by the released 8-task EvoGM code path and are not included in the default requirements.

For the released package, the direct Python dependency surface is covered by `torch`, `transformers`, `peft`, `safetensors`, `accelerate`, `lightning`, `hydra-core`, `omegaconf`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `tqdm`, `rich`, `psutil`, `wandb`, `tensorboard`, `huggingface-hub`, `tokenizers`, `sentencepiece`, `protobuf`, `PyYAML`, and `typing_extensions`. NPU additionally requires a matching Ascend `torch_npu` installation outside the portable pip requirements.

## Smoke Test

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
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_gpu_multi.sh
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_npu_multi.sh
```

## Outputs

By default, outputs are written under `outputs/`.

Important files include:

- `search_process/run_context.json`
- `search_process/best_solution.json`
- `search_process/all_results.json`
- `search_process/summary_*.csv`
- `search_process/memory_trace.jsonl`

Merged model weights are not saved unless `merged_model_save_path` is set.

## Useful Overrides

Short debug run:

```bash
bash scripts/run_gpu_multi.sh method.n_rounds=1 method.population_size=4 method.max_iter=1 method.generator_epochs=5 method.num_gpus=1 taskpool.batch_size=1
```

Disable config printing:

```bash
bash scripts/run_gpu_multi.sh print_config=false
```

Save a final report:

```bash
bash scripts/run_gpu_multi.sh save_report=outputs/final_report.json
```
