"""
EvoGM Worker (Huawei Ascend NPU).

Key difference from CUDA version:
- Uses torch_npu and NPU devices instead of CUDA
- ASCEND_RT_VISIBLE_DEVICES for device isolation
- torch.npu.empty_cache() for memory cleanup
"""

import os
import logging
import json
import shutil
import subprocess
import threading
import queue

import torch
import torch_npu  # noqa: F401  Huawei Ascend NPU support
import torch.multiprocessing as mp
import numpy as np
from typing import Dict, List
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def _resolve_physical_npu_id(local_npu_id: int) -> str:
    """Map a logical visible-device index back to the physical NPU ID for subprocesses."""
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible:
        return str(local_npu_id)

    visible_ids = [x.strip() for x in visible.split(",") if x.strip()]
    if 0 <= local_npu_id < len(visible_ids):
        return visible_ids[local_npu_id]
    return str(local_npu_id)


def _calculate_score(report: dict) -> float:
    """Calculate overall score from evaluation report (lower is better for minimization)."""
    scores = []
    for task_name, result in report.items():
        if isinstance(result, dict):
            if "accuracy" in result:
                scores.append(result["accuracy"])
            elif "spearman_rho" in result:
                scores.append(result["spearman_rho"])
            elif "effective_reliability" in result:
                scores.append(result["effective_reliability"])
    return -np.mean(scores) if scores else float('inf')


def _gpu_evaluation_worker_v3(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    shared_data_queue: mp.Queue,
    pretrained_path: str,
    taskpool_config: dict,
    model_names: List[str],
    dtype_str: str,
    model_type: str = "causal_lm",
    evaluation_backend: str = "taskpool",
    script_tasks: str = None,
    script_num_samples: int = 50,
    output_dir: str = None,
):
    """
    EvoGM persistent worker with shared memory (NPU).

    Unlike V2 which loads pretrained_sd + task_vectors from disk (144GB per worker),
    V3 receives them via shared memory from the main process (zero-copy, ~0 extra RAM).

    Signals:
    - None: Shutdown
    - "RELOAD": Get new shared task vectors from shared_data_queue
    - (idx, coefficients, split, task_name): Normal evaluation
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker NPU {gpu_id}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    log = logging.getLogger(__name__)

    if model_type == "seq2seq_lm":
        from transformers import AutoModelForSeq2SeqLM as ModelClass
    else:
        from transformers import AutoModelForCausalLM as ModelClass

    device = torch.device(f"npu:{gpu_id}")
    device_map = str(device)
    dtype = getattr(torch, dtype_str) if dtype_str else torch.float32

    try:
        # =============================
        # Load NPU model (unavoidable disk I/O)
        # =============================
        torch.npu.set_device(device_map)
        log.info(
            "Binding worker to logical %s (visible devices: %s)",
            device,
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<all>"),
        )
        pretrained_model = ModelClass.from_pretrained(
            pretrained_path,
            torch_dtype=dtype,
            device_map=device_map,
        )

        # =============================
        # Receive shared memory data from main process (ZERO-COPY!)
        # No torch.load, no disk I/O, no RAM duplication
        # =============================
        log.info("Waiting for shared memory data from main process...")
        init_data = shared_data_queue.get(timeout=300)
        pretrained_sd = init_data["pretrained_sd"]
        task_vectors = init_data["task_vectors"]
        log.info(f"Received shared data: pretrained_sd ({len(pretrained_sd)} keys), "
                 f"task_vectors ({len(task_vectors)} experts)")

        # Create taskpool
        taskpool = None
        if evaluation_backend == "taskpool":
            from fusion_bench.taskpool import TaskPoolFactory
            taskpool = TaskPoolFactory.create_taskpool(OmegaConf.create(taskpool_config))

        result_queue.put(("READY", gpu_id))
        log.info("Worker ready")

        # =============================
        # Main event loop
        # =============================
        while True:
            idx = None
            try:
                task = task_queue.get(timeout=60)

                # --- Shutdown ---
                if task is None:
                    log.info("Received shutdown signal")
                    break

                # --- Reload task vectors via shared memory ---
                if task == "RELOAD":
                    log.info("Reloading task vectors from shared memory...")
                    reload_data = shared_data_queue.get(timeout=120)
                    task_vectors = reload_data["task_vectors"]
                    result_queue.put(("RELOAD_DONE", gpu_id))
                    log.info(f"Task vectors reloaded ({len(task_vectors)} experts)")
                    continue

                # --- Normal evaluation ---
                idx, coefficients, split, task_name = task
                log.info(f"Processing task {idx}")

                # Compute weighted sum of task vectors
                total_task_vector = None
                for model_name in model_names:
                    coef = coefficients.get(model_name, 0.0)
                    if coef == 0.0:
                        continue
                    if model_name not in task_vectors:
                        continue

                    tv = task_vectors[model_name]
                    scaled_tv = {k: v * coef for k, v in tv.items()} if coef != 1.0 else tv

                    if total_task_vector is None:
                        total_task_vector = {k: v.clone() for k, v in scaled_tv.items()}
                    else:
                        for k, v in scaled_tv.items():
                            if k in total_task_vector:
                                total_task_vector[k] += v
                            else:
                                total_task_vector[k] = v.clone()

                # Merge: pretrained + total_task_vector (on CPU, then load to NPU)
                if total_task_vector is not None:
                    merged_sd = {}
                    for k, v in pretrained_sd.items():
                        if k in total_task_vector:
                            merged_sd[k] = v + total_task_vector[k]  # CPU arithmetic
                        else:
                            merged_sd[k] = v
                    # load_state_dict handles CPU→NPU transfer in-place
                    pretrained_model.load_state_dict(merged_sd)
                    del merged_sd, total_task_vector
                    torch.npu.empty_cache()

                merged_model = pretrained_model

                # Evaluate
                report = {}
                if evaluation_backend == "script":
                    report = _script_evaluate(
                        merged_model, gpu_id, idx,
                        pretrained_path, output_dir,
                        script_tasks, script_num_samples, log
                    )
                else:
                    with torch.no_grad():
                        if task_name is not None:
                            task_obj = taskpool.load_task_with_split(task_name, split)
                            report = {task_name: task_obj.evaluate(merged_model)}
                        elif hasattr(taskpool, 'evaluate_with_split'):
                            report = taskpool.evaluate_with_split(merged_model, split)
                        else:
                            report = taskpool.evaluate(merged_model)

                score = _calculate_score(report)
                result_queue.put((idx, score, report, None))
                log.info(f"Task {idx} complete, score: {-score:.4f}")

                # Restore original pretrained weights (CPU→NPU in-place)
                pretrained_model.load_state_dict(pretrained_sd)

            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Worker error: {e}")
                import traceback
                log.error(traceback.format_exc())
                if idx is not None:
                    result_queue.put((idx, float('inf'), {}, str(e)))

    except Exception as e:
        log.error(f"CRITICAL WORKER FAILURE: {e}")
        import traceback
        traceback.print_exc()


def _script_evaluate(
    merged_model, gpu_id, idx,
    pretrained_path, output_dir,
    script_tasks, script_num_samples, log
):
    """Script-based evaluation (for T5)."""
    eval_id = f"worker_{gpu_id}_{idx}"
    temp_model_path = os.path.abspath(os.path.join(output_dir, f"temp_eval_{eval_id}"))
    os.makedirs(temp_model_path, exist_ok=True)
    report = {}

    try:
        merged_model.save_pretrained(temp_model_path)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            tokenizer.save_pretrained(temp_model_path)
        except Exception as e:
            log.warning(f"Could not save tokenizer: {e}")

        project_root = os.getcwd()
        script_path = os.path.join(project_root, "scripts/inference/distributed_inference.py")

        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = _resolve_physical_npu_id(gpu_id)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(29500 + gpu_id * 10 + (idx % 10))
        env["WORLD_SIZE"] = "1"
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

        tasks = script_tasks.split(",")
        for t_name in tasks:
            port = 29500 + gpu_id * 10 + (idx % 10)
            cmd = [
                "accelerate", "launch",
                "--num_processes=1",
                f"--main_process_port={port}",
                script_path,
                "--base_model", temp_model_path,
                "--task_name", t_name,
                "--num_samples", str(script_num_samples),
                "--model_type", "t5",
            ]
            result = subprocess.run(cmd, env=env, cwd=temp_model_path,
                                    capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

            res_file = os.path.join(temp_model_path, f"infer_res_{t_name}.jsonl")
            if os.path.exists(res_file):
                all_data = []
                with open(res_file, 'r') as f:
                    for line in f:
                        all_data.append(json.loads(line))

                if t_name == "stsb":
                    from scipy.stats import spearmanr
                    preds, labels = [], []
                    for d in all_data:
                        try:
                            pred = float(d['output'].strip())
                        except:
                            pred = 0.0
                        preds.append(pred)
                        labels.append(float(d['label']))
                    score = spearmanr(preds, labels)[0] if len(preds) > 1 else 0.0
                    if score != score:
                        score = 0.0
                    report[t_name] = {"spearman_rho": score}
                else:
                    correct = sum(1 for d in all_data
                                  if d['output'].strip().lower() == d.get('label_text', '').strip().lower())
                    report[t_name] = {"accuracy": correct / len(all_data) if all_data else 0.0}
            else:
                report[t_name] = {"accuracy": 0.0}

    except Exception as e:
        log.error(f"Script evaluation failed: {e}")
    finally:
        if os.path.exists(temp_model_path):
            try:
                shutil.rmtree(temp_model_path)
            except:
                pass
    return report
