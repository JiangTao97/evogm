"""
This is the CLI script that is executed when the user runs the `fusion-bench` command.
The script is responsible for parsing the command-line arguments, loading the configuration file, and running the fusion algorithm.
"""
#import debugpy
#try:
#    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#    debugpy.listen(("localhost", 9501))
#    print("Waiting for debugger attach")
#    debugpy.wait_for_client()
#except Exception as e:
#    pass

import importlib
import importlib.resources
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, Iterable, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from fusion_bench.method import load_algorithm_from_config
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.modelpool import load_modelpool_from_config
from fusion_bench.taskpool import load_taskpool_from_config
from fusion_bench.utils.rich_utils import print_config_tree

log = logging.getLogger(__name__)


def _is_npu_available() -> bool:
    """Safely check whether Ascend NPU runtime is available."""
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None:
        return False
    try:
        return bool(npu_mod.is_available())
    except Exception:
        return False


def _preferred_compute_device() -> torch.device:
    """
    Detect best available compute device with priority:
    1) CUDA
    2) NPU
    3) CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _is_npu_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def _get_default_config_path():
    for config_dir in ["fusion_bench_config", "config"]:
        config_path = os.path.join(
            importlib.import_module("fusion_bench").__path__[0], "..", config_dir
        )
        if os.path.exists(config_path) and os.path.isdir(config_path):
            return config_path
    raise FileNotFoundError("Default config path not found.")


def run_model_fusion(cfg: DictConfig):
    """
    Run the model fusion process based on the provided configuration.

    1. This function loads a model pool and an model fusion algorithm based on the configuration.
    2. It then uses the algorithm to fuse the models in the model pool into a single model.
    3. If a task pool is specified in the configuration, it loads the task pool and uses it to evaluate the merged model.
    """
    log.warning(
        "This function is deprecated. Use LightningProgram instead. This will be removed in future versions."
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _format_path(path: str, is_file: bool = False) -> str:
        if not isinstance(path, str):
            return path
        if "{timestamp}" in path:
            path = path.format(timestamp=timestamp)
        else:
            if is_file:
                dirname, filename = os.path.split(path)
                path = os.path.join(dirname, timestamp, filename)
            else:
                path = os.path.join(path, timestamp)
        return path

    modelpool = load_modelpool_from_config(cfg.modelpool)

    algorithm = load_algorithm_from_config(cfg.method)
    merged_model = algorithm.run(modelpool)

    # save the merged model
    already_saved = isinstance(merged_model, Dict) and merged_model.get("already_saved", False)
    already_evaluated = isinstance(merged_model, Dict) and merged_model.get("already_evaluated", False)

    if isinstance(merged_model, Dict):
        model_ref = merged_model.get("model")
        model_path = merged_model.get("model_path")
    else:
        model_ref = merged_model
        model_path = merged_model if isinstance(merged_model, str) else None

    if cfg.get("merged_model_save_path", None) is not None and not already_saved and model_ref is not None:
        save_path = _format_path(cfg.merged_model_save_path)
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        modelpool.save_model(
            model_ref, save_path, save_tokenizer=cfg.get("save_tokenizer", True)
        )

    if hasattr(cfg, "taskpool") and cfg.taskpool is not None:
        taskpool = load_taskpool_from_config(cfg.taskpool)
        if hasattr(modelpool, "_fabric") and hasattr(taskpool, "_fabric"):
            if taskpool._fabric is None:
                taskpool._fabric = modelpool._fabric
        modelpool.setup_taskpool(taskpool)
        if already_evaluated:
            report = merged_model.get("report", {"status": "already_evaluated", "model_path": model_path})
        else:
            if isinstance(model_ref, str):
                model_ref = modelpool.load_model(
                    OmegaConf.create({"name": "_merged_", "path": model_ref})
                )
            report = taskpool.evaluate(model_ref)
        if cfg.get("save_report", False):
            save_report_path = cfg.save_report
            if isinstance(save_report_path, bool):
                save_report_path = "report.json"
            save_report_path = _format_path(save_report_path, is_file=True)
            # save report (Dict) to a file
            # if the directory of `save_report` does not exists, create it
            if os.path.dirname(save_report_path):
                os.makedirs(os.path.dirname(save_report_path), exist_ok=True)
            json.dump(report, open(save_report_path, "w"))
    else:
        print("No task pool specified. Skipping evaluation.")


class LightningProgram(LightningFabricMixin):

    def __init__(self, config: DictConfig):
        self.config = config
        self._timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _format_path(self, path: str, is_file: bool = False) -> str:
        """
        Format a path string by replacing placeholders and ensuring it has a timestamp if requested.
        """
        if not isinstance(path, str):
            return path

        format_kwargs = {"timestamp": self._timestamp}
        if self.log_dir is not None:
            format_kwargs["log_dir"] = self.log_dir

        if "{log_dir}" in path or "{timestamp}" in path:
            path = path.format(**format_kwargs)
        else:
            # If no placeholders are present, append a timestamped folder as requested by the user.
            if is_file:
                dirname, filename = os.path.split(path)
                path = os.path.join(dirname, self._timestamp, filename)
            else:
                path = os.path.join(path, self._timestamp)
        return path

    def _load_and_setup(self, load_fn, *args, **kwargs):
        """
        Load an object using a provided loading function and setup its attributes.
        """
        obj = load_fn(*args, **kwargs)
        obj._program = self
        if hasattr(obj, "_fabric") and self.fabric is not None:
            obj._fabric = self.fabric
        return obj

    def save_merged_model(self, merged_model):
        if isinstance(merged_model, Dict) and merged_model.get("already_saved", False):
            model_path = merged_model.get("model_path") or merged_model.get("model")
            log.info(f"Merged model is already saved at {model_path}. Skipping saving.")
            return

        if isinstance(merged_model, str):
            log.info(f"Merged model is already saved at {merged_model}. Skipping saving.")
            return

        if isinstance(merged_model, Dict):
            merged_model = merged_model.get("model")

        if self.config.get("merged_model_save_path", None) is not None:
            save_path = self._format_path(self.config.merged_model_save_path)

            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.modelpool.save_model(
                merged_model, save_path, save_tokenizer=self.config.save_tokenizer
            )
        else:
            print("No save path specified for the merged model. Skipping saving.")

    def evaluate_merged_model(
        self, taskpool, merged_model: Union[nn.Module, Dict, Iterable]
    ):
        """
        Evaluates the merged model using the provided task pool.

        Depending on the type of the merged model, this function handles the evaluation differently:
        - If the merged model is an instance of `nn.Module`, it directly evaluates the model.
        - If the merged model is a dictionary, it extracts the model from the dictionary and evaluates it.
          The evaluation report is then updated with the remaining dictionary items.
        - If the merged model is an iterable, it recursively evaluates each model in the iterable.
        - Raises a `ValueError` if the merged model is of an invalid type.

        Args:
            taskpool: The task pool used for evaluating the merged model.
            merged_model: The merged model to be evaluated. It can be an instance of `nn.Module`, a dictionary, or an iterable.

        Returns:
            The evaluation report. The type of the report depends on the type of the merged model:
            - If the merged model is an instance of `nn.Module`, the report is a dictionary.
            - If the merged model is a dictionary, the report is a dictionary updated with the remaining dictionary items.
            - If the merged model is an iterable, the report is a list of evaluation reports.
        """
        if isinstance(merged_model, Dict) and merged_model.get("already_evaluated", False):
            model_path = merged_model.get("model_path") or merged_model.get("model")
            log.info(f"Merged model was already evaluated by the algorithm. Skipping CLI evaluation for {model_path}.")
            return merged_model.get(
                "report",
                {
                    "status": "already_evaluated",
                    "model_path": model_path,
                    "algorithm": merged_model.get("algorithm"),
                },
            )

        if isinstance(merged_model, str):
            log.info("Loading saved merged model from %s for evaluation.", merged_model)
            merged_model = self.modelpool.load_model(
                OmegaConf.create({"name": "_merged_", "path": merged_model})
            )

        if isinstance(merged_model, nn.Module):
            if isinstance(merged_model, nn.Module) and hasattr(self, "fabric") and self.fabric is not None:
                # Check if model is already on the correct device
                try:
                    param = next(merged_model.parameters())
                    model_device = param.device
                    fabric_device = self.fabric.device
                    preferred_device = _preferred_compute_device()

                    # If fabric resolves to CPU but accelerators are available,
                    # follow explicit device priority: CUDA -> NPU -> CPU.
                    if fabric_device.type == "cpu":
                        target_device = preferred_device
                    else:
                        target_device = fabric_device

                    if model_device != target_device:
                        log.info(
                            "Moving merged model to device: %s (model=%s, fabric=%s, preferred=%s)",
                            target_device,
                            model_device,
                            fabric_device,
                            preferred_device,
                        )
                        merged_model = merged_model.to(target_device)
                except StopIteration:
                    pass

            report = taskpool.evaluate(merged_model)
            print(report)
            return report
        elif isinstance(merged_model, Dict):
            model = merged_model.get("model")
            if model is None:
                raise ValueError(f"Invalid merged model metadata without 'model': {merged_model.keys()}")
            report: dict = taskpool.evaluate(model)
            report.update(
                {
                    k: v
                    for k, v in merged_model.items()
                    if k != "model"
                }
            )
            print(report)
            return report
        elif isinstance(merged_model, Iterable):
            return [
                self.evaluate_merged_model(taskpool, m)
                for m in tqdm(merged_model, desc="Evaluating models")
            ]
        else:
            raise ValueError(f"Invalid type for merged model: {type(merged_model)}")

    def run_model_fusion(self):
        import time
        start_time = time.time()
        cfg = self.config

        self.modelpool = modelpool = self._load_and_setup(
            load_modelpool_from_config, cfg.modelpool
        )
        self.alalgorithm = algorithm = self._load_and_setup(
            load_algorithm_from_config, cfg.method
        )

        # Re-seed AFTER fabric initialization, because L.Fabric.launch()
        # internally calls seed_everything() which overrides our seed
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
        log.info(f"Re-seeded all RNGs with seed={seed} (after Fabric init)")

        # Load taskpool early so algorithms can access it during run() via self._program.taskpool
        # (e.g., lora_evo needs to evaluate models during optimization)
        if hasattr(cfg, "taskpool") and cfg.taskpool is not None:
            self.taskpool = taskpool = self._load_and_setup(
                load_taskpool_from_config, cfg.taskpool
            )
            modelpool.setup_taskpool(taskpool)
        else:
            self.taskpool = None
            taskpool = None

        merged_model = None
        save_path = cfg.merged_model_save_path

        # 仅当 save_path 是一个 HuggingFace 目录时，才尝试直接从磁盘加载，
        # 否则总是重新运行融合算法（适用于 UNet 等 state_dict 场景）。
        if isinstance(save_path, str) and os.path.isdir(save_path):
            config_json = os.path.join(save_path, "config.json")
            if os.path.exists(config_json):
                log.info(f"Existing merged model found at {save_path}. Using it for evaluation.")
                merged_model = save_path

        if merged_model is None:
            merged_model = algorithm.run(modelpool)

        self.save_merged_model(merged_model)

        if taskpool is not None:
            report = self.evaluate_merged_model(taskpool, merged_model)

            end_time = time.time()
            running_time = end_time - start_time
            if isinstance(report, dict):
                report["running_time"] = running_time

            log.info(f"Running time: {running_time:.2f} seconds")
            if cfg.get("save_report", False):
                save_report_path = cfg.save_report
                if isinstance(save_report_path, bool):
                    save_report_path = "report.json"
                save_report_path = self._format_path(save_report_path, is_file=True)
                # save report (Dict) to a file
                # if the directory of `save_report` does not exists, create it
                if os.path.dirname(save_report_path):
                    os.makedirs(os.path.dirname(save_report_path), exist_ok=True)

                json.dump(report, open(save_report_path, "w"), indent=4)
        else:
            end_time = time.time()
            log.info(f"Running time: {end_time - start_time:.2f} seconds")
            print("No task pool specified. Skipping evaluation.")


@hydra.main(
    config_path=_get_default_config_path(),
    config_name="qwen25_evogm",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Set global seed for reproducibility
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    log.info(f"Global seed set to {seed}")

    if cfg.print_config:
        print_config_tree(
            cfg,
            print_order=[
                "method",
                "modelpool",
                "taskpool",
            ],
        )
    if cfg.get("dry_run", False):
        log.info("The program is running in dry-run mode. Exiting.")
        return
    if cfg.use_lightning:
        program = LightningProgram(cfg)
        program.run_model_fusion()
    else:
        run_model_fusion(cfg)


if __name__ == "__main__":
    main()
