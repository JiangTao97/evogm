"""
EvoGM Algorithm (GPU).

Evolutionary Generative Model Merging for LoRA expert merging.

Key features:
- Shared memory task vectors: Workers receive TV via share_memory_() (zero-copy)
- Basis propagation: When top_k == n_experts, the algorithm represents each
  round's experts as a coefficient basis over the original experts, avoiding
  dense task-vector materialization and worker restarts at round transitions.
- Persistent workers with CUDA GPU support
"""

import logging
import os
import json
import copy
import gc
import shutil
import traceback
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from functools import partial
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.multiprocessing as mp
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file

from fusion_bench.method.base_algorithm import ModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool, to_modelpool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from .evogm_worker import _gpu_evaluation_worker_v3, _calculate_score
import threading

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

log = logging.getLogger(__name__)




# ============================================================================
# MLP Generator
# ============================================================================

class Generator(nn.Module):
    """
    5-layer MLP Generator: input coefficients → transformed coefficients.

    Used for both lose2win (bad→good) and win2lose (good→bad) transformations.
    Architecture: dim -> 128 -> 256 -> 256 -> 256 -> 128 -> dim
    Output range: [-1, 1] via Tanh activation
    """

    def __init__(self, dim: int, hidden_dims: tuple = (128, 256, 256, 256, 128)):
        super().__init__()
        layers = []
        prev_dim = dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, dim))
        layers.append(nn.Tanh())  # Output range [-1, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Dual Generator Loss Module
# ============================================================================

class DualGeneratorLoss(nn.Module):
    """
    Dual generator loss module with cycle consistency and optimization loss.

    Components:
    - lose2win: Transforms bad solutions to good solutions
    - win2lose: Transforms good solutions to bad solutions (optional, used in dual mode)
    - Cycle consistency: lose→win→lose ≈ lose, win→lose→win ≈ win (only in dual mode)

    Modes:
    - 'dual': Full dual generator with cycle consistency loss
    - 'single': Only lose2win generator, no cycle loss
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: tuple = (128, 256, 256, 256, 128),
        cycle_scale: float = 100.0,
        mode: str = "dual"  # 'dual' or 'single'
    ):
        super().__init__()
        self.mode = mode
        self.lose2win = Generator(dim, hidden_dims)

        if mode == "dual":
            self.win2lose = Generator(dim, hidden_dims)
            self.cycle_scale = cycle_scale
        else:
            self.win2lose = None
            self.cycle_scale = 0.0  # No cycle loss in single mode

        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        winners: torch.Tensor,
        losers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss based on mode.

        Args:
            winners: Tensor of winner coefficients [batch, dim]
            losers: Tensor of loser coefficients [batch, dim]

        Returns:
            Tuple of (total_loss, fake_winners, fake_losers)
            In single mode: fake_losers will be None
        """
        # Generate fake winners from losers (both modes)
        fake_win = self.lose2win(losers)      # bad → good

        if self.mode == "dual":
            # Dual mode: use both generators with cycle consistency
            fake_lose = self.win2lose(winners)    # good → bad

            # Cycle reconstruction
            cycle_lose = self.win2lose(fake_win)  # fake_good → reconstructed bad
            cycle_win = self.lose2win(fake_lose)  # fake_bad → reconstructed good

            # Cycle consistency loss
            loss_cycle_lose = self.mse(cycle_lose, losers).mean()
            loss_cycle_win = self.mse(cycle_win, winners).mean()

            total_cycle_loss = self.cycle_scale * (loss_cycle_lose + loss_cycle_win)

            return total_cycle_loss, fake_win, fake_lose
        else:
            # Single mode: only lose2win, no cycle loss
            # Train lose2win to generate solutions closer to winners (direct supervision)
            direct_loss = self.mse(fake_win, winners.mean(dim=0, keepdim=True).expand_as(fake_win)).mean()

            return direct_loss, fake_win, None

    def generate_from_losers(self, losers: torch.Tensor) -> torch.Tensor:
        """Generate improved solutions from losers using lose2win."""
        with torch.no_grad():
            return self.lose2win(losers)




# ============================================================================
# Parallel Evaluation Worker (Top-level function for spawn)
# ============================================================================





# ============================================================================
# Main Algorithm Class
# ============================================================================

class EvoGMAlgorithm(
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
):
    """
    EvoGM evolutionary algorithm for LoRA expert merging.

    Key features:
    - Each round, top-K merged models become new experts
    - Task vectors are recomputed from these evolved experts
    - Knowledge accumulates across rounds
    """

    def __init__(self, config):
        super().__init__(config)

        # Iterative parameters
        self.n_rounds = config.get("n_rounds", 10)
        self.top_k = config.get("top_k", None)  # None = n_experts

        # Population parameters
        self.population_size = config.get("population_size", 20)
        self.max_iter = config.get("max_iter", 5)  # Inner iterations per round

        # Generator parameters
        self.hidden_dims = tuple(config.get("hidden_dims", [128, 256, 256, 256, 128]))
        self.generator_lr = config.get("generator_lr", 1e-3)
        self.generator_epochs = config.get("generator_epochs", 50)

        # Loss weights
        self.cycle_scale = config.get("cycle_scale", 100.0)

        # Generator mode: 'dual' (default) or 'single'
        # 'dual': Full dual generator with cycle consistency loss
        # 'single': Only lose2win generator, no cycle loss
        self.generator_mode = config.get("generator_mode", "dual")
        self.opt_scale = config.get("opt_scale", 1.0)

        # Data split ratio
        self.winner_portion = config.get("winner_portion", 0.3)

        # Evaluation mode
        self.evaluation_mode = config.get("evaluation_mode", "multi_task")
        self.target_tasks = config.get("target_tasks", None)

        # Multi-GPU parameters
        self.num_gpus = config.get("num_gpus", 4)

        # Output directory
        self.output_dir = config.get("output_dir", "evogm_results")

        # Wandb parameters
        self.use_wandb = config.get("use_wandb", True) and HAS_WANDB
        self.wandb_project = config.get("wandb_project", "evogm")
        self.wandb_run_name = config.get("wandb_run_name", None)
        self.wandb_offline = config.get("wandb_offline", True)

        # Model type: 'causal_lm' (default, for Qwen/Llama) or 'seq2seq_lm' (for T5)
        self.model_type = config.get("model_type", "causal_lm")

        # Evaluation backend: 'taskpool' (default, for Qwen) or 'script' (for T5)
        # 'taskpool': Use taskpool.evaluate_with_split() - in-memory evaluation
        # 'script': Use fitness.evaluate() - subprocess-based evaluation via infer.sh
        self.evaluation_backend = config.get("evaluation_backend", "taskpool")

        # Script evaluation parameters (only used when evaluation_backend='script')
        self.script_tasks = config.get("script_tasks", "cola,mnli,mrpc,qnli,qqp,rte,sst2,stsb")
        self.script_num_samples = config.get("script_num_samples", 50)

        # Runtime state
        self.modelpool = None
        self.taskpool = None
        self.model_names = None
        self.n_experts = None
        self.device = None
        self.eval_count = 0
        self._wandb_run = None

        # Task vector cache (updated each round)
        self._task_vectors = None  # Dict[expert_name, state_dict]
        self._pretrained_sd = None  # Pretrained model state_dict
        self._task_vectors_dir = None  # Directory for task vectors (for parallel workers)

        # Current expert models (updated each round)
        self._current_expert_sds = None  # Dict[expert_name, state_dict]

        # History buffer for cumulative training
        self.history_population = None
        self.history_scores = None

        # Pretrained model path (for tokenizer saving)
        self.pretrained_path = None
        self.pretrained_dtype = None

        # Persistent worker state + shared memory
        self._persistent_workers = []  # List of worker processes
        self._task_queue = None
        self._result_queue = None
        self._shared_data_queues = []  # Per-worker queues for shared memory data
        self._worker_gpus = []  # GPU IDs used by workers
        self._original_tv_dir = None  # Directory for original (unmodified) task vectors

        # Basis propagation state (active when top_k == n_experts)
        self._original_model_names = None
        self._base_task_vectors = None
        self._current_expert_basis = None
        self._pending_expert_basis = None
        self._skip_worker_stop_once = False
        self._skip_worker_restart_once = False

    def _get_model_base_name(self) -> str:
        """
        Extract base model name from pretrained path for output directory naming.
        e.g., 'models/Qwen2.5/Qwen2.5-1.5B-Instruct' -> 'qwen25_1.5b'
        """
        if self.pretrained_path:
            base_name = os.path.basename(self.pretrained_path)
            name_lower = base_name.lower().replace('.', '').replace('-', '_')
            if 'qwen' in name_lower:
                return 'qwen25_1.5b'
            if 't5' in name_lower or 'flan' in name_lower:
                return 'flan_t5_base'
            return name_lower.split('_')[0][:20]
        return 'model'

    def _get_model_class(self):
        """
        Get the appropriate model class based on model_type config.

        Returns:
            AutoModelForCausalLM or AutoModelForSeq2SeqLM class
        """
        if self.model_type == "seq2seq_lm":
            from transformers import AutoModelForSeq2SeqLM
            return AutoModelForSeq2SeqLM
        else:  # causal_lm (default)
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM

    def _get_pretrained_dtype(self):
        """Return the pretrained model dtype even after the CPU model is released."""
        if self.pretrained_dtype is not None:
            return self.pretrained_dtype
        if getattr(self, "pretrained_model", None) is not None:
            return self.pretrained_model.dtype
        return torch.bfloat16

    @staticmethod
    def _share_tensor_inplace(tensor: torch.Tensor) -> torch.Tensor:
        """Share a CPU tensor in-place; fall back to a shared clone only if needed."""
        try:
            if not tensor.is_shared():
                tensor.share_memory_()
            return tensor
        except RuntimeError:
            shared = tensor.clone()
            shared.share_memory_()
            return shared

    def _share_state_dict_inplace(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, value in list(state_dict.items()):
            state_dict[key] = self._share_tensor_inplace(value)
        return state_dict

    def _share_task_vectors_inplace(
        self,
        task_vectors: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        for model_name, task_vector in task_vectors.items():
            task_vectors[model_name] = self._share_state_dict_inplace(task_vector)
        return task_vectors

    @staticmethod
    def _sync_file_handle(file_obj):
        file_obj.flush()
        os.fsync(file_obj.fileno())

    @staticmethod
    def _empty_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_memory(self):
        gc.collect()
        self._empty_device_cache()

    @staticmethod
    def _get_process_memory_mb() -> Optional[float]:
        try:
            import psutil  # type: ignore

            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except Exception:
            try:
                import resource
                import sys

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return rss / (1024 ** 2) if sys.platform == "darwin" else rss / 1024
            except Exception:
                return None

    def _append_jsonl(self, path: str, data: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")
            self._sync_file_handle(f)

    def _write_json(self, path: str, data: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            self._sync_file_handle(f)

    def _save_large_tensor_dict(self, obj, path: str):
        """Save large tensor dicts via legacy serialization."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            torch.save(
                obj,
                f,
                pickle_protocol=4,
                _use_new_zipfile_serialization=False,
            )
            self._sync_file_handle(f)

    @staticmethod
    def _load_large_tensor_dict(path: str):
        return torch.load(path, map_location="cpu", mmap=False)

    def _write_run_context(self, task_suffix: str, extra: Optional[Dict] = None):
        context = {
            "timestamp": datetime.now().isoformat(),
            "task": task_suffix,
            "algorithm": type(self).__name__,
            "pid": os.getpid(),
            "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "num_gpus": self.num_gpus,
            "n_experts": self.n_experts,
            "model_names": list(self.model_names or []),
            "evaluation_mode": self.evaluation_mode,
            "population_size": getattr(self, "population_size", None),
            "max_iter": getattr(self, "max_iter", None),
            "n_rounds": getattr(self, "n_rounds", None),
            "top_k": getattr(self, "top_k", None),
        }
        if extra:
            context.update(extra)
        self._write_json(os.path.join(self.output_dir, "run_context.json"), context)

    def _record_memory_trace(self, event: str, task_suffix: Optional[str] = None, extra: Optional[Dict] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "task": task_suffix,
            "pid": os.getpid(),
            "rss_mb": self._get_process_memory_mb(),
            "eval_count": self.eval_count,
        }
        if torch.cuda.is_available():
            for attr_name, field_name in (
                ("memory_allocated", "gpu_memory_allocated_mb"),
                ("memory_reserved", "gpu_memory_reserved_mb"),
            ):
                if hasattr(torch.cuda, attr_name):
                    try:
                        entry[field_name] = getattr(torch.cuda, attr_name)() / (1024 ** 2)
                    except Exception:
                        pass
        if extra:
            entry.update(extra)
        self._append_jsonl(os.path.join(self.output_dir, "memory_trace.jsonl"), entry)

    def _write_fatal_error(self, task_suffix: str, exc: Exception):
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "task": task_suffix,
            "eval_count": self.eval_count,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        self._write_json(os.path.join(self.output_dir, "fatal_error.json"), error_data)

    def _init_population(self) -> np.ndarray:
        """
        Initialize population with:
        1. One-hot coefficients for each expert
        2. Average coefficients (1/n for each expert)
        3. Random coefficients in [-1, 1]

        Returns:
            Array of shape [population_size, n_experts]
        """
        population = []

        # Add one-hot coefficients for each expert
        for i in range(self.n_experts):
            one_hot = np.zeros(self.n_experts)
            one_hot[i] = 1.0
            population.append(one_hot)

        # Add average coefficients (1/n for each expert)
        avg_coef = np.full(self.n_experts, 1.0 / self.n_experts)
        population.append(avg_coef)

        # Fill remaining with random coefficients in range [-1, 1]
        remaining = self.population_size - len(population)
        if remaining > 0:
            for _ in range(remaining):
                coef = np.random.uniform(-1, 1, self.n_experts)
                population.append(coef)

        # Truncate if experts + 1 > population_size
        population = population[:self.population_size]

        return np.array(population)

    def _coefficients_to_dict(self, coef_vector: np.ndarray) -> Dict[str, float]:
        """Convert coefficient vector to named dict, applying basis propagation if active."""
        effective = self._vector_to_effective_coefficients(coef_vector)
        names = self._original_model_names or self.model_names
        return {name: float(effective[i]) for i, name in enumerate(names)}

    def _vector_to_effective_coefficients(self, coef_vector: np.ndarray) -> np.ndarray:
        """Apply basis propagation to map current-round coefficients to original expert space."""
        coef_array = np.asarray(coef_vector, dtype=np.float32)
        if self._current_expert_basis is None:
            return coef_array
        if coef_array.shape[-1] != self._current_expert_basis.shape[0]:
            raise ValueError(
                f"Coefficient dimension mismatch for basis propagation: "
                f"{coef_array.shape[-1]} vs {self._current_expert_basis.shape[0]}"
            )
        return coef_array @ self._current_expert_basis

    def _dict_to_coefficients(self, coef_dict: Dict[str, float]) -> np.ndarray:
        """Convert named dict to coefficient vector."""
        return np.array([coef_dict.get(name, 0.0) for name in self.model_names])

    def _split_winners_losers(
        self,
        population: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split population into winners and losers based on scores."""
        n_winners = max(1, int(len(population) * self.winner_portion))
        sorted_indices = np.argsort(scores)

        winners = population[sorted_indices[:n_winners]]
        losers = population[sorted_indices[n_winners:]]
        winner_scores = scores[sorted_indices[:n_winners]]
        loser_scores = scores[sorted_indices[n_winners:]]

        return winners, losers, winner_scores, loser_scores

    def _load_state_dict_from_path(self, model_path: str) -> Dict[str, torch.Tensor]:
        """
        Load state dict from a model directory (safetensors or pytorch_bin).

        This is much more memory efficient than loading the full model object.
        Supports sharded models (index.json).
        """
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path is not a directory: {model_path}")

        log.info(f"      Loading state dict from: {model_path}")

        # 1. Try loading safetensors (single file)
        safetensors_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_file):
            log.info(f"      Found single safetensors file")
            return load_file(safetensors_file)

        # 2. Try loading sharded safetensors
        safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(safetensors_index):
            log.info(f"      Found sharded safetensors index")
            import json
            with open(safetensors_index, 'r') as f:
                index = json.load(f)

            # Safetensors index format: {weight_map: {tensor_name: filename}}
            # We need to load each shard file once and extract weights
            state_dict = {}
            weight_map = index.get('weight_map', {})

            # Get unique filenames (each shard file may contain multiple tensors)
            unique_files = set(weight_map.values())
            log.info(f"      Found {len(unique_files)} sharded safetensors files")

            for filename in sorted(unique_files):
                sf_path = os.path.join(model_path, filename)
                log.info(f"      Loading sharded safetensors: {filename}")
                loaded = load_file(sf_path)
                # Add all weights from this shard
                state_dict.update(loaded)

            return state_dict

        # 3. Try loading pytorch_bin (single file)
        pytorch_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_file):
            log.info(f"      Found single pytorch_bin file")
            return torch.load(pytorch_file, map_location="cpu")

        # 4. Try loading sharded pytorch_bin
        pytorch_index = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.exists(pytorch_index):
            log.info(f"      Found sharded pytorch_bin index")
            import json
            with open(pytorch_index, 'r') as f:
                index = json.load(f)
            state_dict = {}
            for filename, weight_names in index['weight_map'].items():
                bin_path = os.path.join(model_path, filename)
                log.info(f"      Loading part: {filename}")
                loaded = torch.load(bin_path, map_location="cpu")
                for name in weight_names:
                    state_dict[name] = loaded[name]
            return state_dict

        raise FileNotFoundError(f"No model weights found in {model_path}")

    def _compute_initial_task_vectors_with_model_objects(self):
        """
        [BACKUP] Original method: Compute task vectors by instantiating full model objects.

        WARNING: This method is memory intensive as it loads full model objects into memory.
        Use _compute_initial_task_vectors() instead for large models.
        """
        log.info("Computing initial task vectors (Original Method - Memory Intensive)...")

        self._pretrained_sd = {k: v.cpu().clone() for k, v in self.pretrained_model.state_dict().items()}
        self._task_vectors = {}
        self._current_expert_sds = {}

        self._task_vectors_dir = os.path.join(self.output_dir, "task_vectors")
        os.makedirs(self._task_vectors_dir, exist_ok=True)

        pretrained_path = os.path.join(self._task_vectors_dir, "_pretrained_.pt")
        self._save_large_tensor_dict(self._pretrained_sd, pretrained_path)

        pretrained_config = self.modelpool.get_model_config("_pretrained_")
        base_model_path = self.modelpool._resolve_path(pretrained_config["path"])

        def load_model_for_tv(model_name, model_config):
            model_class = self._get_model_class()
            model_path = self.modelpool._resolve_path(model_config["path"])
            is_lora = self.modelpool._is_lora_adapter(model_path)

            if is_lora:
                log.info(f"  [BACKUP] Loading LoRA adapter for: {model_name}")
                base_model = model_class.from_pretrained(base_model_path, torch_dtype=self._get_pretrained_dtype(), device_map="cpu")
                lora_model = PeftModel.from_pretrained(base_model, model_path)
                fine_tuned_model = lora_model.merge_and_unload()
                del base_model, lora_model
                return fine_tuned_model
            else:
                log.info(f"  [BACKUP] Loading full model for: {model_name}")
                return model_class.from_pretrained(model_path, torch_dtype=self._get_pretrained_dtype(), device_map="cpu")

        for model_name in self.modelpool.model_names:
            model_config = self.modelpool.get_model_config(model_name)
            log.info(f"  Computing task vector for: {model_name}")
            fine_tuned_model = load_model_for_tv(model_name, model_config)
            fine_tuned_sd = {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()}
            self._current_expert_sds[model_name] = fine_tuned_sd
            task_vector = {k: fine_tuned_sd[k] - self._pretrained_sd[k] for k in fine_tuned_sd if k in self._pretrained_sd}
            self._task_vectors[model_name] = task_vector
            tv_path = os.path.join(self._task_vectors_dir, f"{model_name}.pt")
            self._save_large_tensor_dict(task_vector, tv_path)
            del fine_tuned_model, fine_tuned_sd
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        log.info(f"Initial task vectors computed for {len(self._task_vectors)} experts (saved to {self._task_vectors_dir})")

    def _compute_initial_task_vectors(self):
        """
        Compute task vectors from initial models.

        If the expert model is a LoRA adapter (adapter_config.json exists), merge it with the base model.
        If it's a full model, use it directly.

        Task vector = (fine_tuned) - pretrained
        Shared-memory workers do not need these vectors on disk, so this method
        keeps them in RAM and avoids creating a large persistent disk cache.

        Also initializes basis propagation state for efficient round transitions.
        """
        log.info("Computing initial task vectors from models...")

        if getattr(self, "pretrained_model", None) is not None:
            self._pretrained_sd = {k: v.cpu().clone() for k, v in self.pretrained_model.state_dict().items()}
        elif self._pretrained_sd is None:
            raise RuntimeError("Pretrained state dict is unavailable for task vector computation.")

        self._task_vectors = {}
        self._current_expert_sds = {}
        self._task_vectors_dir = None
        self._original_tv_dir = None

        # Get base model path
        pretrained_config = self.modelpool.get_model_config("_pretrained_")
        base_model_path = self.modelpool._resolve_path(pretrained_config["path"])

        for model_name in self.modelpool.model_names:
            model_config = self.modelpool.get_model_config(model_name)
            model_path = self.modelpool._resolve_path(model_config["path"])

            # Check if it's a LoRA adapter
            is_lora = self.modelpool._is_lora_adapter(model_path)

            log.info(f"  Computing task vector for: {model_name} (is_lora={is_lora})")

            if is_lora:
                log.warning(f"  LoRA adapter detected for {model_name}. Merging LoRA requires loading base model.")

                model_class = self._get_model_class()
                base_model = model_class.from_pretrained(
                    base_model_path,
                    torch_dtype=self._get_pretrained_dtype(),
                    device_map="cpu"
                )
                lora_model = PeftModel.from_pretrained(base_model, model_path)
                fine_tuned_model = lora_model.merge_and_unload()
                fine_tuned_sd = {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()}
                del base_model, lora_model, fine_tuned_model
            else:
                fine_tuned_sd = self._load_state_dict_from_path(model_path)
                fine_tuned_sd = {k: v.cpu() for k, v in fine_tuned_sd.items()}

            task_vector = {}
            for k in fine_tuned_sd:
                if k in self._pretrained_sd:
                    task_vector[k] = fine_tuned_sd[k] - self._pretrained_sd[k]
            self._task_vectors[model_name] = task_vector

            del fine_tuned_sd
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        log.info(f"Initial task vectors computed for {len(self._task_vectors)} experts (kept in memory)")

        # Initialize basis propagation
        self._original_model_names = list(self.model_names or [])
        self._base_task_vectors = self._task_vectors
        self._reset_expert_basis()
        log.info(
            "Initialized basis propagation with %s original experts",
            len(self._original_model_names),
        )

    def _reset_expert_basis(self):
        """Reset the expert basis to identity (original experts)."""
        if self.n_experts is None:
            raise RuntimeError("n_experts must be initialized before resetting expert basis.")
        self._current_expert_basis = np.eye(self.n_experts, dtype=np.float32)
        self._pending_expert_basis = None
        self._skip_worker_stop_once = False
        self._skip_worker_restart_once = False

    def _update_task_vectors_from_experts(self, expert_state_dicts: Dict[str, Dict[str, torch.Tensor]]):
        """
        Update task vectors from evolved expert models.
        Also saves updated task vectors to disk for parallel workers.

        Args:
            expert_state_dicts: Dict mapping expert names to their state dicts
        """
        log.info("Updating task vectors from evolved experts...")

        self._task_vectors = {}
        self._current_expert_sds = None

        for model_name, expert_sd in expert_state_dicts.items():
            task_vector = {}
            for k in expert_sd:
                if k in self._pretrained_sd:
                    task_vector[k] = expert_sd[k] - self._pretrained_sd[k]
            self._task_vectors[model_name] = task_vector

            log.info(f"  Updated task vector for: {model_name}")

        log.info(f"Task vectors updated for {len(self._task_vectors)} experts")

    def _set_task_vectors(self, task_vectors: Dict[str, Dict[str, torch.Tensor]]):
        """Replace current task vectors directly without materializing full merged models."""
        log.info("Replacing task vectors for next round...")
        self._task_vectors = task_vectors
        self._current_expert_sds = None

        for model_name in self._task_vectors:
            log.info(f"  Updated task vector for: {model_name}")

        log.info(f"Task vectors replaced for {len(self._task_vectors)} experts")

    def _get_merged_task_vector(self, coefficients: Dict[str, float]) -> Optional[Dict[str, torch.Tensor]]:
        """Compute only the weighted task vector sum for a coefficient set."""
        if self._task_vectors is None:
            raise RuntimeError("Task vectors not computed.")

        total_task_vector = None

        for model_name, coef in coefficients.items():
            if coef == 0.0:
                continue

            task_vector = self._task_vectors.get(model_name)
            if task_vector is None:
                continue

            scaled_tv = state_dict_mul(task_vector, coef) if coef != 1.0 else task_vector

            if total_task_vector is None:
                total_task_vector = {k: v.clone() for k, v in scaled_tv.items()}
            else:
                total_task_vector = state_dict_add(total_task_vector, scaled_tv)

        return total_task_vector

    def _create_merged_model(self, coefficients: Dict[str, float]) -> torch.nn.Module:
        """
        Create a merged model using Task Arithmetic with current task vectors.

        Formula: merged = pretrained + Σ(coef_i × task_vector_i)
        """
        if self._task_vectors is None:
            raise RuntimeError("Task vectors not computed.")

        total_task_vector = self._get_merged_task_vector(coefficients)

        # Create merged model
        model_class = self._get_model_class()
        pretrained_config = self.modelpool.get_model_config("_pretrained_")
        model_clone = model_class.from_pretrained(
            pretrained_config["path"],
            torch_dtype=self._get_pretrained_dtype(),
            device_map="cpu",
        )

        if total_task_vector is not None:
            # Use strict=False to handle tied weights (e.g., T5 encoder/decoder embed_tokens)
            # Keys in pretrained_sd but not in total_task_vector will be copied as-is
            merged_sd = state_dict_add(self._pretrained_sd, total_task_vector, strict=False)
            # Add keys that are in pretrained_sd but not in total_task_vector
            for k in self._pretrained_sd:
                if k not in merged_sd:
                    merged_sd[k] = self._pretrained_sd[k].clone()
            model_clone.load_state_dict(merged_sd)

        # Return model on CPU - caller decides whether to move to GPU
        return model_clone

    def _get_merged_state_dict(self, coefficients: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Get merged model state dict without creating a full model (memory efficient).

        Returns:
            Merged state dict on CPU
        """
        if self._task_vectors is None:
            raise RuntimeError("Task vectors not computed.")

        total_task_vector = self._get_merged_task_vector(coefficients)

        if total_task_vector is not None:
            # Use strict=False to handle tied weights (e.g., T5 encoder/decoder embed_tokens)
            merged_sd = state_dict_add(self._pretrained_sd, total_task_vector, strict=False)
            # Add keys that are in pretrained_sd but not in total_task_vector
            for k in self._pretrained_sd:
                if k not in merged_sd:
                    merged_sd[k] = self._pretrained_sd[k].clone()
            return merged_sd
        else:
            return {k: v.clone() for k, v in self._pretrained_sd.items()}

    def _save_best_task_vector(self, coefficients: Dict[str, float], save_path: str):
        """Persist the best solution as effective coefficients (lightweight) plus a compatibility artifact."""
        payload = {"coefficients": coefficients}
        torch.save(payload, save_path)
        json_path = os.path.splitext(save_path)[0] + ".json"
        self._write_json(json_path, payload)

    def _create_merged_model_from_task_vector_path(self, task_vector_path: str) -> torch.nn.Module:
        """Reconstruct a merged model from a saved coefficient payload."""
        payload = torch.load(task_vector_path, map_location="cpu", mmap=False)
        if isinstance(payload, dict) and "coefficients" in payload:
            return self._create_merged_model(payload["coefficients"])
        # Legacy fallback: dense task vector
        model_class = self._get_model_class()
        pretrained_config = self.modelpool.get_model_config("_pretrained_")
        model_clone = model_class.from_pretrained(
            pretrained_config["path"],
            torch_dtype=self._get_pretrained_dtype(),
            device_map="cpu",
        )
        if payload:
            merged_sd = state_dict_add(self._pretrained_sd, payload, strict=False)
            for k in self._pretrained_sd:
                if k not in merged_sd:
                    merged_sd[k] = self._pretrained_sd[k].clone()
            model_clone.load_state_dict(merged_sd)
            del merged_sd
        del payload
        self._release_memory()
        return model_clone

    def _materialize_next_round_task_vectors(
        self,
        top_k_indices: np.ndarray,
        top_k_populations: np.ndarray,
    ) -> Tuple[Optional[str], object]:
        """
        Prepare next-round experts.

        Fast path (basis propagation, when top_k == n_experts):
          Compute the next-round expert basis in memory instead of materializing
          dense task vectors to disk. Returns (None, {"pending_basis": ...}).

        Slow path (fallback):
          Materialize next-round task vectors into a transient cache directory.
          Returns (temp_dir, [(expert_name, save_path), ...]).
        """
        # Fast path: basis propagation
        if self._use_basis_propagation(top_k_populations):
            del top_k_indices
            self._pending_expert_basis = (
                np.asarray(top_k_populations, dtype=np.float32)
                @ np.asarray(self._current_expert_basis, dtype=np.float32)
            )
            self._skip_worker_stop_once = True
            log.info(
                "Prepared next-round expert basis without dense task-vector materialization (shape=%s)",
                tuple(self._pending_expert_basis.shape),
            )
            return None, {"pending_basis": self._pending_expert_basis}

        # Slow path: materialize to disk
        temp_dir = os.path.join(
            self.output_dir,
            f"_next_round_task_vectors_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        )
        os.makedirs(temp_dir, exist_ok=True)

        saved_paths = []
        try:
            for i, (idx, coef_vector) in enumerate(zip(top_k_indices, top_k_populations)):
                coef_dict = self._coefficients_to_dict(coef_vector)
                expert_name = self.model_names[i] if i < len(self.model_names) else f"expert_{i}"

                log.info(
                    "  Creating expert %s from individual %s (accuracy: %.4f)",
                    expert_name,
                    idx,
                    -self.history_scores[idx],
                )
                merged_task_vector = self._get_merged_task_vector(coef_dict)
                save_path = os.path.join(temp_dir, f"{expert_name}.pt")
                self._save_large_tensor_dict(merged_task_vector if merged_task_vector is not None else {}, save_path)
                saved_paths.append((expert_name, save_path))
                del merged_task_vector
                self._release_memory()
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return temp_dir, saved_paths

    def _use_basis_propagation(self, top_k_populations: np.ndarray) -> bool:
        """Check whether basis propagation fast path is applicable."""
        return (
            self._current_expert_basis is not None
            and self._base_task_vectors is not None
            and self.top_k == self.n_experts
            and top_k_populations.shape[0] == self.n_experts
        )

    def _load_task_vectors_from_saved_paths(self, saved_paths):
        """Load next-round task vectors or swap in pending basis."""
        # Fast path: basis propagation
        if self._pending_expert_basis is not None:
            pending_basis = saved_paths.get("pending_basis") if isinstance(saved_paths, dict) else None
            if pending_basis is None:
                raise RuntimeError("Round switch is missing the pending expert basis.")
            if self._base_task_vectors is None:
                raise RuntimeError("Base task vectors are unavailable during round switch.")

            self._task_vectors = self._base_task_vectors
            self._current_expert_basis = np.asarray(pending_basis, dtype=np.float32)
            self._pending_expert_basis = None
            self._skip_worker_restart_once = True
            log.info("Applied next-round expert basis in memory without reloading task vectors")
            return

        # Slow path: load from disk
        new_task_vectors = {}
        for expert_name, save_path in saved_paths:
            new_task_vectors[expert_name] = self._load_large_tensor_dict(save_path)

        self._set_task_vectors(new_task_vectors)
        del new_task_vectors
        self._release_memory()

    def _build_next_round_task_vectors(
        self,
        top_k_indices: np.ndarray,
        top_k_populations: np.ndarray,
    ):
        """Compatibility wrapper for callers that do not need worker stop/restart sequencing."""
        temp_dir = None
        try:
            temp_dir, saved_paths = self._materialize_next_round_task_vectors(top_k_indices, top_k_populations)
            self._task_vectors = None
            self._release_memory()
            self._load_task_vectors_from_saved_paths(saved_paths)
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _evaluate_with_script(
        self,
        merged_model: torch.nn.Module,
        eval_id: str,
        split: str = "dev"
    ) -> Tuple[float, dict]:
        """
        Evaluate model using subprocess-based script (direct python call).

        This method saves the model to disk, runs the evaluation script directly,
        and parses the results. It avoids accelerate launch conflicts.

        Args:
            merged_model: The merged model to evaluate
            eval_id: Unique identifier for this evaluation (used for temp directory)
            split: Data split to use ('dev' or 'test')

        Returns:
            Tuple of (score, report_dict)
        """
        # Create temp directory for this evaluation
        # Use absolute path to avoid HuggingFace treating it as a repo ID
        temp_model_path = os.path.abspath(os.path.join(self.output_dir, f"temp_eval_{eval_id}"))
        os.makedirs(temp_model_path, exist_ok=True)


        try:
            # Save model to disk for script evaluation
            log.info(f"Saving model for script evaluation to: {temp_model_path}")
            merged_model.save_pretrained(temp_model_path)

            # Also save tokenizer if available
            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                self._tokenizer.save_pretrained(temp_model_path)
            elif self.pretrained_path:
                from transformers import AutoTokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
                    tokenizer.save_pretrained(temp_model_path)
                except Exception as e:
                    log.warning(f"Could not save tokenizer: {e}")

            # Run inference script directly
            import subprocess
            import sys

            project_root = os.getcwd()
            script_path = os.path.join(project_root, "scripts/inference/distributed_inference.py")

            # Use GPU 0 for main process
            gpu_id = 0

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["MASTER_ADDR"] = "127.0.0.1"
            # Use a port range distinct from workers (e.g., 29600+)
            env["MASTER_PORT"] = str(29600 + (self.eval_count % 100))
            env["WORLD_SIZE"] = "1"
            env["RANK"] = "0"
            env["LOCAL_RANK"] = "0"
            env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

            report = {}
            tasks = self.script_tasks.split(",")

            for t_name in tasks:
                # Use accelerate launch for proper distributed setup
                cmd = [
                    sys.executable, "-m", "accelerate.commands.launch",
                    "--num_processes", "1",
                    "--main_process_port", str(29600 + (self.eval_count % 100)),
                    script_path,
                    "--base_model", temp_model_path,
                    "--task_name", t_name,
                    "--num_samples", str(self.script_num_samples),
                    "--model_type", "t5", # Force T5 model type
                ]

                log.info(f"[Main Process] Running inference for {t_name}...")
                # Run with cwd=temp_model_path so outputs go there
                result = subprocess.run(cmd, env=env, cwd=temp_model_path, capture_output=True, text=True)
                if result.returncode != 0:
                    log.error(f"[Main Process] Inference failed for {t_name}")
                    log.error(f"STDOUT: {result.stdout[-2000:] if result.stdout else 'None'}")
                    log.error(f"STDERR: {result.stderr[-2000:] if result.stderr else 'None'}")
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

                # Read result file
                res_file = os.path.join(temp_model_path, f"infer_res_{t_name}.jsonl")
                if os.path.exists(res_file):
                    import json
                    from scipy.stats import spearmanr

                    all_data = []
                    with open(res_file, 'r') as f:
                        for line in f:
                            all_data.append(json.loads(line))

                    # Calculate score using exact match (consistent with eval_t5_glue.sh)
                    if t_name == "stsb":
                        # STS-B: use Spearman correlation
                        preds = []
                        labels = []
                        for d in all_data:
                            try:
                                pred = float(d['output'].strip())
                            except:
                                pred = 0.0
                            preds.append(pred)
                            labels.append(float(d['label']))

                        if len(preds) > 1:
                            score = spearmanr(preds, labels)[0]
                            if score != score:  # check for NaN
                                score = 0.0
                        else:
                            score = 0.0
                        report[t_name] = {"spearman_rho": score}
                    else:
                        # Classification tasks: use exact match (output == label_text)
                        correct = 0
                        total = len(all_data)
                        for d in all_data:
                            output = d['output'].strip().lower()
                            label_text = d.get('label_text', '').strip().lower()
                            if output == label_text:
                                correct += 1
                        score = correct / total if total > 0 else 0.0
                        report[t_name] = {"accuracy": score}
                else:
                    log.error(f"[Main Process] Result file not found for {t_name}")
                    report[t_name] = {"accuracy": 0.0}

            # Calculate normalized score using existing function
            score = _calculate_score(report)

            log.info(f"Script evaluation complete: final_score={score:.4f}")
            return score, report

        except Exception as e:
            log.error(f"Script evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), {"error": str(e)}

        finally:
            # Cleanup temp model directory
            import shutil
            if os.path.exists(temp_model_path):
                try:
                    shutil.rmtree(temp_model_path)
                    log.info(f"Cleaned up temp model directory: {temp_model_path}")
                except Exception as e:
                    log.warning(f"Could not cleanup temp directory {temp_model_path}: {e}")


    def _get_available_gpus(self) -> List[int]:
        """Get logical NPU IDs visible to the current process."""
        gpu_ids = list(range(torch.cuda.device_count()))
        max_workers = min(self.num_gpus, len(gpu_ids))
        return gpu_ids[:max_workers]

    # ========================================================================
    # V3: Persistent Worker Lifecycle with Shared Memory
    # ========================================================================

    def _make_shared_tensors(self):
        """Convert current _pretrained_sd and _task_vectors to shared memory tensors."""
        shared_pretrained_sd = self._share_state_dict_inplace(self._pretrained_sd)
        shared_task_vectors = self._share_task_vectors_inplace(self._task_vectors)
        return shared_pretrained_sd, shared_task_vectors

    def _start_persistent_workers(self):
        """Start persistent workers with shared memory data (call once per task)."""
        if self._skip_worker_restart_once:
            if self._persistent_workers:
                log.info("Skipping worker restart during basis-only round switch")
            self._skip_worker_restart_once = False
            return

        if self._persistent_workers:
            log.warning("Workers already running, stopping first...")
            self._stop_persistent_workers()

        available_gpus = self._get_available_gpus()
        self._worker_gpus = available_gpus[1:]  # GPU 0 reserved for main process

        if not self._worker_gpus:
            log.info("No worker GPUs available (only 1 GPU), will use sequential evaluation")
            return

        pretrained_config = self.modelpool.get_model_config("_pretrained_")
        pretrained_path = pretrained_config["path"]

        from omegaconf import OmegaConf
        taskpool_config = OmegaConf.to_container(self.taskpool.config, resolve=True)
        dtype_str = str(self._get_pretrained_dtype()).split('.')[-1]

        ctx = mp.get_context('spawn')
        self._task_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._shared_data_queues = []

        # V3: Create shared memory tensors (ONE copy for ALL workers)
        log.info("Preparing shared memory tensors...")
        shared_pretrained_sd, shared_task_vectors = self._make_shared_tensors()
        log.info(f"  Shared pretrained_sd: {len(shared_pretrained_sd)} keys")
        log.info(f"  Shared task_vectors: {len(shared_task_vectors)} experts")

        log.info(f"Starting {len(self._worker_gpus)} persistent workers on GPUs {self._worker_gpus}...")

        for gpu_id in self._worker_gpus:
            # Each worker gets its own data queue for receiving shared tensors
            data_queue = ctx.Queue()
            data_queue.put({
                "pretrained_sd": shared_pretrained_sd,
                "task_vectors": shared_task_vectors,
            })
            self._shared_data_queues.append(data_queue)

            p = ctx.Process(
                target=_gpu_evaluation_worker_v3,
                args=(
                    gpu_id, self._task_queue, self._result_queue,
                    data_queue, pretrained_path, taskpool_config,
                    self.model_names, dtype_str,
                    self.model_type, self.evaluation_backend,
                    self.script_tasks, self.script_num_samples, self.output_dir,
                )
            )
            p.start()
            self._persistent_workers.append(p)

        # Wait for all workers to signal READY
        ready_count = 0
        for _ in range(len(self._worker_gpus)):
            try:
                msg = self._result_queue.get(timeout=300)
                if msg[0] == "READY":
                    ready_count += 1
                    log.info(f"  Worker on GPU {msg[1]} is ready ({ready_count}/{len(self._worker_gpus)})")
            except Exception as e:
                log.error(f"  Timeout waiting for worker: {e}")

        log.info(f"All {ready_count} persistent workers are ready (shared memory mode)")

    def _stop_persistent_workers(self):
        """Stop all persistent workers (call at end of task)."""
        if self._skip_worker_stop_once:
            if self._persistent_workers:
                log.info("Skipping worker stop during basis-only round switch")
            self._skip_worker_stop_once = False
            return

        if not self._persistent_workers:
            return

        log.info(f"Stopping {len(self._persistent_workers)} persistent workers...")

        for _ in self._persistent_workers:
            self._task_queue.put(None)  # Shutdown signal

        for p in self._persistent_workers:
            p.join(timeout=30)
            if p.is_alive():
                log.warning(f"Worker {p.pid} did not terminate, forcing...")
                p.terminate()

        self._persistent_workers = []
        self._task_queue = None
        self._result_queue = None
        self._shared_data_queues = []
        self._worker_gpus = []
        log.info("All persistent workers stopped")

    def _reload_worker_task_vectors(self):
        """V3: Send new shared memory task vectors to workers (zero-copy)."""
        if not self._persistent_workers:
            return

        log.info("Creating new shared task vectors for workers...")

        shared_task_vectors = self._share_task_vectors_inplace(self._task_vectors)

        # Signal workers to reload and send new data
        for i, _ in enumerate(self._persistent_workers):
            self._task_queue.put("RELOAD")
            self._shared_data_queues[i].put({"task_vectors": shared_task_vectors})

        # Wait for all RELOAD_DONE acks
        reload_count = 0
        for _ in range(len(self._persistent_workers)):
            try:
                msg = self._result_queue.get(timeout=120)
                if msg[0] == "RELOAD_DONE":
                    reload_count += 1
            except Exception as e:
                log.error(f"Timeout waiting for worker reload: {e}")

        log.info(f"Task vectors reloaded on {reload_count}/{len(self._persistent_workers)} workers (shared memory)")

    def _reset_task_vectors_from_disk(self):
        """Reset task vectors for a new task, restoring basis to identity."""
        if self._base_task_vectors is not None:
            self._task_vectors = self._base_task_vectors
            self._reset_expert_basis()
            self._release_memory()
            log.info("Reset expert basis to original experts without disk reload")
            return

        if self._original_tv_dir is None:
            log.info("No on-disk task-vector cache; recomputing original task vectors from source models...")
            self._compute_initial_task_vectors()
            return
        log.info("Resetting task vectors from disk (original/ directory)...")
        for model_name in self.model_names:
            orig_path = os.path.join(self._original_tv_dir, f"{model_name}.pt")
            self._task_vectors[model_name] = self._load_large_tensor_dict(orig_path)
            if self._task_vectors_dir:
                current_path = os.path.join(self._task_vectors_dir, f"{model_name}.pt")
                self._save_large_tensor_dict(self._task_vectors[model_name], current_path)
        log.info(f"  Task vectors reset for {len(self.model_names)} experts")

    # ========================================================================
    # V2: Optimized Parallel Evaluation
    # ========================================================================

    def _parallel_evaluate(
        self,
        population: np.ndarray,
        split: str = "dev",
        task_name: Optional[str] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        V2: Evaluate population using persistent workers + parallel main process.

        Workers are already running. Main process evaluates pop[0] concurrently
        with workers evaluating pop[1:] via threading.
        """
        pop_size = len(population)
        coef_dicts = [self._coefficients_to_dict(p) for p in population]

        has_workers = len(self._persistent_workers) > 0 and pop_size > 1

        if not has_workers:
            return self._sequential_evaluate(coef_dicts, split, task_name=task_name)

        log.info(f"[V2] Parallel evaluation: {pop_size} individuals on {len(self._persistent_workers)+1} GPUs")

        scores = [None] * pop_size
        reports = [None] * pop_size

        # Submit tasks to workers: pop[1:]
        worker_task_count = pop_size - 1
        for i in range(1, pop_size):
            self._task_queue.put((i, coef_dicts[i], split, task_name))

        # V2: Main process evaluates pop[0] in a thread (parallel with workers)
        main_result = [None, None]  # [score, report]
        main_error = [None]

        def _main_evaluate():
            try:
                with torch.no_grad():
                    merged_model = self._create_merged_model(coef_dicts[0])
                    if self.evaluation_backend == "script":
                        eval_id = f"main_0_{self.eval_count}"
                        score, report = self._evaluate_with_script(merged_model, eval_id, split)
                    else:
                        merged_model = merged_model.to("cuda:0")
                        if task_name is not None:
                            task_obj = self.taskpool.load_task_with_split(task_name, split)
                            report = {task_name: task_obj.evaluate(merged_model)}
                        elif hasattr(self.taskpool, 'evaluate_with_split'):
                            report = self.taskpool.evaluate_with_split(merged_model, split)
                        else:
                            report = self.taskpool.evaluate(merged_model)
                        score = _calculate_score(report)
                    del merged_model
                    self._empty_device_cache()
                main_result[0] = score
                main_result[1] = report
            except Exception as e:
                log.error(f"Main process evaluation error: {e}")
                main_error[0] = str(e)
                main_result[0] = float('inf')
                main_result[1] = {"error": str(e)}

        main_thread = threading.Thread(target=_main_evaluate)
        main_thread.start()

        # Collect worker results while main evaluates
        for _ in range(worker_task_count):
            try:
                result = self._result_queue.get(timeout=600)
                # Skip non-tuple results (leftover ACKs)
                if isinstance(result, tuple) and len(result) == 4 and isinstance(result[0], int):
                    idx, score, report, error = result
                    scores[idx] = score
                    reports[idx] = report
                    self.eval_count += 1
                    log.info(f"[Eval {self.eval_count}] Individual {idx} score: {-score:.4f}")
                    self._save_evolution_log(coef_dicts[idx], score, report, split)
            except Exception as e:
                log.error(f"Error collecting worker result: {e}")

        # Wait for main thread
        main_thread.join()
        scores[0] = main_result[0]
        reports[0] = main_result[1]
        self.eval_count += 1
        log.info(f"[Eval {self.eval_count}] Individual 0 (main) score: {-main_result[0]:.4f}")
        self._save_evolution_log(coef_dicts[0], main_result[0], main_result[1], split)

        # Handle missing results
        for i in range(pop_size):
            if scores[i] is None:
                log.warning(f"Missing result for individual {i}, setting to inf")
                scores[i] = float('inf')
                reports[i] = {"error": "missing"}

        return np.array(scores), reports

    def _sequential_evaluate(
        self,
        coef_dicts: List[Dict[str, float]],
        split: str,
        task_name: Optional[str] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """Sequential evaluation fallback."""
        scores = []
        reports = []

        for i, coef_dict in enumerate(coef_dicts):
            self.eval_count += 1
            log.info(f"[Eval {self.eval_count}] Evaluating individual {i+1}/{len(coef_dicts)}")

            with torch.no_grad():
                merged_model = self._create_merged_model(coef_dict)

            # Choose evaluation backend
            if self.evaluation_backend == "script":
                # Use script-based evaluation (for T5) - model stays on CPU
                eval_id = f"seq_{i}_{self.eval_count}"
                score, report = self._evaluate_with_script(merged_model, eval_id, split)
            else:
                # Use taskpool-based evaluation (for Qwen) - need model on GPU
                merged_model = merged_model.to(self.device)
                if task_name is not None:
                    task_obj = self.taskpool.load_task_with_split(task_name, split)
                    report = {task_name: task_obj.evaluate(merged_model)}
                elif hasattr(self.taskpool, 'evaluate_with_split'):
                    report = self.taskpool.evaluate_with_split(merged_model, split)
                else:
                    report = self.taskpool.evaluate(merged_model)
                score = _calculate_score(report)
            scores.append(score)
            reports.append(report)

            log.info(f"[Eval {self.eval_count}] Accuracy: {-score:.4f}")

            # Save to evolution log
            self._save_evolution_log(coef_dict, score, report, split)

            # Cleanup
            del merged_model
            self._empty_device_cache()

        return np.array(scores), reports

    def _evaluate_population(
        self,
        population: np.ndarray,
        split: str = "dev",
        task_name: Optional[str] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Evaluate population using parallel evaluation (falls back to sequential if needed).

        Args:
            population: Array of coefficient vectors [pop_size, n_experts]
            split: Data split to use ('dev' or 'test')
            task_name: Optional task name to evaluate (for single_task mode)

        Returns:
            Tuple of (scores array, list of reports)
        """
        return self._parallel_evaluate(population, split, task_name)

    def _create_dual_gen(self) -> nn.Module:
        """Create generator loss module (dual or single based on config)."""
        return DualGeneratorLoss(
            dim=self.n_experts,
            hidden_dims=self.hidden_dims,
            cycle_scale=self.cycle_scale,
            mode=self.generator_mode,
        )

    def _train_generators(
        self,
        dual_gen: nn.Module,
        winners: np.ndarray,
        losers: np.ndarray,
        winner_scores: np.ndarray,
        loser_scores: np.ndarray,
        optimizer: optim.Optimizer,
    ) -> float:
        """Train dual generators for one epoch."""
        dual_gen.train()

        winners_t = torch.tensor(winners, dtype=torch.float32, device=self.device)
        losers_t = torch.tensor(losers, dtype=torch.float32, device=self.device)

        min_size = min(len(winners_t), len(losers_t))
        if len(winners_t) > min_size:
            indices = torch.randperm(len(winners_t))[:min_size]
            winners_t = winners_t[indices]
        if len(losers_t) > min_size:
            indices = torch.randperm(len(losers_t))[:min_size]
            losers_t = losers_t[indices]

        optimizer.zero_grad()

        cycle_loss, fake_win, fake_lose = dual_gen(winners_t, losers_t)
        opt_loss = self.opt_scale * nn.functional.mse_loss(
            fake_win, winners_t.mean(dim=0, keepdim=True).expand_as(fake_win)
        )

        total_loss = cycle_loss + opt_loss
        total_loss.backward()
        optimizer.step()

        return total_loss.item()

    def _save_evolution_log(
        self,
        coefficients: Dict[str, float],
        score: float,
        report: dict,
        split: str
    ):
        """Save evaluation result to evolution log file and print detailed scores."""
        os.makedirs(self.output_dir, exist_ok=True)
        result_file = os.path.join(self.output_dir, "evolution_log.jsonl")
        csv_file = os.path.join(self.output_dir, "detailed_scores.csv")

        def serialize_report(r):
            if isinstance(r, dict):
                return {k: serialize_report(v) for k, v in r.items()}
            elif isinstance(r, (int, float)):
                return float(r)
            else:
                return str(r)

        per_task_scores = {}
        for task_name, result in report.items():
            if isinstance(result, dict):
                if "accuracy" in result:
                    per_task_scores[task_name] = float(result["accuracy"])
                elif "spearman_rho" in result:  # Support for regression tasks (e.g., STS-B)
                    per_task_scores[task_name] = float(result["spearman_rho"])
                elif "effective_reliability" in result:
                    per_task_scores[task_name] = float(result["effective_reliability"])

        # Print detailed scores to console
        score_str = ", ".join([f"{k}: {v:.4f}" for k, v in per_task_scores.items()])
        log.info(f"Detailed scores: {score_str}")
        print(f">>> Detailed scores: {score_str}", flush=True)

        result_data = {
            "eval_id": self.eval_count,
            "timestamp": datetime.now().isoformat(),
            "split": split,
            "coefficients": coefficients,
            "per_task_scores": per_task_scores,
            "score": float(score),
            "accuracy": float(-score),
            "report": serialize_report(report),
        }

        # Save to JSONL
        self._append_jsonl(result_file, result_data)

        # Save to CSV
        file_exists = os.path.exists(csv_file)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Define header: eval_id, timestamp, score, [task_scores...], [coefficients...]
            task_names = sorted(per_task_scores.keys())
            coef_names = sorted(coefficients.keys())

            if not file_exists:
                header = ["eval_id", "timestamp", "avg_score"] + task_names + coef_names
                writer.writerow(header)

            row = [self.eval_count, result_data["timestamp"], -score]
            row += [per_task_scores.get(tn, "") for tn in task_names]
            row += [coefficients.get(cn, "") for cn in coef_names]
            writer.writerow(row)
            self._sync_file_handle(f)


    def _save_round_results(self, round_num: int, best_coefficients: Dict[str, float], best_score: float):
        """Save results for a round."""
        os.makedirs(self.output_dir, exist_ok=True)

        round_result = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "best_coefficients": best_coefficients,
            "best_accuracy": float(-best_score),
        }

        self._append_jsonl(os.path.join(self.output_dir, "round_results.jsonl"), round_result)

    def _save_best_solutions(
        self,
        initial_best: Dict,
        final_best: Dict,
        final_test_result: Dict
    ):
        """Save best solutions summary to file."""
        os.makedirs(self.output_dir, exist_ok=True)

        best_solutions = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": self.eval_count,
            "n_rounds": self.n_rounds,
            "initial": initial_best,
            "final": final_best,
            "test_result": final_test_result,
        }

        self._write_json(os.path.join(self.output_dir, "best_solutions.json"), best_solutions)

        log.info(f"Best solutions saved to {self.output_dir}/best_solutions.json")

    def _init_wandb(self):
        """Initialize wandb run if enabled."""
        if not self.use_wandb:
            return

        config = {
            "algorithm": "evogm",
            "n_rounds": self.n_rounds,
            "top_k": self.top_k,
            "population_size": self.population_size,
            "max_iter": self.max_iter,
            "hidden_dims": self.hidden_dims,
            "generator_lr": self.generator_lr,
            "generator_epochs": self.generator_epochs,
            "cycle_scale": self.cycle_scale,
            "winner_portion": self.winner_portion,
            "n_experts": self.n_experts,
            "expert_names": self.model_names,
        }

        run_name = self.wandb_run_name or f"evogm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        self._wandb_run = wandb.init(
            project=self.wandb_project,
            name=run_name,
            config=config,
            dir=self.output_dir,
        )
        log.info(f"Wandb initialized: {self._wandb_run.url if not self.wandb_offline else 'offline mode'}")

    def _log_wandb(self, data: Dict, step: Optional[int] = None):
        """Log data to wandb if enabled."""
        if not self.use_wandb or self._wandb_run is None:
            return
        wandb.log(data, step=step)

    def _finish_wandb(self, summary: Dict):
        """Finish wandb run with summary."""
        if not self.use_wandb or self._wandb_run is None:
            return
        for key, value in summary.items():
            wandb.run.summary[key] = value
        wandb.finish()

    def run(self, modelpool, taskpool=None):
        """
        Run the EvoGM optimization.

        Algorithm:
        1. Initialize with original LoRA adapters as experts
        2. For each round:
           a. Run evolution to find best coefficient combinations
           b. Select top-K merged models
           c. Update task vectors from top-K models
        3. Final evaluation on test set

        In single_task mode: run the complete algorithm for each task independently
        In multi_task mode: optimize for average score across all tasks
        """
        self.modelpool = to_modelpool(modelpool)
        self.model_names = list(self.modelpool.model_names)
        self.n_experts = len(self.model_names)

        # Set top_k to n_experts if not specified
        if self.top_k is None:
            self.top_k = self.n_experts

        # Get taskpool
        if taskpool is None:
            if hasattr(self, '_program') and self._program is not None:
                taskpool = self._program.taskpool
            else:
                raise ValueError("taskpool must be provided or accessible via _program")
        self.taskpool = taskpool

        log.info("=" * 60)
        log.info("EvoGM Algorithm")
        log.info("=" * 60)
        log.info(f"  Experts: {self.model_names}")
        log.info(f"  Number of rounds: {self.n_rounds}")
        log.info(f"  Top-K per round: {self.top_k}")
        log.info(f"  Population size: {self.population_size}")
        log.info(f"  Inner iterations per round: {self.max_iter}")
        log.info(f"  Generator epochs: {self.generator_epochs}")
        log.info(f"  Output directory: {self.output_dir}")
        log.info(f"  Evaluation mode: {self.evaluation_mode}")
        log.info("=" * 60)

        # Setup device
        # Main process uses GPU 0 for generator training, workers use GPU 1-3
        # This allows parallel evaluation while keeping generator on GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # Use first GPU for generator
        else:
            self.device = torch.device("cpu")
            log.warning("GPU not available, running on CPU")

        log.info(f"Main process uses {self.device} for generator training")

        # Load pretrained model (kept on CPU - only used for task vector computation)
        with self.profile("load_pretrained"):
            pretrained_config = self.modelpool.get_model_config("_pretrained_")
            self.pretrained_path = pretrained_config["path"]
            self.pretrained_model = self.modelpool.load_model("_pretrained_")
            self.pretrained_model = self.pretrained_model.to("cpu")  # CPU is enough for state dict ops
            log.info(f"Pretrained model loaded (CPU), main process uses {self.device} for generator training")

        # Compute initial task vectors from LoRA adapters
        with self.profile("compute_initial_task_vectors"):
            self._compute_initial_task_vectors()
        self.pretrained_dtype = self.pretrained_model.dtype
        del self.pretrained_model
        self.pretrained_model = None
        gc.collect()
        log.info("Released CPU pretrained model after extracting task vectors")

        # Determine tasks to run based on evaluation_mode
        if self.evaluation_mode == "single_task":
            tasks_to_run = list(self.taskpool.task_names)
            if self.target_tasks:
                if isinstance(self.target_tasks, str):
                    selected_tasks = [t.strip() for t in self.target_tasks.split(",") if t.strip()]
                else:
                    selected_tasks = list(self.target_tasks)
                unknown_tasks = sorted(set(selected_tasks) - set(tasks_to_run))
                if unknown_tasks:
                    raise ValueError(
                        f"Unknown target_tasks: {unknown_tasks}. Available tasks: {tasks_to_run}"
                    )
                tasks_to_run = [task for task in tasks_to_run if task in selected_tasks]
            log.info(f"Single task mode: will run algorithm for each of {len(tasks_to_run)} tasks")
        else:
            tasks_to_run = [None]  # None means multi-task (evaluate all tasks together)

        # V2: No in-memory clone needed - original task vectors are backed up to disk
        # in _compute_initial_task_vectors() -> self._original_tv_dir

        # Initialize wandb once for all tasks
        self._init_wandb()

        merged_model = None
        all_task_results = {}

        # ========================================
        # Main task loop (single_task mode) or single iteration (multi_task mode)
        # ========================================
        # Generate base output directory with model + method naming
        model_base = self._get_model_base_name()
        method_name = "evogm"
        initial_test_report = None
        base_output_dir = self.output_dir
        aggregate_output_dir = self.config.get("output_root_dir", base_output_dir)

        for task_idx, task_name in enumerate(tasks_to_run):
            task_suffix = task_name if task_name else "multi_task"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # If output_root_dir is explicitly set in config, use structured path under it.
            # Otherwise, respect the YAML-configured output_dir (e.g. ${log_dir}/search_process).
            if "output_root_dir" in self.config and self.config.output_root_dir is not None:
                output_root = self.config.output_root_dir
                self.output_dir = os.path.join(
                    output_root,
                    f"{model_base}_{method_name}_{task_suffix}",
                    timestamp,
                )
            else:
                # Append task suffix to the YAML-configured output_dir
                self.output_dir = os.path.join(
                    base_output_dir,
                    f"{task_suffix}_{timestamp}",
                )
            os.makedirs(self.output_dir, exist_ok=True)

            self.eval_count = 0
            initial_best = None
            initial_test_report = None
            dev_report = None
            test_report = None
            final_best = None
            final_test_result = None
            merged_model = None
            global_best_score = float("inf")
            global_best_coefficients = None
            global_best_task_vector_path = os.path.join(self.output_dir, "global_best_task_vector.pt")
            prefix = f"{task_name}/" if task_name else ""

            if task_name is not None:
                log.info(f"\n{'#'*60}")
                log.info(f"# STARTING EVOGO ITERATIVE FOR TASK: {task_name}")
                log.info(f"{'#'*60}\n")
                log.info(f"Output directory: {self.output_dir}")
                if task_idx > 0:
                    self._reset_task_vectors_from_disk()
                self._current_expert_sds = {}
            else:
                log.info(f"\n{'#'*60}")
                log.info(f"# STARTING EVOGO ITERATIVE FOR MULTI-TASK")
                log.info(f"{'#'*60}\n")
                log.info(f"Output directory: {self.output_dir}")

            log_file = os.path.join(self.output_dir, "evolution_log.jsonl")
            if os.path.exists(log_file):
                os.remove(log_file)

            self._write_run_context(task_suffix)
            self._record_memory_trace("task_start", task_suffix)
            self._release_memory()

            try:
                self._record_memory_trace("before_worker_start", task_suffix)
                self._start_persistent_workers()
                self._record_memory_trace("after_worker_start", task_suffix)

                for round_num in range(self.n_rounds):
                    log.info("\n" + "=" * 60)
                    log.info(f"ROUND {round_num + 1}/{self.n_rounds}" + (f" (Task: {task_name})" if task_name else ""))
                    log.info("=" * 60)

                    self.history_population = None
                    self.history_scores = None

                    population = self._init_population()
                    log.info(f"Population initialized: {len(population)} individuals")

                    log.info("Evaluating initial population...")
                    self._record_memory_trace(
                        "before_initial_population_eval",
                        task_suffix,
                        {"round": round_num + 1},
                    )
                    scores, reports = self._evaluate_population(population, split="dev", task_name=task_name)
                    self._record_memory_trace(
                        "after_initial_population_eval",
                        task_suffix,
                        {"round": round_num + 1},
                    )

                    self.history_population = population.copy()
                    self.history_scores = scores.copy()

                    best_idx = np.argmin(scores)
                    round_best_score = scores[best_idx]
                    round_best_coefficients = population[best_idx].copy()

                    if round_num == 0:
                        initial_best = {
                            "coefficients": self._coefficients_to_dict(round_best_coefficients),
                            "dev_score": float(-round_best_score),
                        }

                        log.info("Evaluating initial best on TEST split...")
                        with torch.no_grad():
                            initial_merged_model = self._create_merged_model(initial_best["coefficients"])

                        if self.evaluation_backend == "script":
                            initial_test_score, initial_test_report = self._evaluate_with_script(
                                initial_merged_model,
                                "initial_test",
                                "test",
                            )
                        else:
                            initial_merged_model = initial_merged_model.to(self.device)
                            if task_name is not None:
                                task_obj = self.taskpool.load_task_with_split(task_name, "test")
                                initial_test_report = {task_name: task_obj.evaluate(initial_merged_model)}
                            elif hasattr(self.taskpool, "evaluate_with_split"):
                                initial_test_report = self.taskpool.evaluate_with_split(initial_merged_model, "test")
                            else:
                                initial_test_report = self.taskpool.evaluate(initial_merged_model)
                            initial_test_score = _calculate_score(initial_test_report)
                        initial_best["test_score"] = float(-initial_test_score)

                        log.info(">>> Initial Best:")
                        log.info(f"  - Initial Dev:  {initial_best['dev_score']:.4f}")
                        log.info(f"  - Initial Test: {initial_best['test_score']:.4f}")

                        del initial_merged_model
                        self._release_memory()

                    log.info(f"Round {round_num + 1} initial best accuracy: {-round_best_score:.4f}")
                    self._log_wandb({
                        f"{prefix}round_{round_num + 1}/initial_accuracy": float(-round_best_score),
                        f"{prefix}round_{round_num + 1}/mean_accuracy": float(-scores.mean()),
                    })

                    dual_gen = self._create_dual_gen().to(self.device)
                    optimizer = optim.Adam(dual_gen.parameters(), lr=self.generator_lr)

                    for iteration in range(self.max_iter):
                        log.info(f"\n--- Round {round_num + 1}, Iteration {iteration + 1}/{self.max_iter} ---")

                        winners, losers, winner_scores, loser_scores = self._split_winners_losers(
                            self.history_population,
                            self.history_scores,
                        )
                        log.info(f"Training with {len(self.history_population)} historical samples")

                        for epoch in range(self.generator_epochs):
                            loss = self._train_generators(
                                dual_gen,
                                winners,
                                losers,
                                winner_scores,
                                loser_scores,
                                optimizer,
                            )
                            if (epoch + 1) % 20 == 0:
                                log.info(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

                        with torch.no_grad():
                            population_t = torch.tensor(population, dtype=torch.float32, device=self.device)
                            new_candidates = dual_gen.generate_from_losers(population_t).cpu().numpy()
                        new_candidates = np.clip(new_candidates, -1, 1)

                        new_scores, _ = self._evaluate_population(new_candidates, split="dev", task_name=task_name)

                        self.history_population = np.vstack([self.history_population, new_candidates])
                        self.history_scores = np.concatenate([self.history_scores, new_scores])

                        all_population = np.vstack([population, new_candidates])
                        all_scores = np.concatenate([scores, new_scores])

                        top_indices = np.argsort(all_scores)[:self.population_size]
                        population = all_population[top_indices]
                        scores = all_scores[top_indices]

                        if scores[0] < round_best_score:
                            round_best_score = scores[0]
                            round_best_coefficients = population[0].copy()
                            log.info(f"New round best! Accuracy: {-round_best_score:.4f}")

                        log.info(f"Iteration {iteration + 1} best accuracy: {-scores[0]:.4f}")

                    del dual_gen, optimizer
                    self._release_memory()

                    if round_best_score < global_best_score:
                        global_best_score = round_best_score
                        global_best_coefficients = round_best_coefficients.copy()
                        best_coef_dict = self._coefficients_to_dict(round_best_coefficients)
                        self._save_best_task_vector(best_coef_dict, global_best_task_vector_path)
                        log.info(
                            "New global best! Accuracy: %.4f (task vector saved to %s)",
                            -global_best_score,
                            global_best_task_vector_path,
                        )
                        self._record_memory_trace(
                            "saved_global_best_task_vector",
                            task_suffix,
                            {"round": round_num + 1},
                        )

                    self._save_round_results(
                        round_num + 1,
                        self._coefficients_to_dict(round_best_coefficients),
                        round_best_score,
                    )

                    self._log_wandb({
                        f"{prefix}round_{round_num + 1}/final_accuracy": float(-round_best_score),
                        f"{prefix}global_best_accuracy": float(-global_best_score),
                    })

                    if round_num < self.n_rounds - 1:
                        log.info(f"\nSelecting top-{self.top_k} models for next round...")
                        self._record_memory_trace(
                            "before_round_switch",
                            task_suffix,
                            {"round": round_num + 1},
                        )

                        top_k_indices = np.argsort(self.history_scores)[:self.top_k]
                        top_k_populations = self.history_population[top_k_indices]
                        temp_dir = None
                        try:
                            temp_dir, saved_paths = self._materialize_next_round_task_vectors(
                                top_k_indices,
                                top_k_populations,
                            )
                            self._record_memory_trace(
                                "after_next_round_task_vectors_materialized",
                                task_suffix,
                                {"round": round_num + 1},
                            )
                            self._record_memory_trace(
                                "before_round_switch_worker_stop",
                                task_suffix,
                                {"round": round_num + 1},
                            )
                            self._stop_persistent_workers()
                            self._release_memory()
                            self._record_memory_trace(
                                "after_round_switch_worker_stop",
                                task_suffix,
                                {"round": round_num + 1},
                            )

                            self._task_vectors = None
                            self._release_memory()
                            self._record_memory_trace(
                                "before_next_round_task_vectors_load",
                                task_suffix,
                                {"round": round_num + 1},
                            )

                            self._load_task_vectors_from_saved_paths(saved_paths)
                            self._record_memory_trace(
                                "after_next_round_task_vectors_load",
                                task_suffix,
                                {"round": round_num + 1},
                            )

                            self._start_persistent_workers()
                            self._record_memory_trace(
                                "after_round_switch_worker_restart",
                                task_suffix,
                                {"round": round_num + 1},
                            )
                        finally:
                            if temp_dir is not None:
                                shutil.rmtree(temp_dir, ignore_errors=True)
                        self._record_memory_trace(
                            "after_round_switch",
                            task_suffix,
                            {"round": round_num + 1},
                        )

                        log.info(f"Experts updated for round {round_num + 2}")

                log.info("\n" + "=" * 60)
                log.info(f"FINAL EVALUATION ON TEST" + (f" (Task: {task_name})" if task_name else ""))
                log.info("=" * 60)

                self._stop_persistent_workers()
                self._release_memory()

                final_coefficients = self._coefficients_to_dict(global_best_coefficients)
                self._record_memory_trace("before_final_test", task_suffix)

                log.info("Loading best model from saved task vector...")
                merged_model = self._create_merged_model_from_task_vector_path(global_best_task_vector_path)
                log.info("Best model loaded successfully from saved task vector")

                log.info("Evaluating best on DEV...")
                if self.evaluation_backend == "script":
                    dev_score, dev_report = self._evaluate_with_script(merged_model, "final_dev", "dev")
                else:
                    merged_model = merged_model.to(self.device)
                    if task_name is not None:
                        task_obj = self.taskpool.load_task_with_split(task_name, "dev")
                        dev_report = {task_name: task_obj.evaluate(merged_model)}
                    elif hasattr(self.taskpool, "evaluate_with_split"):
                        dev_report = self.taskpool.evaluate_with_split(merged_model, "dev")
                    else:
                        dev_report = self.taskpool.evaluate(merged_model)
                    dev_score = _calculate_score(dev_report)

                log.info("Evaluating best on TEST...")
                if self.evaluation_backend == "script":
                    test_score, test_report = self._evaluate_with_script(merged_model, "final_test", "test")
                else:
                    if task_name is not None:
                        task_obj = self.taskpool.load_task_with_split(task_name, "test")
                        test_report = {task_name: task_obj.evaluate(merged_model)}
                    elif hasattr(self.taskpool, "evaluate_with_split"):
                        test_report = self.taskpool.evaluate_with_split(merged_model, "test")
                    else:
                        test_report = self.taskpool.evaluate(merged_model)
                    test_score = _calculate_score(test_report)

                final_best = {
                    "coefficients": final_coefficients,
                    "dev_score": float(-dev_score),
                }

                final_test_result = {
                    "coefficients": final_coefficients,
                    "test_score": float(-test_score),
                    "test_report": {k: dict(v) if isinstance(v, dict) else v for k, v in test_report.items()},
                }

                self._save_best_solutions(initial_best, final_best, final_test_result)

                store_key = task_name if task_name else "multi_task"
                all_task_results[store_key] = {
                    "initial_dev": initial_best["dev_score"],
                    "initial_test": initial_best.get("test_score", 0),
                    "final_dev": final_best["dev_score"],
                    "test": final_test_result["test_score"],
                    "coefficients": final_coefficients,
                }

                log.info("=" * 60)
                log.info(f"EVOLUTION COMPLETE - SUMMARY" + (f" ({task_name})" if task_name else ""))
                log.info("=" * 60)
                log.info(f"Total evaluations: {self.eval_count}")
                log.info(f"Initial dev accuracy: {initial_best['dev_score']:.4f}")
                log.info(f"Initial test accuracy: {initial_best.get('test_score', 0):.4f}")
                log.info(f"Final dev accuracy: {final_best['dev_score']:.4f}")
                log.info(f"Final test accuracy: {final_test_result['test_score']:.4f}")
                log.info(f"Best coefficients: {final_coefficients}")
                log.info(f"Improvement (dev): {final_best['dev_score'] - initial_best['dev_score']:+.4f}")
                log.info(f"Improvement (test): {final_test_result['test_score'] - initial_best.get('test_score', 0):+.4f}")
                log.info("=" * 60)

                model_save_path = os.path.join(self.output_dir, "merged_model")
                os.makedirs(model_save_path, exist_ok=True)
                merged_model.save_pretrained(model_save_path)
                log.info(f"Merged model saved to: {model_save_path}")

                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
                    tokenizer.save_pretrained(model_save_path)
                    log.info(f"Tokenizer saved to: {model_save_path}")
                except Exception as e:
                    log.warning(f"Failed to save tokenizer: {e}")

                best_solution = {
                    "task": task_suffix,
                    "initial_coefficients": initial_best["coefficients"],
                    "final_coefficients": final_coefficients,
                    "initial_dev_fitness": initial_best["dev_score"],
                    "initial_test_fitness": initial_best.get("test_score", 0),
                    "final_dev_fitness": float(-dev_score),
                    "final_test_fitness": float(-test_score),
                    "initial_test_scores": {
                        k: v.get("accuracy", v.get("spearman_rho", v.get("effective_reliability", 0)))
                        for k, v in (initial_test_report or {}).items()
                        if isinstance(v, dict)
                    },
                    "final_test_scores": {
                        k: v.get("accuracy", v.get("spearman_rho", v.get("effective_reliability", 0)))
                        for k, v in test_report.items()
                        if isinstance(v, dict)
                    },
                    "n_rounds": self.n_rounds,
                    "max_iter": self.max_iter,
                    "total_evaluations": self.eval_count,
                    "timestamp": timestamp,
                }
                self._write_json(os.path.join(self.output_dir, "best_solution.json"), best_solution)

                self._log_wandb({
                    f"{prefix}final/dev_accuracy": final_best["dev_score"],
                    f"{prefix}final/test_accuracy": final_test_result["test_score"],
                    f"{prefix}final/improvement_dev": final_best["dev_score"] - initial_best["dev_score"],
                    f"{prefix}final/improvement_test": final_test_result["test_score"] - initial_best.get("test_score", 0),
                })
                self._record_memory_trace("after_final_test", task_suffix)
            except Exception as exc:
                self._record_memory_trace(
                    "fatal_error",
                    task_suffix,
                    {"error_type": type(exc).__name__},
                )
                self._write_fatal_error(task_suffix, exc)
                raise
            finally:
                self._stop_persistent_workers()
                initial_test_report = None
                dev_report = None
                test_report = None
                final_test_result = None
                final_best = None
                if merged_model is not None:
                    del merged_model
                    merged_model = None
                self._release_memory()

        # ========================================
        # Print overall summary for single_task mode
        # ========================================
        if self.evaluation_mode == "single_task" and all_task_results:
            log.info("\n" + "#" * 60)
            log.info("# OVERALL SUMMARY - ALL TASKS")
            log.info("#" * 60)
            total_test_scores = []
            for task_name, result in all_task_results.items():
                log.info(f"  {task_name}: dev={result['final_dev']:.4f}, test={result['test']:.4f}")
                total_test_scores.append(result['test'])
            avg_test = np.mean(total_test_scores)
            log.info(f"  Average test accuracy: {avg_test:.4f}")
            log.info("#" * 60)

            summary_path = os.path.join(
                aggregate_output_dir,
                f"{model_base}_{method_name}_all_results.json",
            )
            os.makedirs(aggregate_output_dir, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump({
                    "all_task_results": all_task_results,
                    "average_test_accuracy": avg_test,
                }, f, indent=2)
            log.info(f"Overall summary saved to: {summary_path}")

        # ========================================
        # Generate Summary Table CSV
        # ========================================
        if all_task_results:
            os.makedirs(aggregate_output_dir, exist_ok=True)
            summary_csv_path = os.path.join(
                aggregate_output_dir,
                f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )

            summary_data = []
            for task_name, result in all_task_results.items():
                row = {
                    "task": task_name,
                    "initial_dev": result['initial_dev'],
                    "initial_test": result.get('initial_test', 0),
                    "final_dev": result['final_dev'],
                    "final_test": result['test'],
                    "improvement_test": result['test'] - result.get('initial_test', 0),
                }
                # Add coefficients
                for k, v in result['coefficients'].items():
                    row[f"coef_{k}"] = v
                summary_data.append(row)

            import pandas as pd
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(summary_csv_path, index=False)
            log.info(f"Summary table saved to: {summary_csv_path}")

        # Finish wandb
        self._finish_wandb({
            "total_evaluations": self.eval_count,
        })

        self.print_profile_summary()
        return merged_model
