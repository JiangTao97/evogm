import logging
import copy
import lightning as L
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from tqdm.autonotebook import tqdm
from .base_pool import TaskPool
from fusion_bench.tasks.qwen_eval import QwenEvaluationTask

log = logging.getLogger(__name__)

class QwenEvaluationTaskPool(TaskPool):
    """
    Task pool for Qwen evaluation tasks.
    Manages resources like tokenizer and fabric, and loads QwenEvaluationTask instances.
    """

    def __init__(self, taskpool_config: DictConfig):
        super().__init__(taskpool_config)
        self._tokenizer = None
        self._fabric = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tokenizer_path = self.config.get("tokenizer", "Qwen/Qwen2.5-1.5B-Instruct")
            log.info(f"Loading tokenizer from {tokenizer_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def fabric(self):
        if self._fabric is None:
            # Default to a simple fabric if not provided by the program
            self._fabric = L.Fabric(devices=1)
            self._fabric.launch()
        return self._fabric

    def load_task(self, task_name_or_config: str | DictConfig):
        if isinstance(task_name_or_config, str):
            task_config = self.get_task_config(task_name_or_config)
        else:
            task_config = task_name_or_config

        task = QwenEvaluationTask(task_config)
        task._taskpool = self
        return task

    def load_task_with_split(self, task_name: str, split: str):
        """
        Load a task with a specific split (dev/test).

        Args:
            task_name: Name of the task to load.
            split: The data split to use ('dev' or 'test').

        Returns:
            QwenEvaluationTask configured with the specified split.
        """
        task_config = self.get_task_config(task_name)
        # Create a mutable copy and modify the split
        if isinstance(task_config, DictConfig):
            task_config_copy = OmegaConf.to_container(task_config, resolve=True)
        else:
            task_config_copy = copy.deepcopy(task_config)

        task_config_copy["split"] = split
        task_config_copy = OmegaConf.create(task_config_copy)

        task = QwenEvaluationTask(task_config_copy)
        task._taskpool = self
        return task

    def evaluate_with_split(self, model, split: str = "dev"):
        """
        Evaluate the model on all tasks using a specific data split.

        Args:
            model: The model to evaluate.
            split: The data split to use ('dev' or 'test').

        Returns:
            report (dict): A dictionary containing the results of the evaluation for each task.
        """
        report = {}
        for task_name in tqdm(self.task_names, desc=f"Evaluating tasks ({split})"):
            task = self.load_task_with_split(task_name, split)
            result = task.evaluate(model)
            report[task_name] = result
        return report

    def evaluate(self, model):
        """
        Evaluate the model on all tasks in the pool.
        """
        # Ensure model is on the right device/fabric if needed
        # In fusion_bench, the model is usually already set up by the program
        return super().evaluate(model)
