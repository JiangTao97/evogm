import functools
import logging
import os
from typing import Optional, cast

from omegaconf import DictConfig
from torch.nn.modules import Module
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    MistralForCausalLM,
    PreTrainedModel,
)
from typing_extensions import override

from fusion_bench.modelpool.base_pool import ModelPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.dtype import parse_dtype

log = logging.getLogger(__name__)


class AutoModelForCausalLMPool(ModelPool):
    """
    ModelPool for HuggingFace CausalLM models.

    Supports both full models and LoRA adapters. LoRA adapters are automatically
    detected by checking for `adapter_config.json` in the model path, or can be
    explicitly enabled via `lora: true` in the model config.

    Config options:
        - lora: bool - Global flag to treat all experts as LoRA adapters
        - merge_lora: bool - Whether to merge LoRA weights into base model (default: True)

    Per-model config:
        - lora: bool - Override global lora setting for this specific model
    """

    def __init__(self, modelpool_config: Optional[DictConfig] = None):
        super().__init__(modelpool_config)
        self.processor = None
        self.feature_extractor = None
        self.image_processor = None
        self._base_model_cache = None  # Cache for base model when loading LoRA

    def _resolve_path(self, path: str) -> str:
        return os.path.expanduser(path)

    def _is_lora_adapter(self, model_path: str) -> bool:
        """Check if the model path contains a LoRA adapter (adapter_config.json exists)."""
        adapter_config = os.path.join(model_path, "adapter_config.json")
        return os.path.exists(adapter_config)

    def _should_load_as_lora(self, model_config: DictConfig, model_path: str) -> bool:
        """
        Determine if the model should be loaded as a LoRA adapter.

        Priority:
        1. Per-model `lora` config (if explicitly set)
        2. Global `lora` config from modelpool
        3. Auto-detect via adapter_config.json
        """
        # Per-model explicit setting takes priority
        if model_config.get("lora", None) is not None:
            return model_config.lora

        # Global modelpool setting
        if self.config.get("lora", None) is not None:
            return self.config.lora

        # Auto-detect
        return self._is_lora_adapter(model_path)

    def _maybe_load_multimodal_assets(self, source_path: str):
        """
        Load processor/image processors once so they can be re-used when saving.
        """
        resolved_path = self._resolve_path(source_path)
        if self.processor is None:
            self.processor = self._safe_hf_load(AutoProcessor, resolved_path, "processor")
        if self.feature_extractor is None:
            self.feature_extractor = self._safe_hf_load(
                AutoFeatureExtractor, resolved_path, "feature_extractor"
            )
        if self.image_processor is None:
            self.image_processor = self._safe_hf_load(
                AutoImageProcessor, resolved_path, "image_processor"
            )

    def _safe_hf_load(self, cls, path: str, name: str):
        try:
            return cls.from_pretrained(path)
        except Exception as exc:
            log.debug("Skipping %s load for %s: %s", name, path, exc)
            return None

    def _load_base_model(self, **kwargs) -> Module:
        """Load the pretrained base model, with caching for LoRA scenarios."""
        if self._base_model_cache is not None:
            return self._base_model_cache

        pretrained_config = self.get_model_config("_pretrained_")
        base_path = self._resolve_path(pretrained_config.path)

        with timeit_context(f"loading base model from {base_path}"):
            base_model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)

        return base_model

    def load_model(self, model_config: str | DictConfig) -> Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        kwargs = {}
        if self.config.get("dtype", None) is not None:
            kwargs["torch_dtype"] = parse_dtype(self.config.dtype)

        model_path = self._resolve_path(model_config.path)
        model_name = model_config.get("name", model_path)

        # Check if this should be loaded as LoRA
        # Skip LoRA loading for _pretrained_ model itself
        is_lora = (
            model_name != "_pretrained_"
            and self._should_load_as_lora(model_config, model_path)
        )

        if is_lora:
            from peft import PeftModel

            with timeit_context(f"loading LoRA adapter from {model_path}"):
                # Load base model
                base_model = self._load_base_model(**kwargs)

                # Load LoRA adapter on top of base model
                model = PeftModel.from_pretrained(base_model, model_path)

                # Merge LoRA weights if configured (default: True)
                if self.config.get("merge_lora", True):
                    log.info(f"Merging LoRA adapter into base model: {model_name}")
                    model = model.merge_and_unload()
        else:
            with timeit_context(f"loading model from {model_path}"):
                model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

        self._maybe_load_multimodal_assets(model_config.path)
        return model

    # @override
    # def save_model(
    #     self,
    #     model: PreTrainedModel,
    #     path: str,
    #     push_to_hub: bool = False,
    #     save_tokenizer: bool = False,
    #     **kwargs,
    # ):
    #     """
    #     Save the model to the specified path.

    #     Args:
    #         model (PreTrainedModel): The model to be saved.
    #         path (str): The path where the model will be saved.
    #         push_to_hub (bool, optional): Whether to push the model to the Hugging Face Hub. Defaults to False.
    #         save_tokenizer (bool, optional): Whether to save the tokenizer along with the model. Defaults to False.
    #         **kwargs: Additional keyword arguments passed to the `save_pretrained` method.
    #     """
    #     path = os.path.expanduser(path)
    #     model.save_pretrained(
    #         path,
    #         push_to_hub=push_to_hub,
    #         **kwargs,
    #     )
    #     if save_tokenizer:
    #         if self.has_pretrained:
    #             tokenizer = AutoTokenizer.from_pretrained(
    #                 os.path.expanduser(self.get_model_config("_pretrained_").path)
    #             )
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(
    #                 os.path.expanduser(self.get_model_config(self.model_names[0]).path)
    #             )
    #         tokenizer.save_pretrained(
    #             path,
    #             push_to_hub=push_to_hub,
    #         )

    @override
    def save_model(
        self,
        model: PreTrainedModel,
        path: str,
        push_to_hub: bool = False,
        save_tokenizer: bool = False,
        **kwargs,
    ):
        """
        Save the model, tokenizer, processor, and feature extractor.
        """

        path = os.path.expanduser(path)

        # 1. Save model
        model.save_pretrained(
            path,
            push_to_hub=push_to_hub,
            **kwargs,
        )

        # 2. Save tokenizer
        if save_tokenizer:
            if self.has_pretrained:
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.expanduser(self.get_model_config("_pretrained_").path)
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.expanduser(self.get_model_config(self.model_names[0]).path)
                )
            tokenizer.save_pretrained(
                path,
                push_to_hub=push_to_hub,
            )

        # 3. Save multimodal *preprocessor* (e.g., CLIPImageProcessor, Qwen2VLProcessor)
        if hasattr(self, "processor") and self.processor is not None:
            try:
                self.processor.save_pretrained(path)
            except Exception as e:
                print(f"[WARNING] Failed to save processor: {e}")

        # 4. Save feature extractor (used by older CLIP/ViT models)
        if hasattr(self, "feature_extractor") and self.feature_extractor is not None:
            try:
                self.feature_extractor.save_pretrained(path)
            except Exception as e:
                print(f"[WARNING] Failed to save feature extractor: {e}")


class LLamaForCausalLMPool(AutoModelForCausalLMPool):
    @override
    def load_model(
        self,
        model_config: str | DictConfig,
        backbone_only: bool = False,
    ):
        model = super().load_model(model_config)
        model = cast(LlamaForCausalLM, model)
        if backbone_only:
            model = model.model
        return model


class MistralForCausalLMPool(AutoModelForCausalLMPool):
    @override
    def load_model(
        self,
        model_config: str | DictConfig,
        backbone_only: bool = False,
    ):
        model = super().load_model(model_config)
        model = cast(MistralForCausalLM, model)
        if backbone_only:
            model = model.model
        return model
