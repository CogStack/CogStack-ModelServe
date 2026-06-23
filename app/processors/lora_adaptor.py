import logging
from typing import List, Tuple, Optional, Dict, Any
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model # type: ignore
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from app.exception import ManagedModelException

logger = logging.getLogger("cms")


class LoraAdaptor:

    @staticmethod
    def apply(
        model: PreTrainedModel,
        task_type: str,
        target_modules: Optional[List[str]] = None,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> Tuple[PreTrainedModel, Any]:
        """
        Applies LoRA adaptation to the given model.

        Args:
            model (PreTrainedModel): The model to apply LoRA adaptation to.
            task_type (str): The type of task to apply LoRA adaptation for.
            target_modules (Optional[List[str]]): The names of the modules to apply LoRA adaptation to.
            r (int): The rank of the LoRA adaptation.
            lora_alpha (int): The alpha parameter for the LoRA adaptation.
            lora_dropout (float): The dropout rate for the LoRA adaptation.

        Returns:
            Tuple[PreTrainedModel, LoraConfig]: The adapted model and the LoRA configuration.
        """
        resolved_target_modules = target_modules or LoraAdaptor._get_target_modules_from_mapping(
            model,
            TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
        ) or LoraAdaptor._infer_target_modules(model)
        if not resolved_target_modules:
            raise ManagedModelException(
                "Could not determine LoRA target modules from PEFT mapping or model module names."
            )

        lora_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=resolved_target_modules,
        )
        try:
            peft_model = get_peft_model(model, lora_config)
        except Exception:
            detected_target_modules = LoraAdaptor._infer_target_modules(model)
            logger.warning(
                "Cannot get the PEFT model with target modules %s; retrying with detected modules %s",
                resolved_target_modules,
                detected_target_modules,
            )
            lora_config = LoraConfig(
                task_type=task_type,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=detected_target_modules,
            )
            peft_model = get_peft_model(model, lora_config)

        return peft_model, lora_config

    @staticmethod
    def _get_target_modules_from_mapping(
        model: PreTrainedModel,
        target_modules_mapping: Dict[str, List[str]],
    ) -> List[str]:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        return list(target_modules_mapping.get(model_type, [])) if model_type else []

    @classmethod
    def _infer_target_modules(cls, model: PreTrainedModel) -> List[str]:
        target_module_candidates = [
            ["query", "key", "value"],
            ["q_proj", "k_proj", "v_proj"],
            ["q_lin", "k_lin", "v_lin"],
            ["c_attn"],
        ]
        leaf_module_names = {
            module_name.split(".")[-1]
            for module_name, module in model.named_modules()
            if module_name and len(list(module.children())) == 0
        }
        for candidate_group in target_module_candidates:
            matched = [name for name in candidate_group if name in leaf_module_names]
            if len(matched) == len(candidate_group):
                return matched
        for candidate_group in target_module_candidates:
            matched = [name for name in candidate_group if name in leaf_module_names]
            if matched:
                return matched
        return []
