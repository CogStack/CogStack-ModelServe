import os
import shutil
import logging
import torch
import numpy as np
from typing import Tuple, List, Dict, Iterable, Optional, final, Any
from scipy.special import softmax
from transformers import AutoModelForTokenClassification, PreTrainedModel
from medcat.components.ner.trf.tokenizer import TransformersTokenizer
from app import __version__ as app_version
from app.model_services.base import AbstractModelService
from app.domain import ModelCard, ModelType, Annotation
from app.config import Settings
from app.utils import cls_deprecated, non_default_device_is_available, load_pydantic_object_from_dict

logger = logging.getLogger("cms")


@cls_deprecated("TransformersModelDeIdentification has been deprecated. Use MedCATModelDeIdentification instead.")
@final
class TransformersModelDeIdentification(AbstractModelService):
    """
    DEPRECATED: This class is deprecated and will be removed in a future version.

    Consider using `MedCATModelDeIdentification` instead.
    """

    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._config = config
        model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_file_path = os.path.join(self._model_parent_dir, base_model_file or config.BASE_MODEL_FILE)
        if non_default_device_is_available(config.DEVICE):
            self._device = torch.device(config.DEVICE)
        self.model_name = model_name or "De-identification model"
        self._model: PreTrainedModel
        self._tokenizer: TransformersTokenizer
        self._id2cui: Dict[str, str]

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @model.setter
    def model(self, model: PreTrainedModel) -> None:
        self._model = model

    @model.deleter
    def model(self) -> None:
        del self._model

    @property
    def api_version(self) -> str:
        # APP version is used although each model service could have its own API versioning
        return app_version

    def info(self) -> ModelCard:
        return ModelCard(
            model_description=self.model_name,
            model_type=ModelType.TRANSFORMERS_DEID,
            api_version=self.api_version,
        )

    @staticmethod
    def load_model(
        model_file_path: str,
        *args: Tuple,
        **kwargs: Dict[str, Any],
    ) -> Tuple[TransformersTokenizer, PreTrainedModel]:
        model_file_dir = os.path.dirname(model_file_path)
        model_file_name = os.path.basename(model_file_path).replace(".zip", "")
        unpacked_model_dir = os.path.join(model_file_dir, model_file_name)
        if not os.path.isdir(unpacked_model_dir):
            shutil.unpack_archive(model_file_path, extract_dir=unpacked_model_dir)
        tokenizer_path = os.path.join(unpacked_model_dir, "tokenizer.dat")
        tokenizer = TransformersTokenizer.load(tokenizer_path)
        logger.info("Tokenizer loaded from %s", tokenizer_path)
        model = AutoModelForTokenClassification.from_pretrained(unpacked_model_dir)
        logger.info("Model loaded from %s", unpacked_model_dir)
        return tokenizer, model

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self, "_model") and isinstance(self._model, PreTrainedModel):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._tokenizer, self._model = self.load_model(self._model_file_path)
            self._id2cui = {str(cui_id): cui for cui, cui_id in self._tokenizer.label_map.items()}
            self._model.to(self._device)

    def annotate(self, text: str) -> List[Annotation]:
        return self._get_annotations(text)

    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        annotation_list = []
        for text in texts:
            annotation_list.append(self._get_annotations(text))
        return annotation_list

    def _get_annotations(self, text: str) -> List[Annotation]:
        if not text.strip():
            return []
        self._model.eval()
        device = self._config.DEVICE
        cas = self._config.CONCAT_SIMILAR_ENTITIES == "true"
        ist = self._config.INCLUDE_SPAN_TEXT == "true"
        annotations: List[Annotation] = []

        for dataset, offset_mappings in self._get_chunked_tokens(text):
            predictions = self._model(
                torch.tensor([dataset["input_ids"]]).to(device),
                torch.tensor([dataset["attention_mask"]]).to(device),
            )
            predictions = softmax(predictions.logits.detach().numpy()[0], axis=-1)
            predictions = np.argmax(predictions, axis=-1)

            input_ids = dataset["input_ids"]
            for t_idx, cur_cui_id in enumerate(predictions):
                if cur_cui_id not in [0, -100]:
                    t_text = self._tokenizer.hf_tokenizer.decode(input_ids[t_idx])  # type: ignore
                    if t_text.strip() in ["", "[PAD]"]:
                        continue
                    annotation = load_pydantic_object_from_dict(
                        Annotation,
                        {
                            "label_name": self._tokenizer.cui2name.get(self._id2cui[cur_cui_id]),   # type: ignore
                            "label_id": self._id2cui[cur_cui_id],
                            "start": offset_mappings[t_idx][0],
                            "end": offset_mappings[t_idx][1],
                        },
                    )
                    if ist:
                        annotation.text = t_text
                    if annotations:
                        token_type = self._tokenizer.id2type.get(input_ids[t_idx])  # type: ignore
                        if any([
                            self._should_expand_with_partial(cur_cui_id, token_type, annotation, annotations),  # type: ignore
                            self._should_expand_with_whole(cas, annotation, annotations),
                        ]):
                            annotations[-1].end = annotation.end
                            if ist:
                                annotations[-1].text = text[annotations[-1].start:annotations[-1].end]
                            del annotation
                            continue
                        elif cur_cui_id != 1:
                            annotations.append(annotation)
                            continue
                    else:
                        if cur_cui_id != 1:
                            annotations.append(annotation)
                            continue

        return annotations

    def _get_chunked_tokens(self, text: str) -> Iterable[Tuple[Dict, List[Tuple]]]:
        tokens = self._tokenizer.hf_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)  # type: ignore
        model_max_length = self._tokenizer.max_len
        pad_token_id = self._tokenizer.hf_tokenizer.pad_token_id    # type: ignore
        partial = len(tokens["input_ids"]) % model_max_length
        for i in range(0, len(tokens["input_ids"]) - partial, model_max_length):
            dataset = {
                "input_ids": tokens["input_ids"][i:i+model_max_length],
                "attention_mask": tokens["attention_mask"][i:i+model_max_length],
            }
            offset_mappings = tokens["offset_mapping"][i:i+model_max_length]
            yield dataset, offset_mappings
        if partial:
            dataset = {
                "input_ids": tokens["input_ids"][-partial:] + [pad_token_id]*(model_max_length-partial),
                "attention_mask": tokens["attention_mask"][-partial:] + [0]*(model_max_length-partial),
            }
            offset_mappings = (tokens["offset_mapping"][-partial:] + [(tokens["offset_mapping"][-1][1]+i, tokens["offset_mapping"][-1][1]+i+1) for i in range(model_max_length-partial)])
            yield dataset, offset_mappings

    @staticmethod
    def _should_expand_with_partial(
        cur_cui_id: int,
        cur_token_type: str,
        annotation: Annotation,
        annotations: List[Annotation],
    ) -> bool:
        return all([cur_cui_id == 1, cur_token_type == "sub", (annotation.start - annotations[-1].end) in [0, 1]])

    @staticmethod
    def _should_expand_with_whole(
        is_enabled: bool,
        annotation: Annotation,
        annotations: List[Annotation],
    ) -> bool:
        return all([is_enabled, annotation.label_id == annotations[-1].label_id, (annotation.start - annotations[-1].end) in [0, 1]])
