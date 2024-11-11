import os
import logging
import zipfile
import torch
import pandas as pd

from functools import partial
from typing import Dict, List, Optional, Tuple, Any, TextIO
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)
from transformers.pipelines import Pipeline

from exception import ConfigurationException
from model_services.base import AbstractModelService
from domain import ModelCard, ModelType
from config import Settings
from utils import get_settings


logger = logging.getLogger("cms")


class HuggingfaceTransformerModel(AbstractModelService):

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None,
                 base_model_file: Optional[str] = None) -> None:
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_pack_path = os.path.join(self._model_parent_dir, config.BASE_MODEL_FILE if base_model_file is None else base_model_file)
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._supervised_trainer = None
        self._unsupervised_trainer = None
        self._model: PreTrainedModel = None
        self._tokenizer: PreTrainedTokenizer = None
        self._ner_pipeline: Pipeline = None
        self._whitelisted_tuis = set([tui.strip() for tui in config.TYPE_UNIQUE_ID_WHITELIST.split(",")])
        self.model_name = model_name or "Huggingface Transformer model"

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
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        self._tokenizer = tokenizer

    @tokenizer.deleter
    def tokenizer(self) -> None:
        del self._tokenizer

    @property
    def api_version(self) -> str:
        return "0.0.1"

    @classmethod
    def from_model(cls, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> "HuggingfaceTransformerModel":
        model_service = cls(get_settings(), enable_trainer=False)
        model_service.model = model
        model_service.tokenizer = tokenizer
        return model_service

    @staticmethod
    def load_model(model_file_path: str, *args: Tuple, **kwargs: Dict[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_path = os.path.join(os.path.dirname(model_file_path), os.path.basename(model_file_path).split(".")[0])
        with zipfile.ZipFile(model_file_path, "r") as f:
            f.extractall(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, add_special_tokens=False)
        logger.info("Model pack loaded from %s", os.path.normpath(model_file_path))
        return model, tokenizer

    def init_model(self) -> None:
        if all([hasattr(self, "_model"),
                hasattr(self, "_tokenizer"),
                isinstance(self._model, PreTrainedModel),
                isinstance(self._tokenizer, PreTrainedTokenizer)]):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model, self._tokenizer = self.load_model(self._model_pack_path)
            _pipeline = partial(pipeline,
                                task="ner",
                                model=self._model,
                                tokenizer=self._tokenizer,
                                stride=10,
                                aggregation_strategy="simple")
            if (get_settings().DEVICE.startswith("cuda") and torch.cuda.is_available()) or \
               (get_settings().DEVICE.startswith("mps") and torch.backends.mps.is_available()) or \
               (get_settings().DEVICE.startswith("cpu")):
                self._ner_pipeline = _pipeline(device=get_settings().DEVICE)
            else:
                self._ner_pipeline = _pipeline()
            if self._enable_trainer:
                self._supervised_trainer = None
                self._unsupervised_trainer = None

    def info(self) -> ModelCard:
        return ModelCard(model_description=self.model_name,
                         model_type=ModelType.HF_TRANSFORMER,
                         api_version=self.api_version,
                         model_card=self._model.config.to_dict())

    def annotate(self, text: str) -> Dict:
        entities = self._ner_pipeline(text)
        df = pd.DataFrame(entities)

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        else:
            for _, row in df.iterrows():
                if "athena_ids" in row and row["athena_ids"]:
                    row["athena_ids"] = [athena_id["code"] for athena_id in row["athena_ids"]]
            df.rename(columns={"word": "label_name", "entity_group": "label_id", "score": "accuracy"}, inplace=True)
        records = df.to_dict("records")
        return records

    def batch_annotate(self, texts: List[str]) -> List[Dict]:
        raise NotImplementedError("Batch annotation is not yet implemented for Huggingface transformer models")

    def train_supervised(self,
                         data_file: TextIO,
                         epochs: int,
                         log_frequency: int,
                         training_id: str,
                         input_file_name: str,
                         raw_data_files: Optional[List[TextIO]] = None,
                         description: Optional[str] = None,
                         synchronised: bool = False,
                         **hyperparams: Dict[str, Any]) -> bool:
        if self._supervised_trainer is None:
            raise ConfigurationException("The supervised trainer is not enabled")
        return self._supervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name, raw_data_files, description, synchronised, **hyperparams)

    def train_unsupervised(self,
                           data_file: TextIO,
                           epochs: int,
                           log_frequency: int,
                           training_id: str,
                           input_file_name: str,
                           raw_data_files: Optional[List[TextIO]] = None,
                           description: Optional[str] = None,
                           synchronised: bool = False,
                           **hyperparams: Dict[str, Any]) -> bool:
        if self._unsupervised_trainer is None:
            raise ConfigurationException("The unsupervised trainer is not enabled")
        return self._unsupervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name, raw_data_files, description, synchronised, **hyperparams)
