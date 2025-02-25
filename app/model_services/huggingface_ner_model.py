import os
import logging
import pandas as pd

from functools import partial
from typing import Dict, List, Optional, Tuple, Any, TextIO
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
)
from transformers.pipelines import Pipeline
from app import __version__ as app_version
from app.exception import ConfigurationException
from app.model_services.base import AbstractModelService
from app.trainers.huggingface_ner_trainer import HuggingFaceNerUnsupervisedTrainer, HuggingFaceNerSupervisedTrainer
from app.domain import ModelCard, ModelType, Annotation
from app.config import Settings
from app.utils import get_settings, non_default_device_is_available, get_hf_pipeline_device_id, unpack_model_data_package

logger = logging.getLogger("cms")


class HuggingFaceNerModel(AbstractModelService):

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None,
                 base_model_file: Optional[str] = None) -> None:
        super().__init__(config)
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_pack_path = os.path.join(self._model_parent_dir, base_model_file or config.BASE_MODEL_FILE)
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._model: PreTrainedModel = None
        self._tokenizer: PreTrainedTokenizerBase = None
        self._ner_pipeline: Pipeline = None
        self._whitelisted_tuis = set([tui.strip() for tui in config.TYPE_UNIQUE_ID_WHITELIST.split(",")])
        self._multi_label_threshold = 0.5
        self.model_name = model_name or "Hugging Face NER model"

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
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @tokenizer.deleter
    def tokenizer(self) -> None:
        del self._tokenizer

    @property
    def api_version(self) -> str:
        # APP version is used although each model service could have its own API versioning
        return app_version

    @classmethod
    def from_model(cls, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> "HuggingFaceNerModel":
        model_service = cls(get_settings(), enable_trainer=False)
        model_service.model = model
        model_service.tokenizer = tokenizer
        _pipeline = partial(pipeline,
                            task="ner",
                            model=model_service.model,
                            tokenizer=model_service.tokenizer,
                            stride=10,
                            aggregation_strategy=get_settings().HF_PIPELINE_AGGREGATION_STRATEGY)
        if non_default_device_is_available(get_settings().DEVICE):
            model_service._ner_pipeline = _pipeline(device=get_hf_pipeline_device_id(get_settings().DEVICE))
        else:
            model_service._ner_pipeline = _pipeline()
        return model_service

    @staticmethod
    def load_model(model_file_path: str, *args: Tuple, **kwargs: Dict[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_path = os.path.join(os.path.dirname(model_file_path), os.path.basename(model_file_path).split(".")[0])
        if unpack_model_data_package(model_file_path, model_path):
            try:
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                          model_max_length=model.config.max_position_embeddings,
                                                          add_special_tokens=False, do_lower_case=False)
                logger.info("Model package loaded from %s", os.path.normpath(model_file_path))
                return model, tokenizer
            except ValueError as e:
                logger.error(e)
                raise ConfigurationException(f"Model package is not valid or not supported: {model_file_path}")
        else:
            raise ConfigurationException(f"Model package archive format is not supported: {model_file_path}")

    def init_model(self) -> None:
        if all([hasattr(self, "_model"),
                hasattr(self, "_tokenizer"),
                isinstance(self._model, PreTrainedModel),
                isinstance(self._tokenizer, PreTrainedTokenizerBase)]):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model, self._tokenizer = self.load_model(self._model_pack_path)
            _pipeline = partial(pipeline,
                                task="ner",
                                model=self._model,
                                tokenizer=self._tokenizer,
                                stride=10,
                                aggregation_strategy=self._config.HF_PIPELINE_AGGREGATION_STRATEGY)
            if non_default_device_is_available(get_settings().DEVICE):
                self._ner_pipeline = _pipeline(device=get_hf_pipeline_device_id(get_settings().DEVICE))
            else:
                self._ner_pipeline = _pipeline()
            if self._enable_trainer:
                self._supervised_trainer = HuggingFaceNerSupervisedTrainer(self)
                self._unsupervised_trainer = HuggingFaceNerUnsupervisedTrainer(self)

    def info(self) -> ModelCard:
        return ModelCard(model_description=self.model_name,
                         model_type=ModelType.HUGGINGFACE_NER,
                         api_version=self.api_version,
                         model_card=self._model.config.to_dict())

    def annotate(self, text: str) -> List[Annotation]:
        entities = self._ner_pipeline(text)
        df = pd.DataFrame(entities)

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        else:
            for idx, row in df.iterrows():
                df.loc[idx, "label_id"] = row["entity_group"]
            df.rename(columns={"entity_group": "label_name", "score": "accuracy"}, inplace=True)
            df = df[df["accuracy"] >= self._multi_label_threshold]
        records = df.to_dict("records")
        return [Annotation.parse_obj(record) for record in records]

    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        raise NotImplementedError("Batch annotation is not yet implemented for Hugging Face NER models")

    def train_supervised(self,
                         data_file: TextIO,
                         epochs: int,
                         log_frequency: int,
                         training_id: str,
                         input_file_name: str,
                         raw_data_files: Optional[List[TextIO]] = None,
                         description: Optional[str] = None,
                         synchronised: bool = False,
                         **hyperparams: Dict[str, Any]) -> Tuple[bool, str, str]:
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
                           **hyperparams: Dict[str, Any]) -> Tuple[bool, str, str]:
        if self._unsupervised_trainer is None:
            raise ConfigurationException("The unsupervised trainer is not enabled")
        return self._unsupervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name, raw_data_files, description, synchronised, **hyperparams)
