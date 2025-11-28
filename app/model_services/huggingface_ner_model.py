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
from app.domain import ModelCard, ModelType, Annotation, Device, TaggingScheme
from app.config import Settings
from app.utils import (
    get_settings,
    non_default_device_is_available,
    get_hf_pipeline_device_id,
    unpack_model_data_package,
    ensure_tensor_contiguity,
    get_model_data_package_base_name,
    load_pydantic_object_from_dict,
)
from app.processors.tagging import TagProcessor

logger = logging.getLogger("cms")


class HuggingFaceNerModel(AbstractModelService):
    """A model service for Hugging Face NER models."""

    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        enable_trainer: Optional[bool] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialises the HuggingFace NER model service with specified configurations.

        Args:
            config (Settings): The configuration for the model service.
            model_parent_dir (Optional[str]): The directory where the model package is stored. Defaults to None.
            enable_trainer (Optional[bool]): The flag to enable or disable trainers. Defaults to None.
            model_name (Optional[str]): The name of the model. Defaults to None.
            base_model_file (Optional[str]): The model package file name. Defaults to None.
            confidence_threshold (float): The threshold for the confidence score. Defaults to 0.7.
        """

        super().__init__(config)
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_pack_path = os.path.join(self._model_parent_dir, base_model_file or config.BASE_MODEL_FILE)
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._model: PreTrainedModel = None
        self._tokenizer: PreTrainedTokenizerBase = None
        self._ner_pipeline: Pipeline = None
        self._whitelisted_tuis = set([tui.strip() for tui in config.TYPE_UNIQUE_ID_WHITELIST.split(",")])
        self._confidence_threshold = confidence_threshold
        self.model_name = model_name or "HuggingFace NER model"

    @property
    def model(self) -> PreTrainedModel:
        """Getter for the HuggingFace pre-trained model."""

        return self._model

    @model.setter
    def model(self, model: PreTrainedModel) -> None:
        """Setter for the HuggingFace pre-trained model."""

        self._model = model

    @model.deleter
    def model(self) -> None:
        """Deleter for the HuggingFace pre-trained model."""

        del self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Getter for the HuggingFace tokenizer."""

        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Setter for the HuggingFace tokenizer."""

        self._tokenizer = tokenizer

    @tokenizer.deleter
    def tokenizer(self) -> None:
        """Deleter for the HuggingFace tokenizer."""

        del self._tokenizer

    @property
    def api_version(self) -> str:
        """Getter for the API version of the model service."""

        # APP version is used although each model service could have its own API versioning
        return app_version

    @classmethod
    def from_model(cls, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> "HuggingFaceNerModel":
        """
        Creates a model service from a provided HuggingFace pre-trained model and its tokenizer.

        Args:
            model (PreTrainedModel): The HuggingFace pre-trained model.
            tokenizer (PreTrainedTokenizerBase): The tokenizer for the HuggingFace pre-trained model.

        Returns:
            HuggingFaceNerModel: A HuggingFace NER model service.
        """

        _config = get_settings()
        model_service = cls(_config, enable_trainer=False)
        model_service.model = model
        model_service.tokenizer = tokenizer
        _pipeline = partial(
            pipeline,
            task="ner",
            model=model_service.model,
            tokenizer=model_service.tokenizer,
            stride=32,
            aggregation_strategy=_config.HF_PIPELINE_AGGREGATION_STRATEGY,
        )
        if non_default_device_is_available(_config.DEVICE):
            model_service._ner_pipeline = _pipeline(device=get_hf_pipeline_device_id(_config.DEVICE))
        else:
            model_service._ner_pipeline = _pipeline()
        return model_service

    @staticmethod
    def load_model(model_file_path: str, *args: Tuple, **kwargs: Dict[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads a pre-trained model and its tokenizer from a model package file.

        Args:
            model_file_path (str): The path to the model package file.
            *args (Tuple): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: A tuple containing the HuggingFace pre-trained model and its tokenizer.

        Raises:
            ConfigurationException: If the model package is not valid or not supported.
        """

        model_path = os.path.join(os.path.dirname(model_file_path), get_model_data_package_base_name(model_file_path))
        if unpack_model_data_package(model_file_path, model_path):
            try:
                if get_settings().DEVICE == Device.DEFAULT.value:
                    model = AutoModelForTokenClassification.from_pretrained(model_path, device_map="auto")
                else:
                    model = AutoModelForTokenClassification.from_pretrained(model_path)
                ensure_tensor_contiguity(model)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    model_max_length=model.config.max_position_embeddings,
                    do_lower_case=False,
                )
                logger.info("Model package loaded from %s", os.path.normpath(model_file_path))
                return model, tokenizer
            except ValueError as e:
                logger.error(e)
                raise ConfigurationException(f"Model package is not valid or not supported: {model_file_path}")
        else:
            raise ConfigurationException(f"Model package archive format is not supported: {model_file_path}")

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialises the HuggingFace model, its tokenizer and a NER pipeline based on the configuration.

        Args:
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.
        """

        if all([
            hasattr(self, "_model"),
            hasattr(self, "_tokenizer"),
            isinstance(self._model, PreTrainedModel),
            isinstance(self._tokenizer, PreTrainedTokenizerBase),
        ]):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model, self._tokenizer = self.load_model(self._model_pack_path)
            _pipeline = partial(
                pipeline,
                task="ner",
                model=self._model,
                tokenizer=self._tokenizer,
                stride=32,
                aggregation_strategy=self._config.HF_PIPELINE_AGGREGATION_STRATEGY,
            )
            if non_default_device_is_available(get_settings().DEVICE):
                self._ner_pipeline = _pipeline(device=get_hf_pipeline_device_id(get_settings().DEVICE))
            else:
                self._ner_pipeline = _pipeline()
            if self._enable_trainer:
                self._supervised_trainer = HuggingFaceNerSupervisedTrainer(self)
                self._unsupervised_trainer = HuggingFaceNerUnsupervisedTrainer(self)

    def info(self) -> ModelCard:
        """
        Retrieves a ModelCard containing information about the model.

        Returns:
            ModelCard: Information about the model.
        """
        return ModelCard(
            model_description=self.model_name,
            model_type=ModelType.HUGGINGFACE_NER,
            api_version=self.api_version,
            model_card=self._model.config.to_dict(),
        )

    def annotate(self, text: str) -> List[Annotation]:
        """
        Annotates the given text with extracted named entities.

        Args:
            text (str): The input text to be annotated.

        Returns:
            List[Annotation]: A list of annotations containing the extracted named entities.
        """

        if TaggingScheme(self._config.TRAINING_HF_TAGGING_SCHEME.lower()) == TaggingScheme.IOBES:
            entities = self._ner_pipeline(text, aggregation_strategy="none")
        else:
            entities = self._ner_pipeline(text)
        df = pd.DataFrame(entities)

        if df.empty:
            columns = ["label_name", "label_id", "start", "end", "accuracy"]
            df = pd.DataFrame(columns=(columns + ["text"]) if self._config.INCLUDE_SPAN_TEXT == "true" else columns)
        elif TaggingScheme(self._config.TRAINING_HF_TAGGING_SCHEME.lower()) == TaggingScheme.IOBES:
            aggregated_entities = TagProcessor.aggregate_bioes_predictions(
                df,
                text,
                self._config.INCLUDE_SPAN_TEXT == "true",
            )
            df = pd.DataFrame(aggregated_entities)
            if df.empty:
                columns = ["label_name", "label_id", "start", "end", "accuracy"]
                df = pd.DataFrame(
                    columns=(columns + ["text"]) if self._config.INCLUDE_SPAN_TEXT == "true" else columns
                )
            else:
                df = df[df["accuracy"] >= self._confidence_threshold]
        else:
            for idx, row in df.iterrows():
                df.loc[idx, "label_id"] = row["entity_group"]
                if self._config.INCLUDE_SPAN_TEXT == "true":
                    df.loc[idx, "text"] = text[row["start"]:row["end"]]

            df.rename(columns={"entity_group": "label_name", "score": "accuracy"}, inplace=True)
            df = df[df["accuracy"] >= self._confidence_threshold]

        records = df.to_dict("records")
        return [load_pydantic_object_from_dict(Annotation, record) for record in records]

    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        raise NotImplementedError("Batch annotation is not yet implemented for HuggingFace NER models")

    def train_supervised(
        self,
        data_file: TextIO,
        epochs: int,
        log_frequency: int,
        training_id: str,
        input_file_name: str,
        raw_data_files: Optional[List[TextIO]] = None,
        description: Optional[str] = None,
        synchronised: bool = False,
        **hyperparams: Dict[str, Any],
    ) -> Tuple[bool, str, str]:
        """
        Initiates supervised training on the model.

        Args:
            data_file (TextIO): The file containing the trainer export data.
            epochs (int): The number of training epochs.
            log_frequency (int): The number of epochs after which training metrics will be logged.
            training_id (str): A unique identifier for the training process.
            input_file_name (str): The name of the input file to be logged.
            raw_data_files (Optional[List[TextIO]]): Additional raw data files to be logged. Defaults to None.
            description (Optional[str]): The description of the training or change logs. Defaults to empty.
            synchronised (bool): Whether to wait for the training to complete.
            **hyperparams (Dict[str, Any]): Additional hyperparameters for training.

        Returns:
            Tuple[bool, str, str]: A tuple with the first element indicating success or failure.

        Raises:
            ConfigurationException: If the supervised trainer is not enabled.
        """
        if self._supervised_trainer is None:
            raise ConfigurationException("The supervised trainer is not enabled")
        return self._supervised_trainer.train(
            data_file,
            epochs,
            log_frequency,
            training_id,
            input_file_name,
            raw_data_files,
            description,
            synchronised,
            **hyperparams,
        )

    def train_unsupervised(
        self,
        data_file: TextIO,
        epochs: int,
        log_frequency: int,
        training_id: str,
        input_file_name: str,
        raw_data_files: Optional[List[TextIO]] = None,
        description: Optional[str] = None,
        synchronised: bool = False,
        **hyperparams: Dict[str, Any],
    ) -> Tuple[bool, str, str]:
        """
        Initiates unsupervised training on the model.

        Args:
            data_file (TextIO): The file containing a JSON list of texts.
            epochs (int): The number of training epochs.
            log_frequency (int): The number of epochs after which training metrics will be logged.
            training_id (str): A unique identifier for the training process.
            input_file_name (str): The name of the input file to be logged.
            raw_data_files (Optional[List[TextIO]]): Additional raw data files to be logged. Defaults to None.
            description (Optional[str]): The description of the training or change logs. Defaults to empty.
            synchronised (bool): Whether to wait for the training to complete.
            **hyperparams (Dict[str, Any]): Additional hyperparameters for training.

        Returns:
            Tuple[bool, str, str]:  A tuple with the first element indicating success or failure.

        Raises:
            ConfigurationException: If the unsupervised trainer is not enabled.
        """
        if self._unsupervised_trainer is None:
            raise ConfigurationException("The unsupervised trainer is not enabled")
        return self._unsupervised_trainer.train(
            data_file,
            epochs,
            log_frequency,
            training_id,
            input_file_name,
            raw_data_files,
            description,
            synchronised,
            **hyperparams,
        )
