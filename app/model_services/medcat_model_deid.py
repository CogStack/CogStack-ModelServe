import logging
import inspect
import threading
import torch
from typing import Dict, List, TextIO, Tuple, Optional, Any, final, Callable, cast
from functools import partial
from transformers import pipeline
from medcat.cat import CAT
from medcat.components.types import CoreComponentType
from app import __version__ as app_version
from app.config import Settings
from app.model_services.medcat_model import MedCATModel
from app.trainers.medcat_deid_trainer import MedcatDeIdentificationSupervisedTrainer
from app.domain import ModelCard, ModelType, Annotation
from app.utils import non_default_device_is_available, get_hf_pipeline_device_id, load_pydantic_object_from_dict
from app.exception import ConfigurationException

logger = logging.getLogger("cms")


@final
class MedCATModelDeIdentification(MedCATModel):
    """A model service for MedCAT De-Identification (AnonCAT) models."""

    CHUNK_SIZE = 500
    LEFT_CONTEXT_WORDS = 5

    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        enable_trainer: Optional[bool] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
    ) -> None:
        """
        Initialises the MedCAT De-Identification (AnonCAT) model service with specified configurations.

        Args:
            config (Settings): The configuration for the model service.
            model_parent_dir (Optional[str]): The directory where the model package is stored. Defaults to None.
            enable_trainer (Optional[bool]): The flag to enable or disable trainers. Defaults to None.
            model_name (Optional[str]): The name of the model. Defaults to None.
            base_model_file (Optional[str]): The model package file name. Defaults to None.
        """

        super().__init__(config, model_parent_dir=model_parent_dir, enable_trainer=enable_trainer, model_name=model_name, base_model_file=base_model_file)
        self.model_name = model_name or "De-Identification MedCAT model"
        self._lock = threading.RLock()

    @property
    def api_version(self) -> str:
        """Getter for the API version of the model service."""

        # APP version is used although each model service could have its own API versioning
        return app_version

    def info(self) -> ModelCard:
        """
        Retrieves information about the MedCAT De-Identification (AnonCAT) model.

        Returns:
            ModelCard: A card containing information about the MedCAT De-Identification (AnonCAT) model.
        """

        assert self.model is not None, "Model is not initialised"
        model_card = self.model.get_model_card(as_dict=True)
        model_card["Basic CDB Stats"]["Average training examples per concept"] = 0
        return ModelCard(
            model_description=self.model_name,
            model_type=ModelType.ANONCAT,
            api_version=self.api_version,
            model_card=dict(model_card),
            labels={cui: info['preferred_name'] for cui, info in self.model.cdb.cui2info.items()},
        )

    def annotate(self, text: str) -> List[Annotation]:
        """
        Annotates the given text with extracted PII entities.

        Args:
            text (str): The input text to be annotated.

        Returns:
            List[Annotation]: A list of annotations containing the extracted PII entities.
        """

        assert self.model is not None, "Model is not initialised"
        doc = self.model.get_entities(text)
        if doc["entities"]:
            for _, entity in doc["entities"].items():
                entity["type_ids"] = ["PII"]

        records = self.get_records_from_doc({"entities": doc["entities"]})
        return [load_pydantic_object_from_dict(Annotation, record) for record in records]

    def annotate_with_local_chunking(self, text: str) -> List[Annotation]:
        """
        Annotates the given text with PII entities using custom chunking.

        Args:
            text (str): The input text to be annotated.

        Returns:
            List[Annotation]: A list of annotation containing the extracted PII entities.
        """

        assert self.model is not None, "Model is not initialised"
        tokenizer = self.model.pipe.get_component(CoreComponentType.ner)._component.tokenizer.hf_tokenizer # type: ignore
        leading_ws_len = len(text) - len(text.lstrip())
        text = text.lstrip()
        tokenized = self._with_lock(tokenizer, text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        offset_mapping = tokenized["offset_mapping"]
        chunk = []
        aggregated_entities = {}
        ent_key = 0
        processed_char_len = leading_ws_len

        for input_id, (start, end) in zip(input_ids, offset_mapping):
            chunk.append((input_id, (start, end)))
            if len(chunk) == MedCATModelDeIdentification.CHUNK_SIZE:
                last_token_start_idx = 0
                window_overlap_start_idx = 0
                number_of_seen_words = 0
                for i in range(MedCATModelDeIdentification.CHUNK_SIZE-1, -1, -1):
                    if " " in tokenizer.decode([chunk[i][0]], skip_special_tokens=True):
                        if last_token_start_idx == 0:
                            last_token_start_idx = i
                        if number_of_seen_words < MedCATModelDeIdentification.LEFT_CONTEXT_WORDS:
                            window_overlap_start_idx = i
                        else:
                            break
                        number_of_seen_words += 1
                c_text = text[chunk[:last_token_start_idx][0][1][0]:chunk[:last_token_start_idx][-1][1][1]]
                doc = self._with_lock(self.model.get_entities, c_text)
                doc["entities"] = {_id: entity for _id, entity in doc["entities"].items() if (entity["end"] + processed_char_len) < chunk[window_overlap_start_idx][1][0]}
                for entity in doc["entities"].values():
                    entity["start"] += processed_char_len
                    entity["end"] += processed_char_len
                    entity["type_ids"] = ["PII"]
                    aggregated_entities[ent_key] = entity
                    ent_key += 1
                processed_char_len = chunk[:window_overlap_start_idx][-1][1][1] + leading_ws_len + 1
                chunk = chunk[window_overlap_start_idx:]
        if chunk:
            c_text = text[chunk[0][1][0]:chunk[-1][1][1]]
            doc = self.model.get_entities(c_text)
            if doc["entities"]:
                for entity in doc["entities"].values():
                    entity["start"] += processed_char_len
                    entity["end"] += processed_char_len
                    entity["type_ids"] = ["PII"]
                    aggregated_entities[ent_key] = entity
                    ent_key += 1
            processed_char_len += len(c_text)

        assert processed_char_len == (len(text) + leading_ws_len), f"{len(text) + leading_ws_len - processed_char_len} characters were not processed:\n{text}"

        records = self.get_records_from_doc({"entities": aggregated_entities})
        return [load_pydantic_object_from_dict(Annotation, record) for record in records]

    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        """
        Annotates texts in batches and returns a list of lists of annotations.

        Args:
            texts (List[str]): The list of texts to be annotated.

        Returns:
            List[List[Annotation]]: A list where each element is a list of annotations containing the extracted PII entities.
        """

        annotations_list = []
        assert self.model is not None, "Model is not initialised"
        entities_list = self.model.get_entities_multi_texts(texts)
        for _, entities in entities_list:
            for _, entity in entities["entities"].items():
                entity = cast(Dict[str, Any], entity)
                entity["type_ids"] = ["PII"]
            annotations_list.append([
                load_pydantic_object_from_dict(Annotation, record) for record in self.get_records_from_doc(entities)
            ])

        return annotations_list

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the MedCAT De-Identification (AnonCAT) model based on the configuration.

        Args:
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.
        """

        if hasattr(self, "_model") and isinstance(self._model, CAT):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model = self.load_model(self._model_pack_path)
            ner = self._model.pipe.get_component(CoreComponentType.ner)._component  # type: ignore
            ner.tokenizer.hf_tokenizer._in_target_context_manager = getattr(ner.tokenizer.hf_tokenizer, "_in_target_context_manager", False)
            ner.tokenizer.hf_tokenizer.clean_up_tokenization_spaces = getattr(ner.tokenizer.hf_tokenizer, "clean_up_tokenization_spaces", None)
            ner.tokenizer.hf_tokenizer.split_special_tokens = getattr(ner.tokenizer.hf_tokenizer, "split_special_tokens", False)
            if non_default_device_is_available(self._config.DEVICE):
                ner.model.to(torch.device(self._config.DEVICE))
                ner.ner_pipe = pipeline(
                    model=ner.model,
                    framework="pt",
                    task="ner",
                    tokenizer=ner.tokenizer.hf_tokenizer,
                    device=get_hf_pipeline_device_id(self._config.DEVICE),
                    aggregation_strategy=self._config.HF_PIPELINE_AGGREGATION_STRATEGY,
                )
            else:
                if self._config.DEVICE != "default":
                    logger.warning("DEVICE is set to '%s' but it is not available. Using 'default' instead.", self._config.DEVICE)
            _save_pretrained = ner.model.save_pretrained
            if ("safe_serialization" in inspect.signature(_save_pretrained).parameters):
                ner.model.save_pretrained = partial(_save_pretrained, safe_serialization=(self._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"))
            if self._enable_trainer:
                self._supervised_trainer = MedcatDeIdentificationSupervisedTrainer(self)

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
            raise ConfigurationException("Trainers are not enabled")
        return self._supervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name, raw_data_files, description, synchronised, **hyperparams)

    def _with_lock(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        # Temporarily tackle https://github.com/huggingface/tokenizers/issues/537 but it reduces parallelism
        with self._lock:
            return func(*args, **kwargs)
