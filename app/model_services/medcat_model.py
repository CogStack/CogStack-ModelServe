import os
import logging
import pandas as pd

from multiprocessing import cpu_count
from typing import Dict, List, Optional, TextIO, Tuple, Any, Set, Union
from medcat.cat import CAT
from medcat.data.entities import Entities, OnlyCUIEntities
from app import __version__ as app_version
from app.model_services.base import AbstractModelService
from app.trainers.medcat_trainer import MedcatSupervisedTrainer, MedcatUnsupervisedTrainer
from app.trainers.metacat_trainer import MetacatTrainer
from app.domain import ModelCard, Annotation
from app.config import Settings
from app.utils import (
    get_settings,
    TYPE_ID_TO_NAME_PATCH,
    non_default_device_is_available,
    unpack_model_data_package,
    get_model_data_package_base_name,
    load_pydantic_object_from_dict,
)
from app.exception import ConfigurationException

logger = logging.getLogger("cms")


class MedCATModel(AbstractModelService):
    """A model service for MedCAT models."""

    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        enable_trainer: Optional[bool] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
    ) -> None:
        """
        Initialises the MedCAT model service with specified configurations.

        Args:
            config (Settings): The configuration for the model service.
            model_parent_dir (Optional[str]): The directory where the model package is stored. Defaults to None.
            enable_trainer (Optional[bool]): The flag to enable or disable trainers. Defaults to None.
            model_name (Optional[str]): The name of the model. Defaults to None.
            base_model_file (Optional[str]): The model package file name. Defaults to None.
        """
        super().__init__(config)
        self._model: Optional[CAT] = None
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_pack_path = os.path.join(self._model_parent_dir, base_model_file or config.BASE_MODEL_FILE)
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._whitelisted_tuis = set([tui.strip() for tui in config.TYPE_UNIQUE_ID_WHITELIST.split(",")])
        self.model_name = model_name or "MedCAT model"

    @property
    def model(self) -> Optional[CAT]:
        """Getter for the MedCAT model."""

        return self._model

    @model.setter
    def model(self, model: CAT) -> None:
        """Setter for the MedCAT model."""

        self._model = model

    @model.deleter
    def model(self) -> None:
        """Deleter for the MedCAT model."""

        del self._model

    @property
    def api_version(self) -> str:
        """Getter for the API version of the model service."""

        # APP version is used although each model service could have its own API versioning
        return app_version

    @classmethod
    def from_model(cls, model: CAT) -> "MedCATModel":
        """
        Creates a model service from a MedCAT model instance.

        Args:
            model (CAT): A MedCAT model instance.

        Returns:
            MedCATModel: A MedCAT model service.
        """
        model_service = cls(get_settings(), enable_trainer=False)
        model_service.model = model
        return model_service

    @staticmethod
    def load_model(model_file_path: str, *args: Tuple, **kwargs: Dict[str, Any]) -> CAT:
        """
        Loads a MedCAT model from a model package file.

        Args:
            model_file_path (str): The path to the model package file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            CAT: The loaded MedCAT model instance.

        Raises:
            ConfigurationException: If the model package archive format is not supported.
        """

        model_path = os.path.join(os.path.dirname(model_file_path), get_model_data_package_base_name(model_file_path))
        if unpack_model_data_package(model_file_path, model_path):
            cat = CAT.load_model_pack(model_file_path.replace(".tar.gz", ".zip"), **kwargs)
            logger.info("Model package loaded from %s", os.path.normpath(model_file_path))
            return cat
        else:
            raise ConfigurationException("Model package archive format is not supported")

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the MedCAT model based on the configuration.

        Args:
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.
        """

        if hasattr(self, "_model") and isinstance(self._model, CAT):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            if non_default_device_is_available(get_settings().DEVICE):
                self._model = self.load_model(self._model_pack_path)
                for addon in self._model.get_addons():
                    addon.config.general.device = get_settings().DEVICE # type: ignore
                self._model.config.general.device = get_settings().DEVICE   # type: ignore
            else:
                self._model = self.load_model(self._model_pack_path)
            self._set_tuis_filtering()
            if self._enable_trainer:
                self._supervised_trainer = MedcatSupervisedTrainer(self)
                self._unsupervised_trainer = MedcatUnsupervisedTrainer(self)
                self._metacat_trainer = MetacatTrainer(self)
            self._model.config.general.map_to_other_ontologies = [  # type: ignore
                tui.strip() for tui in self._config.MEDCAT2_MAPPED_ONTOLOGIES.split(",")
            ]

    def info(self) -> ModelCard:
        """
        Retrieves information about the model and should be implemented by subclasses.

        Returns:
            ModelCard: A card containing information about the model.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def annotate(self, text: str) -> List[Annotation]:
        """
        Annotates the given text with extracted named entities.

        Args:
            text (str): The input text to be annotated.

        Returns:
            List[Annotation]: A list of annotations containing the extracted named entities.
        """

        assert self.model is not None, "Model is not initialised"
        doc = self.model.get_entities(text)
        return [load_pydantic_object_from_dict(Annotation, record) for record in self.get_records_from_doc(doc)]

    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        """
        Annotates texts in batches and returns a list of lists of annotations.

        Args:
            texts (List[str]): The list of texts to be annotated.

        Returns:
            List[List[Annotation]]: A list where each element is a list of annotations containing the extracted named entities.
        """

        batch_size_chars = 500000

        assert self.model is not None, "Model is not initialised"
        docs = {i: result for i, (_, result) in enumerate(self.model.get_entities_multi_texts(
            texts,
            batch_size_chars=batch_size_chars,
            n_process=max(int(cpu_count() / 2), 1),
        ))}
        docs = dict(sorted(docs.items(), key=lambda x: x[0]))
        annotations_list = []
        for _, doc in docs.items():
            annotations_list.append([
                load_pydantic_object_from_dict(Annotation, record) for record in self.get_records_from_doc(doc)
            ])
        return annotations_list

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
            log_frequency (int): The number of processed documents after which training metrics will be logged.
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
            log_frequency (int): The number of processed documents after which training metrics will be logged.
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

    def train_metacat(
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
        Initiates metacat training on the model.

        Args:
            data_file (TextIO): The file containing a JSON list of texts.
            epochs (int): The number of training epochs.
            log_frequency (int): The number of processed documents after which training metrics will be logged.
            training_id (str): A unique identifier for the training process.
            input_file_name (str): The name of the input file to be logged.
            raw_data_files (Optional[List[TextIO]]): Additional raw data files to be logged. Defaults to None.
            description (Optional[str]): The description of the training or change logs. Defaults to empty.
            synchronised (bool): Whether to wait for the training to complete.
            **hyperparams (Dict[str, Any]): Additional hyperparameters for training.

        Returns:
            Tuple[bool, str, str]:  A tuple with the first element indicating success or failure.

        Raises:
            ConfigurationException: If the metacat trainer is not enabled.
        """

        if self._metacat_trainer is None:
            raise ConfigurationException("The metacat trainer is not enabled")
        return self._metacat_trainer.train(
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

    def get_records_from_doc(self, doc: Union[Dict, Entities, OnlyCUIEntities]) -> List[Dict]:
        """
        Extracts and formats entity records from a document dictionary.

        Args:
            doc (Union[Dict, Entities, OnlyCUIEntities]): The document dictionary containing extracted named entities.

        Returns:
            List[Dict]: A list of formatted entity records.
        """

        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        else:
            for idx, row in df.iterrows():
                if "athena_ids" in row and row["athena_ids"]:
                    df.loc[idx, "athena_ids"] = [athena_id["code"] for athena_id in row["athena_ids"]]
            if self._config.INCLUDE_SPAN_TEXT == "true":
                df.rename(columns={"pretty_name": "label_name", "cui": "label_id", "source_value": "text", "type_ids": "categories", "acc": "accuracy", "athena_ids": "athena_ids"}, inplace=True)
            else:
                df.rename(columns={"pretty_name": "label_name", "cui": "label_id", "type_ids": "categories", "acc": "accuracy", "athena_ids": "athena_ids"}, inplace=True)
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict("records")
        return records

    @staticmethod
    def _retrieve_meta_annotations(df: pd.DataFrame) -> pd.DataFrame:
        meta_annotations = []
        for i, r in df.iterrows():
            meta_dict = {}
            for k, v in r.meta_anns.items():
                meta_dict[k] = v["value"]
            meta_annotations.append(meta_dict)

        df["new_meta_anns"] = meta_annotations
        return pd.concat([df.drop(["new_meta_anns"], axis=1), df["new_meta_anns"].apply(pd.Series)], axis=1)


    def _set_tuis_filtering(self) -> None:
        # this patching may not be needed after the base 1.4.x model is fixed in the future
        assert self._model is not None, "Model is not initialised"
        if self._model.cdb.addl_info.get("type_id2name", {}) == {}:
            self._model.cdb.addl_info["type_id2name"] = TYPE_ID_TO_NAME_PATCH

        type_id2info = self._model.cdb.type_id2info
        model_tuis = set(type_id2info.keys())
        if self._whitelisted_tuis == {""}:
            return
        assert self._whitelisted_tuis.issubset(model_tuis), f"Unrecognisable Type Unique Identifier(s): {self._whitelisted_tuis - model_tuis}"
        whitelisted_cuis: Set = set()
        for tui in self._whitelisted_tuis:
            type_info = type_id2info.get(tui)
            if type_info is None:
                continue
            whitelisted_cuis.update(type_info.cuis)
        self._model.config.components.linking.filters.cuis = whitelisted_cuis
