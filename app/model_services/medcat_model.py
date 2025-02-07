import os
import logging
import pandas as pd

from multiprocessing import cpu_count
from typing import Dict, List, Optional, TextIO, Tuple, Any
from medcat.cat import CAT
from model_services.base import AbstractModelService
from trainers.medcat_trainer import MedcatSupervisedTrainer, MedcatUnsupervisedTrainer
from trainers.metacat_trainer import MetacatTrainer
from domain import ModelCard
from config import Settings
from utils import get_settings, TYPE_ID_TO_NAME_PATCH, non_default_device_is_available, unpack_model_data_package
from exception import ConfigurationException

logger = logging.getLogger("cms")


class MedCATModel(AbstractModelService):

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None,
                 base_model_file: Optional[str] = None) -> None:
        super().__init__(config)
        self._model: CAT = None
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_pack_path = os.path.join(self._model_parent_dir, base_model_file or config.BASE_MODEL_FILE)
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._whitelisted_tuis = set([tui.strip() for tui in config.TYPE_UNIQUE_ID_WHITELIST.split(",")])
        self.model_name = model_name or "MedCAT model"

    @property
    def model(self) -> CAT:
        return self._model

    @model.setter
    def model(self, model: CAT) -> None:
        self._model = model

    @model.deleter
    def model(self) -> None:
        del self._model

    @property
    def api_version(self) -> str:
        return "0.0.1"

    @classmethod
    def from_model(cls, model: CAT) -> "MedCATModel":
        model_service = cls(get_settings(), enable_trainer=False)
        model_service.model = model
        return model_service

    @staticmethod
    def load_model(model_file_path: str, *args: Tuple, **kwargs: Dict[str, Any]) -> CAT:
        model_path = os.path.join(os.path.dirname(model_file_path), os.path.basename(model_file_path).split(".")[0])
        if unpack_model_data_package(model_file_path, model_path):
            cat = CAT.load_model_pack(model_file_path.replace(".tar.gz", ".zip"), *args, **kwargs)
            logger.info("Model package loaded from %s", os.path.normpath(model_file_path))
            return cat
        else:
            raise ConfigurationException("Model package archive format is not supported")


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

    def init_model(self) -> None:
        if hasattr(self, "_model") and isinstance(self._model, CAT):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            if non_default_device_is_available(get_settings().DEVICE):
                self._model = self.load_model(self._model_pack_path, meta_cat_config_dict={"general": {"device": get_settings().DEVICE}})
                self._model.config.general["device"] = get_settings().DEVICE
            else:
                self._model = self.load_model(self._model_pack_path)
            self._set_tuis_filtering()
            if self._enable_trainer:
                self._supervised_trainer = MedcatSupervisedTrainer(self)
                self._unsupervised_trainer = MedcatUnsupervisedTrainer(self)
                self._metacat_trainer = MetacatTrainer(self)

    def info(self) -> ModelCard:
        raise NotImplementedError

    def annotate(self, text: str) -> Dict:
        doc = self.model.get_entities(text,
                                      addl_info=["cui2icd10", "cui2ontologies", "cui2snomed", "cui2athena_ids"])
        return self.get_records_from_doc(doc)

    def batch_annotate(self, texts: List[str]) -> List[Dict]:
        batch_size_chars = 500000

        docs = self.model.multiprocessing_batch_char_size(
            self._data_iterator(texts),
            batch_size_chars=batch_size_chars,
            nproc=max(int(cpu_count() / 2), 1),
            addl_info=["cui2icd10", "cui2ontologies", "cui2snomed", "cui2athena_ids"]
        )
        annotations_list = []
        for _, doc in docs.items():
            annotations_list.append(self.get_records_from_doc(doc))
        return annotations_list

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

    def train_metacat(self,
                      data_file: TextIO,
                      epochs: int,
                      log_frequency: int,
                      training_id: str,
                      input_file_name: str,
                      raw_data_files: Optional[List[TextIO]] = None,
                      description: Optional[str] = None,
                      synchronised: bool = False,
                      **hyperparams: Dict[str, Any]) -> Tuple[bool, str, str]:
        if self._metacat_trainer is None:
            raise ConfigurationException("The metacat trainer is not enabled")
        return self._metacat_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name, raw_data_files, description, synchronised, **hyperparams)

    def get_records_from_doc(self, doc: Dict) -> Dict:
        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        else:
            for idx, row in df.iterrows():
                if "athena_ids" in row and row["athena_ids"]:
                    df.loc[idx, "athena_ids"] = [athena_id["code"] for athena_id in row["athena_ids"]]
            if self._config.INCLUDE_SPAN_TEXT == "true":
                df.rename(columns={"pretty_name": "label_name", "cui": "label_id", "source_value": "text", "types": "categories", "acc": "accuracy", "athena_ids": "athena_ids"}, inplace=True)
            else:
                df.rename(columns={"pretty_name": "label_name", "cui": "label_id", "types": "categories", "acc": "accuracy", "athena_ids": "athena_ids"}, inplace=True)
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict("records")
        return records

    def _set_tuis_filtering(self) -> None:
        # this patching may not be needed after the base 1.4.x model is fixed in the future
        if self._model.cdb.addl_info.get("type_id2name", {}) == {}:
            self._model.cdb.addl_info["type_id2name"] = TYPE_ID_TO_NAME_PATCH

        tuis2cuis = self._model.cdb.addl_info.get("type_id2cuis")
        model_tuis = set(tuis2cuis.keys())
        if self._whitelisted_tuis == {""}:
            return
        assert self._whitelisted_tuis.issubset(model_tuis), f"Unrecognisable Type Unique Identifier(s): {self._whitelisted_tuis - model_tuis}"
        whitelisted_cuis = set()
        for tui in self._whitelisted_tuis:
            whitelisted_cuis.update(tuis2cuis.get(tui, {}))
        self._model.cdb.config.linking.filters = {"cuis": whitelisted_cuis}
