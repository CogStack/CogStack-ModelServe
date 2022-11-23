import os
import logging
import pandas as pd

from typing import Dict, List, Optional, TextIO
from medcat.cat import CAT
from model_services.base import AbstractModelService
from trainer.medcat_trainer import MedcatSupervisedTrainer, MedcatUnsupervisedTrainer
from domain import ModelCard
from config import Settings
from exception import ConfigurationException

logger = logging.getLogger(__name__)


class MedCATModel(AbstractModelService):

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None) -> None:
        self._model: CAT = None
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.join(os.path.dirname(__file__), "..", "model")
        self._model_pack_path = os.path.join(self._model_parent_dir, config.BASE_MODEL_FILE)
        self._meta_cat_config_dict = {"general": {"device": config.DEVICE}}
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._supervised_trainer = None
        self._unsupervised_trainer = None

    @property
    def model(self) -> CAT:
        return self._model

    @model.setter
    def model(self, m) -> None:
        self._model = m

    @model.deleter
    def model(self) -> None:
        del self._model

    @property
    def model_name(self) -> str:
        return "SNOMED MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"

    @classmethod
    def of(cls, model: CAT) -> "MedCATModel":
        model_service = cls(Settings(), enable_trainer=False)
        model_service.model = model
        return model_service

    @staticmethod
    def load_model(model_file_path: str, *args, **kwargs) -> CAT:
        cat = CAT.load_model_pack(model_file_path, *args, **kwargs)
        logger.info(f"Model pack loaded from {os.path.normpath(model_file_path)}")
        return cat

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
            self._model = self.load_model(self._model_pack_path, meta_cat_config_dict=self._meta_cat_config_dict)
            if self._enable_trainer:
                self._supervised_trainer = MedcatSupervisedTrainer(self)
                self._unsupervised_trainer = MedcatUnsupervisedTrainer(self)

    def info(self) -> ModelCard:
        return ModelCard(model_description=self.model_name,
                         model_type="MedCAT",
                         api_version=self.api_version,
                         model_card=self.model.get_model_card(as_dict=True))

    def annotate(self, text: str) -> Dict:
        doc = self.model.get_entities(text)
        return self.get_records_from_doc(doc)

    def batch_annotate(self, texts: List[str]) -> List[Dict]:
        batch_size_chars = 500000

        docs = self.model.multiprocessing(self._data_iterator(texts),
                                          batch_size_chars=batch_size_chars,
                                          nproc=2,
                                          addl_info=["cui2icd10", "cui2ontologies", "cui2snomed"])
        annotations_list = []
        for _, doc in docs.items():
            annotations_list.append(self.get_records_from_doc(doc))
        return annotations_list

    def train_supervised(self,
                         data_file: TextIO,
                         epochs: int,
                         log_frequency: int,
                         training_id: str,
                         input_file_name: str) -> bool:
        if self._supervised_trainer is None:
            raise ConfigurationException("Trainers are not enabled")
        return self._supervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name)

    def train_unsupervised(self,
                           data_file: TextIO,
                           epochs: int,
                           log_frequency: int,
                           training_id: str,
                           input_file_name: str) -> bool:
        if self._unsupervised_trainer is None:
            raise ConfigurationException("Trainers are not enabled")
        return self._unsupervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name)

    def get_records_from_doc(self, doc: Dict) -> Dict:
        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end"])
        else:
            df.rename(columns={"pretty_name": "label_name", "cui": "label_id"}, inplace=True)
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict("records")
        return records
