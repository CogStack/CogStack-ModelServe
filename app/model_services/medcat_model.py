import os
import json
import logging
import asyncio
import pandas as pd
import threading

from functools import partial
from typing import Dict, List, Iterable, TextIO, Union, Callable
from medcat.cat import CAT
from model_services.base import AbstractModelService
from domain import ModelCard
from config import Settings

logger = logging.getLogger(__name__)


class MedCATModel(AbstractModelService):

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.model_pack_dir = os.path.join(os.path.dirname(__file__), "..", "model")
        self.model_pack_path = os.path.join(self.model_pack_dir, config.BASE_MODEL_FILE)
        self.meta_cat_config_dict = {"general": {"device": config.DEVICE}}
        self.model = self.load_model(self.model_pack_path, meta_cat_config_dict=self.meta_cat_config_dict)
        self.training_lock = threading.Lock()
        self.training_in_progress = False

    def info(self) -> Dict:
        return ModelCard(model_description=f"{self.config.CODE_TYPE.upper()} model",
                         model_type="MedCAT",
                         api_version="0.0.1",
                         model_card=self.model.get_model_card(as_dict=True))

    @staticmethod
    def load_model(model_file_path: str, *args, **kwargs) -> CAT:
        cat = CAT.load_model_pack(model_file_path, *args, **kwargs)
        logger.info(f"Model pack loaded from {model_file_path}")
        return cat

    def annotate(self, text: str) -> Dict:
        doc = self.model.get_entities(text)
        return self._get_records_from_doc(doc)

    def batch_annotate(self, texts: List[str]) -> List[Dict]:
        batch_size_chars = 500000

        docs = self.model.multiprocessing(self._data_iterator(texts),
                                          batch_size_chars=batch_size_chars,
                                          nproc=2,
                                          addl_info=["cui2icd10", "cui2ontologies", "cui2snomed"])
        annotations_list = []
        for _, doc in docs.items():
            annotations_list.append(self._get_records_from_doc(doc))
        return annotations_list

    def train_supervised(self, data_file: TextIO, redeploy: bool, skip_save_model: bool) -> bool:
        return self._start_training(self._train_supervised, data_file, redeploy, skip_save_model)

    def train_unsupervised(self, texts: Iterable[str], redeploy: bool, skip_save_model: bool) -> bool:
        return self._start_training(self._train_unsupervised, texts, redeploy, skip_save_model)

    def train_meta_models(self, annotations: Dict) -> None:
        pass

    def _start_training(self,
                        runner: Callable,
                        dataset: Union[Iterable[str], TextIO],
                        redeploy: bool,
                        skip_save_model: bool) -> bool:
        loop = asyncio.get_event_loop()
        with self.training_lock:
            if self.training_in_progress:
                return False
            else:
                self.training_in_progress = True
        loop.run_in_executor(None, partial(runner, self, dataset, redeploy, skip_save_model))
        return True

    @staticmethod
    def _train_supervised(medcat_model: "MedCATModel",
                          data_file: TextIO,
                          redeploy: bool,
                          skip_save_model: bool) -> None:
        try:
            training_params = {
                "data_path": data_file.name,
                "reset_cui_count": False,
                "nepochs": 1,
                "print_stats": True,
                "use_filters": False,

            }
            logger.info("Cloning the current model")
            model = medcat_model.load_model(medcat_model.model_pack_path,
                                            meta_cat_config_dict=medcat_model.meta_cat_config_dict)
            logger.info("Starting supervised training")
            model.train_supervised(**training_params)
            logger.info("Supervised training finished")
            model._versioning()
            logger.info(model.get_model_card())
            data = json.load(open(data_file.name))
            logger.debug(model._print_stats(data, extra_cui_filter=True))
            data_file.close()
            MedCATModel._perform_post_training_actions(medcat_model, model, redeploy, skip_save_model)
        finally:
            with medcat_model.training_lock:
                medcat_model.training_in_progress = False

    @staticmethod
    def _train_unsupervised(medcat_model: "MedCATModel",
                            texts: Iterable[str],
                            redeploy: bool,
                            skip_save_model: bool) -> None:
        try:
            training_params = {
                "nepochs": 1,
                "progress_print": 1000,
            }
            logger.info("Cloning the running model...")
            model = medcat_model.load_model(medcat_model.model_pack_path,
                                            meta_cat_config_dict=medcat_model.meta_cat_config_dict)
            logger.info("Starting unsupervised training...")
            model.train(texts, **training_params)
            logger.info("Unsupervised training finished")
            model._versioning()
            logger.info(model.get_model_card())
            MedCATModel._perform_post_training_actions(medcat_model, model, redeploy, skip_save_model)
        finally:
            with medcat_model.training_lock:
                medcat_model.training_in_progress = False

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

    @staticmethod
    def _perform_post_training_actions(current_model: "MedCATModel",
                                       trained_model: CAT,
                                       redeploy: bool,
                                       skip_save_model: bool):
        if not skip_save_model:
            logger.info(f"Saving retrained model to {current_model.model_pack_dir}...")
            model_pack_name = trained_model.create_model_pack(current_model.model_pack_dir, "model")
            logger.info(f"Retrained model saved to {os.path.join(current_model.model_pack_dir, model_pack_name)}")
        if redeploy:
            del current_model.model
            current_model.model = trained_model
            logger.info("Retrained model deployed")

    def _get_records_from_doc(self, doc: Dict) -> Dict:
        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end"])
        else:
            if self.config.CODE_TYPE == "icd10":
                output = pd.DataFrame()
                for _, row in df.iterrows():
                    if "icd10" not in row:
                        logger.error("No mapped ICD-10 code found in the record")
                    if row["icd10"]:
                        for icd10 in row["icd10"]:
                            output_row = row.copy()
                            if isinstance(icd10, str):
                                output_row["icd10"] = icd10
                            else:
                                output_row["icd10"] = icd10["code"]
                                output_row["pretty_name"] = icd10["name"]
                            output = output.append(output_row, ignore_index=True)
                df = output
                df.rename(columns={"pretty_name": "label_name", "icd10": "label_id"}, inplace=True)
            elif self.config.CODE_TYPE == "snomed":
                df.rename(columns={"pretty_name": "label_name", "cui": "label_id"}, inplace=True)
            else:
                logger.error(f'CODE_TYPE {self.config.CODE_TYPE} is not supported')
                raise ValueError(f"Unknown coding type: {self.config.CODE_TYPE}")
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict("records")
        return records
