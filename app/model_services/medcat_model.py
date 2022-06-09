import io
import os
import json
import logging
import asyncio
import pandas as pd
import threading
import gc
import re
import mlflow

from functools import partial
from copy import deepcopy
from contextlib import redirect_stdout
from typing import Dict, List, Iterable, TextIO, Union, Callable
from medcat.cat import CAT
from model_services.base import AbstractModelService
from domain import ModelCard
from config import Settings


logger = logging.getLogger(__name__)


class MedCATModel(AbstractModelService):

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._model_pack_dir = os.path.join(os.path.dirname(__file__), "..", "model")
        self._retrained_models_dir = os.path.join(self._model_pack_dir, "retrained")
        self._model_pack_path = os.path.join(self._model_pack_dir, config.BASE_MODEL_FILE)
        self._meta_cat_config_dict = {"general": {"device": config.DEVICE}}
        self._model = self.load_model(self._model_pack_path, meta_cat_config_dict=self._meta_cat_config_dict)
        self._training_lock = threading.Lock()
        self._training_in_progress = False
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    @property
    def model(self) -> CAT:
        return self._model

    @model.setter
    def model(self, m) -> None:
        self._model = m

    @model.deleter
    def model(self) -> None:
        del self._model

    @staticmethod
    def load_model(model_file_path: str, *args, **kwargs) -> CAT:
        cat = CAT.load_model_pack(model_file_path, *args, **kwargs)
        logger.info(f"Model pack loaded from {model_file_path}")
        return cat

    def info(self) -> ModelCard:
        return ModelCard(model_description=f"{self._config.CODE_TYPE.upper()} model",
                         model_type="MedCAT",
                         api_version="0.0.1",
                         model_card=self.model.get_model_card(as_dict=True))

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

    def train_supervised(self,
                         data_file: TextIO,
                         epochs: int,
                         redeploy: bool,
                         skip_save_model: bool,
                         job_name: str) -> bool:
        training_type = "supervised"
        training_params = {
            "data_path": data_file.name,
            "nepochs": epochs,
            "print_stats": 1,
        }
        return self._start_training(self._train_supervised, training_type, training_params, data_file, redeploy,
                                    skip_save_model, job_name)

    def train_unsupervised(self,
                           texts: Iterable[str],
                           epochs: int,
                           redeploy: bool,
                           skip_save_model: bool,
                           job_name: str) -> bool:
        training_type = "unsupervised"
        training_params = {
            "nepochs": epochs,
        }
        return self._start_training(self._train_unsupervised, training_type, training_params, texts, redeploy,
                                    skip_save_model, job_name)

    def train_meta_models(self, annotations: Dict) -> None:
        pass

    @staticmethod
    def _train_supervised(medcat_model: "MedCATModel",
                          training_params: Dict,
                          data_file: TextIO,
                          redeploy: bool,
                          skip_save_model: bool) -> None:
        try:
            logger.info("Cloning the current model")
            model = deepcopy(medcat_model.model)
            logger.info("Starting supervised training")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                model.train_supervised(**training_params)
            metric_lines = re.findall(r"Epoch: (\d), Prec: (\d+\.\d+), Rec: (\d+\.\d+), F1: (\d+\.\d+)", buffer.getvalue(), re.IGNORECASE)
            for step, metric in enumerate(metric_lines):
                metrics = {
                    "epoch": int(metric[0]),
                    "precision": float(metric[1]),
                    "recall": float(metric[2]),
                    "f1": float(metric[3]),
                }
                mlflow.log_metrics(metrics, step)
            if not skip_save_model:
                model_pack_path = MedCATModel._save_model(medcat_model, model)
                mlflow.log_artifact(model_pack_path)
            else:
                logger.info("Skipped saving on the retrained model")
            data = json.load(open(data_file.name))
            logger.debug(model._print_stats(data, extra_cui_filter=True))
            if redeploy:
                MedCATModel._deploy_model(medcat_model, model, skip_save_model)
            else:
                del model
                logger.info("Skipped deployment on the retrained model")
            data_file.close()
            gc.collect()
            logger.info("Supervised training finished")
            logger.debug(medcat_model.model.get_model_card())
        except Exception as e:
            logger.error("Supervised training failed")
            logger.error(e, exc_info=True, stack_info=True)
            raise e
        finally:
            with medcat_model._training_lock:
                medcat_model._training_in_progress = False
            mlflow.end_run()

    @staticmethod
    def _train_unsupervised(medcat_model: "MedCATModel",
                            training_params: Dict,
                            texts: Iterable[str],
                            redeploy: bool,
                            skip_save_model: bool) -> None:
        try:
            logger.info("Cloning the running model...")
            model = deepcopy(medcat_model.model)
            logger.info("Starting unsupervised training...")
            model.train(texts, **training_params)
            if not skip_save_model:
                model_pack_path = MedCATModel._save_model(medcat_model, model)
                mlflow.log_artifact(model_pack_path)
            else:
                logger.info("Skipped saving on the retrained model")
            if redeploy:
                MedCATModel._deploy_model(medcat_model, model, skip_save_model)
            else:
                del model
                logger.info("Skipped deployment on the retrained model")
            gc.collect()
            logger.info("Unsupervised training finished")
            logger.debug(medcat_model.model.get_model_card())
        except Exception as e:
            logger.error("Unsupervised training failed")
            logger.error(e, exc_info=True, stack_info=True)
            raise e
        finally:
            with medcat_model._training_lock:
                medcat_model._training_in_progress = False
            mlflow.end_run()

    @staticmethod
    def _save_model(service: "MedCATModel",
                    model: CAT) -> str:
        logger.info(f"Saving retrained model to {service._retrained_models_dir}...")
        model_pack_name = model.create_model_pack(service._retrained_models_dir, "model")
        model_pack_path = f"{os.path.join(service._retrained_models_dir, model_pack_name)}.zip"
        logger.info(f"Retrained model saved to {model_pack_path}")
        return model_pack_path

    @staticmethod
    def _deploy_model(service: "MedCATModel",
                      model: CAT,
                      skip_save_model: bool):
        if skip_save_model:
            model._versioning()
        del service.model
        service.model = model
        logger.info("Retrained model deployed")

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
    def _get_experiment_id(experiment_name: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return mlflow.create_experiment(name=experiment_name) if experiment is None else experiment.experiment_id

    def _get_records_from_doc(self, doc: Dict) -> Dict:
        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end"])
        else:
            if self._config.CODE_TYPE == "icd10":
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
            elif self._config.CODE_TYPE == "snomed":
                df.rename(columns={"pretty_name": "label_name", "cui": "label_id"}, inplace=True)
            else:
                logger.error(f'CODE_TYPE {self._config.CODE_TYPE} is not supported')
                raise ValueError(f"Unknown coding type: {self._config.CODE_TYPE}")
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict("records")
        return records

    def _start_training(self,
                        runner: Callable,
                        training_type: str,
                        training_params: Dict,
                        dataset: Union[Iterable[str], TextIO],
                        redeploy: bool,
                        skip_save_model: bool,
                        job_name: str) -> bool:
        with self._training_lock:
            if self._training_in_progress:
                return False
            else:
                loop = asyncio.get_event_loop()
                experiment_name = f"{self.info().model_description} {training_type}".replace(" ", "_")
                experiment_id = self._get_experiment_id(experiment_name)
                logger.info(f"Starting training job: {job_name} with experiment ID: {experiment_id}")
                mlflow.start_run(experiment_id=experiment_id, run_name=job_name)
                mlflow.log_params(training_params)
                self._training_in_progress = True
                loop.run_in_executor(None, partial(runner, self, training_params, dataset, redeploy, skip_save_model))
                return True
