import os
import logging
import asyncio
import threading
import gc
import shutil
import ijson
import pandas as pd

from functools import partial
from contextlib import redirect_stdout
from typing import Dict, List, TextIO, Callable, Optional
from medcat.cat import CAT
from model_services.base import AbstractModelService
from domain import ModelCard
from config import Settings
from processors.data_batcher import mini_batch
from management.tracker_client import TrackerClient
from management.log_captor import LogCaptor
from management.model_manager import ModelManager
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MedCATModel(AbstractModelService):

    def __init__(self, config: Settings, model_parent_dir: Optional[str] = None) -> None:
        super().__init__(config)
        model_parent_dir = model_parent_dir or os.path.join(os.path.dirname(__file__), "..", "model")
        self._retrained_models_dir = os.path.join(model_parent_dir, "retrained")
        self._model_pack_path = os.path.join(model_parent_dir, config.BASE_MODEL_FILE)
        self._meta_cat_config_dict = {"general": {"device": config.DEVICE}}
        self._training_lock = threading.Lock()
        self._training_in_progress = False
        self._tracker_client = TrackerClient(config.MLFLOW_TRACKING_URI)
        self._pyfunc_model = ModelManager(type(self), config)
        self.executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1)
        self._model: CAT

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
        return "MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"

    @staticmethod
    def load_model(model_file_path: str, *args, **kwargs) -> CAT:
        cat = CAT.load_model_pack(model_file_path, *args, **kwargs)
        logger.info(f"Model pack loaded from {os.path.normpath(model_file_path)}")
        return cat

    def init_model(self) -> None:
        if hasattr(self, "_model") and isinstance(self._model, CAT):
            logger.warning("Model service can be initialised only once")
        else:
            self._model = self.load_model(self._model_pack_path, meta_cat_config_dict=self._meta_cat_config_dict)

    def info(self) -> ModelCard:
        return ModelCard(model_description="SNOMED MedCAT model",
                         model_type="MedCAT",
                         api_version=self.api_version,
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
                         log_frequency: int,
                         training_id: str,
                         input_file_name: str) -> bool:
        training_type = "supervised"
        training_params = {
            "data_path": data_file.name,
            "nepochs": epochs,
        }
        return self._start_training(self._train_supervised, training_type, training_params, data_file, log_frequency, training_id, input_file_name)

    def train_unsupervised(self,
                           data_file: TextIO,
                           epochs: int,
                           log_frequency: int,
                           training_id: str,
                           input_file_name: str) -> bool:
        training_type = "unsupervised"
        training_params = {
            "nepochs": epochs,
        }
        return self._start_training(self._train_unsupervised, training_type, training_params, data_file, log_frequency, training_id, input_file_name)

    def train_meta_models(self, annotations: Dict) -> None:
        pass

    @staticmethod
    def _train_supervised(medcat_model: "MedCATModel",
                          training_params: Dict,
                          data_file: TextIO,
                          log_frequency: int) -> None:
        training_params.update({"print_stats": log_frequency})
        model_pack_path = None
        cdb_config_path = None
        copied_model_pack_path = None
        redeploy = medcat_model._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = medcat_model._config.SKIP_SAVE_MODEL == "true"
        try:
            logger.info("Loading a new model copy for training...")
            copied_model_pack_path = medcat_model._make_model_file_copy(medcat_model._model_pack_path)
            model = medcat_model.load_model(copied_model_pack_path,
                                            meta_cat_config_dict=medcat_model._meta_cat_config_dict)
            logger.info("Performing supervised training...")
            with redirect_stdout(LogCaptor(medcat_model._training_tracker.glean_and_log_metrics)):
                fps, fns, tps, p, r, f1, cc, _ = model.train_supervised(**training_params)
            del _
            gc.collect()
            class_id = 0
            cuis = []
            f1 = {c: f for c, f in sorted(f1.items(), key=lambda item: item[0])}
            for cui, f1_val in f1.items():
                metric = {
                    "per_concept_fp": fps.get(cui, 0),
                    "per_concept_fn": fns.get(cui, 0),
                    "per_concept_tp": tps.get(cui, 0),
                    "per_concept_counts": cc.get(cui, 0),
                    "per_concept_precision": p[cui],
                    "per_concept_recall": r[cui],
                    "per_concept_f1": f1_val,
                }
                medcat_model._tracker_client.send_model_stats(metric, class_id)
                cuis.append(cui)
                class_id += 1
            medcat_model._tracker_client.log_classes(cuis)
            if not skip_save_model:
                model_pack_path = MedCATModel._save_model(medcat_model, model)
                cdb_config_path = model_pack_path.replace(".zip", "_config.json")
                model.cdb.config.save(cdb_config_path)
                medcat_model._tracker_client.save_model(model_pack_path,
                                                        medcat_model.info().model_description,
                                                        medcat_model._pyfunc_model)
                medcat_model._tracker_client.save_model_artifact(cdb_config_path,
                                                                 medcat_model.info().model_description)
            else:
                logger.info("Skipped saving on the retrained model")
            if redeploy:
                MedCATModel._deploy_model(medcat_model, model, skip_save_model)
            else:
                del model
                gc.collect()
                logger.info("Skipped deployment on the retrained model")
            logger.info("Supervised training finished")
            medcat_model._tracker_client.end_with_success()
        except Exception as e:
            logger.error("Supervised training failed")
            logger.error(e, exc_info=True, stack_info=True)
            medcat_model._tracker_client.log_exception(e)
            medcat_model._tracker_client.end_with_failure()
        finally:
            data_file.close()
            with medcat_model._training_lock:
                medcat_model._training_in_progress = False
            medcat_model._housekeep_file(model_pack_path)
            medcat_model._housekeep_file(copied_model_pack_path)
            if cdb_config_path:
                os.remove(cdb_config_path)

    @staticmethod
    def _train_unsupervised(medcat_model: "MedCATModel",
                            training_params: Dict,
                            data_file: TextIO,
                            log_frequency: int) -> None:
        model_pack_path = None
        cdb_config_path = None
        copied_model_pack_path = None
        redeploy = medcat_model._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = medcat_model._config.SKIP_SAVE_MODEL == "true"
        data_file.seek(0)
        texts = ijson.items(data_file, "item")
        try:
            logger.info("Loading a new model copy for training...")
            copied_model_pack_path = medcat_model._make_model_file_copy(medcat_model._model_pack_path)
            model = medcat_model.load_model(copied_model_pack_path,
                                            meta_cat_config_dict=medcat_model._meta_cat_config_dict)
            logger.info("Performing unsupervised training...")
            step = 0
            medcat_model._tracker_client.send_model_stats(model.cdb._make_stats(), step)
            for batch in mini_batch(texts, batch_size=log_frequency):
                step += 1
                model.train(batch, **training_params)
                medcat_model._tracker_client.send_model_stats(model.cdb._make_stats(), step)
            if not skip_save_model:
                model_pack_path = MedCATModel._save_model(medcat_model, model)
                cdb_config_path = model_pack_path.replace(".zip", "_config.json")
                model.cdb.config.save(cdb_config_path)
                medcat_model._tracker_client.save_model(model_pack_path,
                                                        medcat_model.info().model_description,
                                                        medcat_model._pyfunc_model)
                medcat_model._tracker_client.save_model_artifact(cdb_config_path,
                                                                 medcat_model.info().model_description)
            else:
                logger.info("Skipped saving on the retrained model")
            if redeploy:
                MedCATModel._deploy_model(medcat_model, model, skip_save_model)
            else:
                del model
                gc.collect()
                logger.info("Skipped deployment on the retrained model")
            logger.info("Unsupervised training finished")
            medcat_model._tracker_client.end_with_success()
        except Exception as e:
            logger.error("Unsupervised training failed")
            logger.error(e, exc_info=True, stack_info=True)
            medcat_model._tracker_client.log_exception(e)
            medcat_model._tracker_client.end_with_failure()
        finally:
            data_file.close()
            with medcat_model._training_lock:
                medcat_model._training_in_progress = False
            medcat_model._housekeep_file(model_pack_path)
            medcat_model._housekeep_file(copied_model_pack_path)
            if cdb_config_path:
                os.remove(cdb_config_path)

    @staticmethod
    def _make_model_file_copy(model_file_path: str):
        copied_model_pack_path = model_file_path.replace(".zip", "_copied.zip")
        shutil.copy2(model_file_path, copied_model_pack_path)
        if os.path.exists(copied_model_pack_path.replace(".zip", "")):
            shutil.rmtree(copied_model_pack_path.replace(".zip", ""))
        return copied_model_pack_path

    @staticmethod
    def _housekeep_file(file_path: Optional[str]):
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("model pack housekept")
        if file_path and os.path.exists(file_path.replace(".zip", "")):
            shutil.rmtree(file_path.replace(".zip", ""))
            logger.debug("Unpacked model directory housekept")

    @staticmethod
    def _save_model(service: "MedCATModel",
                    model: CAT) -> str:
        logger.info(f"Saving retrained model to {service._retrained_models_dir}...")
        model_pack_name = model.create_model_pack(service._retrained_models_dir, "model")
        model_pack_path = f"{os.path.join(service._retrained_models_dir, model_pack_name)}.zip"
        logger.debug(f"Retrained model saved to {model_pack_path}")
        return model_pack_path

    @staticmethod
    def _deploy_model(service: "MedCATModel",
                      model: CAT,
                      skip_save_model: bool):
        if skip_save_model:
            model._versioning()
        del service.model
        gc.collect()
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

    def _get_records_from_doc(self, doc: Dict) -> Dict:
        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end"])
        else:
            df.rename(columns={"pretty_name": "label_name", "cui": "label_id"}, inplace=True)
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict("records")
        return records

    def _start_training(self,
                        runner: Callable,
                        training_type: str,
                        training_params: Dict,
                        dataset: TextIO,
                        log_frequency: int,
                        training_id: str,
                        input_file_name: str) -> bool:
        with self._training_lock:
            if self._training_in_progress:
                return False
            else:
                loop = asyncio.get_event_loop()
                experiment_id, run_id = self._tracker_client.start_tracking(
                    model_name=self.info().model_description,
                    input_file_name=input_file_name,
                    base_model_original=self._config.BASE_MODEL_FULL_PATH,
                    training_type=training_type,
                    training_params=training_params,
                    run_name=training_id,
                    log_frequency=log_frequency,
                )
                if self._config.SKIP_SAVE_TRAINING_DATASET == "false":
                    self._tracker_client.save_model_artifact(dataset.name, self.info().model_description)
                logger.info(f"Starting training job: {training_id} with experiment ID: {experiment_id}")
                self._training_in_progress = True
                asyncio.ensure_future(loop.run_in_executor(self.executor, partial(runner, self, training_params, dataset, log_frequency)))
                return True
