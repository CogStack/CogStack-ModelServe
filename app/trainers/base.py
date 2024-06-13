import asyncio
import threading
import shutil
import os
import logging
import datasets

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TextIO, Callable, Dict, Optional, Any, List, final
from config import Settings
from management.tracker_client import TrackerClient
from data import doc_dataset, anno_dataset
from domain import TrainingType

logger = logging.getLogger("cms")


class TrainerCommon(object):

    def __init__(self, config: Settings, model_name: str) -> None:
        self._config = config
        self._model_name = model_name
        self._training_lock = threading.Lock()
        self._training_in_progress = False
        self._tracker_client = TrackerClient(self._config.MLFLOW_TRACKING_URI)
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1)

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name

    @final
    def start_training(self,
                       run: Callable,
                       training_type: str,
                       training_params: Dict,
                       data_file: TextIO,
                       log_frequency: int,
                       training_id: str,
                       input_file_name: str,
                       raw_data_files: Optional[List[TextIO]] = None,
                       description: Optional[str] = None,
                       synchronised: bool = False) -> bool:
        with self._training_lock:
            if self._training_in_progress:
                return False
            else:
                loop = asyncio.get_event_loop()
                experiment_id, run_id = self._tracker_client.start_tracking(
                    model_name=self._model_name,
                    input_file_name=input_file_name,
                    base_model_original=self._config.BASE_MODEL_FULL_PATH,
                    training_type=training_type,
                    training_params=training_params,
                    run_name=training_id,
                    log_frequency=log_frequency,
                    description=description,
                )
                if self._config.SKIP_SAVE_TRAINING_DATASET == "false":
                    if raw_data_files is not None:
                        for odf in raw_data_files:
                            self._tracker_client.save_raw_artifact(odf.name, self._model_name)

                    # This may not be needed once Dataset can be stored as an artifact
                    self._tracker_client.save_processed_artifact(data_file.name, self._model_name)

                    if training_type == TrainingType.UNSUPERVISED.value:
                        dataset = datasets.load_dataset(doc_dataset.__file__,
                                                        data_files={"documents": data_file.name},
                                                        split="train",
                                                        cache_dir=None)
                        self._tracker_client.save_train_dataset(dataset)
                    elif training_type == TrainingType.SUPERVISED.value:
                        dataset = datasets.load_dataset(anno_dataset.__file__,
                                                        data_files={"annotations": data_file.name},
                                                        split="train",
                                                        cache_dir=None)
                        self._tracker_client.save_train_dataset(dataset)
                    else:
                        raise ValueError(f"Unknown training type: {training_type}")

                logger.info(f"Starting training job: {training_id} with experiment ID: {experiment_id}")
                self._training_in_progress = True
                training_task = asyncio.ensure_future(loop.run_in_executor(self._executor,
                                                                           partial(run, self, training_params, data_file, log_frequency, run_id, description)))

        if synchronised:
            loop.run_until_complete(training_task)

        return True

    @staticmethod
    def _make_model_file_copy(model_file_path: str, run_id: str) -> str:
        copied_model_pack_path = model_file_path.replace(".zip", f"_copied_{run_id}.zip")
        shutil.copy2(model_file_path, copied_model_pack_path)
        if os.path.exists(copied_model_pack_path.replace(".zip", "")):
            shutil.rmtree(copied_model_pack_path.replace(".zip", ""))
        return copied_model_pack_path


class SupervisedTrainer(ABC, TrainerCommon):

    def __init__(self, config: Settings, model_name: str) -> None:
        super().__init__(config, model_name)

    def train(self,
              data_file: TextIO,
              epochs: int,
              log_frequency: int,
              training_id: str,
              input_file_name: str,
              raw_data_files: Optional[List[TextIO]] = None,
              description: Optional[str] = None,
              synchronised: bool = False,
              **hyperparams: Dict[str, Any]) -> bool:
        training_type = TrainingType.SUPERVISED.value
        training_params = {
            "data_path": data_file.name,
            "nepochs": epochs,
            **hyperparams,
        }
        return self.start_training(run=self.run,
                                   training_type=training_type,
                                   training_params=training_params,
                                   data_file=data_file,
                                   log_frequency=log_frequency,
                                   training_id=training_id,
                                   input_file_name=input_file_name,
                                   raw_data_files=raw_data_files,
                                   description=description,
                                   synchronised=synchronised)

    @staticmethod
    @abstractmethod
    def run(trainer: "SupervisedTrainer",
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str,
            description: Optional[str] = None) -> None:
        raise NotImplementedError


class UnsupervisedTrainer(ABC, TrainerCommon):

    def __init__(self, config: Settings, model_name: str) -> None:
        super().__init__(config, model_name)

    def train(self,
              data_file: TextIO,
              epochs: int,
              log_frequency: int,
              training_id: str,
              input_file_name: str,
              raw_data_files: Optional[List[TextIO]] = None,
              description: Optional[str] = None,
              synchronised: bool = False,
              **hyperparams: Dict[str, Any]) -> bool:
        training_type = TrainingType.UNSUPERVISED.value
        training_params = {
            "nepochs": epochs,
            **hyperparams,
        }
        return self.start_training(run=self.run,
                                   training_type=training_type,
                                   training_params=training_params,
                                   data_file=data_file,
                                   log_frequency=log_frequency,
                                   training_id=training_id,
                                   input_file_name=input_file_name,
                                   raw_data_files=raw_data_files,
                                   description=description,
                                   synchronised=synchronised)

    @staticmethod
    @abstractmethod
    def run(trainer: "UnsupervisedTrainer",
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str,
            description: Optional[str] = None) -> None:
        raise NotImplementedError
