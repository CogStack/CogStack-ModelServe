import asyncio
import threading
import shutil
import os
import logging

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TextIO, Callable, Dict, Optional
from config import Settings
from management.tracker_client import TrackerClient

logger = logging.getLogger(__name__)


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

    @staticmethod
    def get_experiment_name(model_name: str, training_type: Optional[str] = "") -> str:
        return f"{model_name} {training_type}".replace(" ", "_") if training_type else model_name.replace(" ", "_")

    def start_training(self,
                       run: Callable,
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
                    model_name=self._model_name,
                    input_file_name=input_file_name,
                    base_model_original=self._config.BASE_MODEL_FULL_PATH,
                    training_type=training_type,
                    training_params=training_params,
                    run_name=training_id,
                    log_frequency=log_frequency,
                )
                if self._config.SKIP_SAVE_TRAINING_DATASET == "false":
                    self._tracker_client.save_model_artifact(dataset.name, self._model_name)
                logger.info(f"Starting training job: {training_id} with experiment ID: {experiment_id}")
                self._training_in_progress = True
                asyncio.ensure_future(loop.run_in_executor(self._executor,
                                                           partial(run, self, training_params, dataset, log_frequency, run_id)))
                return True

    @staticmethod
    def _make_model_file_copy(model_file_path: str) -> str:
        copied_model_pack_path = model_file_path.replace(".zip", "_copied.zip")
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
              input_file_name: str) -> bool:
        training_type = "supervised"
        training_params = {
            "data_path": data_file.name,
            "nepochs": epochs,
        }
        return self.start_training(self.run, training_type, training_params, data_file, log_frequency,
                                   training_id, input_file_name)

    @staticmethod
    @abstractmethod
    def run(trainer: "SupervisedTrainer",
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str) -> None:
        raise NotImplementedError


class UnsupervisedTrainer(ABC, TrainerCommon):

    def __init__(self, config: Settings, model_name: str) -> None:
        super().__init__(config, model_name)

    def train(self,
              data_file: TextIO,
              epochs: int,
              log_frequency: int,
              training_id: str,
              input_file_name: str) -> bool:
        training_type = "unsupervised"
        training_params = {
            "nepochs": epochs,
        }
        return self.start_training(self.run, training_type, training_params, data_file, log_frequency,
                                   training_id, input_file_name)

    @staticmethod
    @abstractmethod
    def run(trainer: "UnsupervisedTrainer",
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str) -> None:
        raise NotImplementedError
