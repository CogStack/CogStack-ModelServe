import asyncio
import threading
import shutil
import os
import logging
import tempfile
import datasets

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, CancelledError
from functools import partial
from typing import TextIO, Callable, Dict, Tuple, Optional, Any, List, Union, final
from app.config import Settings
from app.management.tracker_client import TrackerClient
from app.data import doc_dataset, anno_dataset
from app.domain import TrainingType
from app.utils import get_model_data_package_extension

logger = logging.getLogger("cms")
logging.getLogger("asyncio").setLevel(logging.ERROR)


class TrainerCommon(object):
    """A base class for common training functionalities across all trainers."""

    def __init__(self, config: Settings, model_name: str) -> None:
        """
        Initialises the common trainer class.

        Args:
            config (Settings): Configuration for the trainer.
            model_name (str): The name of the model to be trained.
        """

        self._config = config
        self._model_name = model_name
        self._training_lock = threading.Lock()
        self._training_in_progress = False
        self._experiment_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._tracker_client = TrackerClient(self._config.MLFLOW_TRACKING_URI)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._cancel_event = threading.Event()

    @property
    def model_name(self) -> str:
        """Getter for the model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        """Setter for the model name."""
        self._model_name = model_name

    @property
    def experiment_id(self) -> str:
        """Getter for the experiment ID."""
        return self._experiment_id or ""

    @property
    def run_id(self) -> str:
        """Getter for the run ID."""
        return self._run_id or ""

    @property
    def tracker_client(self) -> TrackerClient:
        """Getter for the tracker client."""
        return self._tracker_client

    @final
    def start_training(
        self,
        run: Callable,
        training_type: str,
        training_params: Dict,
        data_file: Union[TextIO, tempfile.TemporaryDirectory],
        log_frequency: int,
        training_id: str,
        input_file_name: str,
        raw_data_files: Optional[List[TextIO]] = None,
        description: Optional[str] = None,
        synchronised: bool = False,
    ) -> Tuple[bool, str, str]:
        """
        Starts the training process.

        Args:
            run (Callable): The main training function to be executed.
            training_type (str): The type of training (e.g., supervised, unsupervised).
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (Union[TextIO, tempfile.TemporaryDirectory]): The file-like object or directory containing the training data.
            log_frequency: The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            training_id: A unique identifier for the training job.
            input_file_name: The name of the input file to be logged.
            raw_data_files: The optional list of raw data files to be saved as artifacts.
            description: The optional description of the training or change logs.
            synchronised: If True, this method will wait for the training to complete before returning.

        Returns:
            Tuple[bool, str, str]: A tuple containing a boolean indicating if the training was started successfully,
                 the experiment ID, and the run ID.
        """

        with self._training_lock:
            if self._training_in_progress:
                return False, self.experiment_id, self.run_id
            else:
                loop = asyncio.get_event_loop()
                self._experiment_id, self._run_id = self._tracker_client.start_tracking(
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

                    dataset = None
                    if training_type == TrainingType.UNSUPERVISED.value and isinstance(data_file, tempfile.TemporaryDirectory):
                        dataset = datasets.load_from_disk(data_file.name)
                        self._tracker_client.save_train_dataset(dataset)
                    elif training_type == TrainingType.UNSUPERVISED.value:
                        try:
                            dataset = datasets.load_dataset(
                                doc_dataset.__file__,
                                data_files={"documents": data_file.name},
                                split="train",
                                cache_dir=self._config.TRAINING_CACHE_DIR,
                                trust_remote_code=True,
                            )
                            self._tracker_client.save_train_dataset(dataset)
                        finally:
                            if dataset is not None:
                                dataset.cleanup_cache_files()
                    elif training_type == TrainingType.SUPERVISED.value:
                        try:
                            dataset = datasets.load_dataset(
                                anno_dataset.__file__,
                                data_files={"annotations": data_file.name},
                                split="train",
                                cache_dir=self._config.TRAINING_CACHE_DIR,
                                trust_remote_code=True,
                            )
                            self._tracker_client.save_train_dataset(dataset)
                        finally:
                            if dataset is not None:
                                dataset.cleanup_cache_files()
                    else:
                        raise ValueError(f"Unknown training type: {training_type}")

                logger.info("Starting training job: %s with experiment ID: %s", training_id, self.experiment_id)
                self._training_in_progress = True

        if not synchronised:
            asyncio.ensure_future(
                loop.run_in_executor(
                    self._executor,
                    partial(
                        run,
                        training_params,
                        data_file,
                        log_frequency,
                        self.run_id,
                        description,
                    ),
                )
            )
            return True, self.experiment_id, self.run_id
        else:
            training_task = self._executor.submit(
                partial(
                    run,
                    training_params,
                    data_file,
                    log_frequency,
                    self.run_id,
                    description,
                )
            )
            try:
                training_task.result()
                logger.info("Training task completed with training ID: %s", training_id)
                return True, self.experiment_id, self.run_id
            except CancelledError:
                logger.error("Training task cancelled with training ID: %s", training_id)
                return False, self.experiment_id, self.run_id
            except Exception as e:
                logger.error("Training task failed with training ID: %s and exception %s", training_id, e)
                return False, self.experiment_id, self.run_id

    @final
    def stop_training(self) -> bool:
        """
        Marks the current training process as cancelled.

        Returns:
            bool: True if the training is in progress and marked as cancelled successfully, False otherwise.
        """
        with self._training_lock:
            if self._training_in_progress:
                self._cancel_event.set()
                return True
            return False

    @staticmethod
    def _make_model_file_copy(model_file_path: str, run_id: str) -> str:
        model_pack_file_ext = get_model_data_package_extension(model_file_path)
        copied_model_pack_path = model_file_path.replace(
            model_pack_file_ext,
            f"_copied_{run_id}{model_pack_file_ext}",
        )
        shutil.copy2(model_file_path, copied_model_pack_path)
        if os.path.exists(copied_model_pack_path.replace(model_pack_file_ext, "")):
            shutil.rmtree(copied_model_pack_path.replace(model_pack_file_ext, ""))
        logger.debug("Model file copied to %s", copied_model_pack_path)
        return copied_model_pack_path

    @staticmethod
    def _housekeep_file(file_path: Optional[str]) -> None:
        if file_path:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug("model package housekept")
            model_pack_file_ext = get_model_data_package_extension(file_path)
            if file_path and os.path.exists(file_path.replace(model_pack_file_ext, "")):
                shutil.rmtree(file_path.replace(model_pack_file_ext, ""))
                logger.debug("Unpacked model directory housekept")

    def _clean_up_training_cache(self) -> None:
        for root, dirs, files in os.walk(self._config.TRAINING_CACHE_DIR, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.debug("Housekept file: %s", file_path)
                except Exception as e:
                    logger.error("Error occurred on deleting file: %s : %s", file_path, e)

            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    shutil.rmtree(dir_path)
                    logger.debug("Housekept directory: %s", dir_path)
                except Exception as e:
                    logger.error("Error occurred on deleting directory: %s : %s", dir_path, e)


class SupervisedTrainer(ABC, TrainerCommon):
    """A base class for supervised training."""

    def __init__(self, config: Settings, model_name: str) -> None:
        """
        Initialises the supervised trainer class.

        Args:
            config (Settings): Configuration for the supervised trainer.
            model_name (str): The name of the model to be trained.
        """
        super().__init__(config, model_name)

    def train(
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
        Starts the supervised training process.

        Args:
            data_file (TextIO): The file-like object containing the training data.
            epochs: The number of training epochs.
            log_frequency: The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            training_id: A unique identifier for the training job.
            input_file_name:  The name of the input file to be logged.
            raw_data_files: The optional list of raw data files to be saved as artifacts.
            description: The optional description of the training or change logs.
            synchronised: If True, this method will wait for the training to complete before returning.
            hyperparams: Additional hyperparameters for the training process.

        Returns:
            Tuple[bool, str, str]: A tuple containing a boolean indicating if the training was started successfully,
            the experiment ID, and the run ID.
        """

        training_type = TrainingType.SUPERVISED.value
        training_params = {
            "data_path": data_file.name,
            "nepochs": epochs,
            **hyperparams,
        }
        return self.start_training(
            run=self.run,
            training_type=training_type,
            training_params=training_params,
            data_file=data_file,
            log_frequency=log_frequency,
            training_id=training_id,
            input_file_name=input_file_name,
            raw_data_files=raw_data_files,
            description=description,
            synchronised=synchronised,
        )

    @abstractmethod
    def run(
        self,
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        The main training function to be implemented by a specific supervised trainer.

        Args:
            training_params: A dictionary containing parameters for the training.
            data_file: The file-like object containing the training data.
            log_frequency: The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id: The run ID of the training job.
            description: The optional description of the training or change logs.
        """

        raise NotImplementedError


class UnsupervisedTrainer(ABC, TrainerCommon):
    """A base class for unsupervised training."""

    def __init__(self, config: Settings, model_name: str) -> None:
        """
        Initialises the unsupervised trainer class.

        Args:
            config (Settings): Configuration for the unsupervised trainer.
            model_name (str): The name of the model to be trained.
        """

        super().__init__(config, model_name)

    def train(
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
        Starts the unsupervised training process.

        Args:
            data_file (TextIO): The file-like object containing the training data.
            epochs: The number of training epochs.
            log_frequency: The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            training_id: A unique identifier for the training job.
            input_file_name:  The name of the input file to be logged.
            raw_data_files: The optional list of raw data files to be saved as artifacts.
            description: The optional description of the training or change logs.
            synchronised: If True, this method will wait for the training to complete before returning.
            hyperparams: Additional hyperparameters for the training process.

        Returns:
            Tuple[bool, str, str]: A tuple containing a boolean indicating if the training was started successfully,
            the experiment ID, and the run ID.
        """

        training_type = TrainingType.UNSUPERVISED.value
        training_params = {
            "nepochs": epochs,
            **hyperparams,
        }
        return self.start_training(
            run=self.run,
            training_type=training_type,
            training_params=training_params,
            data_file=data_file,
            log_frequency=log_frequency,
            training_id=training_id,
            input_file_name=input_file_name,
            raw_data_files=raw_data_files,
            description=description,
            synchronised=synchronised,
        )

    @abstractmethod
    def run(
        self,
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        The main training function to be implemented by a specific unsupervised trainer.

        Args:
            training_params: A dictionary containing parameters for the training.
            data_file: The file-like object containing the training data.
            log_frequency: The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id: The run ID of the training job.
            description: The optional description of the training or change logs.
        """

        raise NotImplementedError
