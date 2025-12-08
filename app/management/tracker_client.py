import ast
import os
import socket
import mlflow
import tempfile
import json
import logging
import datasets
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union, final, Any
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME
from mlflow.entities import RunStatus, Metric
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo
from app.management.model_manager import ModelManager
from app.exception import StartTrainingException
from app.domain import TrainerBackend, TrackerBackend

logger = logging.getLogger("cms")
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.CRITICAL)


@final
class TrackerClient(object):
    """A client for tracking training and evaluation jobs."""

    def __init__(self, tracking_uri: str, backend: TrackerBackend = TrackerBackend.MLFLOW) -> None:
        """
        Initialises the TrackerClient with the tracking backend.

        Args:
            tracking_uri (str): The string representing the tracking URI of the backend.
            backend (TrackerBackend): The backend to use for tracking. Currently, only MLflow is supported.
        """

        if backend == TrackerBackend.MLFLOW:
            mlflow.set_tracking_uri(tracking_uri)
            self.mlflow_client = MlflowClient(tracking_uri)
        else:
            raise NotImplementedError(f"Tracking backend {backend} is not supported.")

    @staticmethod
    def start_tracking(
        model_name: str,
        input_file_name: str,
        base_model_original: str,
        training_type: str,
        training_params: Dict,
        run_name: str,
        log_frequency: int,
        description: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Starts tracking a new training or evaluation job.

        Args:
            model_name (str): The name of the model being trained or evaluated.
            input_file_name (str): The name of the input file containing data used for training or evaluation.
            base_model_original (str): The string representing the origin of the base model.
            training_type (str): The string representing the type of training.
            training_params (Dict): The dictionary of training parameters.
            run_name (str): The name of the run for identification purposes.
            log_frequency (int): The frequency of logging metrics.
            description (Optional[str]): Optional description of the run.

        Returns:
            Tuple: A tuple containing the experiment ID and the active run ID.
        """
        experiment_name = TrackerClient.get_experiment_name(model_name, training_type)
        experiment_id = TrackerClient._get_experiment_id(experiment_name)
        try:
            active_run = mlflow.start_run(
                experiment_id=experiment_id,
                tags={
                    MLFLOW_SOURCE_NAME: socket.gethostname(),
                    "mlflow.runName": run_name,
                    "mlflow.note.content": description or "",
                    "training.input_data.filename": input_file_name,
                    "training.base_model.origin": base_model_original,
                    "training.is.tracked": "True",
                    "training.metrics.log_frequency": str(log_frequency),
                },
            )
        except Exception:
            logger.exception("Cannot start a new training")
            raise StartTrainingException("Cannot start a new training")
        mlflow.log_params(training_params)
        return experiment_id, active_run.info.run_id

    @staticmethod
    def end_with_success() -> None:
        """Ends the current tracking with a status of 'FINISHED'."""

        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

    @staticmethod
    def end_with_failure() -> None:
        """Ends the current tracking with a status of 'FAILED'."""
        mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

    @staticmethod
    def end_with_interruption() -> None:
        """Ends the current tracking with a status of 'KILLED'."""
        mlflow.end_run(RunStatus.to_string(RunStatus.KILLED))

    @staticmethod
    def send_model_stats(stats: Dict, step: int) -> None:
        """
        Logs model statistics to the tracking backend.

        Args:
            stats (Dict): The dictionary containing training or evaluation statistics to log.
            step (int): The current step in the training or evaluation process.
        """

        metrics = {key.replace(" ", "_").lower(): val for key, val in stats.items() if isinstance(val, (int, float))}
        mlflow.log_metrics(metrics, step)

    @staticmethod
    def send_hf_metrics_logs(logs: Dict, step: int) -> None:
        """
        Logs Hugging Face metrics to the tracking backend.

        Args:
            logs (Dict): The dictionary containing Hugging Face metrics to log.
            step (int): The current step in the training or evaluation process.
        """

        mlflow.log_metrics(logs, step)

    @staticmethod
    def save_model_local(
        local_dir: str,
        filepath: str,
        model_manager: ModelManager,
    ) -> None:
        """
        Saves a model locally using the model manager.

        Args:
            local_dir (str): The local directory where the model will be saved.
            filepath (str): The artifact path to the model.
            model_manager (ModelManager): The instance of ModelManager used for model saving.
        """

        model_manager.save_model(local_dir, filepath)

    @staticmethod
    def save_model_artifact(filepath: str, model_name: str) -> None:
        """
        Saves a model artifact to the tracking backend.

        Args:
            filepath (str): The filepath of the model artifact.
            model_name (str): The name of the model, used to organise artifacts.
        """

        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(filepath, artifact_path=os.path.join(model_name, "artifacts"))

    @staticmethod
    def save_raw_artifact(filepath: str, model_name: str) -> None:
        """
        Saves a raw artifact to the tracking backend.

        Args:
            filepath (str): The filepath of the raw artifact.
            model_name (str): The name of the model, used to organise artifacts.
        """

        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(filepath, artifact_path=os.path.join(model_name, "artifacts", "raw"))

    @staticmethod
    def save_processed_artifact(filepath: str, model_name: str) -> None:
        """
        Saves a processed artifact to the tracking backend.

        Args:
            filepath (str): The filepath of the processed artifact.
            model_name (str): The name of the model, used to organise artifacts.
        """

        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(filepath, artifact_path=os.path.join(model_name, "artifacts", "processed"))

    @staticmethod
    def save_dataframe_as_csv(file_name: str, data_frame: pd.DataFrame, model_name: str) -> None:
        """
        Saves a Pandas DataFrame as a CSV artifact.

        Args:
            file_name (str): The name of the CSV file to save.
            data_frame (pd.DataFrame): The Pandas DataFrame to save as CSV.
            model_name (str): The name of the model, used to organise artifacts.
        """

        model_name = model_name.replace(" ", "_")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, file_name), "w") as f:
                data_frame.to_csv(f.name, index=False)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path=os.path.join(model_name, "stats"))

    @staticmethod
    def save_dict_as_json(file_name: str, data: Dict, model_name: str) -> None:
        """
        Saves a dictionary as a JSON artifact.

        Args:
            file_name (str): The name of the JSON file to save.
            data (Dict): The dictionary to save as JSON.
            model_name (str): The name of the model, used to organise artifacts.
        """

        model_name = model_name.replace(" ", "_")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, file_name), "w") as f:
                json.dump(data, f)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path=os.path.join(model_name, "stats"))

    @staticmethod
    def save_plot(file_name: str, model_name: str) -> None:
        """
        Saves a plot artifact.

        Args:
            file_name (str): The name of the plot file to save.
            model_name (str): The name of the model, used to organise artifacts.
        """

        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(file_name, artifact_path=os.path.join(model_name, "stats"))

    @staticmethod
    def save_table_dict(table_dict: Dict, model_name: str, file_name: str) -> None:
        """
        Saves a dictionary as a table artifact.

        Args:
            table_dict (Dict): The dictionary to save as a table.
            model_name (str): The name of the model, used to organise artifacts.
            file_name (str): The name of the artifact file to save the table under.
        """

        model_name = model_name.replace(" ", "_")
        mlflow.log_table(data=table_dict, artifact_file=os.path.join(model_name, "tables", file_name))

    @staticmethod
    def save_train_dataset(dataset: datasets.Dataset) -> None:
        """
        Saves a training dataset to the tracking backend.

        Args:
            dataset (datasets.Dataset): The Hugging Face dataset to save.
        """

        ds = mlflow.data.huggingface_dataset.from_huggingface(dataset)
        mlflow.log_input(ds, context="train")

    @staticmethod
    def log_exceptions(es: Union[Exception, List[Exception]]) -> None:
        """
        Logs exceptions thrown during training or evaluation as tags.

        Args:
            es (Union[Exception, List[Exception]]): A single exception or a list of exceptions to save.
        """

        if isinstance(es, list):
            for idx, e in enumerate(es):
                mlflow.set_tag(f"exception_{idx}", str(e))
        else:
            mlflow.set_tag("exception", str(es))

    @staticmethod
    def log_classes(classes: List[str]) -> None:
        """
        Logs the list of concepts used in training or evaluation as a tag.

        Args:
            classes (List[str]): A list of concept IDs.
        """

        mlflow.set_tag("training.entity.classes", str(classes)[:5000])

    @staticmethod
    def log_classes_and_names(class2names: Dict[str, str]) -> None:
        """
        Logs a dictionary mapping concepts to their names as a tag.

        Args:
            class2names (Dict[str, str]): The dictionary where keys are concept IDs and values are names of concepts.
        """

        mlflow.set_tag("training.entity.class2names", str(class2names)[:5000])

    @staticmethod
    def log_trainer_version(trainer_backend: TrainerBackend, trainer_version: str) -> None:
        """
        Logs the version and backend of the trainer as tags.

        Args:
            trainer_backend (TrainerBackend): The backend of the trainer.
            trainer_version (str): The semantic versioning string of the trainer.
        """

        mlflow.set_tags({
            "training.trainer.version": trainer_version,
            "training.trainer.backend": trainer_backend.value,
        })

    @staticmethod
    def log_trainer_mode(training: bool = True) -> None:
        """
        Logs the mode of the trainer (train or eval) as a tag.

        Args:
            training (bool): The boolean indicating if the mode is training (True) or evaluation (False).
        """

        mlflow.set_tag("training.trainer.mode", "train" if training else "eval")

    @staticmethod
    def log_document_size(num_of_docs: int) -> None:
        """
        Logs the number of documents as a tag.

        Args:
            num_of_docs (int): The number of documents used for training or evaluation.
        """

        mlflow.set_tag("training.document.size", str(num_of_docs))

    @staticmethod
    def log_model_config(config: Dict[str, str]) -> None:
        """
        Logs the model configuration as parameters.

        Args:
            config (Dict[str, str]): The dictionary containing the model configuration.
        """

        mlflow.log_params(config)

    @staticmethod
    def _set_model_version_tags(
        client: MlflowClient,
        model_name: str,
        version: str,
        model_type: str,
        validation_status: Optional[str] = None,
    ) -> None:
        """
        Sets standard tags on a model version for serving and discovery.

        Args:
            client (MlflowClient): The MLflow client to use for setting tags.
            model_name (str): The name of the registered model.
            version (str): The version of the model.
            model_type (str): The type of the model (e.g., "medcat_snomed").
            validation_status (Optional[str]): The status of the model validation (e.g., "pending").
        """
        try:
            client.set_model_version_tag(
                name=model_name, version=version, key="model_uri", value=f"models:/{model_name}/{version}"
            )
            client.set_model_version_tag(name=model_name, version=version, key="model_type", value=model_type)
            if validation_status is not None:
                client.set_model_version_tag(
                    name=model_name, version=version, key="validation_status", value=validation_status
            )
        except Exception:
            logger.warning("Failed to set tags on version %s of model %s", version, model_name)

    @staticmethod
    def log_model(
            model_name: str,
            model_path: str,
            model_manager: ModelManager,
            registered_model_name: Optional[str] = None,
    ) -> ModelInfo:
        """
        Logs the model with the specified name and local path to MLflow.

        Args:
            model_name (str): The name of the model to be logged.
            model_path (str): The artifact path to the model.
            model_manager (ModelManager): The instance of ModelManager used for model saving.
            registered_model_name (Optional[str]): The name of the registered model in MLflow.

        Returns:
            ModelInfo: The information instance of the logged model.
        """

        return mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=model_manager,
            artifacts={"model_path": model_path},
            signature=model_manager.model_signature,
            code_path=ModelManager.get_code_path_list(),
            pip_requirements=ModelManager.get_pip_requirements_from_file(),
            registered_model_name=registered_model_name,
        )

    @staticmethod
    def save_pretrained_model(
        model_name: str,
        model_path: str,
        model_manager: ModelManager,
        model_type: str,
        training_type: Optional[str] = "",
        run_name: Optional[str] = "",
        model_config: Optional[Dict] = None,
        model_metrics: Optional[List[Dict]] = None,
        model_tags: Optional[Dict] = None,
    ) -> None:
        """
        Saves a pretrained model to the tracking backend and associated metadata.

        Args:
            model_name (str): The name of the model.
            model_path (str): The path to the pretrained model.
            model_manager (ModelManager): The instance of ModelManager used for model saving.
            model_type (str): The type of the model (e.g., "medcat_snomed").
            training_type (Optional[str]): The type of training used for the model.
            run_name (Optional[str]): The name of the run for identification purposes.
            model_config (Optional[Dict]): The configuration of the model to save.
            model_metrics (Optional[List[Dict]]): The list of dictionaries containing model metrics to save.
            model_tags (Optional[Dict]): The dictionary of tags to set for the model.
        """

        experiment_name = TrackerClient.get_experiment_name(model_name, training_type)
        experiment_id = TrackerClient._get_experiment_id(experiment_name)
        active_run = mlflow.start_run(experiment_id=experiment_id)
        try:
            if model_config is not None:
                TrackerClient.log_model_config(model_config)
            if model_metrics is not None:
                for step, metric in enumerate(model_metrics):
                    TrackerClient.send_model_stats(metric, step)
            tags = {
                MLFLOW_SOURCE_NAME: socket.gethostname(),
                "mlflow.runName": run_name,
                "training.mlflow.run_id": active_run.info.run_id,
                "training.input_data.filename": "Unknown",
                "training.base_model.origin": model_path,
                "training.is.tracked": "False",
            }
            if model_tags is not None:
                tags = {**tags, **model_tags}
            mlflow.set_tags(tags)
            model_name = model_name.replace(" ", "_")
            TrackerClient.log_model(model_name, model_path, model_manager, model_name)
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'", order_by=["version_number DESC"])
            if versions:
                TrackerClient._set_model_version_tags(client, model_name, versions[0].version, model_type)
            TrackerClient.end_with_success()
        except KeyboardInterrupt:
            TrackerClient.end_with_interruption()
        except Exception as e:
            logger.exception("Failed to save the pretrained model")
            TrackerClient.log_exceptions(e)
            TrackerClient.end_with_failure()

    @staticmethod
    def get_experiment_name(model_name: str, training_type: Optional[str] = "") -> str:
        """
        Gets the experiment name based on the model name and the training type provided.

        Args:
            model_name (str): The name of the model.
            training_type (Optional[str]): The type of training used for the model.

        Returns:
            str: The formatted experiment name.
        """

        return f"{model_name} {training_type}".replace(" ", "_") if training_type else model_name.replace(" ", "_")

    @staticmethod
    def get_info_by_job_id(job_id: str) -> List[Dict]:
        """
        Gets information about a training or evaluation job by job ID.

        Args:
            job_id (str): The ID of the job to search for.

        Returns:
            List[Dict]: A list of dictionaries containing the job information.
        """

        try:
            runs = mlflow.search_runs(
                filter_string=f"tags.mlflow.runName = '{job_id}'",
                search_all_experiments=True,
                output_format="list",
            )
            if len(runs) == 0:
                logger.debug("Cannot find any runs with job ID '%s'", job_id)
                return []

            return [{**dict(run.info), "tags": run.data.tags} for run in runs]
        except MlflowException as e:
            logger.exception(e)
            logger.warning("Failed to retrieve the information about job '%s'", job_id)
        return []

    def send_batched_model_stats(self, aggregated_metrics: List[Dict], run_id: str, batch_size: int = 1000) -> None:
        """
        Sends batched model statistics to the tracking backend.

        Args:
            aggregated_metrics (List[Dict]): The list of dictionaries containing aggregated model statistics.
            run_id (str): The ID of the run to send the metrics to.
            batch_size (int): The maximum number of metrics to send in a single batch.
        """

        if batch_size <= 0:
            return
        batch = []
        for step, metrics in enumerate(aggregated_metrics):
            for metric_name, metric_value in metrics.items():
                batch.append(Metric(key=metric_name, value=metric_value, timestamp=0, step=step))
                if len(batch) == batch_size:
                    self.mlflow_client.log_batch(run_id=run_id, metrics=batch)
                    batch.clear()
        if batch:
            self.mlflow_client.log_batch(run_id=run_id, metrics=batch)


    def save_model(
        self,
        filepath: str,
        model_name: str,
        model_manager: ModelManager,
        model_type: str,
        validation_status: str = "pending",
    ) -> str:
        """
        Saves a model and its information to the tracking backend.

        Args:
            filepath (str): The artifact path of the model to save.
            model_name (str): The name of the model.
            model_manager (ModelManager): The instance of ModelManager used for model saving.
            model_type (str): The type of the model (e.g., "medcat_snomed").
            validation_status (str): The status of the model validation (default: "pending").

        Returns:
            str: The artifact URI of the saved model.
        """

        model_name = model_name.replace(" ", "_")

        mlflow.set_tag("training.output.package", os.path.basename(filepath))

        if not mlflow.get_tracking_uri().startswith("file:/"):
            TrackerClient.log_model(model_name, filepath, model_manager, model_name)
            versions = self.mlflow_client.search_model_versions(
                f"name='{model_name}'", order_by=["version_number DESC"]
            )
            if versions:
                TrackerClient._set_model_version_tags(
                    self.mlflow_client, model_name, versions[0].version, model_type, validation_status
                )
        else:
            TrackerClient.log_model(model_name, filepath, model_manager)

        artifact_uri = mlflow.get_artifact_uri(model_name)
        mlflow.set_tag("training.output.model_uri", artifact_uri)
        mlflow.set_tag("training.output.model_type", model_type)

        return artifact_uri

    def get_metrics_by_job_id(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Gets metrics for a training or evaluation job by the ID.

        Args:
            job_id (str): The ID of the training or evaluation job to search for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the job metrics.
        """

        try:
            runs = mlflow.search_runs(
                filter_string=f"tags.mlflow.runName = '{job_id}'",
                search_all_experiments=True,
                output_format="list",
            )
            if len(runs) == 0:
                logger.debug("Cannot find any runs with job ID '%s'", job_id)
                return []

            metrics = []
            for run in runs:
                metrics_history = {}
                for metric in run.data.metrics.keys():
                    metrics_history[metric] = [m.value for m in self.mlflow_client.get_metric_history(run_id=run.info.run_id, key=metric)]
                metrics_history["concepts"] = ast.literal_eval(run.data.tags.get("training.entity.classes", "[]"))
                metrics.append(metrics_history)
            return metrics
        except MlflowException as e:
            logger.exception(e)
            logger.warning("Failed to retrieve the information about job '%s'", job_id)
        return []

    @staticmethod
    def _get_experiment_id(experiment_name: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            experiment_id = experiment.experiment_id
            mlflow.set_experiment(None, experiment_id)
        return experiment_id
