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
from app.management.model_manager import ModelManager
from app.exception import StartTrainingException
from app.domain import TrainerBackend

logger = logging.getLogger("cms")
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.CRITICAL)


@final
class TrackerClient(object):

    def __init__(self, mlflow_tracking_uri: str) -> None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = MlflowClient(mlflow_tracking_uri)

    @staticmethod
    def start_tracking(model_name: str,
                       input_file_name: str,
                       base_model_original: str,
                       training_type: str,
                       training_params: Dict,
                       run_name: str,
                       log_frequency: int,
                       description: Optional[str] = None) -> Tuple[str, str]:
        experiment_name = TrackerClient.get_experiment_name(model_name, training_type)
        experiment_id = TrackerClient._get_experiment_id(experiment_name)
        try:
            active_run = mlflow.start_run(experiment_id=experiment_id)
        except Exception:
            logger.exception("Cannot start a new training")
            raise StartTrainingException("Cannot start a new training")
        mlflow.set_tags({
            MLFLOW_SOURCE_NAME: socket.gethostname(),
            "mlflow.runName": run_name,
            "mlflow.note.content": description or "",
            "training.mlflow.run_id": active_run.info.run_id,
            "training.input_data.filename": input_file_name,
            "training.base_model.origin": base_model_original,
            "training.is.tracked": "True",
            "training.metrics.log_frequency": log_frequency,
        })
        mlflow.log_params(training_params)
        return experiment_id, active_run.info.run_id

    @staticmethod
    def end_with_success() -> None:
        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

    @staticmethod
    def end_with_failure() -> None:
        mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

    @staticmethod
    def end_with_interruption() -> None:
        mlflow.end_run(RunStatus.to_string(RunStatus.KILLED))

    @staticmethod
    def send_model_stats(stats: Dict, step: int) -> None:
        metrics = {key.replace(" ", "_").lower(): val for key, val in stats.items()}
        mlflow.log_metrics(metrics, step)

    @staticmethod
    def send_hf_metrics_logs(logs: Dict, step: int) -> None:
        mlflow.log_metrics(logs, step)

    @staticmethod
    def save_model_local(local_dir: str,
                         filepath: str,
                         model_manager: ModelManager) -> None:
        model_manager.save_model(local_dir, filepath)

    @staticmethod
    def save_model_artifact(filepath: str,
                            model_name: str) -> None:
        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(filepath, artifact_path=os.path.join(model_name, "artifacts"))

    @staticmethod
    def save_raw_artifact(filepath: str,
                          model_name: str) -> None:
        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(filepath, artifact_path=os.path.join(model_name, "artifacts", "raw"))

    @staticmethod
    def save_processed_artifact(filepath: str,
                                model_name: str) -> None:
        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(filepath, artifact_path=os.path.join(model_name, "artifacts", "processed"))

    @staticmethod
    def save_dataframe_as_csv(file_name: str, data_frame: pd.DataFrame, model_name: str) -> None:
        model_name = model_name.replace(" ", "_")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, file_name), "w") as f:
                data_frame.to_csv(f.name, index=False)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path=os.path.join(model_name, "stats"))

    @staticmethod
    def save_dict_as_json(file_name: str, data: Dict, model_name: str) -> None:
        model_name = model_name.replace(" ", "_")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, file_name), "w") as f:
                json.dump(data, f)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path=os.path.join(model_name, "stats"))

    @staticmethod
    def save_plot(file_name: str, model_name: str) -> None:
        model_name = model_name.replace(" ", "_")
        mlflow.log_artifact(file_name, artifact_path=os.path.join(model_name, "stats"))

    @staticmethod
    def save_table_dict(table_dict: Dict, model_name: str, file_name: str) -> None:
        model_name = model_name.replace(" ", "_")
        mlflow.log_table(data=table_dict, artifact_file=os.path.join(model_name, "tables", file_name))

    @staticmethod
    def save_train_dataset(dataset: datasets.Dataset) -> None:
        ds = mlflow.data.huggingface_dataset.from_huggingface(dataset)
        mlflow.log_input(ds, context="train")

    @staticmethod
    def log_exceptions(es: Union[Exception, List[Exception]]) -> None:
        if isinstance(es, list):
            for idx, e in enumerate(es):
                mlflow.set_tag(f"exception_{idx}", str(e))
        else:
            mlflow.set_tag("exception", str(es))

    @staticmethod
    def log_classes(classes: List[str]) -> None:
        mlflow.set_tag("training.entity.classes", str(classes)[:5000])

    @staticmethod
    def log_classes_and_names(class2names: Dict[str, str]) -> None:
        mlflow.set_tag("training.entity.class2names", str(class2names)[:5000])

    @staticmethod
    def log_trainer_version(trainer_backend: TrainerBackend, trainer_version: str) -> None:
        mlflow.set_tags({
            "training.trainer.version": trainer_version,
            "training.trainer.backend": trainer_backend.value,
        })

    @staticmethod
    def log_trainer_mode(training: bool = True) -> None:
        mlflow.set_tag("training.trainer.mode", "train" if training else "eval")

    @staticmethod
    def log_document_size(num_of_docs: int) -> None:
        mlflow.set_tag("training.document.size", num_of_docs)

    @staticmethod
    def log_model_config(config: Dict[str, str]) -> None:
        mlflow.log_params(config)

    @staticmethod
    def save_pretrained_model(model_name: str,
                              model_path: str,
                              model_manager: ModelManager,
                              training_type: Optional[str] = "",
                              run_name: Optional[str] = "",
                              model_config: Optional[Dict] = None,
                              model_metrics: Optional[List[Dict]] = None,
                              model_tags: Optional[Dict] = None, ) -> None:
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
            model_manager.log_model(model_name, model_path, model_name)
            TrackerClient.end_with_success()
        except KeyboardInterrupt:
            TrackerClient.end_with_interruption()
        except Exception as e:
            logger.exception("Failed to save the pretrained model")
            TrackerClient.log_exceptions(e)
            TrackerClient.end_with_failure()

    @staticmethod
    def get_experiment_name(model_name: str, training_type: Optional[str] = "") -> str:
        return f"{model_name} {training_type}".replace(" ", "_") if training_type else model_name.replace(" ", "_")

    @staticmethod
    def get_info_by_job_id(job_id: str) -> List[Dict]:
        try:
            runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{job_id}'",
                                      search_all_experiments=True,
                                      output_format="list")
            if len(runs) == 0:
                logger.debug("Cannot find any runs with job ID '%s'", job_id)
                return []

            return [{**dict(run.info), "tags": run.data.tags} for run in runs]
        except MlflowException as e:
            logger.exception(e)
            logger.warning("Failed to retrieve the information about job '%s'", job_id)
        return []

    def send_batched_model_stats(self, aggregated_metrics: List[Dict], run_id: str, batch_size: int = 1000) -> None:
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

    def save_model(self,
                   filepath: str,
                   model_name: str,
                   model_manager: ModelManager,
                   validation_status: str = "pending") -> str:
        model_name = model_name.replace(" ", "_")

        mlflow.set_tag("training.output.package", os.path.basename(filepath))

        if not mlflow.get_tracking_uri().startswith("file:/"):
            model_manager.log_model(model_name, filepath, model_name)
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            self.mlflow_client.set_model_version_tag(name=model_name,
                                                     version=versions[0].version,
                                                     key="validation_status",
                                                     value=validation_status)
        else:
            model_manager.log_model(model_name, filepath)

        artifact_uri = mlflow.get_artifact_uri(model_name)
        mlflow.set_tag("training.output.model_uri", artifact_uri)

        return artifact_uri

    def get_metrics_by_job_id(self, job_id: str) -> List[Dict[str, Any]]:
        try:
            runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{job_id}'",
                                      search_all_experiments=True,
                                      output_format="list")
            if len(runs) == 0:
                logger.debug("Cannot find any runs with job ID '%s'", job_id)
                return []

            metrics = []
            for run in runs:
                metrics_history = {}
                for metric in run.data.metrics.keys():
                    metrics_history[metric] = [m.value for m in self.mlflow_client.get_metric_history(run_id=run.info.run_id, key=metric)]
                metrics.append(metrics_history)
            return metrics
        except MlflowException as e:
            logger.exception(e)
            logger.warning("Failed to retrieve the information about job '%s'", job_id)
        return []

    @staticmethod
    def _get_experiment_id(experiment_name: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return mlflow.create_experiment(name=experiment_name) if experiment is None else experiment.experiment_id
