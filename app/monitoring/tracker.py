import socket
import mlflow
import re
from typing import Dict
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME
from mlflow.entities import RunStatus


class TrainingTracker(object):

    def __init__(self, mlflow_tracking_uri: str) -> None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    @staticmethod
    def start_tracking(model_name: str,
                       input_file_name: str,
                       training_type: str,
                       training_params: Dict,
                       training_id: str) -> str:
        experiment_name = TrainingTracker._get_experiment_name(model_name, training_type)
        experiment_id = TrainingTracker._get_experiment_id(experiment_name)
        mlflow.start_run(experiment_id=experiment_id, run_name=training_id)
        mlflow.set_tags({
            MLFLOW_SOURCE_NAME: socket.gethostname(),
            "training.input.filename": input_file_name,
        })
        mlflow.log_params(training_params)
        return experiment_id

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
    def send_metrics(log: str) -> None:
        metric_lines = re.findall(r"Epoch: (\d), Prec: (\d+\.\d+), Rec: (\d+\.\d+), F1: (\d+\.\d+)", log, re.IGNORECASE)
        for step, metric in enumerate(metric_lines):
            metrics = {
                "precision": float(metric[1]),
                "recall": float(metric[2]),
                "f1": float(metric[3]),
            }
            mlflow.log_metrics(metrics, step)

    @staticmethod
    def send_model_stats(stats: Dict, step: int) -> None:
        metrics = {key.replace(" ", "_").lower(): val for key, val in stats.items()}
        mlflow.log_metrics(metrics, step)

    @staticmethod
    def send_model_package(filepath: str) -> None:
        mlflow.log_artifact(filepath)

    @staticmethod
    def _get_experiment_id(experiment_name: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return mlflow.create_experiment(name=experiment_name) if experiment is None else experiment.experiment_id

    @staticmethod
    def _get_experiment_name(model_name: str, training_type: str) -> str:
        return f"{model_name} {training_type}".replace(" ", "_")
