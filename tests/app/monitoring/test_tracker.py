import pytest
import mlflow

from app.monitoring.tracker import TrainingTracker


@pytest.fixture()
def mlflow_fixture(mocker):
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.get_experiment_by_name", return_value=None)
    mocker.patch("mlflow.create_experiment", return_value="1")
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.set_tags")
    mocker.patch("mlflow.log_params")
    mocker.patch("mlflow.log_metrics")
    mocker.patch("mlflow.log_artifact")
    mocker.patch("mlflow.end_run")
    yield TrainingTracker("any")


def test_start_new(mlflow_fixture):
    experiment_id = TrainingTracker.start_tracking("model_name", "input_file_name", "training_type", {}, "training_id")
    mlflow.get_experiment_by_name.assert_called_once_with("model_name_training_type")
    mlflow.create_experiment.assert_called_once_with(name="model_name_training_type")
    mlflow.start_run.assert_called_once_with(experiment_id="1", run_name="training_id")
    mlflow.set_tags.assert_called()
    mlflow.log_params.assert_called_once_with({})
    assert experiment_id == "1"


def test_end_with_success(mlflow_fixture):
    TrainingTracker.end_with_success()
    mlflow.end_run.assert_called_once_with("FINISHED")


def test_end_with_failure(mlflow_fixture):
    TrainingTracker.end_with_failure()
    mlflow.end_run.assert_called_once_with("FAILED")


def test_end_with_interruption(mlflow_fixture):
    TrainingTracker.end_with_interruption()
    mlflow.end_run.assert_called_once_with("KILLED")


def test_send_metrics(mlflow_fixture):
    TrainingTracker.send_metrics("Epoch: 1, Prec: 0.01, Rec: 0.01, F1: 0.01")
    mlflow.log_metrics.assert_called_once_with({'precision': 0.01, 'recall': 0.01, 'f1': 0.01}, 0)


def test_send_model_stats(mlflow_fixture):
    TrainingTracker.send_model_stats({"Key name": 1}, 1)
    mlflow.log_metrics.assert_called_once_with({'key_name': 1}, 1)


def test_send_model_package(mlflow_fixture):
    TrainingTracker.send_model_package("filepath")
    mlflow.log_artifact.assert_called_once_with("filepath")
