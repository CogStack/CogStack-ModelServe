import os
import pytest
import mlflow

from app.monitoring.tracker import TrainingTracker
from unittest.mock import Mock


@pytest.fixture
def mlflow_fixture(mocker):
    active_run = Mock()
    active_run.info.run_id = "run_id"
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.get_experiment_by_name", return_value=None)
    mocker.patch("mlflow.create_experiment", return_value="experiment_id")
    mocker.patch("mlflow.start_run", return_value=active_run)
    mocker.patch("mlflow.set_tags")
    mocker.patch("mlflow.set_tag")
    mocker.patch("mlflow.log_params")
    mocker.patch("mlflow.log_metrics")
    mocker.patch("mlflow.log_artifact")
    mocker.patch("mlflow.pyfunc.log_model")
    mocker.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000")
    mocker.patch("mlflow.register_model")
    mocker.patch("mlflow.end_run")


@pytest.fixture
def mlflow_fixture_file_uri(mlflow_fixture, mocker):
    mocker.patch("mlflow.get_tracking_uri", return_value="file:/tmp/mlruns")


def test_start_new(mlflow_fixture):
    experiment_id, run_id = TrainingTracker.start_tracking("model_name", "input_file_name", "base_model_origin",
                                                           "training_type", {"param": "param"}, "training_id", 10)
    mlflow.get_experiment_by_name.assert_called_once_with("model_name_training_type")
    mlflow.create_experiment.assert_called_once_with(name="model_name_training_type")
    mlflow.start_run.assert_called_once_with(experiment_id="experiment_id", run_name="training_id")
    mlflow.set_tags.assert_called()
    mlflow.log_params.assert_called_once_with({"param": "param"})
    assert experiment_id == "experiment_id"
    assert run_id == "run_id"
    assert mlflow.set_tags.call_args.args[0]["training.base_model.origin"] == "base_model_origin"
    assert mlflow.set_tags.call_args.args[0]["training.input_data.filename"] == "input_file_name"
    assert mlflow.set_tags.call_args.args[0]["training.metrics.log_frequency"] == 10


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
    TrainingTracker.glean_and_log_metrics("Epoch: 1, Prec: 0.01, Rec: 0.01, F1: 0.01")
    mlflow.log_metrics.assert_called_once_with({"precision": 0.01, "recall": 0.01, "f1": 0.01}, 0)


def test_send_model_stats(mlflow_fixture):
    TrainingTracker.send_model_stats({"Key name": 1}, 1)
    mlflow.log_metrics.assert_called_once_with({"key_name": 1}, 1)


def test_save_model(mlflow_fixture):
    pyfunc_model = Mock()
    TrainingTracker.save_model("filepath", "run_id", "model name", pyfunc_model)
    mlflow.pyfunc.log_model.assert_called_once_with(artifact_path="model_name",
                                                    python_model=pyfunc_model,
                                                    artifacts={"model_path": "filepath"},
                                                    registered_model_name="model_name")
    mlflow.register_model.assert_not_called()


def test_save_model_artifact(mlflow_fixture):
    TrainingTracker.save_model_artifact("filepath", "model name")
    mlflow.log_artifact.assert_called_once_with("filepath", artifact_path=os.path.join("model_name", "artifacts"))


def test_save_model_local(mlflow_fixture_file_uri):
    pyfunc_model = Mock()
    TrainingTracker.save_model("filepath", "run_id", "model name", pyfunc_model)
    mlflow.pyfunc.log_model.assert_called_once_with(artifact_path="model_name",
                                                    python_model=pyfunc_model,
                                                    artifacts={"model_path": "filepath"})
    mlflow.register_model.assert_not_called()


def test_log_exception(mlflow_fixture):
    TrainingTracker.log_exception(Exception("something wrong"))
    mlflow.set_tag.assert_called_once_with("exception", "something wrong")
