import os
import pytest
import mlflow

from app.management.tracker_client import TrackerClient
from unittest.mock import Mock, call


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
    tracker_client = TrackerClient("")
    experiment_id, run_id = tracker_client.start_tracking("model_name", "input_file_name", "base_model_origin",
                                                           "training_type", {"param": "param"}, "run_name", 10)
    mlflow.get_experiment_by_name.assert_called_once_with("model_name_training_type")
    mlflow.create_experiment.assert_called_once_with(name="model_name_training_type")
    mlflow.start_run.assert_called_once_with(experiment_id="experiment_id")
    mlflow.set_tags.assert_called()
    mlflow.log_params.assert_called_once_with({"param": "param"})
    assert experiment_id == "experiment_id"
    assert run_id == "run_id"
    assert mlflow.set_tags.call_args.args[0]["mlflow.runName"] == "run_name"
    assert mlflow.set_tags.call_args.args[0]["training.base_model.origin"] == "base_model_origin"
    assert mlflow.set_tags.call_args.args[0]["training.input_data.filename"] == "input_file_name"
    assert mlflow.set_tags.call_args.args[0]["training.is.pretrained"] == "False"
    assert mlflow.set_tags.call_args.args[0]["training.metrics.log_frequency"] == 10


def test_end_with_success(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.end_with_success()
    mlflow.end_run.assert_called_once_with("FINISHED")


def test_end_with_failure(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.end_with_failure()
    mlflow.end_run.assert_called_once_with("FAILED")


def test_end_with_interruption(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.end_with_interruption()
    mlflow.end_run.assert_called_once_with("KILLED")


def test_send_model_stats(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.send_model_stats({"Key name": 1}, 1)
    mlflow.log_metrics.assert_called_once_with({"key_name": 1}, 1)


def test_save_model(mlflow_fixture):
    tracker_client = TrackerClient("")
    pyfunc_model = Mock()
    tracker_client.save_model("filepath", "model name", pyfunc_model)
    mlflow.pyfunc.log_model.assert_called_once_with(artifact_path="model_name",
                                                    python_model=pyfunc_model,
                                                    artifacts={"model_path": "filepath"},
                                                    registered_model_name="model_name")
    mlflow.register_model.assert_not_called()


def test_save_model_artifact(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.save_model_artifact("filepath", "model name")
    mlflow.log_artifact.assert_called_once_with("filepath", artifact_path=os.path.join("model_name", "artifacts"))


def test_save_model_local(mlflow_fixture_file_uri):
    tracker_client = TrackerClient("")
    pyfunc_model = Mock()
    tracker_client.save_model("filepath", "model name", pyfunc_model)
    mlflow.pyfunc.log_model.assert_called_once_with(artifact_path="model_name",
                                                    python_model=pyfunc_model,
                                                    artifacts={"model_path": "filepath"})
    mlflow.register_model.assert_not_called()


def test_save_pretrained_model(mlflow_fixture):
    tracker_client = TrackerClient("")
    pyfunc_model = Mock()
    tracker_client.save_pretrained_model("model_name",
                                        "model_path",
                                        pyfunc_model,
                                        "training_type",
                                        "run_name",
                                        {"param": "value"},
                                        [{"p": 0.8, "r": 0.8}, {"p": 0.9, "r": 0.9}],
                                        {"tag_name": "tag_value"})
    mlflow.get_experiment_by_name.assert_called_once_with("model_name_training_type")
    mlflow.start_run.assert_called_once_with(experiment_id="experiment_id")
    mlflow.log_params.assert_called_once_with( {"param": "value"})
    mlflow.log_metrics.assert_has_calls([call({"p": 0.8, "r": 0.8}, 0), call({"p": 0.9, "r": 0.9}, 1)])
    mlflow.set_tags.assert_called()
    assert mlflow.set_tags.call_args.args[0]["mlflow.runName"] == "run_name"
    assert mlflow.set_tags.call_args.args[0]["training.base_model.origin"] == "model_path"
    assert mlflow.set_tags.call_args.args[0]["training.input_data.filename"] == "Unknown"
    assert mlflow.set_tags.call_args.args[0]["training.is.pretrained"] == "True"
    assert mlflow.set_tags.call_args.args[0]["training.mlflow.run_id"] == "run_id"
    assert len(mlflow.set_tags.call_args.args[0]["mlflow.source.name"]) > 0
    assert mlflow.set_tags.call_args.args[0]["tag_name"] == "tag_value"
    mlflow.pyfunc.log_model.assert_called_once_with(artifact_path="model_name",
                                                    python_model=pyfunc_model,
                                                    artifacts={"model_path": "model_path"},
                                                    registered_model_name="model_name")


def test_log_exception(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.log_exception(Exception("something wrong"))
    mlflow.set_tag.assert_called_once_with("exception", "something wrong")


def test_log_classes(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.log_classes(["class_1", "class_2"])
    mlflow.set_tag.assert_called_once_with("training.entity.classes", "['class_1', 'class_2']")


def test_log_classes_and_names(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.log_classes_and_names({"class_1": "class_1_name", "class_2": "class_2_name"})
    mlflow.set_tag.assert_called_once_with("training.entity.class2names", "{'class_1': 'class_1_name', 'class_2': 'class_2_name'}")


def test_log_trainer_version(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.log_trainer_version("1.2.3")
    mlflow.set_tag.assert_called_once_with("training.trainer.version", "1.2.3")


def test_log_model_config(mlflow_fixture):
    tracker_client = TrackerClient("")
    tracker_client.log_model_config({"property": "value"})
    mlflow.log_params.assert_called_once_with({"property": "value"})


def test_send_batched_model_stats(mlflow_fixture):
    tracker_client = TrackerClient("")
    mlflow_client = Mock()
    tracker_client.mlflow_client = mlflow_client
    tracker_client.send_batched_model_stats(
        [{"m1": "v1", "m2": "v1"}, {"m1": "v2", "m2": "v2"}, {"m1": "v3", "m2": "v3"}],
        "run_id", 3)
    mlflow_client.log_batch.assert_has_calls([call(run_id='run_id', metrics=[]), call(run_id='run_id', metrics=[])])


