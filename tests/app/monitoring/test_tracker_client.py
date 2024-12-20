import os
from unittest.mock import Mock, call

import datasets
import mlflow
import pandas as pd
import pytest

from tests.app.helper import StringContains

from data import doc_dataset
from management.tracker_client import TrackerClient


def test_start_new(mlflow_fixture):
    tracker_client = TrackerClient("")

    experiment_id, run_id = tracker_client.start_tracking(
        "model_name",
        "input_file_name",
        "base_model_origin",
        "training_type",
        {"param": "param"},
        "run_name",
        10,
    )

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
    assert mlflow.set_tags.call_args.args[0]["training.is.tracked"] == "True"
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


def test_send_hf_metrics_logs(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.send_hf_metrics_logs({"Key name": 1}, 1)

    mlflow.log_metrics.assert_called_once_with({"Key name": 1}, 1)


def test_save_model_artifact(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.save_model_artifact("filepath", "model name")

    mlflow.log_artifact.assert_called_once_with(
        "filepath", artifact_path=os.path.join("model_name", "artifacts")
    )


def test_save_raw_artifact(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.save_raw_artifact("filepath", "model name")

    mlflow.log_artifact.assert_called_once_with(
        "filepath", artifact_path=os.path.join("model_name", "artifacts", "raw")
    )


def test_save_processed_artifact(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.save_processed_artifact("filepath", "model name")

    mlflow.log_artifact.assert_called_once_with(
        "filepath", artifact_path=os.path.join("model_name", "artifacts", "processed")
    )


def test_save_dataframe_as_csv(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.save_dataframe_as_csv(
        "test.csv", pd.DataFrame({"x": ["x1", "x2"], "y": ["y1", "y2"]}), "model_name"
    )

    mlflow.log_artifact.assert_called_once_with(
        StringContains("test.csv"), artifact_path=os.path.join("model_name", "stats")
    )


def test_save_dict_as_json(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.save_dict_as_json("test.json", {"key": {"value": ["v1", "v2"]}}, "model_name")

    mlflow.log_artifact.assert_called_once_with(
        StringContains("test.json"), artifact_path=os.path.join("model_name", "stats")
    )


def test_save_plot(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.save_plot("test.png", "model_name")

    mlflow.log_artifact.assert_called_once_with(
        StringContains("test.png"), artifact_path=os.path.join("model_name", "stats")
    )


def test_save_table_dict(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.save_table_dict(
        {"col1": ["cell1", "cell2"], "col2": ["cell3", "cell4"]}, "model_name", "table.json"
    )

    mlflow.log_table.assert_called_once_with(
        data={"col1": ["cell1", "cell2"], "col2": ["cell3", "cell4"]},
        artifact_file=os.path.join("model_name", "tables", "table.json"),
    )


def test_save_train_dataset(mlflow_fixture):
    tracker_client = TrackerClient("")
    sample_texts = os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "fixture", "sample_texts.json"
    )
    dataset = datasets.load_dataset(
        doc_dataset.__file__,
        data_files={"documents": sample_texts},
        split="train",
        cache_dir="/tmp",
        trust_remote_code=True,
    )

    tracker_client.save_train_dataset(dataset)

    assert mlflow.log_input.call_count == 1
    assert isinstance(
        mlflow.log_input.call_args[0][0], mlflow.data.huggingface_dataset.HuggingFaceDataset
    )
    assert mlflow.log_input.call_args[1]["context"] == "train"


@pytest.mark.skip(reason="This test is flaky and needs to be fixed")
def test_save_model(mlflow_fixture):
    tracker_client = TrackerClient("")
    model_manager = Mock()
    model_info = Mock()
    model_info.flavors = {"python_function": {"artifacts": {"key": "value"}}}
    model_info.model_uri = "run://1234567890/model"
    model_manager.log_model.return_value = model_info
    mlflow_client = Mock()
    version = Mock()
    version.version = "1"
    mlflow_client.search_model_versions.return_value = [version]
    tracker_client.mlflow_client = mlflow_client

    artifact_uri = tracker_client.save_model(
        "path/to/file.zip", "model_name", model_manager, "validation_status"
    )

    assert "artifacts/model_name" in artifact_uri
    model_manager.log_model.assert_called_once_with("model_name", "path/to/file.zip", "model_name")
    mlflow_client.set_model_version_tag.assert_called_once_with(
        name="model_name", version="1", key="validation_status", value="validation_status"
    )
    mlflow.set_tag.assert_called_once_with("training.output.package", "file.zip")


def test_save_model_local(mlflow_fixture):
    tracker_client = TrackerClient("")
    model_manager = Mock()

    tracker_client.save_model_local("local_dir", "filepath", model_manager)

    model_manager.save_model.assert_called_once_with("local_dir", "filepath")


def test_save_pretrained_model(mlflow_fixture):
    tracker_client = TrackerClient("")
    model_manager = Mock()

    tracker_client.save_pretrained_model(
        "model_name",
        "model_path",
        model_manager,
        "training_type",
        "run_name",
        {"param": "value"},
        [{"p": 0.8, "r": 0.8}, {"p": 0.9, "r": 0.9}],
        {"tag_name": "tag_value"},
    )

    mlflow.get_experiment_by_name.assert_called_once_with("model_name_training_type")
    mlflow.start_run.assert_called_once_with(experiment_id="experiment_id")
    mlflow.log_params.assert_called_once_with({"param": "value"})
    mlflow.log_metrics.assert_has_calls(
        [call({"p": 0.8, "r": 0.8}, 0), call({"p": 0.9, "r": 0.9}, 1)]
    )
    mlflow.set_tags.assert_called()
    assert mlflow.set_tags.call_args.args[0]["mlflow.runName"] == "run_name"
    assert mlflow.set_tags.call_args.args[0]["training.base_model.origin"] == "model_path"
    assert mlflow.set_tags.call_args.args[0]["training.input_data.filename"] == "Unknown"
    assert mlflow.set_tags.call_args.args[0]["training.is.tracked"] == "False"
    assert mlflow.set_tags.call_args.args[0]["training.mlflow.run_id"] == "run_id"
    assert len(mlflow.set_tags.call_args.args[0]["mlflow.source.name"]) > 0
    assert mlflow.set_tags.call_args.args[0]["tag_name"] == "tag_value"
    model_manager.log_model.assert_called_once_with("model_name", "model_path", "model_name")


def test_log_single_exception(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_exceptions(Exception("something wrong"))

    mlflow.set_tag.assert_called_once_with("exception", "something wrong")


def test_log_multiple_exceptions(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_exceptions([Exception("exception_0"), Exception("exception_1")])

    mlflow.set_tag.assert_has_calls(
        [call("exception_0", "exception_0"), call("exception_1", "exception_1")]
    )


def test_log_classes(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_classes(["class_1", "class_2"])

    mlflow.set_tag.assert_called_once_with("training.entity.classes", "['class_1', 'class_2']")


def test_log_classes_and_names(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_classes_and_names({"class_1": "class_1_name", "class_2": "class_2_name"})

    mlflow.set_tag.assert_called_once_with(
        "training.entity.class2names", "{'class_1': 'class_1_name', 'class_2': 'class_2_name'}"
    )


def test_log_trainer_version(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_trainer_version("1.2.3")

    mlflow.set_tag.assert_called_once_with("training.trainer.version", "1.2.3")


def test_log_trainer_mode(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_trainer_mode(training=False)

    mlflow.set_tag.assert_called_once_with("training.trainer.mode", "eval")


def test_log_document_size(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_document_size(10)

    mlflow.set_tag.assert_called_once_with("training.document.size", 10)


def test_log_model_config(mlflow_fixture):
    tracker_client = TrackerClient("")

    tracker_client.log_model_config({"property": "value"})

    mlflow.log_params.assert_called_once_with({"property": "value"})


def test_send_batched_model_stats(mlflow_fixture):
    tracker_client = TrackerClient("")
    mlflow_client = Mock()
    tracker_client.mlflow_client = mlflow_client

    tracker_client.send_batched_model_stats(
        [{"m1": "v1", "m2": "v1"}, {"m1": "v2", "m2": "v2"}, {"m1": "v3", "m2": "v3"}], "run_id", 3
    )

    mlflow_client.log_batch.assert_has_calls(
        [call(run_id="run_id", metrics=[]), call(run_id="run_id", metrics=[])]
    )


def test_get_experiment_name():
    assert TrackerClient.get_experiment_name("SNOMED model") == "SNOMED_model"
    assert (
        TrackerClient.get_experiment_name("SNOMED model", "unsupervised")
        == "SNOMED_model_unsupervised"
    )
