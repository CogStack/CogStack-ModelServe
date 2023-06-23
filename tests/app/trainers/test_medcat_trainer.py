import os
import pytest
from unittest.mock import create_autospec, patch, Mock
from medcat.config import General
from app.config import Settings
from app.model_services.medcat_model import MedCATModel
from app.trainers.medcat_trainer import MedcatSupervisedTrainer, MedcatUnsupervisedTrainer

model_service = create_autospec(MedCATModel,
                                _config=Settings(),
                                _model_parent_dir="model_parent_dir",
                                _enable_trainer=True,
                                _model_pack_path="model_parent_dir/mode.zip",
                                _meta_cat_config_dict={"general": {"device": "cpu"}})
supervised_trainer = MedcatSupervisedTrainer(model_service)
supervised_trainer.model_name = "supervised_trainer"
unsupervised_trainer = MedcatUnsupervisedTrainer(model_service)
unsupervised_trainer.model_name = "unsupervised_trainer"

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


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


def test_get_flattened_config():
    model = Mock()
    model.cdb.config.general = General()
    config = supervised_trainer.get_flattened_config(model)
    assert "word_skipper" in config
    assert "punct_checker" in config
    assert "linking.filters" not in config


def test_deploy_model():
    model = Mock()
    supervised_trainer.deploy_model(model_service, model, True)
    model._versioning.assert_called_once()
    assert model_service.model == model


def test_save_model():
    model = Mock()
    model.create_model_pack.return_value = "model_pack_name"
    supervised_trainer.save_model(model, "retrained_models_dir")
    model.create_model_pack.called_once_with("retrained_models_dir", "model")


def test_medcat_supervised_trainer(mlflow_fixture):
    with patch.object(supervised_trainer, "run", wraps=supervised_trainer.run) as run:
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            supervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
            supervised_trainer._tracker_client.end_with_success()
    run.assert_called_once()


def test_medcat_unsupervised_trainer(mlflow_fixture):
    with patch.object(unsupervised_trainer, "run", wraps=unsupervised_trainer.run) as run:
        with open(os.path.join(data_dir, "sample_texts.json"), "r") as f:
            unsupervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
            unsupervised_trainer._tracker_client.end_with_success()
    run.assert_called_once()
