import os
import pytest
from unittest.mock import create_autospec, patch, Mock
from medcat.config_meta_cat import General, Model, Train
from config import Settings
from model_services.medcat_model import MedCATModel
from trainers.metacat_trainer import MetacatTrainer

model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")
model_service = create_autospec(MedCATModel,
                                _config=Settings(),
                                _model_parent_dir=model_parent_dir,
                                _enable_trainer=True,
                                _model_pack_path=os.path.join(model_parent_dir, "model.zip"),
                                _meta_cat_config_dict={"general": {"device": "cpu"}})
metacat_trainer = MetacatTrainer(model_service)
metacat_trainer.model_name = "metacat_trainer"

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
    model.config.general = General()
    model.config.model = Model()
    model.config.train = Train()
    config = metacat_trainer.get_flattened_config(model, "prefix")
    for key, val in config.items():
        assert "prefix.general." in key or "prefix.model." in key or "prefix.train" in key


def test_deploy_model():
    model = Mock()
    metacat_trainer.deploy_model(model_service, model, True)
    model._versioning.assert_called_once()
    assert model_service.model == model


def test_save_model_pack():
    model = Mock()
    model.create_model_pack.return_value = "model_pack_name"
    metacat_trainer.save_model_pack(model, "retrained_models_dir")
    model.create_model_pack.called_once_with("retrained_models_dir", "model")


def test_metacat_trainer(mlflow_fixture):
    with patch.object(metacat_trainer, "run", wraps=metacat_trainer.run) as run:
        metacat_trainer._tracker_client = Mock()
        metacat_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            metacat_trainer.train(f, 1, 1, "training_id", "input_file_name")
    metacat_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()
