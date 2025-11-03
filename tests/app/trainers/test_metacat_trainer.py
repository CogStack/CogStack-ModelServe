import os
from unittest.mock import create_autospec, patch, Mock
from medcat.config.config_meta_cat import General, Model, Train
from app.config import Settings
from app.model_services.medcat_model import MedCATModel
from app.trainers.metacat_trainer import MetacatTrainer

model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")
model_service = create_autospec(
    MedCATModel,
    _config=Settings(),
    _model_parent_dir=model_parent_dir,
    _enable_trainer=True,
    _model_pack_path=os.path.join(model_parent_dir, "model.zip"),
)
metacat_trainer = MetacatTrainer(model_service)
metacat_trainer.model_name = "metacat_trainer"

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


def test_get_flattened_metacat_config():
    model = Mock()
    model.config.general = General()
    model.config.model = Model()
    model.config.train = Train()
    config = metacat_trainer.get_flattened_metacat_config(model, "prefix")
    for key, val in config.items():
        assert "prefix.general." in key or "prefix.model." in key or "prefix.train" in key


def test_deploy_model():
    model = Mock()
    metacat_trainer.deploy_model(model_service, model, True)
    model._versioning.assert_called_once()
    assert model_service.model == model


def test_save_model_pack():
    model = Mock()
    model.save_model_pack.return_value = "model_pack_name"
    metacat_trainer.save_model_pack(
        model,
        "retrained_models_dir",
        "model.zip",
        "model description",
    )
    model.save_model_pack.assert_called_once_with("retrained_models_dir", "model")
    assert model.config.meta.description == "model description"


def test_metacat_trainer(mlflow_fixture):
    with patch.object(metacat_trainer, "run", wraps=metacat_trainer.run) as run:
        metacat_trainer._tracker_client = Mock()
        metacat_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            metacat_trainer.train(f, 1, 1, "training_id", "input_file_name")
    metacat_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


def test_metacat_supervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "trainer_export.json"), "r") as data_file:
        MetacatTrainer.run(metacat_trainer, {"nepochs": 1, "print_stats": 1}, data_file, 1, "run_id")
