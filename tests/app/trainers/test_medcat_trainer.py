import os
from unittest.mock import create_autospec, patch, Mock
from medcat.config.config import General
from app.config import Settings
from app.model_services.medcat_model import MedCATModel
from app.trainers.medcat_trainer import MedcatSupervisedTrainer, MedcatUnsupervisedTrainer


model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")
model_service = create_autospec(
    MedCATModel,
    _config=Settings(),
    _model_parent_dir=model_parent_dir,
    _enable_trainer=True,
    _model_pack_path=os.path.join(model_parent_dir, "model.zip"),
)
supervised_trainer = MedcatSupervisedTrainer(model_service)
supervised_trainer.model_name = "supervised_trainer"
unsupervised_trainer = MedcatUnsupervisedTrainer(model_service)
unsupervised_trainer.model_name = "unsupervised_trainer"

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


def test_get_flattened_config():
    model = Mock()
    model.cdb.config.general = General()
    config = supervised_trainer.get_flattened_config(model)
    assert len(config) > 0


def test_deploy_model():
    model = Mock()
    supervised_trainer.deploy_model(model_service, model, True)
    model._versioning.assert_called_once()
    assert model_service.model == model


def test_save_model_pack():
    model = Mock()
    model.save_model_pack.return_value = "model_pack_name"
    supervised_trainer.save_model_pack(
        model,
        "retrained_models_dir",
        "model.zip",
        "model description",
    )
    model.save_model_pack.assert_called_once_with("retrained_models_dir", "model")
    assert model.config.meta.description == "model description"


def test_medcat_supervised_trainer(mlflow_fixture):
    with patch.object(supervised_trainer, "run", wraps=supervised_trainer.run) as run:
        supervised_trainer._tracker_client = Mock()
        supervised_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            supervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
            supervised_trainer._tracker_client.end_with_success()
    supervised_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


def test_medcat_unsupervised_trainer(mlflow_fixture):
    with patch.object(unsupervised_trainer, "run", wraps=unsupervised_trainer.run) as run:
        unsupervised_trainer._tracker_client = Mock()
        unsupervised_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "sample_texts.json"), "r") as f:
            unsupervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
    unsupervised_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


def test_medcat_supervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "trainer_export.json"), "r") as data_file:
        MedcatSupervisedTrainer.run(supervised_trainer, {"nepochs": 1, "print_stats": 1}, data_file, 1, "run_id")


def test_medcat_unsupervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "sample_texts.json"), "r") as data_file:
        MedcatUnsupervisedTrainer.run(unsupervised_trainer, {"nepochs": 1, "print_stats": 1}, data_file, 1, "run_id")
