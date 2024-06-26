import os
import mlflow
import pytest
from unittest.mock import create_autospec, patch, Mock
from transformers import TrainingArguments, TrainerState, TrainerControl
from config import Settings
from model_services.medcat_model_deid import MedCATModelDeIdentification
from trainers.medcat_deid_trainer import MedcatDeIdentificationSupervisedTrainer
from trainers.medcat_deid_trainer import MetricsCallback, LabelCountCallback

model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
model_service = create_autospec(MedCATModelDeIdentification,
                                _config=Settings(),
                                _model_parent_dir=model_parent_dir,
                                _enable_trainer=True,
                                _model_pack_path=os.path.join(model_parent_dir, "model.zip"))
deid_trainer = MedcatDeIdentificationSupervisedTrainer(model_service)
deid_trainer.model_name = "deid_trainer"
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


def test_medcat_deid_supervised_trainer(mlflow_fixture):
    with patch.object(deid_trainer, "run", wraps=deid_trainer.run) as run:
        deid_trainer._tracker_client = Mock()
        deid_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            deid_trainer.train(f, 1, 1, "training_id", "input_file_name")
    deid_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


def test_medcat_deid_supervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "trainer_export.json"), "r") as data_file:
        MedcatDeIdentificationSupervisedTrainer.run(deid_trainer, {"nepochs": 1, "print_stats": 1}, data_file, 1, "run_id")


def test_trainer_callbacks(mlflow_fixture):
    trainer = Mock()
    trainer.train_dataset = [{"labels": []}]
    metrics_callback = MetricsCallback(trainer)
    metrics_callback.on_step_end(TrainingArguments("/tmp"), TrainerState(), TrainerControl())
    assert mlflow.log_metrics.call_count == 0
    label_count_callback = LabelCountCallback(trainer)
    label_count_callback.on_step_end(TrainingArguments("/tmp"), TrainerState(), TrainerControl())
    assert mlflow.log_metrics.call_count == 1
