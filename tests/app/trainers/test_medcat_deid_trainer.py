import os
import mlflow
from unittest.mock import create_autospec, patch, Mock
from transformers import TrainingArguments, TrainerState, TrainerControl
from app.config import Settings
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.trainers.medcat_deid_trainer import MedcatDeIdentificationSupervisedTrainer
from app.trainers.medcat_deid_trainer import MetricsCallback, LabelCountCallback

model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
model_service = create_autospec(
    MedCATModelDeIdentification,
    _config=Settings(),
    _model_parent_dir=model_parent_dir,
    _enable_trainer=True,
    _model_pack_path=os.path.join(model_parent_dir, "model.zip"),
)
deid_trainer = MedcatDeIdentificationSupervisedTrainer(model_service)
deid_trainer.model_name = "deid_trainer"
data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


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
