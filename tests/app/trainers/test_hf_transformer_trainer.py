import os
from unittest.mock import create_autospec, patch, Mock
from app.config import Settings
from app.model_services.huggingface_ner_model import HuggingFaceNerModel
from app.trainers.huggingface_ner_trainer import HuggingFaceNerUnsupervisedTrainer, HuggingFaceNerSupervisedTrainer


model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")
model_service = create_autospec(
    HuggingFaceNerModel,
    _config=Settings(),
    _model_parent_dir=model_parent_dir,
    _enable_trainer=True,
    _model_pack_path=os.path.join(model_parent_dir, "model.zip"),
)
model_service.model.config.max_position_embeddings = 512
unsupervised_trainer = HuggingFaceNerUnsupervisedTrainer(model_service)
unsupervised_trainer.model_name = "unsupervised_trainer"
supervised_trainer = HuggingFaceNerSupervisedTrainer(model_service)
supervised_trainer.model_name = "supervised_trainer"

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


def test_deploy_model():
    model = Mock()
    tokenizer = Mock()
    unsupervised_trainer.deploy_model(model_service, model, tokenizer)
    assert model_service.model == model
    assert model_service.tokenizer == tokenizer


def test_huggingface_ner_unsupervised_trainer(mlflow_fixture):
    with patch.object(unsupervised_trainer, "run", wraps=unsupervised_trainer.run) as run:
        unsupervised_trainer._tracker_client = Mock()
        unsupervised_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "sample_texts.json"), "r") as f:
            unsupervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
    unsupervised_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


def test_huggingface_ner_supervised_trainer(mlflow_fixture):
    with patch.object(supervised_trainer, "run", wraps=supervised_trainer.run) as run:
        supervised_trainer._tracker_client = Mock()
        supervised_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            supervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
            supervised_trainer._tracker_client.end_with_success()
    supervised_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


def test_huggingface_ner_unsupervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "sample_texts.json"), "r") as data_file:
        HuggingFaceNerUnsupervisedTrainer.run(unsupervised_trainer, {"nepochs": 1}, data_file, 1, "run_id")


def test_huggingface_ner_supervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "trainer_export.json"), "r") as data_file:
        HuggingFaceNerSupervisedTrainer.run(supervised_trainer, {"nepochs": 1, "print_stats": 1}, data_file, 1, "run_id")
