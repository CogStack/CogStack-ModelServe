import os
from unittest.mock import create_autospec, patch, Mock
from config import Settings
from model_services.hf_transformer_model import HuggingfaceTransformerModel
from trainers.hf_transformer_trainer import HFTransformerUnsupervisedTrainer


model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")
model_service = create_autospec(HuggingfaceTransformerModel,
                                _config=Settings(),
                                _model_parent_dir=model_parent_dir,
                                _enable_trainer=True,
                                _model_pack_path=os.path.join(model_parent_dir, "model.zip"))
unsupervised_trainer = HFTransformerUnsupervisedTrainer(model_service)
unsupervised_trainer.model_name = "unsupervised_trainer"

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


def test_deploy_model():
    model = Mock()
    tokenizer = Mock()
    unsupervised_trainer.deploy_model(model_service, model, tokenizer)
    assert model_service.model == model
    assert model_service.tokenizer == tokenizer


def test_hf_transformer_unsupervised_trainer(mlflow_fixture):
    with patch.object(unsupervised_trainer, "run", wraps=unsupervised_trainer.run) as run:
        unsupervised_trainer._tracker_client = Mock()
        unsupervised_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "sample_texts.json"), "r") as f:
            unsupervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
    unsupervised_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


def test_medcat_unsupervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "sample_texts.json"), "r") as data_file:
        HFTransformerUnsupervisedTrainer.run(unsupervised_trainer, {"nepochs": 1}, data_file, 1, "run_id")
