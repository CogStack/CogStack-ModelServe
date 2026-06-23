import os
import torch
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


def test_freeze_all_except_classification_head():
    class _DummyModule:
        def __init__(self, params):
            self._params = params

        def parameters(self):
            return self._params

    class _DummyModel:
        def __init__(self) -> None:
            self.classifier_w = torch.nn.Parameter(torch.ones(1))
            self.score_w = torch.nn.Parameter(torch.ones(1))
            self.encoder_w = torch.nn.Parameter(torch.ones(1))
            self.classifier = _DummyModule([self.classifier_w])
            self.score = _DummyModule([self.score_w])
            self._named = [
                ("encoder.layer.0.weight", self.encoder_w),
                ("classifier.weight", self.classifier_w),
                ("score.weight", self.score_w),
            ]

        def named_parameters(self):
            for name, param in self._named:
                yield name, param

    model = _DummyModel()
    HuggingFaceNerSupervisedTrainer._freeze_params_or_classifier(model, "any,except_classifier")

    assert model.encoder_w.requires_grad is False
    assert model.classifier_w.requires_grad is True
    assert model.score_w.requires_grad is True


@patch("app.trainers.huggingface_ner_trainer.torch.set_num_threads")
@patch("app.trainers.huggingface_ner_trainer.os.cpu_count", return_value=8)
def test_get_training_args_uses_cpu_thread_based_batching(mock_cpu_count, mock_set_num_threads):
    original_device = supervised_trainer._config.DEVICE
    supervised_trainer._config.DEVICE = "cpu"

    with patch.object(
        HuggingFaceNerSupervisedTrainer,
        "_create_training_arguments",
        side_effect=lambda **kwargs: kwargs,
    ):
        training_args = supervised_trainer._get_training_args(
            "results",
            "logs",
            {"nepochs": 1, "scaling_factor": 3},
            1,
        )

    supervised_trainer._config.DEVICE = original_device

    assert training_args["per_device_train_batch_size"] == 4
    assert training_args["per_device_eval_batch_size"] == 4
    assert training_args["gradient_accumulation_steps"] == 4
    assert training_args["eval_accumulation_steps"] == 4
    mock_set_num_threads.assert_called_once_with(2)


@patch("app.trainers.huggingface_ner_trainer.torch.set_num_threads")
@patch("app.trainers.huggingface_ner_trainer.torch.backends.mps.is_available", return_value=False)
@patch("app.trainers.huggingface_ner_trainer.torch.cuda.is_available", return_value=True)
@patch("app.trainers.huggingface_ner_trainer.os.cpu_count", return_value=8)
def test_get_training_args_uses_cuda_scaling_factor_batching(
    mock_cpu_count,
    mock_cuda_available,
    mock_mps_available,
    mock_set_num_threads,
):
    original_device = supervised_trainer._config.DEVICE
    supervised_trainer._config.DEVICE = "cuda"

    with patch.object(
        HuggingFaceNerSupervisedTrainer,
        "_create_training_arguments",
        side_effect=lambda **kwargs: kwargs,
    ):
        training_args = supervised_trainer._get_training_args(
            "results",
            "logs",
            {"nepochs": 1, "scaling_factor": 3},
            1,
        )

    supervised_trainer._config.DEVICE = original_device

    assert training_args["per_device_train_batch_size"] == 6
    assert training_args["per_device_eval_batch_size"] == 6
    assert training_args["gradient_accumulation_steps"] == 3
    assert training_args["eval_accumulation_steps"] == 3
    mock_set_num_threads.assert_called_once_with(4)


@patch("app.trainers.huggingface_ner_trainer.torch.set_num_threads")
@patch("app.trainers.huggingface_ner_trainer.torch.backends.mps.is_available", return_value=True)
@patch("app.trainers.huggingface_ner_trainer.torch.cuda.is_available", return_value=False)
@patch("app.trainers.huggingface_ner_trainer.os.cpu_count", return_value=8)
def test_get_training_args_caps_mps_batch_size(
    mock_cpu_count,
    mock_cuda_available,
    mock_mps_available,
    mock_set_num_threads,
):
    original_device = supervised_trainer._config.DEVICE
    supervised_trainer._config.DEVICE = "mps"

    with patch.object(
        HuggingFaceNerSupervisedTrainer,
        "_create_training_arguments",
        side_effect=lambda **kwargs: kwargs,
    ):
        training_args = supervised_trainer._get_training_args(
            "results",
            "logs",
            {"nepochs": 1, "scaling_factor": 10},
            1,
        )

    supervised_trainer._config.DEVICE = original_device

    assert training_args["per_device_train_batch_size"] == 8
    assert training_args["per_device_eval_batch_size"] == 8
    assert training_args["gradient_accumulation_steps"] == 2
    assert training_args["eval_accumulation_steps"] == 2
    mock_set_num_threads.assert_called_once_with(4)
