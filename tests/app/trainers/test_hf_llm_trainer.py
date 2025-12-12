import os
from unittest import skipIf
from unittest.mock import create_autospec, patch, Mock
from transformers import PreTrainedTokenizerFast
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel
from app.trainers.huggingface_llm_trainer import HuggingFaceLlmSupervisedTrainer
from app.config import Settings


def _triton_installed():
    try:
        import triton
        return True
    except ImportError:
        return False

model_parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")
config = Settings()
config.MLFLOW_TRACKING_URI = "http://localhost:5000"
config.TRAINING_CACHE_DIR = "/tmp/test_cache"
config.REDEPLOY_TRAINED_MODEL = "false"
config.SKIP_SAVE_MODEL = "false"
config.DEVICE = "cpu"
config.TRAINING_SAFE_MODEL_SERIALISATION = "true"

model_service = create_autospec(
    HuggingFaceLlmModel,
    _config=config,
    _model_parent_dir=model_parent_dir,
    _enable_trainer=True,
    _model_pack_path=os.path.join(model_parent_dir, "model.zip"),
)
model_service.tokenizer = Mock(spec=PreTrainedTokenizerFast)
model_service.model = Mock()
model_service.model.config.max_position_embeddings = 512
model_service.model_name = "llm_test_model"

supervised_trainer = HuggingFaceLlmSupervisedTrainer(model_service)
supervised_trainer.model_name = "supervised_trainer"

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")

def test_deploy_model():
    model = Mock()
    tokenizer = Mock()
    HuggingFaceLlmSupervisedTrainer.deploy_model(model_service, model, tokenizer)
    assert model_service.model == model
    assert model_service.tokenizer == tokenizer


@skipIf(not _triton_installed(), "This requires triton to be installed")
def test_huggingface_llm_supervised_trainer(mlflow_fixture):
    with patch.object(supervised_trainer, "run", wraps=supervised_trainer.run) as run:
        supervised_trainer._tracker_client = Mock()
        supervised_trainer._tracker_client.start_tracking = Mock(return_value=("experiment_id", "run_id"))
        with open(os.path.join(data_dir, "sample_qa.json"), "r") as f:
            supervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
    supervised_trainer._tracker_client.start_tracking.assert_called_once()
    run.assert_called_once()


@skipIf(config.DEVICE != "cuda", "This requires a CUDA device to run")
def test_huggingface_llm_supervised_run(mlflow_fixture):
    with open(os.path.join(data_dir, "sample_qa.json"), "r") as data_file:
        HuggingFaceLlmSupervisedTrainer.run(supervised_trainer, {"nepochs": 1, "print_stats": 1}, data_file, 1, "run_id")
