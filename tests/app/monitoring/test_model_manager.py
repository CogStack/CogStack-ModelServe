import pytest
import mlflow
import tempfile
import pandas as pd
from unittest.mock import Mock, call
from mlflow.pyfunc import PythonModelContext
from model_services.base import AbstractModelService
from management.model_manager import ModelManager
from config import Settings
from exception import ManagedModelException

pyfunc_model = Mock()


@pytest.fixture
def mlflow_fixture(mocker):
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.pyfunc.load_model", return_value=pyfunc_model)
    mocker.patch("mlflow.pyfunc.log_model")
    mocker.patch("mlflow.artifacts.download_artifacts")
    mocker.patch("mlflow.register_model")
    mocker.patch("mlflow.pyfunc.save_model")


def test_retrieve_model_service_from_uri(mlflow_fixture):
    config = Settings()
    model_service = ModelManager.retrieve_model_service_from_uri("model_uri", "mlflow_tracking_uri", config)
    mlflow.set_tracking_uri.assert_has_calls([call("mlflow_tracking_uri"), call("mlflow_tracking_uri")])
    mlflow.pyfunc.load_model.assert_called_once_with(model_uri="model_uri")
    assert model_service._config.BASE_MODEL_FULL_PATH == "model_uri"
    assert model_service._config == config


def test_download_model_package(mlflow_fixture):
    try:
        ModelManager.download_model_package("mlflow_tracking_uri", "/tmp")
    except ManagedModelException as e:
        assert "Cannot find the model .zip file inside artifacts downloaded from mlflow_tracking_uri" == str(e)


def test_log_model_with_registration(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager.log_model("model_name", "filepath", "model_name")
    mlflow.pyfunc.log_model.assert_called_once_with(artifact_path="model_name",
                                                    python_model=model_manager,
                                                    signature=model_manager.model_signature,
                                                    code_path=model_manager._get_code_path_list(),
                                                    pip_requirements=model_manager._get_pip_requirements(),
                                                    artifacts={"model_path": "filepath"},
                                                    registered_model_name="model_name")


def test_log_model_without_registration(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager.log_model("model_name", "filepath")
    mlflow.pyfunc.log_model.assert_called_once_with(artifact_path="model_name",
                                                    python_model=model_manager,
                                                    signature=model_manager.model_signature,
                                                    code_path=model_manager._get_code_path_list(),
                                                    pip_requirements=model_manager._get_pip_requirements(),
                                                    artifacts={"model_path": "filepath"},
                                                    registered_model_name=None)


def test_save_model(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    with tempfile.TemporaryDirectory() as local_dir:
        model_manager.save_model(local_dir, ".")
        mlflow.pyfunc.save_model.assert_called_once_with(path=local_dir,
                                                         python_model=model_manager,
                                                         signature=model_manager.model_signature,
                                                         code_path=model_manager._get_code_path_list(),
                                                         pip_requirements=model_manager._get_pip_requirements(),
                                                         artifacts={"model_path": "."})


def test_load_context(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager.load_context(PythonModelContext({"model_path": "artifacts/model.zip"}, None))
    assert type(model_manager._model_service) == _MockedModelService


def test_get_model_signature():
    model_manager = ModelManager(_MockedModelService, Settings())
    assert model_manager.model_signature.inputs.to_dict() == [{"type": "string", "name": "name", "optional": True}, {"type": "string", "name": "text"}]
    assert model_manager.model_signature.outputs.to_dict() == [
        {"type": "string", "name": "doc_name"},
        {"type": "integer", "name": "start"},
        {"type": "integer", "name": "end"},
        {"type": "string", "name": "label_name"},
        {"type": "string", "name": "label_id"},
        {"type": "string", "name": "categories", "optional": True},
        {"type": "float", "name": "accuracy", "optional": True},
        {"type": "string", "name": "text", "optional": True},
        {"type": "string", "name": "meta_anns", "optional": True},
    ]


def test_predict(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager._model_service = Mock()
    model_manager._model_service.annotate = Mock()
    model_manager._model_service.annotate.return_value = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 0,
        "end": 15,
        "accuracy": 1.0,
        "meta_anns": {
            "Status": {
                "value": "Affirmed",
                "confidence": 0.9999833106994629,
                "name": "Status"
            }
        },
    }]
    output = model_manager.predict(None, pd.DataFrame([{"name": "doc_1", "text": "text_1"}, {"name": "doc_2", "text": "text_2"}]))
    assert output.to_dict() == {
        "doc_name": {0: "doc_1", 1: "doc_2"},
        "label_name": {0: "Spinal stenosis", 1: "Spinal stenosis"},
        "label_id": {0: "76107001", 1: "76107001"},
        "start": {0: 0, 1: 0}, "end": {0: 15, 1: 15},
        "accuracy": {0: 1.0, 1: 1.0},
        "meta_anns": {0: {"Status": {"value": "Affirmed", "confidence": 0.9999833106994629, "name": "Status"}}, 1: {"Status": {"value": "Affirmed", "confidence": 0.9999833106994629, "name": "Status"}}}}


class _MockedModelService(AbstractModelService):

    def __init__(self, config: Settings, *args, **kwargs) -> None:
        self._config = config
        self.model_name = "Mocked Model"

    @staticmethod
    def load_model(model_file_path, *args, **kwargs):
        return Mock()

    def info(self):
        return None

    def annotate(self, text):
        return None

    def batch_annotate(self, texts):
        return None

    def init_model(self):
        return None
