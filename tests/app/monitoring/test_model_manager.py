import mlflow
import tempfile
import pandas as pd
from typing import Generator
from unittest.mock import Mock, call
from mlflow.pyfunc import PythonModelContext
from app.model_services.base import AbstractModelService
from app.management.model_manager import ModelManager
from app.config import Settings
from app.exception import ManagedModelException
from app.domain import Annotation
from app.utils import load_pydantic_object_from_dict


def test_retrieve_python_model_from_uri(mlflow_fixture):
    config = Settings()
    ModelManager.retrieve_python_model_from_uri("model_uri", config)
    mlflow.set_tracking_uri.assert_has_calls([call(config.MLFLOW_TRACKING_URI), call(config.MLFLOW_TRACKING_URI)])
    mlflow.pyfunc.load_model.assert_called_once_with(model_uri="model_uri")


def test_retrieve_model_service_from_uri(mlflow_fixture):
    config = Settings()
    model_service = ModelManager.retrieve_model_service_from_uri("model_uri", config)
    mlflow.set_tracking_uri.assert_has_calls([call(config.MLFLOW_TRACKING_URI), call(config.MLFLOW_TRACKING_URI)])
    mlflow.pyfunc.load_model.assert_called_once_with(model_uri="model_uri")
    assert model_service._config.BASE_MODEL_FULL_PATH == "model_uri"
    assert model_service._config == config


def test_download_model_package(mlflow_fixture):
    try:
        ModelManager.download_model_package("mlflow_tracking_uri", "/tmp")
    except ManagedModelException as e:
        assert "Cannot find the model package file inside artifacts downloaded from mlflow_tracking_uri" == str(e)


def test_get_code_path_list(mlflow_fixture):
    assert len(ModelManager.get_code_path_list()) == 12


def test_get_pip_requirements_from_file(mlflow_fixture):
    assert len(ModelManager.get_pip_requirements_from_file()) > 0


def test_save_model(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    with tempfile.TemporaryDirectory() as local_dir:
        model_manager.save_model(local_dir, ".")
        mlflow.pyfunc.save_model.assert_called_once_with(
            path=local_dir,
            python_model=model_manager,
            signature=model_manager.model_signature,
            code_path=model_manager.get_code_path_list(),
            pip_requirements=model_manager.get_pip_requirements_from_file(),
            artifacts={"model_path": "."},
        )


def test_load_context(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager.load_context(PythonModelContext({"model_path": "artifacts/model.zip"}, None))
    assert isinstance(model_manager._model_service, _MockedModelService)


def test_get_model_signature():
    model_manager = ModelManager(_MockedModelService, Settings())
    assert model_manager.model_signature.inputs.to_dict() == [
        {"type": "string", "name": "name", "required": False},
        {"type": "string", "name": "text", "required": True}
    ]
    assert model_manager.model_signature.outputs.to_dict() == [
        {"type": "string", "name": "doc_name", "required": True},
        {"type": "integer", "name": "start", "required": True},
        {"type": "integer", "name": "end", "required": True},
        {"type": "string", "name": "label_name", "required": True},
        {"type": "string", "name": "label_id", "required": True},
        {"type": "string", "name": "categories", "required": False},
        {"type": "float", "name": "accuracy", "required": False},
        {"type": "string", "name": "text", "required": False},
        {"type": "string", "name": "meta_anns", "required": False},
    ]


def test_predict(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager._model_service = Mock()
    model_manager._model_service.annotate = Mock()
    model_manager._model_service.annotate.return_value = [
        load_pydantic_object_from_dict(
            Annotation,
            {
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
            },
        )
    ]
    output = model_manager.predict(None, pd.DataFrame([{"name": "doc_1", "text": "text_1"}, {"name": "doc_2", "text": "text_2"}]))
    assert output.to_dict() == {
        "doc_name": {0: "doc_1", 1: "doc_2"},
        "label_name": {0: "Spinal stenosis", 1: "Spinal stenosis"},
        "label_id": {0: "76107001", 1: "76107001"},
        "start": {0: 0, 1: 0}, "end": {0: 15, 1: 15},
        "accuracy": {0: 1.0, 1: 1.0},
        "meta_anns": {0: {"Status": {"value": "Affirmed", "confidence": 0.9999833106994629, "name": "Status"}}, 1: {"Status": {"value": "Affirmed", "confidence": 0.9999833106994629, "name": "Status"}}}}


def test_predict_stream(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager._model_service = Mock()
    model_manager._model_service.annotate = Mock()
    model_manager._model_service.annotate.return_value = [
        load_pydantic_object_from_dict(
            Annotation,
            {
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
            },
        )
    ]
    output = model_manager.predict_stream(None, pd.DataFrame([{"name": "doc_1", "text": "text_1"}, {"name": "doc_2", "text": "text_2"}]))
    assert isinstance(output, Generator)
    assert list(output) == [
        {"doc_name": "doc_1", "label_name": "Spinal stenosis", "label_id": "76107001", "start": 0, "end": 15, "accuracy": 1.0, "meta_anns": {"Status": {"value": "Affirmed", "confidence": 0.9999833106994629, "name": "Status"}}},
        {"doc_name": "doc_2", "label_name": "Spinal stenosis", "label_id": "76107001", "start": 0, "end": 15, "accuracy": 1.0, "meta_anns": {"Status": {"value": "Affirmed", "confidence": 0.9999833106994629, "name": "Status"}}},
    ]


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
