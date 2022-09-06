import pytest
import mlflow
import pandas as pd
from unittest.mock import Mock, patch
from mlflow.pyfunc import PythonModelContext
from app.model_services.base import AbstractModelService
from app.management.model_manager import ModelManager
from app.config import Settings

pyfunc_model = Mock()

@pytest.fixture
def mlflow_fixture(mocker):
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.pyfunc.load_model", return_value=pyfunc_model)
    mocker.patch("mlflow.artifacts.download_artifacts")


def test_get_model_service(mlflow_fixture):
    config = Settings()
    model_service = ModelManager.get_model_service("mlflow_tracking_uri", "model_uri", config)
    mlflow.set_tracking_uri.assert_called_once_with("mlflow_tracking_uri")
    mlflow.pyfunc.load_model.assert_called_once_with(model_uri="model_uri")
    pyfunc_model.predict.assert_called_once()
    assert model_service._config.BASE_MODEL_FULL_PATH == "model_uri"
    assert model_service._config == config


def test_download_model_package(mlflow_fixture):
    try:
        ModelManager.download_model_package("mlflow_tracking_uri", "/tmp")
    except ValueError as e:
        assert "Cannot find the model .zip file inside artifacts downloaded from mlflow_tracking_uri" == str(e)


def test_load_context(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_manager.load_context(PythonModelContext({"model_path": "model_path"}))
    assert type(model_manager._model_service) == _MockedModelService


def test_predict(mlflow_fixture):
    model_manager = ModelManager(_MockedModelService, Settings())
    model_service = model_manager.predict(PythonModelContext({"model_path": "model_path"}), pd.DataFrame())
    assert model_service == model_manager._model_service


class _MockedModelService(AbstractModelService):

    @staticmethod
    def load_model(model_file_path, *args, **kwargs):
        return Mock()

    def info(self):
        return None

    def annotate(self, text: str):
        return None

    def batch_annotate(self, texts):
        return None

    def init_model(self):
        return None
