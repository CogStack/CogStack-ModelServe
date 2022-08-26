import pytest
import mlflow
import pandas as pd
from unittest.mock import Mock
from mlflow.pyfunc import PythonModelContext
from app.model_services.base import AbstractModelService
from app.monitoring.model_wrapper import ModelWrapper
from app.config import Settings

pyfunc_model = Mock()


@pytest.fixture
def mlflow_fixture(mocker):
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.pyfunc.load_model", return_value=pyfunc_model)


def test_get_model_service(mlflow_fixture):
    ModelWrapper.get_model_service("mlflow_tracking_uri", "model_uri")
    mlflow.set_tracking_uri.assert_called_once_with("mlflow_tracking_uri")
    mlflow.pyfunc.load_model.assert_called_once_with(model_uri="model_uri")
    pyfunc_model.predict.assert_called_once()


def test_load_context(mlflow_fixture):
    model_wrapper = ModelWrapper(_MockedModelService, Settings())
    model_wrapper.load_context(PythonModelContext({"model_path": "model_path"}))
    assert type(model_wrapper._model_service) == _MockedModelService


def test_predict(mlflow_fixture):
    model_wrapper = ModelWrapper(_MockedModelService, Settings())
    model_service = model_wrapper.predict(PythonModelContext({"model_path": "model_path"}), pd.DataFrame())
    assert model_service == model_wrapper._model_service


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
