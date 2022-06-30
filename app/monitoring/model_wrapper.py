import mlflow
import pandas as pd
from typing import Type
from pandas import DataFrame
from mlflow.pyfunc import PythonModel, PythonModelContext
from model_services.base import AbstractModelService
from config import Settings


class ModelWrapper(PythonModel):

    def __init__(self, model_service_type: Type, config: Settings) -> None:
        self._model_service_type = model_service_type
        self._config = config
        self._model_service = None

    @staticmethod
    def get_model_service(mlflow_tracking_uri: str, model_uri: str) -> AbstractModelService:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_uri)
        return pyfunc_model.predict(pd.DataFrame())

    def load_context(self, context: PythonModelContext) -> None:
        model_service = self._model_service_type(self._config)
        model_service.model = self._model_service_type.load_model(context.artifacts["model_path"])
        self._model_service = model_service

    # This is hacky and used for getting a model service rather than making prediction
    def predict(self, context: PythonModelContext, model_input: DataFrame) -> AbstractModelService:
        del context
        del model_input
        return self._model_service
