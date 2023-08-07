import glob
import os
import shutil
import tempfile
import mlflow
import pandas as pd
from typing import Type, Optional
from pandas import DataFrame
from mlflow.pyfunc import PythonModel, PythonModelContext
from model_services.base import AbstractModelService
from config import Settings


class ModelManager(PythonModel):

    def __init__(self, model_service_type: Type, config: Settings) -> None:
        self._model_service_type = model_service_type
        self._config = config
        self._model_service = None

    @staticmethod
    def get_model_service(mlflow_tracking_uri: str,
                          mlflow_model_uri: str,
                          config: Optional[Settings] = None,
                          downloaded_model_path: Optional[str] = None) -> AbstractModelService:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri=mlflow_model_uri)
        # In case the load_model overwrote the tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        model_service = pyfunc_model.predict(pd.DataFrame())
        if config is not None:
            config.BASE_MODEL_FULL_PATH = mlflow_model_uri
            model_service._config = config
        if downloaded_model_path:
            ModelManager.download_model_package(os.path.join(mlflow_model_uri, "artifacts"), downloaded_model_path)
        return model_service

    @staticmethod
    def download_model_package(model_artifact_uri: str, dst_file_path: str) -> Optional[str]:
        with tempfile.TemporaryDirectory() as dir_downloaded:
            mlflow.artifacts.download_artifacts(artifact_uri=model_artifact_uri, dst_path=dir_downloaded)
            # This assumes the model package is the sole zip file in the artifacts directory
            file_path = None
            for file_path in glob.glob(os.path.join(dir_downloaded, "**", "*.zip")):
                break
            if file_path:
                shutil.copy(file_path, dst_file_path)
                return dst_file_path
            else:
                raise ValueError(f"Cannot find the model .zip file inside artifacts downloaded from {model_artifact_uri}")

    def load_context(self, context: PythonModelContext) -> None:
        model_service = self._model_service_type(self._config)
        model_service._model_file_path = context.artifacts["model_path"]
        model_service.init_model()
        self._model_service = model_service

    # This is hacky and used for getting a model service rather than making prediction
    def predict(self, context: PythonModelContext, model_input: DataFrame) -> AbstractModelService:
        del context
        del model_input
        return self._model_service
