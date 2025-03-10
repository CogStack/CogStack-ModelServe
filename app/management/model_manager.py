import glob
import os
import shutil
import tempfile
import mlflow
import toml
import pandas as pd
from typing import Type, Optional, Dict, Any, List, Iterator, final, Union
from pandas import DataFrame
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
from mlflow.models.model import ModelInfo
from app.model_services.base import AbstractModelService
from app.config import Settings
from app.exception import ManagedModelException
from app.utils import func_deprecated


@final
class ModelManager(PythonModel):

    input_schema = Schema([
        ColSpec(DataType.string, "name", optional=True),
        ColSpec(DataType.string, "text"),
    ])

    output_schema = Schema([
        ColSpec(DataType.string, "doc_name"),
        ColSpec(DataType.integer, "start"),
        ColSpec(DataType.integer, "end"),
        ColSpec(DataType.string, "label_name"),
        ColSpec(DataType.string, "label_id"),
        ColSpec(DataType.string, "categories", optional=True),
        ColSpec(DataType.float, "accuracy", optional=True),
        ColSpec(DataType.string, "text", optional=True),
        ColSpec(DataType.string, "meta_anns", optional=True)
    ])

    def __init__(self, model_service_type: Type, config: Settings) -> None:
        self._model_service_type = model_service_type
        self._config = config
        self._model_service: Optional[AbstractModelService] = None
        self._model_signature = ModelSignature(inputs=ModelManager.input_schema,
                                               outputs=ModelManager.output_schema,
                                               params=None)

    @property
    def model_service(self) -> Optional[AbstractModelService]:
        return self._model_service

    @model_service.setter
    def model_service(self, model_service: AbstractModelService) -> None:
        self._model_service = model_service

    @property
    def model_signature(self) -> ModelSignature:
        return self._model_signature

    @staticmethod
    def retrieve_python_model_from_uri(mlflow_model_uri: str,
                                       config: Settings) -> PythonModel:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri=mlflow_model_uri)
        # In case the load_model overwrote the tracking URI
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        return pyfunc_model._model_impl.python_model

    @staticmethod
    def retrieve_model_service_from_uri(mlflow_model_uri: str,
                                        config: Settings,
                                        downloaded_model_path: Optional[str] = None) -> AbstractModelService:
        model_manager = ModelManager.retrieve_python_model_from_uri(mlflow_model_uri, config)
        model_service = model_manager.model_service
        config.BASE_MODEL_FULL_PATH = mlflow_model_uri
        model_service._config = config
        if downloaded_model_path:
            ModelManager.download_model_package(os.path.join(mlflow_model_uri, "artifacts"), downloaded_model_path)
        return model_service

    @staticmethod
    def download_model_package(model_artifact_uri: str, dst_file_path: str) -> Optional[str]:
        # This assumes the model package is the sole zip or tar.gz file in the artifacts directory
        with tempfile.TemporaryDirectory() as dir_downloaded:
            mlflow.artifacts.download_artifacts(artifact_uri=model_artifact_uri, dst_path=dir_downloaded)
            file_path = None
            zip_files = glob.glob(os.path.join(dir_downloaded, "**", "*.zip"))
            gztar_files = glob.glob(os.path.join(dir_downloaded, "**", "*.tar.gz"))
            for file_path in (zip_files + gztar_files):
                break
            if file_path:
                shutil.copy(file_path, dst_file_path)
                return dst_file_path
            else:
                raise ManagedModelException(
                    f"Cannot find the model package file inside artifacts downloaded from {model_artifact_uri}"
                )

    def log_model(self,
                  model_name: str,
                  model_path: str,
                  registered_model_name: Optional[str] = None) -> ModelInfo:
        return mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=self,
            artifacts={"model_path": model_path},
            signature=self.model_signature,
            code_path=ModelManager._get_code_path_list(),
            pip_requirements=ModelManager._get_pip_requirements_from_file(),
            registered_model_name=registered_model_name,
        )

    def save_model(self, local_dir: str, model_path: str) -> None:
        mlflow.pyfunc.save_model(
            path=local_dir,
            python_model=self,
            artifacts={"model_path": model_path},
            signature=self.model_signature,
            code_path=ModelManager._get_code_path_list(),
            pip_requirements=ModelManager._get_pip_requirements_from_file(),
        )

    def load_context(self, context: PythonModelContext) -> None:
        artifact_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_service = self._model_service_type(self._config,
                                                 model_parent_dir=os.path.join(artifact_root, os.path.split(context.artifacts["model_path"])[0]),
                                                 base_model_file=os.path.split(context.artifacts["model_path"])[1])
        model_service.init_model()
        self._model_service = model_service

    def predict(self,
                context: PythonModelContext,
                model_input: DataFrame,
                params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        output = []
        for idx, row in model_input.iterrows():
            annotations = self._model_service.annotate(row["text"])  # type: ignore
            for annotation in annotations:
                annotation = {
                    "doc_name": row["name"] if "name" in row else str(idx),
                    **annotation.dict(exclude_none=True)
                }
                output.append(annotation)
        df = pd.DataFrame(output)
        df = df.iloc[:, df.columns.isin(ModelManager.output_schema.input_names())]
        return df

    def predict_stream(self,
                       context: PythonModelContext,
                       model_input: DataFrame,
                       params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        for idx, row in model_input.iterrows():
            annotations = self._model_service.annotate(row["text"])  # type: ignore
            output = []
            for annotation in annotations:
                annotation = {
                    "doc_name": row["name"] if "name" in row else str(idx),
                    **annotation.dict(exclude_none=True)
                }
                output.append(annotation)
            df = pd.DataFrame(output)
            df = df.iloc[:, df.columns.isin(ModelManager.output_schema.input_names())]
            for _, item in df.iterrows():
                yield item.to_dict()

    @staticmethod
    def _get_code_path_list() -> List[str]:
        return [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "management")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_services")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processors")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trainers")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "__init__.py")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.py")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domain.py")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exception.py")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "registry.py")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils.py")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logging.ini")),
        ]

    @staticmethod   # type: ignore
    @func_deprecated
    def _get_pip_requirements() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"))

    @staticmethod
    def _get_pip_requirements_from_file() -> Union[List[str], str]:
        if os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml"))):
            with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")), "r") as file:
                pyproject = toml.load(file)
                return pyproject.get("project", {}).get("dependencies", [])
        elif os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"))):
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"))
        else:
            raise ManagedModelException("Cannot find pip requirements.")
