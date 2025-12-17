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
from app.model_services.base import AbstractModelService
from app.config import Settings
from app.exception import ManagedModelException
from app.utils import func_deprecated, pyproject_dependencies_to_pip_requirements


@final
class ModelManager(PythonModel):
    """
    A model manager class that manages the model service and provides interfaces to log,
    save, and predict on models with the CMS model flavour.

    Attributes:
        input_schema (Schema): The schema defining the expected input for the model.
        output_schema (Schema): The schema defining the expected output from the model.
    """

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
        """
        Initialises a model manager with a specific type of model service and its configuration.

        Args:
            model_service_type (Type): The type of the model service to be managed.
            config (Settings): Configuration for the model service.
        """
        self._model_service_type = model_service_type
        self._config = config
        self._model_service: Optional[AbstractModelService] = None
        self._model_signature = ModelSignature(
            inputs=ModelManager.input_schema,
            outputs=ModelManager.output_schema,
            params=None,
        )

    @property
    def model_service(self) -> Optional[AbstractModelService]:
        """Getter for the model service."""
        return self._model_service

    @model_service.setter
    def model_service(self, model_service: AbstractModelService) -> None:
        """Setter for the model service."""
        self._model_service = model_service

    @property
    def model_signature(self) -> ModelSignature:
        """Getter for the model signature."""
        return self._model_signature

    @staticmethod
    def retrieve_python_model_from_uri(mlflow_model_uri: str, config: Settings) -> PythonModel:
        """
        Retrieves the PythonModel instance from the specified MLflow model URI.

        Args:
            mlflow_model_uri (str): The URI of the MLflow model.
            config (Settings): The configuration for the model service.

        Returns:
            PythonModel: The retrieved PythonModel instance.
        """

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri=mlflow_model_uri)
        # In case the load_model overwrote the tracking URI
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        return pyfunc_model._model_impl.python_model

    @staticmethod
    def retrieve_model_service_from_uri(
        mlflow_model_uri: str,
        config: Settings,
        downloaded_model_path: Optional[str] = None,
    ) -> AbstractModelService:
        """
        Retrieves the model service from the specified MLflow model URI.

        Args:
            mlflow_model_uri (str): The URI of the MLflow model.
            config (Settings): The configuration for the model service.
            downloaded_model_path (Optional[str]): The local path to optionally save the downloaded model package.

        Returns:
            AbstractModelService: The model service retrieved from the URI.
        """

        model_manager = ModelManager.retrieve_python_model_from_uri(mlflow_model_uri, config)
        model_service = model_manager.model_service
        config.BASE_MODEL_FULL_PATH = mlflow_model_uri
        model_service._config = config
        if downloaded_model_path:
            ModelManager.download_model_package(os.path.join(mlflow_model_uri, "artifacts"), downloaded_model_path)
        return model_service

    @staticmethod
    def download_model_package(model_artifact_uri: str, dst_file_path: str) -> Optional[str]:
        """
        Downloads the model package from the specified model artifact URI and save it to the destination file path.

        Args:
            model_artifact_uri (str): The URI of the model artifact.
            dst_file_path (str): The local file path where the model package will be saved.

        Returns:
            Optional[str]: The destination file path if the model package is found and successfully downloaded, otherwise None.

        Raises:
            ManagedModelException: If the model package cannot be found inside the downloaded artifacts.
        """

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
    @staticmethod
    def get_code_path_list() -> List[str]:
        """
        Gets the list of code paths to be included in the registered model.

        Returns:
            List[str]: The list of code paths to be included.
        """

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

    @staticmethod
    def get_pip_requirements_from_file() -> Union[List[str], str]:
        """
        Gets the list of pip requirements from the pyproject.toml file or the requirements.txt file.

        Returns:
            Union[List[str], str]: The list of pip requirements or the path to the requirements.txt file.
        """

        if os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml"))):
            with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")), "r") as file:
                pyproject = toml.load(file)
                dependencies = pyproject.get("project", {}).get("dependencies", [])
                return pyproject_dependencies_to_pip_requirements(dependencies)
        elif os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"))):
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"))
        else:
            raise ManagedModelException("Cannot find pip requirements.")

    def save_model(self, local_dir: str, model_path: str) -> None:
        """
        Saves the model with the specified path into a local directory.

        Args:
            local_dir (str): The local directory where the model will be saved.
            model_path (str): The artifact path to the model.
        """

        mlflow.pyfunc.save_model(
            path=local_dir,
            python_model=self,
            artifacts={"model_path": model_path},
            signature=self.model_signature,
            code_path=ModelManager.get_code_path_list(),
            pip_requirements=ModelManager.get_pip_requirements_from_file(),
        )

    def load_context(self, context: PythonModelContext) -> None:
        """
        Loads artifacts from the context and initialise the model service.

        Args:
            context (PythonModelContext): The context containing the model artifacts.
        """

        artifact_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_service = self._model_service_type(
            self._config,
            model_parent_dir=os.path.join(artifact_root, os.path.split(context.artifacts["model_path"])[0]),
            base_model_file=os.path.split(context.artifacts["model_path"])[1],
        )
        model_service.init_model()
        self._model_service = model_service

    def predict(
        self,
        context: PythonModelContext,
        model_input: DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Predicts using the model service for the provided input data.

        Args:
            context (PythonModelContext): The context containing the model artifacts.
            model_input (DataFrame): The input data for prediction.
            params (Optional[Dict[str, Any]]): Additional parameters for prediction (not used in this implementation).

        Returns:
            pd.DataFrame: The inference results as a DataFrame instance.
        """

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

    def predict_stream(
       self,
       context: PythonModelContext,
       model_input: DataFrame,
       params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Predicts using the model service for the provided input data and yields results one by one.

        Args:
            context (PythonModelContext): The context containing the model artifacts.
            model_input (DataFrame): The input data for prediction.
            params (Optional[Dict[str, Any]]): Additional parameters for prediction (not used in this implementation).

        Returns:
            Iterator[Dict[str, Any]]: The iterator over the inference results as dictionaries.
        """

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
    @func_deprecated()
    def _get_pip_requirements() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "requirements.txt"))
