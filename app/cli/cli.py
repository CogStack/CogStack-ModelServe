import json
import logging.config
import os
import sys
import uuid


from parent_dir import parent_dir  # noqa

import asyncio
import shutil
import warnings
import typer
import globals

from typing import Optional
from hypercorn.config import Config
from hypercorn.asyncio import serve
from domain import ModelType
from api import get_settings, get_model_server
from management.model_manager import ModelManager
from dependencies import ModelServiceDep
from config import Settings
from management.tracker_client import TrackerClient

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
cmd_app = typer.Typer(add_completion=False)


@cmd_app.command("apidoc")
def generate_api_doc(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                     add_training_apis: bool = typer.Option(False, help="Add training APIs to the doc"),
                     exclude_unsupervised_training: bool = typer.Option(False, help="Exclude the unsupervised training API")):
    settings = get_settings()
    settings.ENABLE_TRAINING_APIS = "true" if add_training_apis else "false"
    settings.DISABLE_UNSUPERVISED_TRAINING = "true" if exclude_unsupervised_training else "false"
    model_service_dep = ModelServiceDep(model_type, settings)
    globals.model_service_dep = model_service_dep
    app = get_model_server()
    doc_name = ""
    if model_type == ModelType.MEDCAT_SNOMED.value:
        doc_name = "medcat_snomed_model_apis.json"
    elif model_type == ModelType.MEDCAT_SNOMED.value:
        doc_name = "medcat_umls_model_apis.json"
    elif model_type == ModelType.MEDCAT_ICD10.value:
        doc_name = "medcat_icd10_model_apis.json"
    elif model_type == ModelType.MEDCAT_DEID.value:
        doc_name = "medcat_deidentification_model_apis.json"
    elif model_type == ModelType.TRANSFORMERS_DEID.value:
        doc_name = "de-identification_model_apis.json"
    with open(doc_name, "w") as api_doc:
        json.dump(app.openapi(), api_doc, indent=4)
    print(f"OpenAPI doc exported to {doc_name}")
    sys.exit(0)


@cmd_app.command("serve")
def serve_model(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                model_path: str = typer.Option("", help="The file path to the model package"),
                mlflow_model_uri: str = typer.Option("", help="The URI of the MLflow model to serve", metavar="models:/MODEL_NAME/ENV"),
                host: str = typer.Option("0.0.0.0", help="The hostname of the server"),
                port: str = typer.Option("8000", help="The port of the server"),
                model_name: Optional[str] = typer.Option(None, help="The string representation of the model name"),):
    """
    This script serves various CogStack NLP models
    """
    logging.config.fileConfig(os.path.join(parent_dir, "logging.ini"), disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    settings = get_settings()

    model_service_dep = ModelServiceDep(model_type, settings)
    globals.model_service_dep = model_service_dep
    app = get_model_server()

    dst_model_path = os.path.join(parent_dir, "model", "model.zip")
    if dst_model_path and os.path.exists(dst_model_path.replace(".zip", "")):
        shutil.rmtree(dst_model_path.replace(".zip", ""))
    if model_path:
        try:
            shutil.copy2(model_path, dst_model_path)
        except shutil.SameFileError:
            pass
        model_service = model_service_dep()
        if model_name is not None:
            model_service.model_name = model_name
        model_service.init_model()
    elif mlflow_model_uri:
        model_service = ModelManager.get_model_service(settings.MLFLOW_TRACKING_URI,
                                                       mlflow_model_uri,
                                                       settings,
                                                       dst_model_path)
        if model_name is not None:
            model_service.model_name = model_name
        model_service_dep.model_service = model_service
        app = get_model_server()
    else:
        print("Error: Neither the model path or the mlflow model uri was passed in")
        sys.exit(1)

    config = Config()
    config.bind = [f"{host}:{port}"]
    config.access_log_format = "%(R)s %(s)s %(st)s %(D)s %({Header}o)s"
    config.accesslog = logger
    asyncio.run(serve(app, config))  # type: ignore


@cmd_app.command("register")
def register_model(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                   model_path: str = typer.Option(..., help="The file path to the model package"),
                   model_name: str = typer.Option(..., help="The string representation of the registered model"),
                   training_type: Optional[str] = typer.Option(None, help="The type of training the model went through"),
                   model_config: Optional[str] = typer.Option(None, help="The string representation of a JSON object"),
                   model_metrics: Optional[str] = typer.Option(None, help="The string representation of a JSON array"),
                   model_tags: Optional[str] = typer.Option(None, help="The string representation of a JSON object")):
    """
    This script pushes a pretrained NLP model to the Cogstack ModelServe registry
    """

    config = Settings()
    tracker_client = TrackerClient(config.MLFLOW_TRACKING_URI)

    if (model_type == ModelType.MEDCAT_SNOMED.value
            or model_type == ModelType.MEDCAT_UMLS.value):
        from model_services.medcat_model import MedCATModel
        model_service_type = MedCATModel
    elif model_type == ModelType.MEDCAT_ICD10.value:
        from model_services.medcat_model_icd10 import MedCATModelIcd10
        model_service_type = MedCATModelIcd10
    elif model_type == ModelType.MEDCAT_DEID.value:
        from model_services.medcat_model_deid import MedCATModelDeIdentification
        model_service_type = MedCATModelDeIdentification
    elif model_type == ModelType.TRANSFORMERS_DEID.value:
        from model_services.trf_model_deid import TransformersModelDeIdentification
        model_service_type = TransformersModelDeIdentification
    else:
        print(f"Unknown model type: {model_type}")
        exit(1)

    m_config = json.loads(model_config) if model_config is not None else None
    m_metrics = json.loads(model_metrics) if model_metrics is not None else None
    m_tags = json.loads(model_tags) if model_tags is not None else None
    t_type = training_type if training_type is not None else ""

    run_name = str(uuid.uuid4())
    tracker_client.save_pretrained_model(model_name=model_name,
                                         model_path=model_path,
                                         pyfunc_model=ModelManager(model_service_type, config),
                                         training_type=t_type,
                                         run_name=run_name,
                                         model_config=m_config,
                                         model_metrics=m_metrics,
                                         model_tags=m_tags)
    print(f"Pushed {model_path} as a new model version ({run_name})")


if __name__ == "__main__":
    cmd_app()
