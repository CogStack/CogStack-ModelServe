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
import graypy
import globals

from typing import Optional
from urllib.parse import urlparse
from hypercorn.config import Config
from hypercorn.asyncio import serve
from fastapi.routing import APIRoute
from domain import ModelType
from registry import model_service_registry
from api import get_model_server
from utils import get_settings, send_gelf_message
from management.model_manager import ModelManager
from dependencies import ModelServiceDep
from config import Settings
from management.tracker_client import TrackerClient

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
cmd_app = typer.Typer(add_completion=False)


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

    if "GELF_INPUT_URI" in os.environ:
        try:
            uri = urlparse(os.environ["GELF_INPUT_URI"])
            send_gelf_message(f"Model service {model_type} is starting", uri)
            logger.addHandler(graypy.GELFTCPHandler(uri.hostname, uri.port))

            lrf = logging.getLogRecordFactory()

            def log_record_factory(*args, **kwargs):
                record = lrf(*args, **kwargs)
                record.model_type = model_type
                record.model_name = model_name or "NULL"
                return record

            logging.setLogRecordFactory(log_record_factory)
        except Exception:
            print(f"ERROR: $GELF_INPUT_URI is set to \"{os.environ['GELF_INPUT_URI']}\" but it's not ready to receive logs")

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
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service.init_model()
    elif mlflow_model_uri:
        model_service = ModelManager.get_model_service(settings.MLFLOW_TRACKING_URI,
                                                       mlflow_model_uri,
                                                       settings,
                                                       dst_model_path)
        model_service.model_name = model_name if model_name is not None else "CMS model"
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

    if model_type in model_service_registry.keys():
        model_service_type = model_service_registry[model_type]
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)

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


@cmd_app.command("export-model-apis")
def generate_api_doc_per_model(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                               add_training_apis: bool = typer.Option(False, help="Add training APIs to the doc"),
                               add_evaluation_apis: bool = typer.Option(False, help="Add evaluation APIs to the doc"),
                               add_previews_apis: bool = typer.Option(False, help="Add preview APIs to the doc"),
                               add_user_authentication: bool = typer.Option(False, help="Add user authentication APIs to the doc"),
                               exclude_unsupervised_training: bool = typer.Option(False, help="Exclude the unsupervised training API"),
                               exclude_metacat_training: bool = typer.Option(False, help="Exclude the metacat training API"),
                               model_name: Optional[str] = typer.Option(None, help="The string representation of the model name")):
    """
    This script generates API docs for enabled endpoints
    """

    settings = get_settings()
    settings.ENABLE_TRAINING_APIS = "true" if add_training_apis else "false"
    settings.DISABLE_UNSUPERVISED_TRAINING = "true" if exclude_unsupervised_training else "false"
    settings.DISABLE_METACAT_TRAINING = "true" if exclude_metacat_training else "false"
    settings.ENABLE_EVALUATION_APIS = "true" if add_evaluation_apis else "false"
    settings.ENABLE_PREVIEWS_APIS = "true" if add_previews_apis else "false"
    settings.AUTH_USER_ENABLED = "true" if add_user_authentication else "false"

    model_service_dep = ModelServiceDep(model_type, settings, model_name)
    globals.model_service_dep = model_service_dep
    doc_name = f"{model_name or model_type}_model_apis.json"
    app = get_model_server()
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    with open(doc_name, "w") as api_doc:
        json.dump(app.openapi(), api_doc, indent=4)
    print(f"OpenAPI doc exported to {doc_name}")


@cmd_app.command("export-openapi-spec")
def generate_api_doc(api_title: str = typer.Option("CogStack Model Serve APIs", help="The string representation of the API title")):
    """
    This script generates API docs for enabled endpoints
    """

    settings = get_settings()
    settings.ENABLE_TRAINING_APIS = "true"
    settings.DISABLE_UNSUPERVISED_TRAINING = "false"
    settings.DISABLE_METACAT_TRAINING = "false"
    settings.ENABLE_EVALUATION_APIS = "true"
    settings.ENABLE_PREVIEWS_APIS = "true"
    settings.AUTH_USER_ENABLED = "true"

    model_service_dep = ModelServiceDep(ModelType.MEDCAT_SNOMED, settings, api_title)
    globals.model_service_dep = model_service_dep
    doc_name = f"{api_title.lower().replace(' ', '_')}.json"
    app = get_model_server()
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    with open(doc_name, "w") as api_doc:
        openapi = app.openapi()
        openapi["info"]["title"] = api_title
        json.dump(app.openapi(), api_doc, indent=4)
    print(f"OpenAPI doc exported to {doc_name}")


if __name__ == "__main__":
    cmd_app()
