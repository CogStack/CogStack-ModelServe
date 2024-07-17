import json
import logging.config
import os
import sys
import uuid
import inspect
import warnings

current_frame = inspect.currentframe()
if current_frame is None:  # noqa
    raise Exception("Cannot detect the parent directory!")  # noqa
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(current_frame))))  # noqa
sys.path.insert(0, parent_dir)  # noqa
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import uvicorn  # noqa
import shutil  # noqa
import typer  # noqa
import graypy  # noqa
import aiohttp  # noqa
import asyncio  # noqa
import websockets  # noqa
import api.globals as cms_globals  # noqa

from logging import LogRecord  # noqa
from typing import Optional, Tuple, Dict, Any  # noqa
from urllib.parse import urlparse  # noqa
from fastapi.routing import APIRoute  # noqa
from domain import ModelType, TrainingType  # noqa
from registry import model_service_registry  # noqa
from api.api import get_model_server, get_stream_server # noqa
from utils import get_settings, send_gelf_message  # noqa
from management.model_manager import ModelManager  # noqa
from api.dependencies import ModelServiceDep, ModelManagerDep  # noqa
from management.tracker_client import TrackerClient  # noqa

cmd_app = typer.Typer(name="python cli.py", help="CLI for various CogStack ModelServe operations", add_completion=False)
stream_app = typer.Typer(name="python cli.py stream", help="This groups various stream operations", add_completion=False)
cmd_app.add_typer(stream_app, name="stream")
logging.config.fileConfig(os.path.join(parent_dir, "logging.ini"), disable_existing_loggers=False)
logger = logging.getLogger("cms")


@cmd_app.command("serve")
def serve_model(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                model_path: str = typer.Option(..., help="The file path to the model package"),
                mlflow_model_uri: str = typer.Option("", help="The URI of the MLflow model to serve", metavar="models:/MODEL_NAME/ENV"),
                host: str = typer.Option("127.0.0.1", help="The hostname of the server"),
                port: str = typer.Option("8000", help="The port of the server"),
                model_name: Optional[str] = typer.Option(None, help="The string representation of the model name"),
                streamable: bool = typer.Option(False, help="Serve the streamable endpoints only")) -> None:
    """
    This serves various CogStack NLP models
    """
    lrf = logging.getLogRecordFactory()

    def log_record_factory(*args: Tuple, **kwargs: Dict[str, Any]) -> LogRecord:
        record = lrf(*args, **kwargs)
        record.model_type = model_type
        record.model_name = model_name or "NULL"
        return record
    logging.setLogRecordFactory(log_record_factory)

    if "GELF_INPUT_URI" in os.environ and os.environ["GELF_INPUT_URI"]:
        try:
            uri = urlparse(os.environ["GELF_INPUT_URI"])
            send_gelf_message(f"Model service {model_type} is starting", uri)
            gelf_tcp_handler = graypy.GELFTCPHandler(uri.hostname, uri.port)
            logger.addHandler(gelf_tcp_handler)
            logging.getLogger("uvicorn").addHandler(gelf_tcp_handler)
        except Exception as e:
            logger.error(f"$GELF_INPUT_URI is set to \"{os.environ['GELF_INPUT_URI']}\" but it's not ready to receive logs")
            logger.exception(e)

    config = get_settings()

    model_service_dep = ModelServiceDep(model_type, config, model_name)
    cms_globals.model_service_dep = model_service_dep
    model_server_app = get_model_server()

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
        cms_globals.model_manager_dep = ModelManagerDep(model_service)
    elif mlflow_model_uri:
        model_service = ModelManager.retrieve_model_service_from_uri(mlflow_model_uri, config, dst_model_path)
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service_dep.model_service = model_service
        cms_globals.model_manager_dep = ModelManagerDep(model_service)
        model_server_app = get_model_server()
    else:
        logger.error("Neither the model path or the mlflow model uri was passed in")
        sys.exit(1)

    logger.info(f'Start serving model "{model_type}" on {host}:{port}')
    # interrupted = False
    # while not interrupted:
    uvicorn.run(model_server_app if not streamable else get_stream_server(), host=host, port=int(port), log_config=None)
    # interrupted = True
    print("Shutting down due to either keyboard interrupt or system exit")


@cmd_app.command("train")
def train_model(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                base_model_path: str = typer.Option("", help="The file path to the base model package to be trained on"),
                mlflow_model_uri: str = typer.Option("", help="The URI of the MLflow model to train", metavar="models:/MODEL_NAME/ENV"),
                training_type: TrainingType = typer.Option(..., help="The type of training"),
                data_file_path: str = typer.Option(..., help="The path to the training asset file"),
                epochs: int = typer.Option(1, help="The number of training epochs"),
                log_frequency: int = typer.Option(1, help="The number of processed documents after which training metrics will be logged"),
                hyperparameters: str = typer.Option("{}", help="The overriding hyperparameters serialised as JSON string"),
                description: Optional[str] = typer.Option(None, help="The description of the training or change logs"),
                model_name: Optional[str] = typer.Option(None, help="The string representation of the model name")) -> None:
    """
    This pretrains or fine-tunes various CogStack NLP models
    """
    lrf = logging.getLogRecordFactory()

    def log_record_factory(*args: Tuple, **kwargs: Dict[str, Any]) -> LogRecord:
        record = lrf(*args, **kwargs)
        record.model_type = model_type
        record.model_name = model_name or "NULL"
        return record
    logging.setLogRecordFactory(log_record_factory)

    config = get_settings()

    model_service_dep = ModelServiceDep(model_type, config)
    cms_globals.model_service_dep = model_service_dep

    dst_model_path = os.path.join(parent_dir, "model", "model.zip")
    if dst_model_path and os.path.exists(dst_model_path.replace(".zip", "")):
        shutil.rmtree(dst_model_path.replace(".zip", ""))
    if base_model_path:
        try:
            shutil.copy2(base_model_path, dst_model_path)
        except shutil.SameFileError:
            pass
        model_service = model_service_dep()
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service.init_model()
    elif mlflow_model_uri:
        model_service = ModelManager.retrieve_model_service_from_uri(mlflow_model_uri, config, dst_model_path)
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service_dep.model_service = model_service
    else:
        logger.error("Neither the model path or the mlflow model uri was passed in")
        sys.exit(1)

    training_id = str(uuid.uuid4())
    with open(data_file_path, "r") as data_file:
        training_args = [data_file, epochs, log_frequency, training_id, data_file.name, [data_file], description, True]
        if training_type == TrainingType.SUPERVISED and model_service._supervised_trainer is not None:
            model_service.train_supervised(*training_args, **json.loads(hyperparameters))
        elif training_type == TrainingType.UNSUPERVISED and model_service._unsupervised_trainer is not None:
            model_service.train_unsupervised(*training_args, **json.loads(hyperparameters))
        elif training_type == TrainingType.META_SUPERVISED and model_service._metacat_trainer is not None:
            model_service.train_metacat(*training_args, **json.loads(hyperparameters))
        else:
            logger.error(f"Training type {training_type} is not supported or the corresponding trainer has not been enabled in the .env file.")
            sys.exit(1)


@cmd_app.command("register")
def register_model(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                   model_path: str = typer.Option(..., help="The file path to the model package"),
                   model_name: str = typer.Option(..., help="The string representation of the registered model"),
                   training_type: Optional[str] = typer.Option(None, help="The type of training the model went through"),
                   model_config: Optional[str] = typer.Option(None, help="The string representation of a JSON object"),
                   model_metrics: Optional[str] = typer.Option(None, help="The string representation of a JSON array"),
                   model_tags: Optional[str] = typer.Option(None, help="The string representation of a JSON object")) -> None:
    """
    This pushes a pretrained NLP model to the CogStack ModelServe registry
    """

    config = get_settings()
    tracker_client = TrackerClient(config.MLFLOW_TRACKING_URI)

    if model_type in model_service_registry.keys():
        model_service_type = model_service_registry[model_type]
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)

    m_config = json.loads(model_config) if model_config is not None else None
    m_metrics = json.loads(model_metrics) if model_metrics is not None else None
    m_tags = json.loads(model_tags) if model_tags is not None else None
    t_type = training_type if training_type is not None else ""

    run_name = str(uuid.uuid4())
    tracker_client.save_pretrained_model(model_name=model_name,
                                         model_path=model_path,
                                         model_manager=ModelManager(model_service_type, config),
                                         training_type=t_type,
                                         run_name=run_name,
                                         model_config=m_config,
                                         model_metrics=m_metrics,
                                         model_tags=m_tags)
    print(f"Pushed {model_path} as a new model version ({run_name})")


@stream_app.command("json_lines", help="This gets NER entities as a JSON Lines stream")
def stream_jsonl_annotations(jsonl_file_path: str = typer.Option(..., help="The path to the JSON Lines file"),
                             base_url: str = typer.Option("http://127.0.0.1:8000", help="The CMS base url"),
                             timeout_in_secs: int = typer.Option(0, help="The max time to wait before disconnection")) -> None:
    async def get_jsonl_stream(base_url: str, jsonl_file_path: str) -> None:
        with open(jsonl_file_path) as file:
            headers = {"Content-Type": "application/x-ndjson"}
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=timeout_in_secs)
                    async with session.post(f"{base_url}/stream/process",
                                            data=file.read().encode("utf-8"),
                                            headers=headers,
                                            timeout=timeout) as response:
                        response.raise_for_status()
                        async for line in response.content:
                            print(line.decode("utf-8"), end="")
            finally:
                logger.debug("Closing the session...")
                await session.close()
                logger.debug("Session closed")

    asyncio.run(get_jsonl_stream(base_url, jsonl_file_path))


@stream_app.command("chat", help="This gets NER entities by chatting with the model")
def chat_to_get_jsonl_annotations(base_url: str = typer.Option("ws://127.0.0.1:8000", help="The CMS base url")) -> None:
    async def chat_with_model(base_url: str) -> None:
        try:
            chat_endpoint = f"{base_url}/stream/ws"
            async with websockets.connect(chat_endpoint, ping_interval=None) as websocket:
                async def keep_alive() -> None:
                    while True:
                        try:
                            await websocket.ping()
                            await asyncio.sleep(10)
                        except asyncio.CancelledError:
                            break

                keep_alive_task = asyncio.create_task(keep_alive())
                logging.info("Connected to CMS. Start typing you input and press <ENTER> to submit:")
                try:
                    while True:
                        text = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                        if text.strip() == "":
                            continue
                        try:
                            await websocket.send(text)
                            response = await websocket.recv()
                            print("CMS =>")
                            print(response)
                        except websockets.ConnectionClosed as e:
                            logger.error(f"Connection closed: {e}")
                            break
                        except Exception as e:
                            logger.error(f"Error while sending message: {e}")
                finally:
                    keep_alive_task.cancel()
                    await keep_alive_task
        except websockets.InvalidURI:
            logger.error(f"Invalid URI: {chat_endpoint}")
        except Exception as e:
            logger.error(f"Error: {e}")

    asyncio.run(chat_with_model(base_url))


@cmd_app.command("export-model-apis")
def generate_api_doc_per_model(model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
                               add_training_apis: bool = typer.Option(False, help="Add training APIs to the doc"),
                               add_evaluation_apis: bool = typer.Option(False, help="Add evaluation APIs to the doc"),
                               add_previews_apis: bool = typer.Option(False, help="Add preview APIs to the doc"),
                               add_user_authentication: bool = typer.Option(False, help="Add user authentication APIs to the doc"),
                               exclude_unsupervised_training: bool = typer.Option(False, help="Exclude the unsupervised training API"),
                               exclude_metacat_training: bool = typer.Option(False, help="Exclude the metacat training API"),
                               model_name: Optional[str] = typer.Option(None, help="The string representation of the model name")) -> None:
    """
    This generates model-specific API docs for enabled endpoints
    """

    settings = get_settings()
    settings.ENABLE_TRAINING_APIS = "true" if add_training_apis else "false"
    settings.DISABLE_UNSUPERVISED_TRAINING = "true" if exclude_unsupervised_training else "false"
    settings.DISABLE_METACAT_TRAINING = "true" if exclude_metacat_training else "false"
    settings.ENABLE_EVALUATION_APIS = "true" if add_evaluation_apis else "false"
    settings.ENABLE_PREVIEWS_APIS = "true" if add_previews_apis else "false"
    settings.AUTH_USER_ENABLED = "true" if add_user_authentication else "false"

    model_service_dep = ModelServiceDep(model_type, settings, model_name)
    cms_globals.model_service_dep = model_service_dep
    doc_name = f"{model_name or model_type}_model_apis.json"
    app = get_model_server()
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    with open(doc_name, "w") as api_doc:
        json.dump(app.openapi(), api_doc, indent=4)
    print(f"OpenAPI doc exported to {doc_name}")


@cmd_app.command("export-openapi-spec")
def generate_api_doc(api_title: str = typer.Option("CogStack Model Serve APIs", help="The string representation of the API title")) -> None:
    """
    This generates a single API doc for all endpoints
    """

    settings = get_settings()
    settings.ENABLE_TRAINING_APIS = "true"
    settings.DISABLE_UNSUPERVISED_TRAINING = "false"
    settings.DISABLE_METACAT_TRAINING = "false"
    settings.ENABLE_EVALUATION_APIS = "true"
    settings.ENABLE_PREVIEWS_APIS = "true"
    settings.AUTH_USER_ENABLED = "true"

    model_service_dep = ModelServiceDep(ModelType.MEDCAT_SNOMED, settings, api_title)
    cms_globals.model_service_dep = model_service_dep
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
