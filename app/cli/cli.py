#!/usr/bin/env python

import json
import logging.config
import os
import sys
import uuid
import inspect
import warnings
import subprocess

current_frame = inspect.currentframe()
if current_frame is None:  # noqa
    raise Exception("Cannot detect the parent directory!")  # noqa
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(current_frame))))  # noqa
sys.path.insert(0, os.path.join(parent_dir, ".."))  # noqa
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import uvicorn  # noqa
import shutil  # noqa
import tempfile  # noqa
import typer  # noqa
import graypy  # noqa
import aiohttp  # noqa
import asyncio  # noqa
import websockets  # noqa
import app.api.globals as cms_globals  # noqa

from logging import LogRecord  # noqa
from typing import Optional, Tuple, Dict, Any  # noqa
from urllib.parse import urlparse  # noqa
from fastapi import FastAPI # noqa
from fastapi.routing import APIRoute  # noqa
from huggingface_hub import snapshot_download  # noqa
from datasets import load_dataset  # noqa
from app import __version__  # noqa
from app.config import Settings  # noqa
from app.domain import ModelType, TrainingType, BuildBackend, Device, ArchiveFormat, LlmEngine  # noqa
from app.registry import model_service_registry  # noqa
from app.api.api import (
    get_model_server,
    get_stream_server,
    get_generative_server,
    get_vllm_server,
    get_app_for_api_docs,
)   # noqa
from app.utils import get_settings, send_gelf_message, download_model_package, get_model_data_package_base_name  # noqa
from app.management.model_manager import ModelManager  # noqa
from app.api.dependencies import ModelServiceDep, ModelManagerDep  # noqa
from app.management.tracker_client import TrackerClient  # noqa

cmd_app = typer.Typer(name="cms", help="CLI for various CogStack ModelServe operations", add_completion=True)
stream_app = typer.Typer(name="stream", help="This groups various stream operations", add_completion=True)
cmd_app.add_typer(stream_app, name="stream")
package_app = typer.Typer(name="package", help="This groups various package operations", add_completion=True)
cmd_app.add_typer(package_app, name="package")
logging.config.fileConfig(os.path.join(parent_dir, "logging.ini"), disable_existing_loggers=False)

@cmd_app.command("serve", help="This serves various CogStack NLP models")
def serve_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
    model_path: str = typer.Option("", help="Either the file path to the local model package or the URL to the remote one"),
    mlflow_model_uri: str = typer.Option("", help="The URI of the MLflow model to serve", metavar="models:/MODEL_NAME/ENV"),
    host: str = typer.Option("127.0.0.1", help="The hostname of the server"),
    port: str = typer.Option("8000", help="The port of the server"),
    model_name: Optional[str] = typer.Option(None, help="The string representation of the model name"),
    streamable: bool = typer.Option(False, help="Serve the streamable endpoints only"),
    device: Device = typer.Option(Device.DEFAULT.value, help="The device to serve the model on"),
    llm_engine: Optional[LlmEngine] = typer.Option(LlmEngine.CMS.value, help="The engine to use for text generation"),
    load_in_4bit: Optional[bool] = typer.Option(False, help="Load the model in 4-bit precision, used by 'huggingface_llm' models"),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
    """
    Starts model serving endpoints.

    This function initialises the model service from either a local model package or the model registry.

    Args:
        model_type (ModelType): The type of the model to serve.
        model_path (str): Either the file path to the local model package or the URL to the remote one. Not required if mlflow_model_uri is provided.
        mlflow_model_uri (str): The URI of the MLflow model to serve. Not required if model_path is provided.
        host (str): The hostname of the server. Defaults to "127.0.0.1".
        port (str): The port of the server. Defaults to "8000".
        model_name (Optional[str]): The optional string representation of the model name.
        streamable (bool): Serve the streamable endpoints only. Defaults to False.
        device (Device): The device to serve the model on. Defaults to Device.DEFAULT.
        llm_engine (LlmEngine): The inference engine to use. Defaults to LlmEngine.CMS.
        load_in_4bit (bool): Load the model in 4-bit precision, used by 'huggingface_llm' models. Defaults to False.
        debug (Optional[bool]): Run in debug mode if set to True.
    """

    _display_info_table(model_type, model_name, model_path, mlflow_model_uri, host, port)

    model_name = model_name or "CMS model"
    logger = _get_logger(debug, model_type, model_name)
    config = get_settings()
    config.DEVICE = device
    if model_type in [
        ModelType.HUGGINGFACE_NER,
        ModelType.MEDCAT_DEID,
        ModelType.TRANSFORMERS_DEID,
    ]:
        config.DISABLE_METACAT_TRAINING = "true"

    if "GELF_INPUT_URI" in os.environ and os.environ["GELF_INPUT_URI"]:
        try:
            uri = urlparse(os.environ["GELF_INPUT_URI"])
            send_gelf_message(f"Model service {model_type} is starting", uri)
            gelf_tcp_handler = graypy.GELFTCPHandler(uri.hostname, uri.port)
            logger.addHandler(gelf_tcp_handler)
            logging.getLogger("uvicorn").addHandler(gelf_tcp_handler)
        except Exception:
            logger.exception("$GELF_INPUT_URI is set to \"%s\" but it's not ready to receive logs", os.environ['GELF_INPUT_URI'])

    logging.info("Preparing the model service for %s...", model_name)
    model_service_dep = ModelServiceDep(model_type, config, model_name)
    cms_globals.model_service_dep = model_service_dep

    dst_model_path = _ensure_dst_model_path(model_path, parent_dir, config)

    if model_path:
        if model_path.startswith("http://") or model_path.startswith("https://"):
            try:
                download_model_package(model_path, dst_model_path)
                logger.info("Model package successfully downloaded from %s to %s", model_path, dst_model_path)
            except Exception as e:
                logger.error("Failed to download model package from %s: %s", model_path, e)
                typer.Exit(code=1)
        else:
            try:
                shutil.copy2(model_path, dst_model_path)
            except shutil.SameFileError:
                logger.warning("Source and destination are the same model package file.")
                pass

    if llm_engine is not LlmEngine.VLLM:
        if model_path:
            model_service = model_service_dep()
            model_service.model_name = model_name
            model_service.init_model(load_in_4bit=load_in_4bit)
            cms_globals.model_manager_dep = ModelManagerDep(model_service)
        elif mlflow_model_uri:
            model_service = ModelManager.retrieve_model_service_from_uri(mlflow_model_uri, config, dst_model_path)
            model_service.model_name = model_name
            model_service_dep.model_service = model_service
            cms_globals.model_manager_dep = ModelManagerDep(model_service)
        else:
            logger.error("Neither the model path or the mlflow model uri was passed in")
            typer.Exit(code=1)

    model_server_app: Optional[FastAPI] = None
    if model_type in [ModelType.HUGGINGFACE_LLM]:
        if llm_engine == LlmEngine.CMS:
            model_server_app = get_generative_server(config)
        elif llm_engine == LlmEngine.VLLM:
            model_server_app = get_vllm_server(
                config,
                dst_model_path,
                model_name,
                log_level="debug" if debug else "info"
            )
        else:
            logger.error("Unknown LLM engine: %s" % llm_engine)
            typer.Exit(code=1)
    elif streamable:
        model_server_app = get_stream_server(config)
    else:
        model_server_app = get_model_server(config)

    # if model_server_app is not None:
    logger.info('Start serving model "%s" on %s:%s', model_type, host, port)
    # interrupted = False
    # while not interrupted:
    uvicorn.run(model_server_app, host=host, port=int(port), log_config=None)   # type: ignore
    # interrupted = True
    typer.echo("Shutting down due to either keyboard interrupt or system exit")


@cmd_app.command("train", help="This pretrains or fine-tunes various CogStack NLP models")
def train_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to train"),
    base_model_path: str = typer.Option("", help="The file path to the base model package to be trained on"),
    mlflow_model_uri: str = typer.Option("", help="The URI of the MLflow model to train", metavar="models:/MODEL_NAME/ENV"),
    training_type: TrainingType = typer.Option(..., help="The type of training"),
    data_file_path: str = typer.Option(..., help="The path to the training asset file"),
    epochs: int = typer.Option(1, help="The number of training epochs"),
    log_frequency: int = typer.Option(1, help="The number of processed documents or epochs after which training metrics will be logged"),
    hyperparameters: str = typer.Option("{}", help="The overriding hyperparameters serialised as JSON string"),
    description: Optional[str] = typer.Option(None, help="The description of the training or change logs"),
    model_name: Optional[str] = typer.Option(None, help="The string representation of the model name"),
    device: Device = typer.Option(Device.DEFAULT.value, help="The device to train the model on"),
    load_in_4bit: Optional[bool] = typer.Option(False, help="Load the model in 4-bit precision, used by 'huggingface_llm' models"),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
    """
    Executes model retraining or fine-tuning.

    This function runs retraining or fine-tuning and waits for its completion.

    Args:
        model_type (ModelType): The type of the model to train.
        base_model_path (str): The file path to the model package. Not required if mlflow_model_uri is provided.
        mlflow_model_uri (str): The URI of the MLflow model to serve. Not required if model_path is provided.
        training_type (TrainingType): The training methodology (supervised, unsupervised, meta_supervised).
        data_file_path (str): The path to training data in the supported format.
        epochs (int): THe number of complete passes through training data.
        log_frequency (int): The number of processed documents or epochs after which training metrics will be logged.
        hyperparameters (str): The JSON string of hyperparameter overrides, e.g., {\"lr_override\": 0.00005, \"test_size\": 0.3}.
        description (Optional[str]): The optional description of the training or change logs.
        model_name (Optional[str]): The optional string representation of the model name.
        device (Device): The device to train the model on. Defaults to Device.DEFAULT.
        load_in_4bit (bool): Load the model in 4-bit precision, used by 'huggingface_llm' models. Defaults to False.
        debug (Optional[bool]): Run in debug mode if set to True.
    """

    logger = _get_logger(debug, model_type, model_name)

    config = get_settings()
    config.DEVICE = device

    model_service_dep = ModelServiceDep(model_type, config)
    cms_globals.model_service_dep = model_service_dep

    dst_model_path = _ensure_dst_model_path(base_model_path, parent_dir, config)

    if base_model_path:
        try:
            shutil.copy2(base_model_path, dst_model_path)
        except shutil.SameFileError:
            logger.warning("Source and destination are the same model package file.")
            pass
        model_service = model_service_dep()
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service.init_model(load_in_4bit=load_in_4bit)
    elif mlflow_model_uri:
        model_service = ModelManager.retrieve_model_service_from_uri(mlflow_model_uri, config, dst_model_path)
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service_dep.model_service = model_service
    else:
        logger.error("Neither the model path or the mlflow model uri was passed in")
        typer.Exit(code=1)

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
            logger.error("Training type %s is not supported or the corresponding trainer has not been enabled in the .env file.", training_type)
            typer.Exit(code=1)


@cmd_app.command("register", help="This pushes a pretrained NLP model to the CogStack ModelServe registry")
def register_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to register"),
    model_path: str = typer.Option(..., help="The file path to the model package"),
    model_name: str = typer.Option(..., help="The string representation of the registered model"),
    training_type: Optional[TrainingType] = typer.Option(None, help="The type of training the model went through"),
    model_config: Optional[str] = typer.Option(None, help="The string representation of a JSON object"),
    model_metrics: Optional[str] = typer.Option(None, help="The string representation of a JSON array"),
    model_tags: Optional[str] = typer.Option(None, help="The string representation of a JSON object"),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
    """
    Registers a pretrained model with the model registry.

    This function handles the registration of a pretrained model by saving it to the model registry.

    Args:
        model_type (ModelType): The type of the model to register.
        model_path (str): The file path to the model package.
        model_name (str): The string representation of the registered model.
        training_type (Optional[TrainingType]): The type of training the model went through.
        model_config (Optional[str]): The string representation of a JSON object containing model configuration.
        model_metrics (Optional[str]): The string representation of a JSON array containing model metrics.
        model_tags (Optional[str]): The string representation of a JSON object containing model tags.
        debug (Optional[bool]): Run in debug mode if set to True.
    """

    logger = _get_logger(debug, model_type, model_name)
    config = get_settings()
    tracker_client = TrackerClient(config.MLFLOW_TRACKING_URI)

    if model_type in model_service_registry.keys():
        model_service_type = model_service_registry[model_type]
    else:
        logger.error("Unknown model type: %s", model_type)
        typer.Exit(code=1)

    m_config = json.loads(model_config) if model_config is not None else None
    m_metrics = json.loads(model_metrics) if model_metrics is not None else None
    m_tags = json.loads(model_tags) if model_tags is not None else None
    t_type = training_type if training_type is not None else ""

    run_name = str(uuid.uuid4())
    tracker_client.save_pretrained_model(
        model_name=model_name,
        model_path=model_path,
        model_manager=ModelManager(model_service_type, config),
        model_type=model_type.value,
        training_type=t_type,
        run_name=run_name,
        model_config=m_config,
        model_metrics=m_metrics,
        model_tags=m_tags,
    )
    typer.echo(f"Pushed {model_path} as a new model version ({run_name})")


@stream_app.command("json-lines", help="This gets NER entities as a JSON Lines stream")
def stream_jsonl_annotations(
    jsonl_file_path: str = typer.Option(..., help="The path to the JSON Lines file"),
    base_url: str = typer.Option("http://127.0.0.1:8000", help="The CMS base url"),
    timeout_in_secs: int = typer.Option(0, help="The max time to wait before disconnection"),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
    """
    Streams NER entities extracted from a JSON Lines file.

    Args:
        jsonl_file_path (str): The path to the JSON Lines file containing lines each having the format of {\"name\": \"DOC\", \"text\": \"TEXT\"}.
        base_url (str): The base URL of the CMS stream server.
        timeout_in_secs (int): The maximum time to wait for a response before disconnecting. Defaults to 0 (no timeout).
        debug (Optional[bool]): Run in debug mode if set to True.
    """

    logger = _get_logger(debug)

    async def get_jsonl_stream(base_url: str, jsonl_file_path: str) -> None:
        with open(jsonl_file_path) as file:
            headers = {"Content-Type": "application/x-ndjson"}
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=timeout_in_secs)
                    async with session.post(
                        f"{base_url}/stream/process",
                        data=file.read().encode("utf-8"),
                        headers=headers,
                        timeout=timeout,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.content:
                            typer.echo(line.decode("utf-8"), nl=False)
            finally:
                logger.debug("Closing the session...")
                await session.close()
                logger.debug("Session closed")

    asyncio.run(get_jsonl_stream(base_url, jsonl_file_path))


@stream_app.command("chat", help="This gets NER entities by chatting with the model")
def chat_to_get_jsonl_annotations(
    base_url: str = typer.Option("ws://127.0.0.1:8000", help="The CMS base url"),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
    """
    Streams NER entities extracted from a text input by the user in the interactive mode.

    Args:
        base_url (str): The base URL of the CMS stream server.
        debug (Optional[bool]): Run in debug mode if set to True.
    """

    logger = _get_logger(debug)
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
                            typer.echo("CMS =>")
                            typer.echo(response)
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


@cmd_app.command("export-model-apis", help="This generates a model-specific API document for enabled endpoints")
def generate_api_doc_per_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
    add_training_apis: bool = typer.Option(False, help="Add training APIs to the doc"),
    add_evaluation_apis: bool = typer.Option(False, help="Add evaluation APIs to the doc"),
    add_previews_apis: bool = typer.Option(False, help="Add preview APIs to the doc"),
    add_user_authentication: bool = typer.Option(False, help="Add user authentication APIs to the doc"),
    exclude_unsupervised_training: bool = typer.Option(False, help="Exclude the unsupervised training API"),
    exclude_metacat_training: bool = typer.Option(False, help="Exclude the metacat training API"),
    model_name: Optional[str] = typer.Option(None, help="The string representation of the model name"),
) -> None:
    """
    Generates a model-specific API document for enabled endpoints.

    This function creates an OpenAPI document for the specified model type,
    including or excluding certain types of APIs based on the parameters provided.

    Args:
        model_type (ModelType): The type of the model to serve.
        add_training_apis (str): Whether to include training APIs in the documentation. Defaults to False.
        add_evaluation_apis (str): Whether to include evaluation APIs in the documentation. Defaults to False.
        add_previews_apis (str): Whether to include preview APIs in the documentation. Defaults to False.
        add_user_authentication (str): Whether to include user authentication APIs in the documentation. Defaults to False.
        exclude_unsupervised_training (str): Whether to exclude the unsupervised training API. Defaults to False.
        exclude_metacat_training (str): Whether to exclude the metacat training API. Defaults to False.
        model_name (Optional[str]): The optional string representation of the model name.
    """

    config = get_settings()
    config.ENABLE_TRAINING_APIS = "true" if add_training_apis else "false"
    config.DISABLE_UNSUPERVISED_TRAINING = "true" if exclude_unsupervised_training else "false"
    config.DISABLE_METACAT_TRAINING = "true" if exclude_metacat_training else "false"
    config.ENABLE_EVALUATION_APIS = "true" if add_evaluation_apis else "false"
    config.ENABLE_PREVIEWS_APIS = "true" if add_previews_apis else "false"
    config.AUTH_USER_ENABLED = "true" if add_user_authentication else "false"

    model_service_dep = ModelServiceDep(model_type, config, model_name or model_type)
    cms_globals.model_service_dep = model_service_dep
    doc_name = f"{model_name or model_type}_model_apis.json"

    if model_type == ModelType.HUGGINGFACE_LLM:
        app = get_generative_server(config)
    else:
        app = get_model_server(config)
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    with open(doc_name, "w") as api_doc:
        json.dump(app.openapi(), api_doc, indent=4)
    typer.echo(f"OpenAPI doc exported to {doc_name}")


@package_app.command("hf-model", help="This packages a remotely hosted or locally cached Hugging Face model into a model package")
def package_model(
    hf_repo_id: str = typer.Option("",  help="The repository ID of the model to download from Hugging Face Hub, e.g., 'google-bert/bert-base-cased'"),
    hf_repo_revision: str = typer.Option("", help="The revision of the model to download from Hugging Face Hub"),
    cached_model_dir: str = typer.Option("", help="The path to the cached model directory, will only be used if --hf-repo-id is not provided"),
    output_model_package: str = typer.Option("", help="The path where the model package will be saved, minus any format-specific extension, e.g., './model_packages/bert-base-cased'"),
    archive_format: ArchiveFormat = typer.Option(ArchiveFormat.ZIP.value, help="The archive format of the model package, e.g., 'zip' or 'gztar'"),
    remove_cached: bool = typer.Option(False, help="Whether to remove the downloaded cache after the model package is saved"),
) -> None:
    """
    Packages and saves a Hugging Face model into a specified archive format.

    The model can either be downloaded from the Hugging Face Hub using the repository ID and optional revision,
    or it can be taken from a locally cached model directory if the repository ID is not provided.

    Args:
        hf_repo_id (str): The repository ID of the model to download from Hugging Face Hub, e.g., 'google-bert/bert-base-cased'.
        hf_repo_revision (str): The specific revision of the model to download. If not provided, the latest model will be downloaded.
        cached_model_dir (str): The path to a locally cached model directory. This will be used only if `hf_repo_id` is not provided.
        output_model_package (str): The path where the model package will be saved, minus any format-specific extension, e.g., './model_packages/bert-base-cased'.
        archive_format (ArchiveFormat): The format of the archive for the model package, either 'zip' or 'gztar'. Defaults to 'zip'.
        remove_cached (bool): Whether to remove the downloaded cache after the model package is saved. Defaults to False.
    """

    if hf_repo_id == "" and cached_model_dir == "":
        typer.echo("ERROR: Neither the repository ID of the Hugging Face model nor the cached model directory is passed in.")
        raise typer.Exit(code=1)

    if output_model_package == "":
        typer.echo("ERROR: The output model package path is not passed in.")
        raise typer.Exit(code=1)

    model_package_archive = os.path.abspath(os.path.expanduser(output_model_package))
    if hf_repo_id:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if not hf_repo_revision:
                    download_path = snapshot_download(
                        repo_id=hf_repo_id,
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False,
                    )
                else:
                    download_path = snapshot_download(
                        repo_id=hf_repo_id,
                        revision=hf_repo_revision,
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False,
                    )

                shutil.make_archive(model_package_archive, archive_format.value, download_path)
        finally:
            if remove_cached:
                cached_model_path = os.path.abspath(os.path.join(download_path, "..", ".."))
                shutil.rmtree(cached_model_path)
    elif cached_model_dir:
        cached_model_path = os.path.abspath(os.path.expanduser(cached_model_dir))
        shutil.make_archive(model_package_archive, archive_format.value, cached_model_path)

    typer.echo(f"Model package saved to {model_package_archive}.{'zip' if archive_format == ArchiveFormat.ZIP else 'tar.gz'}")


@package_app.command("hf-dataset", help="This packages a remotely hosted or locally cached Hugging Face dataset into a dataset package")
def package_dataset(
    hf_dataset_id: str = typer.Option("", help="The repository ID of the dataset to download from Hugging Face Hub, e.g., 'stanfordnlp/imdb'"),
    hf_dataset_revision: str = typer.Option("", help="The revision of the dataset to download from Hugging Face Hub"),
    cached_dataset_dir: str = typer.Option("", help="The path to the cached dataset directory, will only be used if --hf-dataset-id is not provided"),
    output_dataset_package: str = typer.Option("", help="The path where the dataset package will be saved, minus any format-specific extension, e.g., './dataset_packages/imdb'"),
    archive_format: ArchiveFormat = typer.Option(ArchiveFormat.ZIP.value, help="The archive format of the dataset package, e.g., 'zip' or 'gztar'"),
    remove_cached: bool = typer.Option(False, help="Whether to remove the downloaded cache after the dataset package is saved"),
    trust_remote_code: bool = typer.Option(False, help="Whether to trust and use the remote script of the dataset"),
) -> None:
    """
    Packages a dataset from Hugging Face Hub or a local cached directory into a specified archive format.

    The dataset can either be downloaded from Hugging Face Hub if the dataset ID is provided, or it can be taken
    from a locally cached dataset directory if the dataset ID is not provided.

    Args:
        hf_dataset_id (str): The repository ID of the dataset to download from Hugging Face Hub, e.g., 'stanfordnlp/imdb'.
        hf_dataset_revision (str): The specific revision of the dataset to download.
        cached_dataset_dir (str): The path to a local cached dataset directory, used only if `hf_dataset_id` is not provided.
        output_dataset_package (str): The path where the dataset package will be saved, minus any format-specific extension, e.g., './dataset_packages/imdb'.
        archive_format (ArchiveFormat): The archive format for the dataset package, either 'zip' or 'gztar'. Defaults to 'zip'.
        remove_cached (bool): Whether to remove the cached dataset after creating the package. Defaults to False.
        trust_remote_code (bool): Whether to trust and execute the remote script of the dataset. Defaults to False.
    """

    if hf_dataset_id == "" and cached_dataset_dir == "":
        typer.echo("ERROR: Neither the repository ID of the Hugging Face dataset nor the cached dataset directory is passed in.")
        raise typer.Exit(code=1)
    if output_dataset_package == "":
        typer.echo("ERROR: The dataset package path is not passed in.")
        raise typer.Exit(code=1)

    dataset_package_archive = os.path.abspath(os.path.expanduser(output_dataset_package))

    if hf_dataset_id != "":
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
        cached_dataset_path = os.path.join(cache_dir, "datasets", hf_dataset_id.replace("/", "_"))

        try:
            if hf_dataset_revision == "":
                dataset = load_dataset(path=hf_dataset_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
            else:
                dataset = load_dataset(
                    path=hf_dataset_id,
                    cache_dir=cache_dir,
                    revision=hf_dataset_revision,
                    trust_remote_code=trust_remote_code,
                )

            dataset.save_to_disk(cached_dataset_path)
            shutil.make_archive(dataset_package_archive, archive_format.value, cached_dataset_path)
        finally:
            if remove_cached:
                shutil.rmtree(cache_dir)
    elif cached_dataset_dir != "":
        cached_dataset_path = os.path.abspath(os.path.expanduser(cached_dataset_dir))
        shutil.make_archive(dataset_package_archive, archive_format.value, cached_dataset_path)

    typer.echo(f"Dataset package saved to {dataset_package_archive}.{'zip' if archive_format == ArchiveFormat.ZIP else 'tar.gz'}")


@cmd_app.command("build", help="This builds an OCI-compliant image to containerise CMS")
def build_image(
    dockerfile_path: str = typer.Option(..., help="The path to the Dockerfile"),
    context_dir: str = typer.Option(..., help="The directory containing the set of files accessible to the build"),
    model_name: str = typer.Option("CMS model", help="The string representation of the model name"),
    user_id: int = typer.Option(1000, help="The ID for the non-root user"),
    group_id: int = typer.Option(1000, help="The group ID for the non-root user"),
    http_proxy: str = typer.Option("", help="The string representation of the HTTP proxy"),
    https_proxy: str = typer.Option("", help="The string representation of the HTTPS proxy"),
    no_proxy: str = typer.Option("localhost,127.0.0.1", help="The string representation of addresses by-passing proxies"),
    version_tag: str = typer.Option("latest", help="The version tag of the built image"),
    backend: BuildBackend = typer.Option(BuildBackend.DOCKER.value, help="The backend used for building the image"),
) -> None:
    """
    Builds an OCI-compliant container image for CMS using the specified backend.

    Args:
        dockerfile_path (str): The path to the Dockerfile used for building the image.
        context_dir (str): The directory containing the build context (files accessible during the build).
        model_name (str): The string representation of the model name. Defaults to "CMS model".
        user_id (int): The ID of the non-root user in the container. Defaults to 1000.
        group_id (int): The group ID of the non-root user in the container. Defaults to 1000.
        http_proxy (str): The HTTP proxy to use during the build. Defaults to empty.
        https_proxy (str): The HTTPS proxy to use during the build. Defaults to empty.
        no_proxy (str): The addresses to bypass the proxy during the build. Defaults to "localhost,127.0.0.1".
        version_tag (str): The version tag for the built image. Defaults to "latest".
        backend (BuildBackend): The backend used for building the image. Defaults to "docker build".
    """

    assert backend is not None
    cmd = [
        *backend.value.split(),
        '-f', dockerfile_path,
        '--progress=plain',
        '-t', f'{model_name.replace(" ", "-").lower()}:{version_tag}',
        '--build-arg', f'CMS_MODEL_NAME={model_name}',
        '--build-arg', f'CMS_UID={str(user_id)}',
        '--build-arg', f'CMS_GID={str(group_id)}',
        '--build-arg', f'HTTP_PROXY={http_proxy}',
        '--build-arg', f'HTTPS_PROXY={https_proxy}',
        '--build-arg', f'NO_PROXY={no_proxy}',
        context_dir,
    ]
    with subprocess.Popen(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
        universal_newlines=True,
        bufsize=1,
    ) as process:
        assert process is not None
        try:
            while True:
                assert process.stdout is not None
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    typer.echo(output.strip())
            process.wait()

            if process.returncode == 0:
                typer.echo(f"The '{backend.value}' command ran successfully.")
            else:
                typer.echo(f"The '{backend.value}' command failed.")
        except FileNotFoundError:
            typer.echo(f"The '{backend.value}' command not found.")
        except KeyboardInterrupt:
            typer.echo("The build was terminated by the user.")
        except Exception as e:
            typer.echo(f"An unexpected error occurred: {e}")
        finally:
            process.kill()


@cmd_app.command("export-openapi-spec", help="This generates an API document for all endpoints defined in CMS")
def generate_api_doc(
    api_title: str = typer.Option("CogStack Model Serve APIs", help="The string representation of the API title")
) -> None:
    """
    Generates an OpenAPI document for all endpoints defined in CMS.

    This function creates an all-in-one OpenAPI document for all CMS endpoints regardless of model types.

    Args:
        api_title (str): The string representation of the API title. Defaults to "CogStack Model Serve APIs".
    """

    config = get_settings()
    config.AUTH_USER_ENABLED = "true"
    model_service_dep = ModelServiceDep("ALL", config, api_title)   # type: ignore
    cms_globals.model_service_dep = model_service_dep
    doc_name = f"{api_title.lower().replace(' ', '_')}.json"
    app = get_app_for_api_docs(None)
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    with open(doc_name, "w") as api_doc:
        openapi = app.openapi()
        openapi["info"]["title"] = api_title
        json.dump(app.openapi(), api_doc, indent=4)
    typer.echo(f"OpenAPI doc exported to {doc_name}")


@cmd_app.callback()
# ruff: noqa
def show_banner(
    model_type: Optional[ModelType] = None,
    host: Optional[str] = None,
    port: Optional[str] = None
) -> None:
    from rich.console import Console, Group
    from rich.align import Align
    from rich.text import Text

    os.environ["COLORTERM"] = "truecolor"
    console = Console()
    banner_lines = [
     r"  _____             _____ _             _       __  __           _      _  _____",
     r" / ____|           / ____| |           | |     |  \/  |         | |    | |/ ____|",
     r"| |     ___   __ _| (___ | |_ __ _  ___| | __  | \  / | ___   __| | ___| | (___   ___ _ ____   _____",
     r"| |    / _ \ / _` |\___ \| __/ _` |/ __| |/ /  | |\/| |/ _ \ / _` |/ _ \ |\___ \ / _ \ '__\ \ / / _ \ ",
     r"| |___| (_) | (_| |____) | || (_| | (__|   <   | |  | | (_) | (_| |  __/ |____) |  __/ |   \ V /  __/",
     r" \_____\___/ \__, |_____/ \__\__,_|\___|_|\_\  |_|  |_|\___/ \__,_|\___|_|_____/ \___|_|    \_/ \___|",
     r"              __/ |",
     r"             |___/",
    ]

    colors = [
        "#00d9ff",   # Bright cyan
        "#00c5f0",   # Cyan-blue
        "#00b1e0",   # Light blue
        "#009dd0",   # Mid-light blue
        "#0089c0",   # Mid blue
        "#0075b0",   # Mid-dark blue
        "#0061a0",   # Dark blue
        "#004d90",   # Deep blue
    ]
    console.print()
    banner_lines_with_styles = []
    for i, line in enumerate(banner_lines):
        styled_line = Text(line, style=f"bold {colors[i]}")
        banner_lines_with_styles.append(styled_line)

    banner_group = Group(*banner_lines_with_styles)
    console.print(Align.center(banner_group))
    console.print()

def _display_info_table(
    model_type: ModelType,
    model_name: Optional[str],
    model_path: Optional[str],
    mlflow_model_uri: Optional[str],
    host: str,
    port: str,
) -> None:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    title_text = Text(f"Welcome to CMS {__version__}", style="bold blue")

    display_model_type = model_type.value
    server_url = f"http://{host}:{port}"
    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold", justify="center")
    info_table.add_column(style="cyan", justify="left")
    info_table.add_column(style="dim", justify="left")

    info_table.add_row("🤖", "Model Name:", model_name or "CMS model")
    info_table.add_row("📦", "Model Type:", display_model_type)
    info_table.add_row("📂", "Model Path:", model_path or mlflow_model_uri)
    info_table.add_row("🔗", "Base URL:", server_url)
    info_table.add_row("📚", "Docs:", f"{server_url}/docs")

    panel_content = Group(
        Align.center(title_text),
        "",
        "",
        Align.center(info_table),
    )

    panel = Panel(
        panel_content,
        border_style="dim",
        padding=(1, 4),
        width=80,
    )
    console = Console(stderr=True)
    console.print(Group("\n", Align.center(panel), "\n"))


def _ensure_dst_model_path(model_path: str, parent_dir: str, config: Settings) -> str:
    if model_path.endswith(".zip"):
        dst_model_path = os.path.join(parent_dir, "model", "model.zip")
        config.BASE_MODEL_FILE = "model.zip"
    else:
        dst_model_path = os.path.join(parent_dir, "model", "model.tar.gz")
        config.BASE_MODEL_FILE = "model.tar.gz"
    model_dir = os.path.join(parent_dir, "model", "model")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if dst_model_path.endswith(".zip") and os.path.exists(dst_model_path.replace(".zip", ".tar.gz")):
        os.remove(dst_model_path.replace(".zip", ".tar.gz"))
    if dst_model_path.endswith(".tar.gz") and os.path.exists(dst_model_path.replace(".tar.gz", ".zip")):
        os.remove(dst_model_path.replace(".tar.gz", ".zip"))
    return dst_model_path


def _get_logger(
    debug: Optional[bool] = None,
    model_type: Optional[ModelType] = None,
    model_name: Optional[str] = None,
) -> logging.Logger:
    if debug is not None:
        get_settings().DEBUG = "true" if debug else "false"
    if get_settings().DEBUG != "true":
        logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("cms")

    lrf = logging.getLogRecordFactory()

    def log_record_factory(*args: Tuple, **kwargs: Dict[str, Any]) -> LogRecord:
        record = lrf(*args, **kwargs)
        record.model_type = model_type
        record.model_name = model_name if model_name is not None else "NULL"
        return record
    logging.setLogRecordFactory(log_record_factory)

    return logger


if __name__ == "__main__":
    cmd_app()
