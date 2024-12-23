import inspect
import json
import logging.config
import os
import subprocess
import sys
import uuid
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
from huggingface_hub import snapshot_download  # noqa
from datasets import load_dataset  # noqa
from domain import ModelType, TrainingType, BuildBackend, Device  # noqa
from registry import model_service_registry  # noqa
from api.api import get_model_server, get_stream_server  # noqa
from utils import get_settings, send_gelf_message  # noqa
from management.model_manager import ModelManager  # noqa
from api.dependencies import ModelServiceDep, ModelManagerDep  # noqa
from management.tracker_client import TrackerClient  # noqa

cmd_app = typer.Typer(
    name="python cli.py",
    help="CLI for various CogStack ModelServe operations",
    add_completion=False,
)
stream_app = typer.Typer(
    name="python cli.py stream", help="This groups various stream operations", add_completion=False
)
cmd_app.add_typer(stream_app, name="stream")
package_app = typer.Typer(
    name="python cli.py package",
    help="This groups various package operations",
    add_completion=False,
)
cmd_app.add_typer(package_app, name="package")
logging.config.fileConfig(os.path.join(parent_dir, "logging.ini"), disable_existing_loggers=False)


@cmd_app.command("serve", help="This serves various CogStack NLP models")
def serve_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
    model_path: str = typer.Option("", help="The file path to the model package"),
    mlflow_model_uri: str = typer.Option(
        "", help="The URI of the MLflow model to serve", metavar="models:/MODEL_NAME/ENV"
    ),
    host: str = typer.Option("127.0.0.1", help="The hostname of the server"),
    port: str = typer.Option("8000", help="The port of the server"),
    model_name: Optional[str] = typer.Option(
        None, help="The string representation of the model name"
    ),
    streamable: bool = typer.Option(False, help="Serve the streamable endpoints only"),
    device: Device = typer.Option(Device.DEFAULT, help="The device to serve the model on"),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
    logger = _get_logger(debug, model_type, model_name)
    get_settings().DEVICE = device.value
    if model_type in [
        ModelType.HUGGINGFACE_NER,
        ModelType.MEDCAT_DEID,
        ModelType.TRANSFORMERS_DEID,
    ]:
        get_settings().DISABLE_METACAT_TRAINING = "true"

    if "GELF_INPUT_URI" in os.environ and os.environ["GELF_INPUT_URI"]:
        try:
            uri = urlparse(os.environ["GELF_INPUT_URI"])
            send_gelf_message(f"Model service {model_type} is starting", uri)
            gelf_tcp_handler = graypy.GELFTCPHandler(uri.hostname, uri.port)
            logger.addHandler(gelf_tcp_handler)
            logging.getLogger("uvicorn").addHandler(gelf_tcp_handler)
        except Exception:
            logger.exception(
                '$GELF_INPUT_URI is set to "%s" but it\'s not ready to receive logs',
                os.environ["GELF_INPUT_URI"],
            )

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
        model_service = ModelManager.retrieve_model_service_from_uri(
            mlflow_model_uri, config, dst_model_path
        )
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service_dep.model_service = model_service
        cms_globals.model_manager_dep = ModelManagerDep(model_service)
        model_server_app = get_model_server()
    else:
        logger.error("Neither the model path or the mlflow model uri was passed in")
        typer.Exit(code=1)

    logger.info('Start serving model "%s" on %s:%s', model_type, host, port)
    # interrupted = False
    # while not interrupted:
    uvicorn.run(
        model_server_app if not streamable else get_stream_server(),
        host=host,
        port=int(port),
        log_config=None,
    )
    # interrupted = True
    typer.echo("Shutting down due to either keyboard interrupt or system exit")


@cmd_app.command("train", help="This pretrains or fine-tunes various CogStack NLP models")
def train_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
    base_model_path: str = typer.Option(
        "", help="The file path to the base model package to be trained on"
    ),
    mlflow_model_uri: str = typer.Option(
        "", help="The URI of the MLflow model to train", metavar="models:/MODEL_NAME/ENV"
    ),
    training_type: TrainingType = typer.Option(..., help="The type of training"),
    data_file_path: str = typer.Option(..., help="The path to the training asset file"),
    epochs: int = typer.Option(1, help="The number of training epochs"),
    log_frequency: int = typer.Option(
        1, help="The number of processed documents after which training metrics will be logged"
    ),
    hyperparameters: str = typer.Option(
        "{}", help="The overriding hyperparameters serialised as JSON string"
    ),
    description: Optional[str] = typer.Option(
        None, help="The description of the training or change logs"
    ),
    model_name: Optional[str] = typer.Option(
        None, help="The string representation of the model name"
    ),
    device: Device = typer.Option(Device.DEFAULT, help="The device to train the model on"),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
    logger = _get_logger(debug, model_type, model_name)

    config = get_settings()
    config.DEVICE = device.value

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
        model_service = ModelManager.retrieve_model_service_from_uri(
            mlflow_model_uri, config, dst_model_path
        )
        model_service.model_name = model_name if model_name is not None else "CMS model"
        model_service_dep.model_service = model_service
    else:
        logger.error("Neither the model path or the mlflow model uri was passed in")
        typer.Exit(code=1)

    training_id = str(uuid.uuid4())
    with open(data_file_path, "r") as data_file:
        training_args = [
            data_file,
            epochs,
            log_frequency,
            training_id,
            data_file.name,
            [data_file],
            description,
            True,
        ]
        if (
            training_type == TrainingType.SUPERVISED
            and model_service._supervised_trainer is not None
        ):
            model_service.train_supervised(*training_args, **json.loads(hyperparameters))
        elif (
            training_type == TrainingType.UNSUPERVISED
            and model_service._unsupervised_trainer is not None
        ):
            model_service.train_unsupervised(*training_args, **json.loads(hyperparameters))
        elif (
            training_type == TrainingType.META_SUPERVISED
            and model_service._metacat_trainer is not None
        ):
            model_service.train_metacat(*training_args, **json.loads(hyperparameters))
        else:
            logger.error(
                "Training type %s is not supported or the corresponding trainer has not been"
                " enabled in the .env file.",
                training_type,
            )
            typer.Exit(code=1)


@cmd_app.command(
    "register", help="This pushes a pretrained NLP model to the CogStack ModelServe registry"
)
def register_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
    model_path: str = typer.Option(..., help="The file path to the model package"),
    model_name: str = typer.Option(..., help="The string representation of the registered model"),
    training_type: Optional[str] = typer.Option(
        None, help="The type of training the model went through"
    ),
    model_config: Optional[str] = typer.Option(
        None, help="The string representation of a JSON object"
    ),
    model_metrics: Optional[str] = typer.Option(
        None, help="The string representation of a JSON array"
    ),
    model_tags: Optional[str] = typer.Option(
        None, help="The string representation of a JSON object"
    ),
    debug: Optional[bool] = typer.Option(None, help="Run in the debug mode"),
) -> None:
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
                logging.info(
                    "Connected to CMS. Start typing you input and press <ENTER> to submit:"
                )
                try:
                    while True:
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, sys.stdin.readline
                        )
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


@cmd_app.command("export-model-apis")
def generate_api_doc_per_model(
    model_type: ModelType = typer.Option(..., help="The type of the model to serve"),
    add_training_apis: bool = typer.Option(False, help="Add training APIs to the doc"),
    add_evaluation_apis: bool = typer.Option(False, help="Add evaluation APIs to the doc"),
    add_previews_apis: bool = typer.Option(False, help="Add preview APIs to the doc"),
    add_user_authentication: bool = typer.Option(
        False, help="Add user authentication APIs to the doc"
    ),
    exclude_unsupervised_training: bool = typer.Option(
        False, help="Exclude the unsupervised training API"
    ),
    exclude_metacat_training: bool = typer.Option(False, help="Exclude the metacat training API"),
    model_name: Optional[str] = typer.Option(
        None, help="The string representation of the model name"
    ),
) -> None:
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
    typer.echo(f"OpenAPI doc exported to {doc_name}")


@package_app.command(
    "hf-model",
    help=(
        "This packages a remotely hosted or locally cached Hugging Face model into a model package"
    ),
)
def package_model(
    hf_repo_id: str = typer.Option(
        "",
        help=(
            "The repository ID of the model to download from Hugging Face Hub,"
            " e.g., 'google-bert/bert-base-cased'"
        ),
    ),
    hf_repo_revision: str = typer.Option(
        "", help="The revision of the model to download from Hugging Face Hub"
    ),
    cached_model_dir: str = typer.Option(
        "",
        help=(
            "Path to the cached model directory, will only be used if --hf-repo-id is not provided"
        ),
    ),
    output_model_package: str = typer.Option(
        "",
        help=(
            "Path to save the model package, minus any format-specific extension,"
            " e.g., './model_packages/bert-base-cased'"
        ),
    ),
    remove_cached: bool = typer.Option(
        False, help="Whether to remove the downloaded cache after the model package is saved"
    ),
) -> None:
    if hf_repo_id == "" and cached_model_dir == "":
        typer.echo(
            "ERROR: Neither the repository ID of the Hugging Face model nor the cached model"
            " directory is passed in."
        )
        raise typer.Exit(code=1)

    if output_model_package == "":
        typer.echo("ERROR: The model package path is not passed in.")
        raise typer.Exit(code=1)

    model_package_archive = os.path.abspath(os.path.expanduser(output_model_package))

    if hf_repo_id:
        try:
            if not hf_repo_revision:
                download_path = snapshot_download(repo_id=hf_repo_id)
            else:
                download_path = snapshot_download(repo_id=hf_repo_id, revision=hf_repo_revision)

            shutil.make_archive(model_package_archive, "zip", download_path)
        finally:
            if remove_cached:
                cached_model_path = os.path.abspath(os.path.join(download_path, "..", ".."))
                shutil.rmtree(cached_model_path)
    elif cached_model_dir:
        cached_model_path = os.path.abspath(os.path.expanduser(cached_model_dir))
        shutil.make_archive(model_package_archive, "zip", cached_model_path)

    typer.echo(f"Model package saved to {model_package_archive}.zip")


@package_app.command(
    "hf-dataset",
    help=(
        "This packages a remotely hosted or locally cached Hugging Face dataset into a dataset"
        " package"
    ),
)
def package_dataset(
    hf_dataset_id: str = typer.Option(
        "",
        help=(
            "The repository ID of the dataset to download from Hugging Face Hub,"
            " e.g., 'stanfordnlp/imdb'"
        ),
    ),
    hf_dataset_revision: str = typer.Option(
        "", help="The revision of the dataset to download from Hugging Face Hub"
    ),
    cached_dataset_dir: str = typer.Option(
        "",
        help=(
            "Path to the cached dataset directory, will only be used if --hf-dataset-id is not"
            " provided"
        ),
    ),
    output_dataset_package: str = typer.Option(
        "",
        help=(
            "Path to save the dataset package, minus any format-specific extension,"
            " e.g., './dataset_packages/imdb'"
        ),
    ),
    remove_cached: bool = typer.Option(
        False, help="Whether to remove the downloaded cache after the dataset package is saved"
    ),
    trust_remote_code: bool = typer.Option(
        False, help="Whether to trust and use the remote script of the dataset"
    ),
) -> None:
    if hf_dataset_id == "" and cached_dataset_dir == "":
        typer.echo(
            "ERROR: Neither the repository ID of the Hugging Face dataset nor the cached dataset"
            " directory is passed in."
        )
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
                dataset = load_dataset(
                    path=hf_dataset_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code
                )
            else:
                dataset = load_dataset(
                    path=hf_dataset_id,
                    cache_dir=cache_dir,
                    revision=hf_dataset_revision,
                    trust_remote_code=trust_remote_code,
                )

            dataset.save_to_disk(cached_dataset_path)
            shutil.make_archive(dataset_package_archive, "zip", cached_dataset_path)
        finally:
            if remove_cached:
                shutil.rmtree(cache_dir)
    elif cached_dataset_dir != "":
        cached_dataset_path = os.path.abspath(os.path.expanduser(cached_dataset_dir))
        shutil.make_archive(dataset_package_archive, "zip", cached_dataset_path)

    typer.echo(f"Dataset package saved to {dataset_package_archive}.zip")


@cmd_app.command("build", help="This builds an OCI-compliant image to containerise CMS")
def build_image(
    dockerfile_path: str = typer.Option(..., help="The path to the Dockerfile"),
    context_dir: str = typer.Option(
        ..., help="The directory containing the set of files accessible to the build"
    ),
    model_name: Optional[str] = typer.Option(
        "cms_model", help="The string representation of the model name"
    ),
    user_id: Optional[int] = typer.Option(1000, help="The ID for the non-root user"),
    group_id: Optional[int] = typer.Option(1000, help="The group ID for the non-root user"),
    http_proxy: Optional[str] = typer.Option(
        "", help="The string representation of the HTTP proxy"
    ),
    https_proxy: Optional[str] = typer.Option(
        "", help="The string representation of the HTTPS proxy"
    ),
    no_proxy: Optional[str] = typer.Option(
        "localhost,127.0.0.1", help="The string representation of addresses by-passing proxies"
    ),
    tag: str = typer.Option(None, help="The tag of the built image"),
    backend: Optional[BuildBackend] = typer.Option(
        BuildBackend.DOCKER, help="The backend used for building the image"
    ),
) -> None:
    assert backend is not None
    cmd = [
        *backend.value.split(),
        "-f",
        dockerfile_path,
        "--progress=plain",
        "-t",
        f"{model_name}:{tag}",
        "--build-arg",
        f"CMS_MODEL_NAME={model_name}",
        "--build-arg",
        f"CMS_UID={str(user_id)}",
        "--build-arg",
        f"CMS_GID={str(group_id)}",
        "--build-arg",
        f"HTTP_PROXY={http_proxy}",
        "--build-arg",
        f"HTTPS_PROXY={https_proxy}",
        "--build-arg",
        f"NO_PROXY={no_proxy}",
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


@cmd_app.command("export-openapi-spec")
def generate_api_doc(
    api_title: str = typer.Option(
        "CogStack Model Serve APIs", help="The string representation of the API title"
    ),
) -> None:
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
    typer.echo(f"OpenAPI doc exported to {doc_name}")


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
