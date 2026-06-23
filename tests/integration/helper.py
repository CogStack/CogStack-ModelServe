import os
import asyncio
import logging
import subprocess
import tempfile
import threading
import time
from functools import partial, wraps
from pytest_bdd import parsers
from urllib.parse import urlparse
import httpx
from app.domain import ModelType
from app.utils import download_model_package


def parse_data_table(text, orient="dict"):
    parsed_text = [
        [x.strip() for x in line.split("|")]
        for line in [x.strip("|") for x in text.splitlines()]
    ]

    header, *data = parsed_text

    if orient == "dict":
        return [
            dict(zip(header, line))
            for line in data
        ]
    else:
        if orient == "columns":
            data = [
                [line[i] for line in data]
                for i in range(len(header))
            ]
        return header, data


def data_table(name, fixture="data", orient="dict"):
    formatted_str = "{name}\n{{{fixture}:DataTable}}".format(
        name=name,
        fixture=fixture,
    )
    data_table_parser = partial(parse_data_table, orient=orient)

    return parsers.cfparse(formatted_str, extra_types=dict(DataTable=data_table_parser))


def async_to_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def ensure_app_config(debug_mode=False):
    os.environ["CMS_CI"] = "true"
    os.environ["DEBUG"] = "true" if debug_mode else "false"
    os.environ["MLFLOW_TRACKING_URI"] = tempfile.TemporaryDirectory().name
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PROCESS_RATE_LIMIT"] = "10000/minute"
    os.environ["PROCESS_BULK_RATE_LIMIT"] = "10000/minute"
    os.environ["GENERATION_RATE_LIMIT"] = "10000/minute"


def get_logger(debug=False, name="cms-integration"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    stdout_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stdout_handler)
    return logger


def download_model(model_url, file_name, max_retries = 5, initial_delay = 1):
    model_path = os.path.join(".pytest_cache", file_name)
    download_model_package(
        model_package_url=model_url,
        destination_path=model_path,
        max_retries=max_retries,
        initial_delay_secs=initial_delay,
        overwrite=False,
    )
    return model_path


def run(conf, logger, streamable=False, generative=False):

    if conf["process"] is None or conf["process"].poll() is not None:
        conf["process"] = subprocess.Popen(
                [ "cms", "serve"] +
                (["--streamable"] if streamable else []) +
                [
                    "--model-type", ModelType.MEDCAT_UMLS.value if not generative else ModelType.HUGGINGFACE_LLM.value,
                    "--model-path", conf["model_path"],
                    "--host", urlparse(conf["base_url"]).hostname,
                    "--port", str(urlparse(conf["base_url"]).port),
                    "--model-name", "test model",
                    "--debug",
                ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=os.environ.copy(),
        )

        def cms_log_listener(pipe, logger, event):
            for line in iter(pipe.readline, ""):
                if "Uvicorn running on" in line:
                    event.set()
                logger.info(line[:-1])
            pipe.close()

        startup_event = threading.Event()
        logging_thread = threading.Thread(target=cms_log_listener, args=(conf["process"].stdout, logger, startup_event))
        logging_thread.daemon = True
        logging_thread.start()
        try:
            timeout = 120
            if not startup_event.wait(timeout=timeout):
                raise RuntimeError(f"CMS process was not ready within {timeout} seconds")
            if conf["process"].poll() is not None:
                raise RuntimeError("CMS process exited before becoming ready")
            return {
                "base_url": conf["base_url"],
            }
        except KeyboardInterrupt:
            conf["process"].terminate()
            conf["process"].wait(timeout=30)
    else:
        logger.info("CMS server is up and running")
        return {
            "base_url": conf["base_url"],
        }


async def wait_for_server_ready(
    base_url: str,
    timeout_secs: int = 60,
    retry_interval_secs: int = 1,
) -> None:
    deadline = time.monotonic() + timeout_secs
    last_error: str = "Unknown"
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            for path in ("/healthz", "/readyz"):
                try:
                    response = await client.get(f"{base_url}{path}")
                    if response.status_code < 500:
                        return
                    last_error = f"{path} returned status {response.status_code}"
                except Exception as exc:
                    last_error = f"{path} connection error: {exc}"
            await asyncio.sleep(retry_interval_secs)

    raise RuntimeError(f"CMS server was not ready within {timeout_secs}s ({last_error})")
