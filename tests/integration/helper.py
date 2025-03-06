import os
import asyncio
import logging
import requests
import subprocess
import tempfile
import threading
import time
from functools import partial, wraps
from pytest_bdd import parsers
from urllib.parse import urlparse
from app.domain import ModelType


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


def get_logger(debug=False):
    logger = logging.getLogger("cms-integration")
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    stdout_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stdout_handler)
    return logger


def download_model(model_url, max_retries = 5, initial_delay = 1):
    model_path = os.path.join(".pytest_cache", "model.zip")
    if os.path.exists(model_path):
        return model_path
    retry_delay = initial_delay
    for attempt in range(max_retries):
        try:
            with requests.get(model_url, stream=True) as response:
                response.raise_for_status()
                with open(model_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            return model_path
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download model from {model_url} after {max_retries} attempts: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2


def run(conf, logger, streamable=False):

    if conf["process"] is None or conf["process"].poll() is not None:
        conf["process"] = subprocess.Popen(
                [ "cms", "serve"] +
                (["--streamable"] if streamable else []) +
                [
                    "--model-type", ModelType.MEDCAT_UMLS.value,
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
                if "Application startup complete" in line:
                    event.set()
                logger.info(line[:-1])
            pipe.close()

        startup_event = threading.Event()
        logging_thread = threading.Thread(target=cms_log_listener, args=(conf["process"].stdout, logger, startup_event))
        logging_thread.daemon = True
        logging_thread.start()
        try:
            startup_event.wait(timeout=60)
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