import os
import argparse
import logging.config
import json
import sys
import asyncio
import shutil
import warnings
import globals

from hypercorn.config import Config
from hypercorn.asyncio import serve
from api import get_settings, get_model_server
from management.model_manager import ModelManager
from dependencies import ModelServiceDep


if __name__ == "__main__":
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.ini"), disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(
        description="This script serves various CogStack NLP models",
    )

    parser.add_argument(
        "-mt",
        "--model_type",
        help="The type of the model to serve",
        choices=["medcat_snomed", "medcat_icd10", "de_id"],
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        help="The file path to the model package",
        type=str,
        default="",
    )

    parser.add_argument(
        "-mmu",
        "--mlflow_model_uri",
        help="The URI of the MLflow model to serve",
        type=str,
        default="",
    )

    parser.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="The hostname of the server",
    )

    parser.add_argument(
        "-p",
        "--port",
        default="8000",
        help="The port of the server",
    )

    parser.add_argument(
        "-d",
        "--doc",
        action="store_true",
        help="Export the OpenAPI doc",
    )

    args = parser.parse_args()
    settings = get_settings()

    model_service_dep = ModelServiceDep(args.model_type, settings)
    globals.model_service_dep = model_service_dep
    app = get_model_server()

    if args.doc:
        doc_name = ""
        if args.model_type == "medcat_snomed":
            doc_name = "medcat_snomed_model_apis.json"
        elif args.model_type == "medcat_icd10":
            doc_name = "medcat_icd10_model_apis.json"
        elif args.model_type == "de_id":
            doc_name = "de-identification_model_apis.json"
        with open(doc_name, "w") as doc:
            json.dump(app.openapi(), doc, indent=4)
        print(f"OpenAPI doc exported to {doc_name}")
        sys.exit(0)
    else:
        dst_model_path = os.path.join(os.path.dirname(__file__), "model", "model.zip")
        if dst_model_path and os.path.exists(dst_model_path.replace(".zip", "")):
            shutil.rmtree(dst_model_path.replace(".zip", ""))
        if args.model_path:
            try:
                shutil.copy2(args.model_path, dst_model_path)
            except shutil.SameFileError:
                pass
            model_service = model_service_dep()
            model_service.init_model()
        elif args.mlflow_model_uri:
            model_service = ModelManager.get_model_service(settings.MLFLOW_TRACKING_URI,
                                                           args.mlflow_model_uri,
                                                           settings,
                                                           dst_model_path)
            model_service_dep.model_service = model_service
            app = get_model_server()
        else:
            print("Error: Neither the model path or the mlflow model uri was passed in")
            sys.exit(1)

        config = Config()
        config.bind = [f"{args.host}:{args.port}"]
        config.access_log_format = "%(R)s %(s)s %(st)s %(D)s %({Header}o)s"
        config.accesslog = logger
        asyncio.run(serve(app, config))  # type: ignore
