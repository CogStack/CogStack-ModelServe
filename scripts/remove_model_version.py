import os
import sys
from argparse import ArgumentParser
from typing import Union
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

ARTIFACTS_DESTINATION = "s3://cms-model-bucket/"
DEFAULT_ARTIFACT_ROOT = "mlflow-artifacts:/"


def _remove_model_version(client: MlflowClient, model_name: str, model_version: Union[int, str]) -> None:
    versions = client.search_model_versions(f"name='{model_name}'")
    model_version = str(model_version)
    if versions is None or len(versions) == 0:
        raise ValueError(f"Cannot find model '{model_name}'")
    deleted = False
    cancelled = False
    run_id = ""
    for m_version in versions:
        if m_version.version == model_version:
            if m_version.current_stage not in ["None", "Archived"]:
                raise ValueError("You cannot delete models which have not been archived!")
            if m_version.status != "READY":
                raise ValueError("You cannot delete models which are not ready yet!")
            confirm = input("""
You cannot undo this action. When you delete a model, all model artifacts stored by the Model Registry
and all the metadata associated with the registered model are deleted. Do you want to proceed? (y/n)
                            """).lower() == ("y" or "yes")
            if confirm:
                client.delete_model_version(name=model_name, version=model_version)
                print(f"Version '{model_version}' of model '{model_name}' was deleted")
                deleted = True
                run_id = m_version.run_id
                break
            else:
                cancelled = True
    if not deleted and not cancelled:
        raise ValueError(f"Cannot find model '{model_name}' with version '{model_version}'")

    if deleted and run_id:
        run = client.get_run(run_id)
        artifact_repo = get_artifact_repository(run.info.artifact_uri.replace(ARTIFACTS_DESTINATION, DEFAULT_ARTIFACT_ROOT))
        artifact_repo.delete_artifacts()
        print(f"Artifacts for version '{model_version}' of model '{model_name}' were deleted")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-u",
        "--mlflow-tracking-uri",
        type=str,
        default="",
        help="The MLflow tracking URI"
    )
    parser.add_argument(
        "-n",
        "--mlflow-model-name",
        type=str,
        default="",
        help="The name of the registered MLflow model"
    )
    parser.add_argument(
        "-v",
        "--mlflow-model-version",
        type=str,
        default="",
        help="The version of the registered MLflow model"
    )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.mlflow_tracking_uri == "":
        print("ERROR: The MLflow tracking URI is not passed in.")
        sys.exit(1)
    if FLAGS.mlflow_model_name == "":
        print("ERROR: The MLflow model name is not passed in.")
        sys.exit(1)
    if FLAGS.mlflow_model_version == "":
        print("ERROR: The MLflow model version is not passed in.")
        sys.exit(1)

    os.environ["MLFLOW_TRACKING_URI"] = FLAGS.mlflow_tracking_uri
    mlflow_client = MlflowClient()
    _remove_model_version(mlflow_client, FLAGS.mlflow_model_name, FLAGS.mlflow_model_version)
