#!/usr/bin/env python

import os
import sys
import argparse
import mlflow
from mlflow.server.auth.client import AuthServiceClient

def main(mlflow_tracking_uri: str, username: str, password: str, revoke: bool, permission: str, insecure: bool) -> None:
    if insecure:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    else:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "false"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    auth_client = AuthServiceClient(mlflow_tracking_uri)

    existing_experiments = mlflow.search_experiments()
    existing_models = mlflow.search_registered_models()
    try:
        user_info = auth_client.get_user(username=username)
    except mlflow.exceptions.RestException:
        user_info = None

    if revoke:
        if user_info:
            for exp in existing_experiments:
                try:
                    auth_client.delete_experiment_permission(experiment_id=exp.experiment_id, username=username)
                except mlflow.exceptions.RestException:
                    continue
            for model in existing_models:
                try:
                    auth_client.delete_registered_model_permission(name=model.name, username=username)
                except mlflow.exceptions.RestException:
                    continue
            print(f"User {username} has got all permissions removed.")
        else:
            print(f"User {username} does not exist.")
            sys.exit(1)
    else:
        if user_info:
            auth_client.update_user_password(username, password)
        else:
            auth_client.create_user(username, password)

        for exp in existing_experiments:
            try:
                existing_exp_permission = auth_client.get_experiment_permission(exp.experiment_id, username)
            except mlflow.exceptions.RestException:
                existing_exp_permission = None

            if existing_exp_permission:
                auth_client.update_experiment_permission(
                    experiment_id=exp.experiment_id,
                    username=username,
                    permission=permission,
                )
            else:
                auth_client.create_experiment_permission(
                    experiment_id=exp.experiment_id,
                    username=username,
                    permission=permission,
                )

        for model in existing_models:
            try:
                existing_model_permission = auth_client.get_registered_model_permission(model.name, username)
            except mlflow.exceptions.RestException:
                existing_model_permission = None

            if existing_model_permission:
                auth_client.update_registered_model_permission(
                    name=model.name,
                    username=username,
                    permission=permission,
                )
            else:
                auth_client.create_registered_model_permission(
                    name=model.name,
                    username=username,
                    permission=permission
                )

        print(f"User {username} has got the permission {permission} on all existing experiments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        required=True,
        help="The URI of the MLflow tracking server.",
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="The username of the user to manage.",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="The password of the user to manage.",
    )
    parser.add_argument(
        "--revoke",
        action="store_true",
        default=False,
        help="Revoke all permissions of an existing user.",
    )
    parser.add_argument(
        "--permission",
        type=str,
        default="READ",
        choices=["READ", "EDIT", "MANAGE", "NO_PERMISSIONS"],
        help="The permission to grant to the user on all existing experiments and models.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=False,
        help="Disable TLS verification when connecting to the tracking server.",
    )

    args = parser.parse_args()
    main(args.mlflow_tracking_uri, args.username, args.password, args.revoke, args.permission, args.insecure)
