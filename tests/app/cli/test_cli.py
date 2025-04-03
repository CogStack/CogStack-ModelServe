import os
import pytest
from typer.testing import CliRunner
from unittest.mock import patch
from app.cli.cli import cmd_app

MODEL_PARENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "model")

runner = CliRunner()


def test_serve_help():
    result = runner.invoke(cmd_app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "This serves various CogStack NLP models" in result.output


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_serve_model():
    with patch("uvicorn.run", side_effect=KeyboardInterrupt):
        result = runner.invoke(
            cmd_app,
            [
                "serve",
                "--model-type",
                "medcat_deid",
                "--model-name",
                "deid model",
                "--model-path",
                os.path.join(MODEL_PARENT_DIR, "deid_model.zip"),
            ],
        )
    assert result.exit_code == 1
    assert "\nAborted.\n" in result.output


def test_register_help():
    result = runner.invoke(cmd_app, ["register", "--help"])
    assert result.exit_code == 0
    assert "This pushes a pretrained NLP model to the CogStack ModelServe registry" in result.output


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_register_nodel():
    result = runner.invoke(
        cmd_app,
        [
            "register",
            "--model-type",
            "medcat_deid",
            "--model-name",
            "deid model",
            "--model-path",
            os.path.join(MODEL_PARENT_DIR, "deid_model.zip"),
        ],
    )
    assert result.exit_code == 0
    assert "as a new model version" in result.output


def test_generate_api_doc_per_model_help():
    result = runner.invoke(cmd_app, ["export-model-apis", "--help"])
    assert result.exit_code == 0
    assert "This generates a model-specific API document for enabled endpoints" in result.output


def test_generate_api_doc_per_model():
    result = runner.invoke(cmd_app, ["export-model-apis", "--model-type", "medcat_deid"])
    assert result.exit_code == 0
    assert "OpenAPI doc exported to" in result.output


def test_stream_chat_help():
    result = runner.invoke(cmd_app, ["stream", "chat", "--help"])
    assert result.exit_code == 0
    assert "This gets NER entities by chatting with the model" in result.output


def test_stream_json_lines_help():
    result = runner.invoke(cmd_app, ["stream", "json-lines", "--help"])
    assert result.exit_code == 0
    assert "This gets NER entities as a JSON Lines stream" in result.output


def test_package_hf_model_help():
    result = runner.invoke(cmd_app, ["package", "hf-model", "--help"])
    assert result.exit_code == 0
    assert "This packages a remotely hosted or locally cached" in result.output


def test_package_hf_dataset_help():
    result = runner.invoke(cmd_app, ["package", "hf-dataset", "--help"])
    assert result.exit_code == 0
    assert "This packages a remotely hosted or locally cached" in result.output


def test_generate_api_doc_help():
    result = runner.invoke(cmd_app, ["export-openapi-spec", "--help"])
    assert result.exit_code == 0
    assert "This generates an API document for all endpoints defined in CMS" in result.output


def test_generate_api_doc():
    result = runner.invoke(cmd_app, ["export-openapi-spec", "--api-title", "TestAPIs"])
    assert result.exit_code == 0
    assert "OpenAPI doc exported to testapis.json" in result.output
