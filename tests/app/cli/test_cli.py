from cli.cli import cmd_app
from typer.testing import CliRunner


runner = CliRunner()


def test_serve_model():
    result = runner.invoke(cmd_app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "This serves various CogStack NLP models" in result.output


def test_register_model():
    result = runner.invoke(cmd_app, ["register", "--help"])
    assert result.exit_code == 0
    assert "This pushes a pretrained NLP model to the Cogstack ModelServe registry" in result.output


def test_generate_api_doc_per_model():
    result = runner.invoke(cmd_app, ["export-model-apis", "--help"])
    assert result.exit_code == 0
    assert "This generates model-specific API docs for enabled endpoints" in result.output


def test_generate_api_doc():
    result = runner.invoke(cmd_app, ["export-openapi-spec", "--help"])
    assert result.exit_code == 0
    assert "This generates a single API doc for all endpoints" in result.output
