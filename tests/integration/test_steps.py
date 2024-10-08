import json
import httpx
import api.globals as cms_globals
from pytest_bdd import scenarios, given, when, then, parsers
from unittest.mock import create_autospec
from fastapi.testclient import TestClient
from management.model_manager import ModelManager
from model_services.medcat_model import MedCATModel
from domain import ModelCard, ModelType
from api.api import get_model_server, get_stream_server
from utils import get_settings
from helper import data_table, async_to_sync

scenarios("features/serving.feature")
scenarios("features/serving_stream.feature")


@given("CMS app is up and running", target_fixture="context")
def cms_is_running():
    config = get_settings()
    config.AUTH_USER_ENABLED = "false"
    model_service = create_autospec(MedCATModel)
    single_annotation = {
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 0,
        "end": 15,
        "accuracy": 1.0,
        "meta_anns": {
            "Status": {
                "value": "Affirmed",
                "confidence": 0.9999833106994629,
                "name": "Status"
            }
        },
    }
    model_service.annotate.return_value = [single_annotation]
    annotations_list = [
        [single_annotation],
        [single_annotation],
    ]
    model_service.batch_annotate.return_value = annotations_list
    model_card = ModelCard.parse_obj({
        "api_version": "0.0.1",
        "model_description": "medcat_model_description",
        "model_type": ModelType.MEDCAT_SNOMED,
        "model_card": None,
    })
    model_service.info.return_value = model_card
    model_manager = ModelManager(None, None)
    model_manager.model_service = model_service
    cms_globals.model_manager_dep = lambda: model_manager
    app = get_model_server(msd_overwritten=lambda: model_service)
    client = TestClient(app)
    return {
        "app": app,
        "client": client,
        "model_service": model_service,
        "single_annotation": single_annotation,
    }


@given("CMS stream app is up and running", target_fixture="context_stream")
def cms_stream_is_running():
    config = get_settings()
    config.AUTH_USER_ENABLED = "false"
    model_service = create_autospec(MedCATModel)
    single_annotation = {
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 0,
        "end": 15,
        "accuracy": 1.0,
        "meta_anns": {
            "Status": {
                "value": "Affirmed",
                "confidence": 0.9999833106994629,
                "name": "Status"
            }
        },
    }
    model_service.async_annotate.return_value = [single_annotation]
    model_manager = ModelManager(None, None)
    model_manager.model_service = model_service
    cms_globals.model_manager_dep = lambda: model_manager
    app = get_stream_server(msd_overwritten=lambda: model_service)
    client = TestClient(app)
    return {
        "app": app,
        "client": client,
        "model_service": model_service,
        "single_annotation": single_annotation,
    }


@then("the response should contain annotations")
def check_response_json(context):
    assert context["response"].json()["annotations"] == [context["single_annotation"]]


@when(parsers.parse("I send a GET request to {endpoint}"))
def send_get_request(context, endpoint):
    context["response"] = context["client"].get(endpoint)


@then(parsers.parse("the response should contain body {body} and status code {status_code:d}"))
def check_status_code(context, body, status_code):
    assert context["response"].content.decode("utf-8") == body
    assert context["response"].status_code == status_code


@when(data_table("I send a POST request with the following content", fixture="request", orient="dict"))
def send_post_request(context, request):
    context["response"] = context["client"].post(request[0]["endpoint"], data=request[0]["data"].replace("\\n", "\n"), headers={"Content-Type": request[0]["content_type"]})


@then("the response should contain json lines")
def check_response_jsonl(context):
    jsonlines = context["response"].text[:-1].split("\n")
    assert len(jsonlines) == 2
    assert json.loads(jsonlines[0]) == {"doc_name": "doc1", **context["single_annotation"]}
    assert json.loads(jsonlines[1]) == {"doc_name": "doc2", **context["single_annotation"]}


@then("the response should contain bulk annotations")
def check_response_bulk(context):
    assert context["response"].json() == [
        {
            "text": "Spinal stenosis",
            "annotations": [context["single_annotation"]]
        },
        {
            "text": "Spinal stenosis",
            "annotations": [context["single_annotation"]]
        },
    ]


@then(parsers.parse("the response should contain text {redaction}"))
def check_response_redacted(context, redaction):
    assert context["response"].text == redaction


@then("the response should contain a preview page")
def check_response_previewed(context):
    assert context["response"].status_code == 200
    assert context["response"].headers["Content-Type"] == "application/octet-stream"


@when(data_table("I send an async POST request with the following content", fixture="request", orient="dict"))
@async_to_sync
async def send_async_post_request(context_stream, request):
    async with httpx.AsyncClient(app=context_stream["app"], base_url="http://test") as ac:
        context_stream["response"] = await ac.post(request[0]["endpoint"],
                                                   data=request[0]["data"].replace("\\n", "\n").encode("utf-8"),
                                                   headers={"Content-Type": request[0]["content_type"]})


@then("the response should contain annotation stream")
@async_to_sync
async def check_response_stream(context_stream):
    assert context_stream["response"].status_code == 200
    jsonlines = b""
    async for chunk in context_stream["response"].aiter_bytes():
        jsonlines += chunk
    assert json.loads(jsonlines.decode("utf-8").splitlines()[-1]) == {"doc_name": "doc2", **context_stream["single_annotation"]}


@when("I send a piece of text to the WS endpoint")
def send_ws_request(context_stream):
    with TestClient(context_stream["app"]) as client:
        with client.websocket_connect("/stream/ws") as websocket:
            websocket.send_text("Spinal stenosis")
            context_stream["response"] = websocket.receive_text()


@then("the response should contain annotated spans")
def check_response_ws(context_stream):
    assert context_stream["response"] == "[Spinal stenosis: Spinal stenosis]"
