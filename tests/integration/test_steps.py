import os
import json
import tempfile
import httpx
import pytest
import requests
import websockets
from pytest_bdd import scenarios, given, when, then, parsers
from helper import ensure_app_config, get_logger, download_model, data_table, async_to_sync, run


ensure_app_config(debug_mode=False)
logger = get_logger(debug=True)

model_pack_url = "https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip"
model_path = download_model(model_pack_url)

@pytest.fixture(scope="session")
def cms():
    conf = {
        "model_path": model_path,
        "base_url": "http://127.0.0.1:8100",
        "process": None,
    }

    yield conf

    if conf["process"] is not None and conf["process"].poll() is None:
        logger.info("Terminating CMS server...")
        conf["process"].terminate()
        conf["process"].wait(timeout=30)


@pytest.fixture(scope="session")
def cms_stream():
    conf = {
        "model_path": model_path,
        "base_url": "http://127.0.0.1:8101",
        "process": None,
    }

    yield conf

    if conf["process"] is not None and conf["process"].poll() is None:
        logger.info("Terminating CMS stream server...")
        conf["process"].terminate()
        conf["process"].wait(timeout=30)

scenarios("features/serving.feature")
scenarios("features/serving_stream.feature")


@given("CMS app is up and running", target_fixture="context")
def cms_is_running(cms):
   return run(cms, logger)

@given("CMS stream app is up and running", target_fixture="context_stream")
def cms_stream_is_running(cms_stream):
   return run(cms_stream, logger, streamable=True)

@then("the response should contain annotations")
def check_response_json(context):
    assert context["response"].headers["Content-Type"] == "application/json"
    annotations = context["response"].json()["annotations"]
    assert len(annotations) >= 1
    assert isinstance(annotations[0], dict)
    assert annotations[0]["start"] == 0
    assert annotations[0]["end"] == 15
    assert annotations[0]["label_name"].lower() == "spinal stenosis"
    assert isinstance(annotations[0]["label_id"], str)
    context["response"].close()

@when(parsers.parse("I send a GET request to {endpoint}"))
def send_get_request(context, endpoint):
   with requests.get(f"{context['base_url']}{endpoint}") as response:
       context["response"] = response

@when(parsers.parse("I send a GET request to {endpoint} with that ID"))
def send_get_request_with_train_eval_id(context, endpoint):
   with requests.get(f"{context['base_url']}{endpoint}?train_eval_id={context['train_eval_id']}") as response:
       context["response"] = response

@then(parsers.parse("the response should contain the training information"))
def check_response_train_info(context):
    assert context["response"].headers["Content-Type"] == "application/json"
    response_json = context["response"].json()
    assert len(response_json) == 1
    assert "artifact_uri" in response_json[0]
    assert "experiment_id" in response_json[0]
    assert "run_id" in response_json[0]
    assert "run_name" in response_json[0]
    assert "status" in response_json[0]
    assert "tags" in response_json[0]
    context["response"].close()

@then(parsers.parse("the response should contain the evaluation information"))
def check_response_eval_info(context):
    check_response_train_info(context)

@then(parsers.parse("the response should contain body {body} and status code {status_code:d}"))
def check_status_code(context, body, status_code):
    assert body in context["response"].content.decode("utf-8")
    assert context["response"].status_code == status_code
    context["response"].close()

@when(data_table("I send a POST request with the following content", fixture="request", orient="dict"))
def send_post_request(context, request):
    context["response"] = requests.post(f"{context['base_url']}{request[0]['endpoint']}",
                                        data=request[0]["data"],
                                        headers={"Content-Type": request[0]["content_type"]})

@when(data_table("I send a POST request with the following jsonlines content", fixture="request", orient="dict"))
def send_post_request(context, request):
    context["response"] = requests.post(f"{context['base_url']}{request[0]['endpoint']}",
                                        data=request[0]["data"].replace("\\n", "\n"),
                                        headers={"Content-Type": request[0]["content_type"]})

@when(data_table("I send a POST request with the following content where data as a file", fixture="request", orient="dict"))
def send_post_request_file(context, request):
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        f.write(request[0]["data"])
        f.seek(0)
        context["response"] = requests.post(f"{context['base_url']}{request[0]['endpoint']}",
                                            files=[("multi_text_file", f)])

@then("the response should contain json lines")
def check_response_jsonl(context):
    assert context["response"].headers["Content-Type"] == "application/x-ndjson; charset=utf-8"
    jsonlines = context["response"].text[:-1].split("\n")
    assert len(jsonlines) == 2
    first_line = json.loads(jsonlines[0])
    second_line = json.loads(jsonlines[1])
    assert first_line["doc_name"] == "doc1"
    assert first_line["start"] == 0
    assert first_line["end"] == 15
    assert first_line["label_name"].lower() == "spinal stenosis"
    assert isinstance(first_line["label_id"], str)
    assert second_line["doc_name"] == "doc2"
    assert second_line["start"] == 0
    assert second_line["end"] == 15
    assert second_line["label_name"].lower() == "spinal stenosis"
    assert isinstance(second_line["label_id"], str)
    context["response"].close()

@then("the response should contain bulk annotations")
def check_response_bulk(context):
    assert context["response"].headers["Content-Type"] == "application/json"
    bulk_results = context["response"].json()
    assert isinstance(bulk_results, list)
    assert len(bulk_results) == 2
    assert bulk_results[0]["text"] == "Spinal stenosis"
    assert bulk_results[0]["annotations"][0]["start"] == 0
    assert bulk_results[0]["annotations"][0]["end"] == 15
    assert bulk_results[0]["annotations"][0]["label_name"].lower() == "spinal stenosis"
    assert isinstance(bulk_results[0]["annotations"][0]["label_id"], str)
    assert bulk_results[1]["text"] == "Spinal stenosis"
    assert bulk_results[1]["annotations"][0]["start"] == 0
    assert bulk_results[1]["annotations"][0]["end"] == 15
    assert bulk_results[1]["annotations"][0]["label_name"].lower() == "spinal stenosis"
    assert isinstance(bulk_results[1]["annotations"][0]["label_id"], str)
    context["response"].close()

@then(parsers.parse("the response should contain text {redaction}"))
def check_response_redacted(context, redaction):
    assert context["response"].text.lower() == redaction
    context["response"].close()

@then("the response should contain a preview page")
def check_response_previewed(context):
    assert context["response"].status_code == 200
    assert context["response"].headers["Content-Type"] == "application/octet-stream"
    assert context["response"].headers["Content-Disposition"].startswith("attachment ; filename=")
    context["response"].close()

@when(data_table("I send a POST request with the following trainer export", fixture="request", orient="dict"))
def send_post_training_request_file(context, request):
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", request[0]["file_name"])
    with open(trainer_export_path, "rb") as f:
        context["response"] = requests.post(f"{context['base_url']}{request[0]['endpoint']}",
                                            files=[("trainer_export", f)])

@when(data_table("I send a POST request with the following training data", fixture="request", orient="dict"))
def send_post_training_request_file(context, request):
    training_data_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", request[0]["file_name"])
    with open(training_data_path, "rb") as f:
        context["response"] = requests.post(
            f"{context['base_url']}{request[0]['endpoint']}?epochs=1&test_size=0.2&log_frequency=1000",
             files=[("training_data", f)]
        )

@then("the response should contain the training ID")
def check_response_training_id(context):
    assert context["response"].status_code == 202
    assert context["response"].headers["Content-Type"] == "application/json"
    assert "training_id" in context["response"].json()
    context['train_eval_id'] = context["response"].json()["training_id"]
    context["response"].close()


@then("the response should contain the evaluation ID")
def check_response_evaluation_id(context):
    assert context["response"].status_code == 202
    assert context["response"].headers["Content-Type"] == "application/json"
    assert "evaluation_id" in context["response"].json()
    context['train_eval_id'] = context["response"].json()["evaluation_id"]
    context["response"].close()

@then("the response should contain the supervised evaluation metrics")
def check_response_evaluation_metrics(context):
    assert context["response"].status_code == 200
    assert context["response"].headers["Content-Type"] == "application/json"
    response_json = context["response"].json()
    assert len(response_json) == 1
    assert "precision" in response_json[0]
    assert "recall" in response_json[0]
    assert "f1" in response_json[0]
    assert "per_concept_precision" in response_json[0]
    assert "per_concept_recall" in response_json[0]
    assert "per_concept_f1" in response_json[0]
    context["response"].close()

@then("the response should contain the unsupervised evaluation metrics")
def check_response_evaluation_metrics(context):
    assert context["response"].status_code == 200
    assert context["response"].headers["Content-Type"] == "application/json"
    response_json = context["response"].json()
    assert len(response_json) == 1
    assert "number_of_names" in response_json[0]
    assert "number_of_seen_training_examples_in_total" in response_json[0]
    assert "average_training_examples_per_concept" in response_json[0]
    assert "per_concept_train_count_after" in response_json[0]
    assert "per_concept_train_count_before" in response_json[0]
    assert "number_of_concepts_that_received_training" in response_json[0]
    assert "number_of_concepts" in response_json[0]
    context["response"].close()

@then("the response should contain encrypted labels")
def check_response_training_id(context):
    assert context["response"].status_code == 200
    assert context["response"].headers["Content-Type"] == "application/json"
    response_json = context["response"].json()
    assert "redacted_text" in response_json
    assert "encryptions" in response_json
    assert "label" in response_json["encryptions"][0]
    assert "encryption" in response_json["encryptions"][0]
    context["response"].close()

@when(data_table("I send an async POST request with the following jsonlines content", fixture="request", orient="dict"))
@async_to_sync
async def send_async_post_request(context_stream, request):
    async with httpx.AsyncClient(base_url=context_stream["base_url"]) as ac:
        context_stream["response"] = await ac.post(f"{context_stream['base_url']}{request[0]['endpoint']}",
                                                   data=request[0]["data"].replace("\\n", "\n").encode("utf-8"),
                                                   headers={"Content-Type": request[0]["content_type"]})

@then("the response should contain annotation stream")
@async_to_sync
async def check_response_stream(context_stream):
    assert context_stream["response"].status_code == 200

    async for line in context_stream["response"].aiter_lines():
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        assert data["doc_name"] in ["doc1", "doc2"]

@when("I send a piece of text to the WS endpoint")
@async_to_sync
async def send_ws_request(context_stream):
    ws_url = context_stream["base_url"].replace("http", "ws") + "/stream/ws"
    async with websockets.connect(ws_url) as websocket:
        await websocket.send("Spinal stenosis")
        context_stream["response"] = await websocket.recv()

@then("the response should contain annotated spans")
def check_response_ws(context_stream):
    assert context_stream["response"].lower() == "[spinal stenosis: spinal stenosis]"
