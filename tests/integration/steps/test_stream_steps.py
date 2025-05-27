import json
import httpx
import pytest
import socket
import websockets
from pytest_bdd import scenarios, given, when, then
from helper import ensure_app_config, get_logger, download_model, data_table, async_to_sync, run


scenarios("../features/serving_stream.feature")
ensure_app_config(debug_mode=False)
logger = get_logger(debug=True, name="cms-integration-stream")

@pytest.fixture(scope="module")
def cms_stream():
    model_pack_url = "https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip"
    model_path = download_model(model_pack_url, "cms_stream_model.zip")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    conf = {
        "model_path": model_path,
        "base_url": f"http://127.0.0.1:{port}",
        "process": None,
    }

    yield conf

    if conf["process"] is not None and conf["process"].poll() is None:
        logger.info("Terminating CMS stream server...")
        conf["process"].terminate()
        conf["process"].wait(timeout=30)

@given("CMS stream app is up and running", target_fixture="context_stream")
def cms_stream_is_running(cms_stream):
    return run(cms_stream, logger, streamable=True)

@when(data_table("I send an async POST request with the following jsonlines content", fixture="request", orient="dict"))
@async_to_sync
async def send_async_post_request(context_stream, request):
    async with httpx.AsyncClient(base_url=context_stream["base_url"]) as ac:
        context_stream["response"] = await ac.post(
            f"{context_stream['base_url']}{request[0]['endpoint']}",
            data=request[0]["data"].replace("\\n", "\n").encode("utf-8"),
            headers={"Content-Type": request[0]["content_type"]},
        )

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