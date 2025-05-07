import os
import pytest
import requests
import socket
from pytest_bdd import scenarios, given, when, then
from helper import ensure_app_config, get_logger, data_table, run


scenarios("../features/serving_llm.feature")
ensure_app_config(debug_mode=False)
logger = get_logger(debug=True, name="cms-integration-llm")

@pytest.fixture(scope="module")
def cms_llm():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "model", "huggingface_llm_model.tar.gz"
    )
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
        logger.info("Terminating CMS LLM server...")
        conf["process"].terminate()
        conf["process"].wait(timeout=30)

@given("CMS LLM app is up and running", target_fixture="context_llm")
def cms_llm_is_running(cms_llm):
    return run(cms_llm, logger, generative=True)


@when(data_table("I send a POST request with the following prompt", fixture="request", orient="dict"))
def send_post_request_prompt(context_llm, request):
    context_llm["response"] = requests.post(
        f"{context_llm['base_url']}{request[0]['endpoint']}",
        data=request[0]["prompt"],
        headers={"Content-Type": request[0]["content_type"]},
    )

@then("the response should contain generated text")
def check_response_generated_text(context_llm):
    assert context_llm["response"].headers["Content-Type"] == "text/plain; charset=utf-8"
    assert len(context_llm["response"].text) >= 1

@then("the response should contain generated text stream")
def check_response_generated_text_stream(context_llm):
    assert context_llm["response"].headers["Content-Type"] == "text/event-stream; charset=utf-8"
    buffer = ""
    for line in context_llm["response"].iter_lines(decode_unicode=True):
        buffer += line + '\n'
    assert len(buffer) >= 1
