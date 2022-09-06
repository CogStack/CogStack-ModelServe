from contextlib import redirect_stdout
from app.management.log_captor import LogCaptor


def test_capture_and_process():
    output = []

    def _process(data):
        output.append(data)

    with redirect_stdout(LogCaptor(_process)):
        print("content 1")
        print("content 2")
        print("content 3", end="")
        print("content 4")

    assert output == ["content 1\n", "content 2\n", "content 3content 4\n"]
