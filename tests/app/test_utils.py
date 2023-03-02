from urllib.parse import urlparse
from app.utils import (
    get_code_base_uri,
    annotations_to_entities,
    send_gelf_message,
)


def test_get_code_base_uri():
    assert get_code_base_uri("SNOMED model") == "http://snomed.info/id"
    assert get_code_base_uri("ICD-10 model") == "https://icdcodelookup.com/icd-10/codes"
    assert get_code_base_uri("UMLS model") == "https://uts.nlm.nih.gov/uts/umls/concept"


def test_annotations_to_entities():
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 1,
        "end": 15,
    }]
    expected = [{
        "start": 1,
        "end": 15,
        "label": "Spinal stenosis (76107001)",
        "kb_id": "76107001",
        "kb_url": "http://snomed.info/id/76107001",
    }]
    assert annotations_to_entities(annotations, "SNOMED model") == expected


def test_send_gelf_message(mocker):
    mocked_socket = mocker.Mock()
    mocked_socket_socket = mocker.Mock(return_value=mocked_socket)
    mocker.patch("socket.socket", new=mocked_socket_socket)
    send_gelf_message("message", urlparse("http://127.0.0.1:12201"))
    mocked_socket.connect.assert_called_once_with(("127.0.0.1", 12201))
    mocked_socket.sendall.assert_called_once()
    assert b"\x1e\x0f" in mocked_socket.sendall.call_args[0][0]
    assert b"\x00\x00" in mocked_socket.sendall.call_args[0][0]
    mocked_socket.close.assert_called_once()
