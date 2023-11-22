import os
import json

from utils import get_settings
from urllib.parse import urlparse
from utils import (
    get_code_base_uri,
    annotations_to_entities,
    send_gelf_message,
    get_rate_limiter,
    get_func_params_as_dict,
    json_normalize_medcat_entities,
    json_normalize_trainer_export,
    json_denormalize,
    filter_by_concept_ids,
    encrypt, decrypt,
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
        "label": "Spinal stenosis",
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


def test_get_per_address_rate_limiter():
    limiter = get_rate_limiter(auth_user_enabled=False)
    assert limiter._key_func.__name__ == "get_remote_address"


def test_get_per_user_rate_limiter():
    limiter = get_rate_limiter(auth_user_enabled=True)
    assert limiter._key_func.__name__ == "get_user_auth"


def test_get_func_params_as_dict():
    def func(arg1, arg2=None, arg3="arg3"):
        pass
    params = get_func_params_as_dict(func)
    assert params == {"arg2": None, "arg3": "arg3"}


def test_json_normalize_medcat_entities():
    medcat_entities_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "medcat_entities.json")
    with open(medcat_entities_path, "r") as f:
        medcat_entities = json.load(f)
    df = json_normalize_medcat_entities(medcat_entities)
    assert len(df) == 25
    assert df.columns.tolist() == ["pretty_name", "cui", "type_ids", "types", "source_value", "detected_name", "acc", "context_similarity", "start", "end", "icd10", "ontologies", "snomed", "id", "meta_anns.Presence.value", "meta_anns.Presence.confidence", "meta_anns.Presence.name", "meta_anns.Subject.value", "meta_anns.Subject.confidence", "meta_anns.Subject.name", "meta_anns.Time.value", "meta_anns.Time.confidence", "meta_anns.Time.name"]


def test_json_normalize_trainer_export():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    df = json_normalize_trainer_export(trainer_export)
    assert len(df) == 30
    assert df.columns.tolist() == ["id", "user", "cui", "value", "start", "end", "validated", "correct", "deleted", "alternative", "killed", "last_modified", "manually_created", "acc", "meta_anns.Status.name", "meta_anns.Status.value", "meta_anns.Status.acc", "meta_anns.Status.validated", "projects.name", "projects.id", "projects.cuis", "projects.tuis", "projects.documents.id", "projects.documents.name", "projects.documents.text", "projects.documents.last_modified"]


def test_json_denormalize():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    df = json_normalize_trainer_export(trainer_export)
    trainer_export = json_denormalize(df)
    assert len(trainer_export) == 30


def test_filter_by_concept_ids():
    backup = get_settings().TRAINING_CONCEPT_ID_WHITELIST
    get_settings().TRAINING_CONCEPT_ID_WHITELIST = "C0017168"
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    filtered = filter_by_concept_ids(trainer_export)
    for project in filtered["projects"]:
        for document in project["documents"]:
            for annotation in document["annotations"]:
                assert annotation["cui"] == "C0017168"
    get_settings().TRAINING_CONCEPT_ID_WHITELIST = backup


def test_encrypt():
    fake_public_key_pem = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ITkTP8Tm/5FygcwY2EQ
7LgVsuCF0OH7psUqvlXnOPNCfX86CobHBiSFjG9o5ZeajPtTXaf1thUodgpJZVZS
qpVTXwGKo8r0COMO87IcwYigkZZgG/WmZgoZART+AA0+JvjFGxflJAxSv7puGlf8
2E+u5Wz2psLBSDO5qrnmaDZTvPh5eX84cocahVVI7X09/kI+sZiKauM69yoy1bdx
16YIIeNm0M9qqS3tTrjouQiJfZ8jUKSZ44Na/81LMVw5O46+5GvwD+OsR43kQ0Te
xMwgtHxQQsiXLWHCDNy2ZzkzukDYRwA3V2lwVjtQN0WjxHg24BTBDBM+v7iQ7cbw
eQIDAQAB
-----END PUBLIC KEY-----"""
    encrypted = encrypt("test", fake_public_key_pem)
    assert isinstance(encrypted, str)
    assert len(encrypted) > 0


def test_decrypt():
    fake_private_key_pem = """-----BEGIN PRIVATE KEY-----
MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDchORM/xOb/kXK
BzBjYRDsuBWy4IXQ4fumxSq+Vec480J9fzoKhscGJIWMb2jll5qM+1Ndp/W2FSh2
CkllVlKqlVNfAYqjyvQI4w7zshzBiKCRlmAb9aZmChkBFP4ADT4m+MUbF+UkDFK/
um4aV/zYT67lbPamwsFIM7mqueZoNlO8+Hl5fzhyhxqFVUjtfT3+Qj6xmIpq4zr3
KjLVt3HXpggh42bQz2qpLe1OuOi5CIl9nyNQpJnjg1r/zUsxXDk7jr7ka/AP46xH
jeRDRN7EzCC0fFBCyJctYcIM3LZnOTO6QNhHADdXaXBWO1A3RaPEeDbgFMEMEz6/
uJDtxvB5AgMBAAECggEABLc80J610yStZmQf90gYng9Tu3cMtYpXoNnfj6Fzp+af
2eIyIg5+zBVU28t4IUzMK86mGj8gxIuQSXHv3uBpNSerWEFGrzkEXfpJFBIPhl3/
HQ3rsT1gGReHMFw8EFE4LoosYOdyaYJv9JSujRarnA6cLWDWp3tLudkNU+bU1A6n
MyXwM1jyM5RkLKSY5tTuzNZ3fL/Yz+Spuxw9yKFE6l6Rcb0weLYMNVrPlSr4SfJ3
R9WyfRKqO2WXZCJ5sGEOx30Zas6ivsorVZ+b9VWkAaDvCpcbg4ahyfGjhWFWFpCo
+zxFlmfGyouY8OtL7Tq7QSnHxoFvMBv7p/CpTuezDwKBgQDrWGjGsAZrD9sIKuYC
yAo7SkaN8s1tm224MYlFd26vzvVxMUv2ZYgRGDPD3L8KDgzIPpU9ltnyPnKmso6c
92+Uit3p1lCLvrRZI+ArYaXkk7pl/XjAd9FNzIWp5mBCOIeEdpeOpBscaOe1yxDG
VvK1RKBqZNX1vkmcjSSRA6So1wKBgQDv32A76d4UQNzjIQeDn/Q4LGZOKPMyC+ys
u/Pf91hGnu6LvcmKjs2HhgOUlH1Nd5voR+bb0AxbdrOV8EtoYoWAg8c5t/jzWspK
UXIRe37EQeKSV6MwU+93Tcjr2fohdGznc6etECa8b9n05qLZa6pt7MtMM2vI69mR
aCGbtnB3LwKBgQDWUeLI3dBae0v6OibQ7Z7zs4ZhCnYtlNfsX6Ak1MjF7fDyrfQB
ZSDugF3TxhlrbLQTP3rlZZUA2AHM8NqS83p3iabhpjwfpwHSE6u3letfJ3EeJCBt
FjBTaydmO9f5NkWjSeRnD+dojdhFY7HZDaFlliOIAGAgtLOQj7B3JxwybQKBgQDc
bwh+xqJhNmJHD5laKmpCHPs/JH6pJTAwZODult02uOM65AQMIsNZoZw0tGiaAiry
QPE0W3KfsuvCBHsnyDIrMe6pahmLeYmg1kvfKQAL1wghuAutY9USbBcSNtSYXeee
ozgZ4FfYn2lKl5BcAYczUYJZ2n9YuvTLnUgVUojz3QKBgQDmewPhaqYJOKDHeY6D
QySZIZwb2mZd3nozPMzBJuTh5QK+KPkzSeJTihuIZh8ZImD0LX3TX8KSdz9oZQQR
cExDsxcGU7ZcTO9WVwDhqF/9ofkXfLOFKxugLNEA5RA3gRcpCxMRLS4k6dfN9N9o
3RQZkF/usTTvyvFQR96frZb2FQ==
-----END PRIVATE KEY-----"""
    encrypted = "TLlMBh4GDf3BSsO/RKlqG5H7Sxv7OXGbl8qE/6YLQPm3coBbnrRRReX7pLamnjLPUU0PtIRIg2H/hWBWE/3cRtXDPT7jMtmGHMIPO/95A0DkrndIkOeQ29J6TBPBBG6YqBNRb2dyhDBwDIEDjPTiRe68sYz4KkxzSOkcz31314kSkZvdIDtQOgeRDa0/7U0VrJePL2N7SJvEiHf4Xa3vW3/20S3O8s/Yp0Azb/kS9dFa54VO1fNNhJ46OtPpdekiFDR5yvQfHwFVeSDdY+eAuYLTWa6bz/LrQkRAdRi9EW5Iz/q8WgKhZXQJfcXtiKfVuFar2N2KodY7C/45vMOfvw=="
    decrypted = decrypt(encrypted, fake_private_key_pem)
    assert decrypted == "test"
