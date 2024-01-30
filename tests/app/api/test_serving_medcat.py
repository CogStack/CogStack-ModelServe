import os
import tempfile

import httpx
import json
import pytest
import api.globals as cms_globals
from fastapi.testclient import TestClient
from api.api import get_model_server
from utils import get_settings
from model_services.medcat_model import MedCATModel
from unittest.mock import create_autospec

model_service = create_autospec(MedCATModel)
config = get_settings()
config.ENABLE_TRAINING_APIS = "true"
config.DISABLE_UNSUPERVISED_TRAINING = "false"
config.ENABLE_EVALUATION_APIS = "true"
config.ENABLE_PREVIEWS_APIS = "true"
config.AUTH_USER_ENABLED = "true"
app = get_model_server(msd_overwritten=lambda: model_service)
app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
client = TestClient(app)
TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export.json")
NOTE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "note.txt")
ANOTHER_TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "another_trainer_export.json")
TRAINER_EXPORT_MULTI_PROJS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export_multi_projs.json")
MULTI_TEXTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "sample_texts.json")


def test_healthz():
    assert client.get("/healthz").content.decode("utf-8") == "OK"


def test_readyz():
    model_card = {
        "api_version": "0.0.1",
        "model_description": "medcat_model_description",
        "model_type": "model_type",
        "model_card": None,
    }
    model_service.info.return_value = model_card
    assert client.get("/readyz").content.decode("utf-8") == "model_type"


def test_info():
    model_card = {
        "api_version": "0.0.1",
        "model_description": "medcat_model_description",
        "model_type": "model_type",
        "model_card": None,
    }
    model_service.info.return_value = model_card
    response = client.get("/info")
    assert response.json() == model_card


def test_process():
    annotations = [{
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
    }]
    model_service.annotate.return_value = annotations
    response = client.post("/process",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.json() == {
        "text": "Spinal stenosis",
        "annotations": annotations
    }


def test_process_bulk():
    annotations_list = [
        [{
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
        }],
        [{
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
        }]
    ]
    model_service.batch_annotate.return_value = annotations_list
    response = client.post("/process_bulk", json=["Spinal stenosis", "Spinal stenosis"])
    assert response.json() == [
        {
            "text": "Spinal stenosis",
            "annotations": [{
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
            }]
        },
        {
            "text": "Spinal stenosis",
            "annotations": [{
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
            }]
        }
    ]


def test_redact():
    annotations = [{
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
    }]
    model_service.annotate.return_value = annotations
    response = client.post("/redact",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.text == "[Spinal stenosis]"


def test_redact_with_mask():
    annotations = [{
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
    }]
    model_service.annotate.return_value = annotations
    response = client.post("/redact?mask=***",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.text == "***"


def test_redact_with_hash():
    annotations = [{
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
    }]
    model_service.annotate.return_value = annotations
    response = client.post("/redact?mask=any&hash=true",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.text == "4c86af83314100034ad83fae3227e595fc54cb864c69ea912cd5290b8d0f41a4"


def test_redact_with_encryption():
    annotations = [{
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
    }]
    body = {
      "text": "Spinal stenosis",
      "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ITkTP8Tm/5FygcwY2EQ7LgVsuCF0OH7psUqvlXnOPNCfX86CobHBiSFjG9o5ZeajPtTXaf1thUodgpJZVZSqpVTXwGKo8r0COMO87IcwYigkZZgG/WmZgoZART+AA0+JvjFGxflJAxSv7puGlf82E+u5Wz2psLBSDO5qrnmaDZTvPh5eX84cocahVVI7X09/kI+sZiKauM69yoy1bdx16YIIeNm0M9qqS3tTrjouQiJfZ8jUKSZ44Na/81LMVw5O46+5GvwD+OsR43kQ0TexMwgtHxQQsiXLWHCDNy2ZzkzukDYRwA3V2lwVjtQN0WjxHg24BTBDBM+v7iQ7cbweQIDAQAB\n-----END PUBLIC KEY-----"
    }
    model_service.annotate.return_value = annotations
    response = client.post("/redact_with_encryption",
                           json=body,
                           headers={"Content-Type": "application/json"})
    assert response.json()["redacted_text"] == "[REDACTED_0]"
    assert len(response.json()["encryptions"]) == 1
    assert response.json()["encryptions"][0]["label"] == "[REDACTED_0]"
    assert isinstance(response.json()["encryptions"][0]["encryption"], str)
    assert len(response.json()["encryptions"][0]["encryption"]) > 0


def test_preview():
    annotations = [{
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
    }]
    model_service.annotate.return_value = annotations
    response = client.post("/preview",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"


def test_preview_trainer_export():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_project_id():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?project_id=14", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_document_id():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 1


def test_preview_trainer_export_with_project_and_document_ids():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?project_id=14&document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 1


@pytest.mark.parametrize("pid,did", [(14, 1), (1, 3205)])
def test_preview_trainer_export_on_missing_project_or_document(pid, did):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post(f"/preview_trainer_export?project_id={pid}&document_id={did}", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 404
    assert response.json() == {"message": "Cannot find any matching documents to preview"}


def test_train_supervised():
    with tempfile.TemporaryFile("r+b") as f:
        f.write(str.encode('{"projects":[{"name":"Project1","id":1,"cuis":"","tuis":"","documents":[{"id":1,"name":"1",' +
                           '"text":"Spinal stenosis","last_modified":"","annotations":[{"id":1,"cui":"76107001","start":1,' +
                           '"end":15,"validated":true,"correct":true,"deleted":false,"alternative":false,"killed":false,' +
                           '"last_modified":"","manually_created":false,"acc":1,"meta_anns":[{"name":"Status","value":"Other",' +
                           '"acc":1,"validated":true}]}]}]}]}'))
        response = client.post("/train_supervised", files=[("trainer_export", open(TRAINER_EXPORT_PATH, "rb"))])
    model_service.train_supervised.assert_called()
    assert response.status_code == 202
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_train_unsupervised():
    with tempfile.TemporaryFile("r+b") as f:
        f.write(str.encode("[\"Spinal stenosis\"]"))
        response = client.post("/train_unsupervised", files=[("training_data", f)])
    model_service.train_unsupervised.assert_called()
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_train_metacat():
    with tempfile.TemporaryFile("r+b") as f:
        f.write(str.encode('{"projects":[{"name":"Project1","id":1,"cuis":"","tuis":"","documents":[{"id":1,"name":"1",' +
                           '"text":"Spinal stenosis","last_modified":"","annotations":[{"id":1,"cui":"76107001","start":1,' +
                           '"end":15,"validated":true,"correct":true,"deleted":false,"alternative":false,"killed":false,' +
                           '"last_modified":"","manually_created":false,"acc":1,"meta_anns":[{"name":"Status","value":"Other",' +
                           '"acc":1,"validated":true}]}]}]}]}'))
        response = client.post("/train_metacat", files=[("trainer_export", open(TRAINER_EXPORT_PATH, "rb"))])
    model_service.train_metacat.assert_called()
    assert response.status_code == 202
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_evaluate_with_trainer_export():
    with open(TRAINER_EXPORT_PATH, "rb") as _:
        response = client.post("/evaluate", files=[("trainer_export", open(TRAINER_EXPORT_PATH, "rb"))])
    assert response.status_code == 202
    assert response.json()["message"] == "Your evaluation started successfully."
    assert "evaluation_id" in response.json()


def test_sanity_check_with_trainer_export():
    with open(TRAINER_EXPORT_PATH, "rb") as _:
        response = client.post("/sanity-check", files=[("trainer_export", open(TRAINER_EXPORT_PATH, "rb"))])
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,name,precision,recall,f1"


def test_inter_annotator_agreement_scores_per_concept():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_concept", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


@pytest.mark.parametrize("pid_a,pid_b,error_message", [(0, 2, "Cannot find the project with ID: 0"), (1, 3, "Cannot find the project with ID: 3")])
def test_project_not_found_on_getting_iaa_scores(pid_a, pid_b, error_message):
    with open(TRAINER_EXPORT_MULTI_PROJS_PATH, "rb") as f:
        response = client.post(f"/iaa-scores?annotator_a_project_id={pid_a}&annotator_b_project_id={pid_b}&scope=per_concept", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"message": error_message}


def test_unknown_scope_on_getting_iaa_scores():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=unknown", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"message": "Unknown scope: \"unknown\""}


def test_inter_annotator_agreement_scores_per_doc():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_document", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "doc_id,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


def test_inter_annotator_agreement_scores_per_span():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_span", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "doc_id,span_start,span_end,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


def test_concat_trainer_exports():
    response = client.post("/concat_trainer_exports", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json; charset=utf-8"
    assert len(response.text) == 36842


def test_extract_entities_from_text_list_file_as_json_file():
    annotations_list = [
        [{
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
        }]
    ] * 15
    model_service.batch_annotate.return_value = annotations_list

    response = client.post("/process_bulk_file", files=[
        ("multi_text_file", open(MULTI_TEXTS_FILE_PATH, "rb")),
    ])

    assert isinstance(response, httpx.Response)
    assert json.loads(response.content) == [{
            "text": "Description: Intracerebral hemorrhage (very acute clinical changes occurred immediately).\nCC: Left hand numbness on presentation; then developed lethargy later that day.\nHX: On the day of presentation, this 72 y/o RHM suddenly developed generalized weakness and lightheadedness, and could not rise from a chair. Four hours later he experienced sudden left hand numbness lasting two hours. There were no other associated symptoms except for the generalized weakness and lightheadedness. He denied vertigo.\nHe had been experiencing falling spells without associated LOC up to several times a month for the past year.\nMEDS: procardia SR, Lasix, Ecotrin, KCL, Digoxin, Colace, Coumadin.\nPMH: 1)8/92 evaluation for presyncope (Echocardiogram showed: AV fibrosis/calcification, AV stenosis/insufficiency, MV stenosis with annular calcification and regurgitation, moderate TR, Decreased LV systolic function, severe LAE. MRI brain: focal areas of increased T2 signal in the left cerebellum and in the brainstem probably representing microvascular ischemic disease. IVG (MUGA scan)revealed: global hypokinesis of the LV and biventricular dysfunction, RV ejection Fx 45% and LV ejection Fx 39%. He was subsequently placed on coumadin severe valvular heart disease), 2)HTN, 3)Rheumatic fever and heart disease, 4)COPD, 5)ETOH abuse, 6)colonic polyps, 7)CAD, 8)CHF, 9)Appendectomy, 10)Junctional tachycardia.",
            "annotations": [{
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
            }]
        }
    ] * 15
