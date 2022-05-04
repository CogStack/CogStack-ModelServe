from fastapi.testclient import TestClient
from app.serve import get_model_server
from app.nlpmodel import NLPModel
from unittest.mock import create_autospec


model = create_autospec(NLPModel)
client = TestClient(get_model_server(model))


def test_process():
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 1,
        "end": 15,
        "meta_anns": None
    }]
    model.annotate.return_value = annotations
    response = client.post("/process?text=Spinal%20stenosis")
    print(response)
    assert response.json() == {
        "text": "Spinal stenosis",
        "annotations": annotations
    }


def test_info():
    model.info.return_value = {
        "model_description": "medmen model",
        "model_type": "medcat"
    }
    response = client.get("/info")
    assert response.json() == {
        "model_description": "medmen model",
        "model_type": "medcat"
    }