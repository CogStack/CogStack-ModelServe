import os
import tempfile
import json
from unittest.mock import create_autospec
from app.model_services.base import AbstractModelService
from app.processors.metrics_collector import evaluate_model_with_trainer_export, concat_trainer_exports


def test_evaluate_model_with_trainer_export():
    model_service = create_autospec(AbstractModelService)
    annotations = [
        {
            "label_name": "gastroesophageal reflux",
            "label_id": "C0017168",
            "start": 332,
            "end": 355,
        },
        {
            "label_name": "hypertension",
            "label_id": "C0020538",
            "start": 255,
            "end": 267,
        }
    ]
    model_service.annotate.return_value = annotations
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")

    precision, recall, f1, per_cui_prec, per_cui_rec, per_cui_f1 = evaluate_model_with_trainer_export(path, model_service)
    assert precision == 0.5
    assert recall == 0.07142857142857142
    assert f1 == 0.125
    assert set(per_cui_prec.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_rec.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_f1.keys()) == {"C0017168", "C0020538"}

def test_concat_trainer_exports():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    with tempfile.NamedTemporaryFile() as f:
        concat_trainer_exports([path, path, path], f.name)
        new_export = json.load(f)
        assert len(new_export["projects"]) == 3
