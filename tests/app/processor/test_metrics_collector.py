import os
import tempfile
import json
from unittest.mock import create_autospec
from app.model_services.base import AbstractModelService
from app.processors.metrics_collector import (
    evaluate_model_with_trainer_export,
    concat_trainer_exports,
    get_cuis_from_trainer_export,
)


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

    precision, recall, f1, per_cui_prec, per_cui_rec, per_cui_f1, per_cui_name = evaluate_model_with_trainer_export(path, model_service)
    assert precision == 0.5
    assert recall == 0.07142857142857142
    assert f1 == 0.125
    assert set(per_cui_prec.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_rec.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_f1.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_name.keys()) == {"C0017168", "C0020538"}


def test_evaluate_model_and_return_dataframe():
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

    result = evaluate_model_with_trainer_export(path, model_service, return_df=True)

    assert set(result["concept"].to_list()) == {"C0020538", "C0017168"}
    assert set(result["name"].to_list()) == {"gastroesophageal reflux", "hypertension"}
    assert set(result["precision"].to_list()) == {0.5, 0.5}
    assert set(result["recall"].to_list()) == {0.25, 1.0}
    assert set(result["f1"].to_list()) == {0.3333333333333333, 0.6666666666666666}


def test_concat_trainer_exports():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    with tempfile.NamedTemporaryFile() as f:
        concat_trainer_exports([path, path, path], f.name)
        new_export = json.load(f)
        assert len(new_export["projects"]) == 3


def test_get_cuis_from_trainer_export():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    cuis = get_cuis_from_trainer_export(path)
    assert cuis == {'C0010068', 'C0011860', 'C0003864', 'C0011849', 'C0878544', 'C0020473', 'C0155626', 'C0007222',
                    'C0012634', 'C0020538', 'C0038454', 'C0042029', 'C0007787', 'C0027051', 'C0017168', 'C0338614',
                    'C0037284'}