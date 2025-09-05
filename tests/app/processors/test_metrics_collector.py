import os
import tempfile
import json
import pytest
from unittest.mock import create_autospec
from app.model_services.base import AbstractModelService
from app.processors.metrics_collector import (
    sanity_check_model_with_trainer_export,
    concat_trainer_exports,
    concat_json_lists,
    get_stats_from_trainer_export,
    get_iaa_scores_per_concept,
    get_iaa_scores_per_doc,
    get_iaa_scores_per_span,
)
from app.exception import AnnotationException
from app.domain import Annotation
from app.utils import load_pydantic_object_from_dict


@pytest.fixture
def model_service():
    return create_autospec(AbstractModelService)


def test_sanity_check_model_with_trainer_export_path(model_service):
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "gastroesophageal reflux",
                "label_id": "C0017168",
                "start": 332,
                "end": 355,
            },
        ),
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "hypertension",
                "label_id": "C0020538",
                "start": 255,
                "end": 267,
            },
        ),
    ]
    model_service.annotate.return_value = annotations
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")

    precision, recall, f1, per_cui_prec, per_cui_rec, per_cui_f1, per_cui_name, per_cui_anchors = sanity_check_model_with_trainer_export(path, model_service)
    assert precision == 0.5
    assert recall == 0.07142857142857142
    assert f1 == 0.125
    assert set(per_cui_prec.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_rec.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_f1.keys()) == {"C0017168", "C0020538"}
    assert set(per_cui_name.keys()) == {"C0017168", "C0020538"}
    assert per_cui_anchors is None


def test_evaluate_model_and_return_dataframe(model_service):
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "gastroesophageal reflux",
                "label_id": "C0017168",
                "start": 332,
                "end": 355,
            }),
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "hypertension",
                "label_id": "C0020538",
                "start": 255,
                "end": 267,
            },
        ),
    ]
    model_service.annotate.return_value = annotations
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")

    result = sanity_check_model_with_trainer_export(path, model_service, return_df=True)

    assert set(result["concept"].to_list()) == {"C0020538", "C0017168"}
    assert set(result["name"].to_list()) == {"gastroesophageal reflux", "hypertension"}
    assert set(result["precision"].to_list()) == {0.5, 0.5}
    assert set(result["recall"].to_list()) == {0.25, 1.0}
    assert set(result["f1"].to_list()) == {0.3333333333333333, 0.6666666666666666}
    assert "anchors" not in result


def test_sanity_check_model_with_trainer_export_file(model_service):
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "gastroesophageal reflux",
                "label_id": "C0017168",
                "start": 332,
                "end": 355,
            }
        ),
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "hypertension",
                "label_id": "C0020538",
                "start": 255,
                "end": 267,
            }
        ),
    ]
    model_service.annotate.return_value = annotations
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")

    with open(path, "r") as file:
        result = sanity_check_model_with_trainer_export(file, model_service, return_df=True)

        assert set(result["concept"].to_list()) == {"C0020538", "C0017168"}
        assert set(result["name"].to_list()) == {"gastroesophageal reflux", "hypertension"}
        assert set(result["precision"].to_list()) == {0.5, 0.5}
        assert set(result["recall"].to_list()) == {0.25, 1.0}
        assert set(result["f1"].to_list()) == {0.3333333333333333, 0.6666666666666666}
        assert "anchors" not in result


def test_sanity_check_model_with_trainer_export_dict(model_service):
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "gastroesophageal reflux",
                "label_id": "C0017168",
                "start": 332,
                "end": 355,
            },
        ),
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "hypertension",
                "label_id": "C0020538",
                "start": 255,
                "end": 267,
            },
        ),
    ]
    model_service.annotate.return_value = annotations
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")

    with open(path, "r") as file:
        result = sanity_check_model_with_trainer_export(json.load(file), model_service, return_df=True)

        assert set(result["concept"].to_list()) == {"C0020538", "C0017168"}
        assert set(result["name"].to_list()) == {"gastroesophageal reflux", "hypertension"}
        assert set(result["precision"].to_list()) == {0.5, 0.5}
        assert set(result["recall"].to_list()) == {0.25, 1.0}
        assert set(result["f1"].to_list()) == {0.3333333333333333, 0.6666666666666666}
        assert "anchors" not in result


def test_evaluate_model_and_include_anchors(model_service):
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "gastroesophageal reflux",
                "label_id": "C0017168",
                "start": 332,
                "end": 355,
            },
        ),
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "hypertension",
                "label_id": "C0020538",
                "start": 255,
                "end": 267,
            },
        ),
    ]
    model_service.annotate.return_value = annotations
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")

    result = sanity_check_model_with_trainer_export(path, model_service, return_df=True, include_anchors=True)

    assert set(result["concept"].to_list()) == {"C0020538", "C0017168"}
    assert set(result["name"].to_list()) == {"gastroesophageal reflux", "hypertension"}
    assert set(result["precision"].to_list()) == {0.5, 0.5}
    assert set(result["recall"].to_list()) == {0.25, 1.0}
    assert set(result["f1"].to_list()) == {0.3333333333333333, 0.6666666666666666}
    assert set(result["anchors"].to_list()) == {"P14/D3204/S255/E267;P14/D3205/S255/E267", "P14/D3204/S332/E355;P14/D3205/S332/E355"}


def test_concat_trainer_exports():
    path_1 = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    path_2 = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export_multi_projs.json")
    with tempfile.NamedTemporaryFile() as f:
        concat_trainer_exports([path_1, path_2], f.name, True)
        new_export = json.load(f)
        assert len(new_export["projects"]) == 3


def test_concat_trainer_exports_with_duplicated_project_ids():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    with pytest.raises(AnnotationException) as e:
        concat_trainer_exports([path, path, path])
    assert "Found multiple projects share the same ID:" in str(e.value)


def test_concat_trainer_exports_with_recurring_document_ids():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    another_path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export_multi_projs.json")
    with pytest.raises(AnnotationException) as e:
        concat_trainer_exports([path, another_path], allow_recurring_doc_ids=False)
    assert str(e.value) == "Found multiple documents share the same ID(s): [3204, 3205]"


def test_get_stats_from_trainer_export():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(path)
    assert cui_counts == {
        "C0003864": 2,
        "C0007222": 1,
        "C0007787": 1,
        "C0010068": 1,
        "C0011849": 1,
        "C0011860": 3,
        "C0012634": 1,
        "C0017168": 1,
        "C0020473": 3,
        "C0020538": 4,
        "C0027051": 1,
        "C0037284": 2,
        "C0038454": 1,
        "C0042029": 4,
        "C0155626": 2,
        "C0338614": 1,
        "C0878544": 1
    }
    assert cui_unique_counts == {
        "C0017168": 1,
        "C0020538": 1,
        "C0012634": 1,
        "C0038454": 1,
        "C0007787": 1,
        "C0155626": 1,
        "C0011860": 3,
        "C0042029": 2,
        "C0010068": 1,
        "C0007222": 1,
        "C0027051": 1,
        "C0878544": 1,
        "C0020473": 1,
        "C0037284": 2,
        "C0003864": 1,
        "C0011849": 1,
        "C0338614": 1
    }
    assert cui_ignorance_counts == {"C0012634": 1, "C0338614": 1}
    assert num_of_docs == 2


def test_get_stats_from_trainer_export_as_dataframe():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export.json")
    result = get_stats_from_trainer_export(path, return_df=True)
    assert result["concept"].tolist() == ["C0017168", "C0020538", "C0012634", "C0038454", "C0007787", "C0155626", "C0011860", "C0042029", "C0010068", "C0007222", "C0027051", "C0878544", "C0020473", "C0037284", "C0003864", "C0011849", "C0338614"]
    assert result["anno_count"].tolist() == [1, 4, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 3, 2, 2, 1, 1]
    assert result["anno_unique_counts"].tolist() == [1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1]
    assert result["anno_ignorance_counts"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_get_iaa_scores_per_concept():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export_multi_projs.json")
    per_cui_anno_iia_pct, per_cui_anno_cohens_kappa, per_cui_metaanno_iia_pct, per_cui_metaanno_cohens_kappa = get_iaa_scores_per_concept(path, 1, 2)
    assert set(per_cui_anno_iia_pct.keys()) == {"C0003864", "C0007222", "C0007787", "C0010068", "C0011849", "C0011860", "C0012634", "C0017168", "C0020473", "C0020538", "C0027051", "C0037284", "C0038454", "C0042029", "C0155626", "C0338614", "C0878544"}
    assert set(per_cui_anno_cohens_kappa.keys()) == {"C0003864", "C0007222", "C0007787", "C0010068", "C0011849", "C0011860", "C0012634", "C0017168", "C0020473", "C0020538", "C0027051", "C0037284", "C0038454", "C0042029", "C0155626", "C0338614", "C0878544"}
    assert set(per_cui_metaanno_iia_pct.keys()) == {"C0003864", "C0007222", "C0007787", "C0010068", "C0011849", "C0011860", "C0012634", "C0017168", "C0020473", "C0020538", "C0027051", "C0037284", "C0038454", "C0042029", "C0155626", "C0338614", "C0878544"}
    assert set(per_cui_metaanno_cohens_kappa.keys()) == {"C0003864", "C0007222", "C0007787", "C0010068", "C0011849", "C0011860", "C0012634", "C0017168", "C0020473", "C0020538", "C0027051", "C0037284", "C0038454", "C0042029", "C0155626", "C0338614", "C0878544"}


def test_get_iaa_scores_per_concept_and_return_dataframe():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture",
                        "trainer_export_multi_projs.json")
    result = get_iaa_scores_per_concept(path, 1, 2, return_df=True)
    assert set(result["concept"]) == {"C0003864", "C0007222", "C0007787", "C0010068", "C0011849", "C0011860",
                                      "C0012634", "C0017168", "C0020473", "C0020538", "C0027051", "C0037284",
                                      "C0038454", "C0042029", "C0155626", "C0338614", "C0878544"}
    assert len(result["iaa_percentage"]) == 17
    assert len(result["cohens_kappa"]) == 17
    assert len(result["iaa_percentage_meta"]) == 17
    assert len(result["cohens_kappa_meta"]) == 17


def test_get_iaa_scores_per_doc():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export_multi_projs.json")
    per_doc_anno_iia_pct, per_doc_anno_cohens_kappa, per_doc_metaanno_iia_pct, per_doc_metaanno_cohens_kappa = get_iaa_scores_per_doc(path, 1, 2)
    assert set(per_doc_anno_iia_pct.keys()) == {"3204", "3205"}
    assert set(per_doc_anno_cohens_kappa.keys()) == {"3204", "3205"}
    assert set(per_doc_metaanno_iia_pct.keys()) == {"3204", "3205"}
    assert set(per_doc_metaanno_cohens_kappa.keys()) == {"3204", "3205"}


def test_get_iaa_scores_per_doc_and_return_dataframe():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export_multi_projs.json")
    result = get_iaa_scores_per_doc(path, 1, 2, return_df=True)
    assert len(result["doc_id"]) == 2
    assert len(result["iaa_percentage"]) == 2
    assert len(result["cohens_kappa"]) == 2
    assert len(result["iaa_percentage_meta"]) == 2
    assert len(result["cohens_kappa_meta"]) == 2


def test_get_iaa_scores_per_span():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export_multi_projs.json")
    per_doc_anno_iia_pct, per_doc_anno_cohens_kappa, per_doc_metaanno_iia_pct, per_doc_metaanno_cohens_kappa = get_iaa_scores_per_span(path, 1, 2)
    assert len(per_doc_anno_iia_pct.keys()) == 30
    assert len(per_doc_anno_cohens_kappa.keys()) == 30
    assert len(per_doc_metaanno_iia_pct.keys()) == 30
    assert len(per_doc_metaanno_cohens_kappa.keys()) == 30


def test_get_iaa_scores_per_span_and_return_dataframe():
    path = os.path.join(os.path.join(os.path.dirname(__file__), "..", "..", "resources"), "fixture", "trainer_export_multi_projs.json")
    result = get_iaa_scores_per_span(path, 1, 2, return_df=True)
    assert len(result["doc_id"]) == 30
    assert len(result["span_start"]) == 30
    assert len(result["span_end"]) == 30
    assert len(result["iaa_percentage"]) == 30
    assert len(result["cohens_kappa"]) == 30
    assert len(result["iaa_percentage_meta"]) == 30
    assert len(result["cohens_kappa_meta"]) == 30


def test_concat_json_lists_return_list():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
        json.dump([{"question": "question_1", "answer": "answer_1"}, {"question": "question_2", "answer": "answer_2"}], f1)
        file1_path = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
        json.dump([{"question": "question_3", "answer": "answer_3"}], f2)
        file2_path = f2.name

    try:
        result = concat_json_lists([file1_path, file2_path])

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"question": "question_1", "answer": "answer_1"}
        assert result[1] == {"question": "question_2", "answer": "answer_2"}
        assert result[2] == {"question": "question_3", "answer": "answer_3"}
    finally:
        os.unlink(file1_path)
        os.unlink(file2_path)


def test_concat_json_lists_save_to_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
        json.dump([{"question": "question_1", "answer": "answer_1"}], f1)
        file1_path = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
        json.dump([{"question": "question_2", "answer": "answer_2"}], f2)
        file2_path = f2.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_path = output_file.name

    try:
        result = concat_json_lists([file1_path, file2_path], output_path)

        assert isinstance(result, str)
        assert result == output_path

        with open(output_path, 'r') as f:
            saved_data = json.load(f)

        assert isinstance(saved_data, list)
        assert len(saved_data) == 2
        assert saved_data[0] == {"question": "question_1", "answer": "answer_1"}
        assert saved_data[1] == {"question": "question_2", "answer": "answer_2"}
    finally:
        os.unlink(file1_path)
        os.unlink(file2_path)
        os.unlink(output_path)
