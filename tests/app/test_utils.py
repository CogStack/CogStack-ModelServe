from app.utils import (
    get_code_base_uri,
    annotations_to_entities,
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
