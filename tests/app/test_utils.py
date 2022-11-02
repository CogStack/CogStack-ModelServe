import os
from app.utils import annotations_to_entities


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
        "kb_url": "#",
    }]
    assert annotations_to_entities(annotations) == expected
