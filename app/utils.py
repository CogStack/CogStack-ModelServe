from typing import List
from domain import Annotation, Entity


def annotations_to_entities(annotations: List[Annotation]) -> List[Entity]:
    entities = []
    for _, annotation in enumerate(annotations):
        entities.append({
            "start": annotation["start"],
            "end": annotation["end"],
            "label": f"{annotation['label_name']} ({annotation['label_id']})",  # remove label_id after upgrading spaCy
            "kb_id": annotation["label_id"],
            "kb_url": "#"
        })
    return entities
