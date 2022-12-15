from typing import List, Optional
from domain import Annotation, Entity, CodeType


def get_code_base_uri(model_name: str) -> Optional[str]:
    code_base_uris = {
        CodeType.SNOMED.value: "http://snomed.info/id",
        CodeType.ICD10.value: "https://icdcodelookup.com/icd-10/codes",
        CodeType.UMLS.value: "https://uts.nlm.nih.gov/uts/umls/concept",
    }
    for code_name, base_uri in code_base_uris.items():
        if code_name.lower() in model_name.lower():
            return base_uri
    return None


def annotations_to_entities(annotations: List[Annotation], model_name: str) -> List[Entity]:
    entities = []
    code_base_uri = get_code_base_uri(model_name)
    for _, annotation in enumerate(annotations):
        entities.append({
            "start": annotation["start"],
            "end": annotation["end"],
            "label": f"{annotation['label_name']} ({annotation['label_id']})",  # remove label_id after upgrading spaCy
            "kb_id": annotation["label_id"],
            "kb_url": f"{code_base_uri}/{annotation['label_id']}" if code_base_uri is not None else "#"
        })
    return entities
