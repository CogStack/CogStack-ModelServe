from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Annotation(BaseModel):
    start: int
    end: int
    label_name: str
    label_id: str
    text: Optional[str] = None
    meta_anns: Optional[dict] = None


class TextwithAnnotations(BaseModel):
    text: str
    annotations: List[Annotation]


class ModelCard(BaseModel):
    api_version: str
    model_description: str
    model_type: str
    model_card: Optional[dict] = None


class Entity(BaseModel):
    start: int
    end: int
    label: str
    kb_id: str
    kb_url: str


class Doc(BaseModel):
    text: str
    ents: List[Entity]
    title: Optional[str]


class Tags(str, Enum):
    Metadata = "Get the model card."
    Annotations = "Retrieve recognised entities by running the model."
    Rendering = "Get embeddable annotation snippet in HTML."
    Training = "Trigger model training on input annotations."
    Evaluating = "Evaluate the deployed model using trainer export"


class ModelType(str, Enum):
    MEDCAT_SNOMED = "medcat_snomed"
    MEDCAT_UMLS = "medcat_umls"
    MEDCAT_ICD10 = "medcat_icd10"
    MEDCAT_DEID = "medcat_deid"
    TRANSFORMERS_DEID = "transformers_deid"


class CodeType(str, Enum):
    SNOMED = "SNOMED"
    UMLS = "UMLS"
    ICD10 = "ICD-10"
    OPCS4 = "OPCS-4"
