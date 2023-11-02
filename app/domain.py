from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, root_validator


class Annotation(BaseModel):
    start: int
    end: int
    label_name: str
    label_id: str
    categories: Optional[List[str]] = None
    accuracy: Optional[float] = None
    text: Optional[str] = None
    meta_anns: Optional[dict] = None

    @root_validator()
    def _validate(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["start"] >= values["end"]:
            raise ValueError("The start index should be lower than the end index")
        return values


class TextWithAnnotations(BaseModel):
    text: str
    annotations: List[Annotation]


class TextWithPublicKey(BaseModel):
    text: str
    public_key_pem: str


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

    @root_validator()
    def _validate(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["start"] >= values["end"]:
            raise ValueError("The start index should be lower than the end index")
        return values


class Doc(BaseModel):
    text: str
    ents: List[Entity]
    title: Optional[str]


class Tags(str, Enum):
    Metadata = "Get the model card"
    Annotations = "Retrieve NER entities by running the model"
    Redaction = "Redact the extracted NER entities"
    Rendering = "Preview embeddable annotation snippet in HTML"
    Training = "Trigger model training on input annotations"
    Evaluating = "Evaluate the deployed model with trainer export"
    Authentication = "Authenticate registered users"


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


class Scope(str, Enum):
    PER_CONCEPT = "per_concept"
    PER_DOCUMENT = "per_document"
    PER_SPAN = "per_span"
