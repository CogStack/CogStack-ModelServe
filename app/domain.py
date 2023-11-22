from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST
from pydantic import BaseModel, Field, root_validator


class Annotation(BaseModel):
    start: int = Field(description="The start index of the annotation span")
    end: int = Field(description="The first index after the annotation span")
    label_name: str = Field(description="The pretty name of the annotation concept")
    label_id: str = Field(description="The ID of the annotation concept")
    categories: Optional[List[str]] = Field(default=None, description="The categories to which the annotation concept belongs")
    accuracy: Optional[float] = Field(default=None, description="The confidence score of the annotation")
    text: Optional[str] = Field(default=None, description="The string literal of the annotation span")
    meta_anns: Optional[dict] = Field(default=None, description="The meta annotations")

    @root_validator()
    def _validate(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["start"] >= values["end"]:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="The start index should be lower than the end index")
        return values


class TextWithAnnotations(BaseModel):
    text: str = Field(description="The text from which the annotations are extracted")
    annotations: List[Annotation] = Field(description="The list of extracted annotations")


class TextWithPublicKey(BaseModel):
    text: str = Field(description="The plain text to be sent to the model for NER and redaction")
    public_key_pem: str = Field(description="the public PEM key used for encrypting detected spans")


class ModelCard(BaseModel):
    api_version: str = Field(description="The version of the model serve APIs")
    model_description: str = Field(description="The description about the served model")
    model_type: str = Field(description="The type of the served model")
    model_card: Optional[dict] = Field(default=None, description="The metadata of the served model")


class Entity(BaseModel):
    start: int = Field(description="The start index of the preview entity")
    end: int = Field(description="The first index after the preview entity")
    label: str = Field(description="The pretty name of the preview entity")
    kb_id: str = Field(description="The knowledge base ID of the preview entity")
    kb_url: str = Field(description="The knowledge base URL of the preview entity")

    @root_validator()
    def _validate(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["start"] >= values["end"]:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="The start index should be lower than the end index")
        return values


class Doc(BaseModel):
    text: str = Field(description="The text from which the entities are extracted")
    ents: List[Entity] = Field(description="The list of extracted entities")
    title: Optional[str] = Field(description="The headline of the text")


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


class TrainingType(str, Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
