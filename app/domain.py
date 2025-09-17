from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST
from pydantic import BaseModel, Field, root_validator


class ModelType(str, Enum):
    MEDCAT_SNOMED = "medcat_snomed"
    MEDCAT_UMLS = "medcat_umls"
    MEDCAT_ICD10 = "medcat_icd10"
    MEDCAT_OPCS4 = "medcat_opcs4"
    MEDCAT_DEID = "medcat_deid"
    ANONCAT = "anoncat"
    TRANSFORMERS_DEID = "transformers_deid"
    HUGGINGFACE_NER = "huggingface_ner"
    HUGGINGFACE_LLM = "huggingface_llm"


class Tags(str, Enum):
    Metadata = "Get the model card"
    Annotations = "Retrieve NER entities by running the model"
    Redaction = "Redact the extracted NER entities"
    Rendering = "Preview embeddable annotation snippet in HTML"
    Training = "Trigger model training on input annotations"
    Evaluating = "Evaluate the deployed model with trainer export"
    Authentication = "Authenticate registered users"
    Generative = "Generate text based on the input prompt"


class TagsStreamable(str, Enum):
    Streaming = "Retrieve NER entities as a stream by running the model"


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
    META_SUPERVISED = "meta_supervised"


class BuildBackend(Enum):
    DOCKER = "docker build"
    DOCKER_BUILDX = "docker buildx build"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Device(str, Enum):
    DEFAULT = "default"
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"


class HfTransformerBackbone(Enum):
    ALBERT = "albert"
    BIG_BIRD = "bert"
    BERT = "bert"
    DISTILBERT = "distilbert"
    FUNNEL = "funnel"
    LAYOUTLM = "layoutlm"
    LONGFORMER = "longformer"
    DEBERTA = "deberta"
    MOBILEBERT = "mobilebert"
    ROBERTA = "roberta"
    SQUEEZEBERT = "transformer"
    XLM_ROBERTA = "xlm_roberta"


class ArchiveFormat(Enum):
    ZIP = "zip"
    TAR_GZ = "gztar"


class TrainerBackend(Enum):
    MEDCAT = "MedCAT"
    TRANSFORMERS = "Transformers"


class TrackerBackend(Enum):
    MLFLOW = "MLflow"


class LlmEngine(Enum):
    CMS = "CMS"
    VLLM = "vLLM"


class Annotation(BaseModel):
    doc_name: Optional[str] = Field(default=None, description="The name of the document to which the annotation belongs")
    start: int = Field(description="The start index of the annotation span")
    end: int = Field(description="The first index after the annotation span")
    label_name: str = Field(description="The pretty name of the annotation concept")
    label_id: str = Field(description="The code of the annotation concept")
    categories: Optional[List[str]] = Field(default=None, description="The categories to which the annotation concept belongs")
    accuracy: Optional[float] = Field(default=None, description="The confidence score of the annotation")
    text: Optional[str] = Field(default=None, description="The string literal of the annotation span")
    meta_anns: Optional[Dict] = Field(default=None, description="The meta annotations")
    athena_ids: Optional[List[Dict]] = Field(default=None, description="The OHDSI Athena concept IDs")

    @root_validator(pre=True)
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


class TextStreamItem(BaseModel):
    text: str = Field(description="The text from which the annotations are extracted")
    name: Optional[str] = Field(default=None, description="The name of the document containing the text")

    class Config:
        extra = "forbid"


class ModelCard(BaseModel):
    api_version: str = Field(description="The version of the model serve APIs")
    model_type: ModelType = Field(description="The type of the served model")
    model_description: Optional[str] = Field(default=None, description="The description about the served model")
    model_card: Optional[dict] = Field(default=None, description="The metadata of the served model")
    labels: Optional[Dict[str, str]] = Field(default=None, description="The mapping of CUIs to names")


class Entity(BaseModel):
    start: int = Field(description="The start index of the preview entity")
    end: int = Field(description="The first index after the preview entity")
    label: str = Field(description="The pretty name of the preview entity")
    kb_id: str = Field(description="The knowledge base ID of the preview entity")
    kb_url: str = Field(description="The knowledge base URL of the preview entity")

    @root_validator(pre=True)
    def _validate(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["start"] >= values["end"]:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="The start index should be lower than the end index")
        return values


class Doc(BaseModel):
    text: str = Field(description="The text from which the entities are extracted")
    ents: List[Entity] = Field(description="The list of extracted entities")
    title: Optional[str] = Field(default=None, description="The headline of the text")
