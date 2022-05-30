from pydantic import BaseModel
from typing import List, Optional


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
