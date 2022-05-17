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
    model_description: str
    model_type: str
