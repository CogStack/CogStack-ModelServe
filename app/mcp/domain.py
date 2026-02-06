from dataclasses import dataclass
from enum import Enum
from cms_client import (
    MetadataApi,
    AnnotationsApi,
    RedactionApi,
    TrainingApi,
)


@dataclass
class AppContext:
    metadata_api: MetadataApi
    annotation_api: AnnotationsApi
    redaction_api: RedactionApi
    training_api: TrainingApi


class TransportType(str, Enum):
    STREAMABLE_HTTP = "http"
    STDIO = "stdio"
    SSE = "sse"
