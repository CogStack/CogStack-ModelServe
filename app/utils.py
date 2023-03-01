import json
import socket
import random
import struct

from urllib.parse import ParseResult
from functools import lru_cache
from typing import List, Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
from domain import Annotation, Entity, CodeType
from config import Settings


@lru_cache()
def get_settings():
    return Settings()


@lru_cache
def get_rate_limiter():
    return Limiter(key_func=get_remote_address)


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


def send_gelf_message(message: str, gelf_input_uri: ParseResult) -> None:
    message = {
        "version": "1.1",
        "host": socket.gethostname(),
        "short_message": message,
        "level": 1,
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((gelf_input_uri.hostname, gelf_input_uri.port))

    message_id = struct.pack("<Q", random.getrandbits(64))
    sock.sendall(b'\x1e\x0f' + message_id + b'\x00\x00' + bytes(json.dumps(message), "utf-8"))
    sock.close()
