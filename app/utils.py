import json
import socket
import random
import struct
import hashlib
import re
import inspect
import os

from urllib.parse import ParseResult
from functools import lru_cache
from typing import List, Optional, Dict, Callable
from fastapi import Request
from starlette.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from domain import Annotation, Entity, CodeType
from config import Settings
from fastapi_users.jwt import decode_jwt


@lru_cache()
def get_settings() -> Settings:
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    return Settings()


@lru_cache()
def get_rate_limiter(auth_user_enabled: Optional[bool] = None) -> Limiter:
    def get_user_auth(request: Request) -> str:
        request_headers = request.scope.get("headers", [])
        limiter_prefix = request.scope.get("root_path", "") + request.scope.get("path") + ":"

        for headers in request_headers:
            if headers[0].decode() == "authorization":
                token = headers[1].decode().split("Bearer ")[1]
                payload = decode_jwt(token, get_settings().AUTH_JWT_SECRET, ["fastapi-users:auth"])
                sub = payload.get("sub")
                assert sub is not None, "Cannot find 'sub' in the decoded payload"
                hash_object = hashlib.sha256(sub.encode())
                current_key = hash_object.hexdigest()
                break

        limiter_key = re.sub(r":+", ":", re.sub(r"/+", ":", limiter_prefix + current_key))
        return limiter_key

    auth_user_enabled = get_settings().AUTH_USER_ENABLED == "true" if auth_user_enabled is None else auth_user_enabled
    return Limiter(key_func=get_user_auth, strategy="moving-window") if auth_user_enabled else Limiter(key_func=get_remote_address, strategy="moving-window")


def rate_limit_exceeded_handler(*args, **kwargs) -> JSONResponse:
    return JSONResponse({"error": "Too many requests. Please wait and try your request again."}, status_code=429)


def adjust_rate_limit_str(rate_limit: str) -> str:
    print(rate_limit)
    if "per" in rate_limit:
        return f"{int(rate_limit.split('per')[0]) * 2} per {rate_limit.split('per')[1]}"
    else:
        return f"{int(rate_limit.split('/')[0]) * 2}/{rate_limit.split('/')[1]}"


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
            "label": f"{annotation['label_name']}",
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


def get_func_params_as_dict(func: Callable) -> Dict:
    signature = inspect.signature(func)
    params = {name: param.default for name, param in signature.parameters.items() if param.default is not inspect.Parameter.empty}
    return params


TYPE_ID_TO_NAME_PATCH = {
    "32816260": "physical object",
    "2680757": "observable entity",
    "37552161": "body structure",
    "91776366": "product",
    "81102976": "organism",
    "28321150": "procedure",
    "67667581": "finding",
    "7882689": "qualifier value",
    "91187746": "substance",
    "29422548": "core metadata concept",
    "40357424": "foundation metadata concept",
    "33782986": "morphologic abnormality",
    "9090192": "disorder",
    "90170645": "record artifact",
    "66527446": "body structure",
    "3061879": "situation",
    "16939031": "occupation",
    "31601201": "person",
    "37785117": "medicinal product",
    "17030977": "assessment scale",
    "47503797": "regime/therapy",
    "33797723": "event",
    "82417248": "navigational concept",
    "75168589": "environment",
    "9593000": "medicinal product form",
    "99220404": "cell",
    "13371933": "social concept",
    "46922199": "religion/philosophy",
    "20410104": "ethnic group",
    "27603525": "clinical drug",
    "43039974": "qualifier value",
    "43857361": "physical force",
    "40584095": "metadata",
    "337250": "specimen",
    "46506674": "disposition",
    "87776218": "role",
    "30703196": "tumor staging",
    "31685163": "staging scale",
    "21114934": "dose form",
    "70426313": "namespace concept",
    "51120815": "intended site",
    "45958968": "administration method",
    "51885115": "OWL metadata concept",
    "8067332": "Lyophilized Dosage Form Category",
    "95475658": "product name",
    "43744943": "supplier",
    "66203715": "transformation",
    "64755083": "release characteristic",
    "49144999": "state of matter",
    "39041339": "unit of presentation",
    "18854038": "geographic location",
    "3242456": "life style",
    "28695783": "link assertion",
    "14654508": "racial group",
    "92873870": "special concept",
    "78096516": "environment / location",
    "72706784": "context-dependent category",
    "25624495": '© 2002-2020 International Health Terminology Standards Development Organisation (IHTSDO). All rights reserved. SNOMED CT®, was originally created by The College of American Pathologists. "SNOMED" and "SNOMED CT" are registered trademarks of the IHTSDO.',
    "55540447": "linkage concept"
}
