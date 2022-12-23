import uuid
import json
from typing import Union

from fastapi import APIRouter, Depends, Body, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from spacy import displacy
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND

import globals
from domain import Doc, Tags
from model_services.base import AbstractModelService
from utils import annotations_to_entities

router = APIRouter()


@router.post("/preview",
             tags=[Tags.Rendering.name],
             response_class=HTMLResponse)
async def preview_processing_result(text: str = Body(..., media_type="text/plain"),
                                    model_service: AbstractModelService = Depends(globals.model_service_dep)) -> HTMLResponse:
    annotations = model_service.annotate(text)
    entities = annotations_to_entities(annotations, model_service.model_name)
    ent_input = Doc(text=text, ents=entities)
    data = displacy.render(ent_input.dict(), style="ent", manual=True)
    response = HTMLResponse(content=data, status_code=HTTP_200_OK)
    response.headers["Content-Disposition"] = f'attachment ; filename="preview_{str(uuid.uuid4())}.html"'
    return response


@router.post("/preview_trainer_export",
             tags=[Tags.Rendering.name],
             response_class=HTMLResponse)
async def preview_trainer_export(trainer_export: UploadFile,
                                 project_id: Union[int, None] = None,
                                 document_id: Union[int, None] = None) -> HTMLResponse:
    data = json.load(trainer_export.file)
    htmls = []
    for project in data["projects"]:
        if project_id is not None and project_id != project["id"]:
            continue
        for document in project["documents"]:
            if document_id is not None and document_id != document["id"]:
                continue
            entities = []
            for annotation in document["annotations"]:
                entities.append({
                    "start": annotation["start"],
                    "end": annotation["end"],
                    "label": f"{annotation['cui']} ({'correct' if annotation['correct'] else 'incorrect'}{'; terminated' if annotation['killed'] else ''})",
                    "kb_id": annotation["cui"],
                    "kb_url": "#",
                })
            # Displacy cannot handle annotations out of appearance order so be this
            entities = sorted(entities, key=lambda e: e["start"])
            doc = Doc(text=document["text"], ents=entities, title=f"P{project['id']}/D{document['id']}")
            htmls.append(displacy.render(doc.dict(), style="ent", manual=True))
    if htmls:
        response = HTMLResponse(content="<br/>".join(htmls), status_code=HTTP_200_OK)
        response.headers["Content-Disposition"] = f'attachment ; filename="preview_{str(uuid.uuid4())}.html"'
    else:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str("Cannot find any matching documents to preview"))
    return response
