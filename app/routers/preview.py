import uuid
import json
from io import BytesIO
from typing import Union

from fastapi import APIRouter, Depends, Body, UploadFile, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from spacy import displacy
from starlette.status import HTTP_404_NOT_FOUND

import globals
from domain import Doc, Tags
from model_services.base import AbstractModelService
from utils import annotations_to_entities
from auth.users import props

router = APIRouter()


@router.post("/preview",
             tags=[Tags.Rendering.name],
             response_class=StreamingResponse,
             dependencies=[Depends(props.current_active_user)])
async def get_rendered_entities_from_text(request: Request,
                                          text: str = Body(..., media_type="text/plain"),
                                          model_service: AbstractModelService = Depends(globals.model_service_dep)) -> StreamingResponse:
    """
    Extract the NER entities in HTML for preview

    - **text**: the text to be sent to the model for NER
    """
    annotations = model_service.annotate(text)
    entities = annotations_to_entities(annotations, model_service.model_name)
    ent_input = Doc(text=text, ents=entities)
    data = displacy.render(ent_input.dict(), style="ent", manual=True)
    response = StreamingResponse(BytesIO(data.encode()), media_type="application/octet-stream")
    response.headers["Content-Disposition"] = f'attachment ; filename="preview_{str(uuid.uuid4())}.html"'
    return response


@router.post("/preview_trainer_export",
             tags=[Tags.Rendering.name],
             response_class=StreamingResponse,
             dependencies=[Depends(props.current_active_user)])
def get_rendered_entities_from_trainer_export(request: Request,
                                              trainer_export: UploadFile,
                                              project_id: Union[int, None] = None,
                                              document_id: Union[int, None] = None) -> Response:
    """
    Get existing entities in HTML from a trainer export for preview

    - **trainer_export**: the trainer export file to be uploaded
    - **project_id**: the target project ID, and if not provided, all projects will be included
    - **document_id**: the target document ID, and if not provided, all documents of the target project(s) will be included
    """
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
        response = StreamingResponse(BytesIO("<br/>".join(htmls).encode()), media_type="application/octet-stream")
        response.headers["Content-Disposition"] = f'attachment ; filename="preview_{str(uuid.uuid4())}.html"'
    else:
        return JSONResponse(content={"message": "Cannot find any matching documents to preview"}, status_code=HTTP_404_NOT_FOUND)
    return response
