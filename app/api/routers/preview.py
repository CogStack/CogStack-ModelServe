import uuid
import json
import tempfile
import logging
from io import BytesIO
from typing import Union
from typing_extensions import Annotated, Dict, List
from fastapi import APIRouter, Depends, Body, UploadFile, Request, Response, File, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from spacy import displacy
from starlette.status import HTTP_404_NOT_FOUND

import api.globals as cms_globals
from api.dependencies import validate_tracking_id
from domain import Doc, Tags
from model_services.base import AbstractModelService
from processors.metrics_collector import concat_trainer_exports
from utils import annotations_to_entities

router = APIRouter()
logger = logging.getLogger("cms")


@router.post("/preview",
             tags=[Tags.Rendering.name],
             response_class=StreamingResponse,
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Extract the NER entities in HTML for preview")
async def get_rendered_entities_from_text(request: Request,
                                          text: Annotated[str, Body(description="The text to be sent to the model for NER", media_type="text/plain")],
                                          tracking_id: Union[str, None] = Depends(validate_tracking_id),
                                          model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> StreamingResponse:
    annotations = model_service.annotate(text)
    entities = annotations_to_entities(annotations, model_service.model_name)
    logger.debug("Entities extracted for previewing %s", entities)
    ent_input = Doc(text=text, ents=entities)
    data = displacy.render(ent_input.dict(), style="ent", manual=True)
    tracking_id = tracking_id or str(uuid.uuid4())
    response = StreamingResponse(BytesIO(data.encode()), media_type="application/octet-stream")
    response.headers["Content-Disposition"] = f'attachment ; filename="preview_{tracking_id}.html"'
    return response


@router.post("/preview_trainer_export",
             tags=[Tags.Rendering.name],
             response_class=StreamingResponse,
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Get existing entities in HTML from a trainer export for preview")
def get_rendered_entities_from_trainer_export(request: Request,
                                              trainer_export: Annotated[List[UploadFile], File(description="One or more trainer export files to be uploaded")] = [],
                                              trainer_export_str: Annotated[str, Form(description="The trainer export raw JSON string")] = "{\"projects\": []}",
                                              project_id: Annotated[Union[int, None], Query(description="The target project ID, and if not provided, all projects will be included")] = None,
                                              document_id: Annotated[Union[int, None], Query(description="The target document ID, and if not provided, all documents of the target project(s) will be included")] = None,
                                              tracking_id: Union[str, None] = Depends(validate_tracking_id)) -> Response:
    data: Dict = {"projects": []}
    if trainer_export is not None:
        files = []
        try:
            for te in trainer_export:
                temp_te = tempfile.NamedTemporaryFile()
                for line in te.file:
                    temp_te.write(line)
                temp_te.flush()
                files.append(temp_te)
            concatenated = concat_trainer_exports([file.name for file in files], allow_recurring_project_ids=True, allow_recurring_doc_ids=True)
            logger.debug("Training exports concatenated")
        finally:
            for file in files:
                file.close()
        data["projects"] += concatenated["projects"]
    if trainer_export_str is not None:
        data["projects"] += json.loads(trainer_export_str)["projects"]
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
                    "label": f"{annotation['cui']} ({'correct' if annotation.get('correct', True) else 'incorrect'}{'; terminated' if annotation.get('deleted', False) and annotation.get('killed', False) else ''})",
                    "kb_id": annotation["cui"],
                    "kb_url": "#",
                })
            # Displacy cannot handle annotations out of appearance order so be this
            entities = sorted(entities, key=lambda e: e["start"])
            logger.debug("Entities extracted for previewing %s", entities)
            doc = Doc(text=document["text"], ents=entities, title=f"P{project['id']}/D{document['id']}")
            htmls.append(displacy.render(doc.dict(), style="ent", manual=True))
    if htmls:
        tracking_id = tracking_id or str(uuid.uuid4())
        response = StreamingResponse(BytesIO("<br/>".join(htmls).encode()), media_type="application/octet-stream")
        response.headers["Content-Disposition"] = f'attachment ; filename="preview_{tracking_id}.html"'
    else:
        logger.debug("Cannot find any matching documents to preview")
        return JSONResponse(content={"message": "Cannot find any matching documents to preview"}, status_code=HTTP_404_NOT_FOUND)
    return response
