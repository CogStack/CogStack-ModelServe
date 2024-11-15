import json
import tempfile
import uuid
import ijson
import logging
import datasets
from typing import List, Union
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, UploadFile, Query, Request, File
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE
import api.globals as cms_globals
from domain import Tags, ModelType
from model_services.base import AbstractModelService
from utils import get_settings
from exception import ConfigurationException

router = APIRouter()
logger = logging.getLogger("cms")


@router.post("/train_unsupervised",
             status_code=HTTP_202_ACCEPTED,
             response_class=JSONResponse,
             tags=[Tags.Training.name],
             dependencies=[Depends(cms_globals.props.current_active_user)])
async def train_unsupervised(request: Request,
                             training_data: Annotated[List[UploadFile], File(description="One or more files to be uploaded and each contains a list of plain texts, in the format of [\"text_1\", \"text_2\", ..., \"text_n\"]")],
                             epochs: Annotated[int, Query(description="The number of training epochs", ge=0)] = 1,
                             lr_override: Annotated[Union[float, None], Query(description="The override of the initial learning rate", gt=0.0)] = None,
                             test_size: Annotated[Union[float, None], Query(description="The override of the test size in percentage", ge=0.0)] = 0.2,
                             log_frequency: Annotated[int, Query(description="The number of processed documents after which training metrics will be logged", ge=1)] = 1000,
                             description: Annotated[Union[str, None], Query(description="The description of the training or change logs")] = None,
                             model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> JSONResponse:
    """
    Upload one or more plain text files and trigger the unsupervised training
    """
    data_file = tempfile.NamedTemporaryFile(mode="r+")
    files = []
    file_names = []
    data_file.write("[")
    for td_idx, td in enumerate(training_data):
        temp_td = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
        items = ijson.items(td.file, "item")
        temp_td.write("[")
        for text_idx, text in enumerate(items):
            if text_idx > 0 or td_idx > 0:
                data_file.write(",")
            json.dump(text, data_file)
            if text_idx > 0:
                temp_td.write(",")
            json.dump(text, temp_td)
        temp_td.write("]")
        temp_td.flush()
        temp_td.seek(0)
        file_names.append("" if td.filename is None else td.filename)
        files.append(temp_td)
    data_file.write("]")
    logger.debug("Training data concatenated")
    data_file.flush()
    data_file.seek(0)
    training_id = str(uuid.uuid4())
    try:
        training_accepted = model_service.train_unsupervised(data_file,
                                                             epochs,
                                                             log_frequency,
                                                             training_id,
                                                             ",".join(file_names),
                                                             raw_data_files=files,
                                                             synchronised=False,
                                                             lr_override=lr_override,
                                                             test_size=test_size,
                                                             description=description)
    finally:
        for file in files:
            file.close()

    return _get_training_response(training_accepted, training_id)


@router.post("/train_unsupervised_with_hf_hub_dataset",
             status_code=HTTP_202_ACCEPTED,
             response_class=JSONResponse,
             tags=[Tags.Training.name],
             dependencies=[Depends(cms_globals.props.current_active_user)])
async def train_unsupervised_with_hf_dataset(request: Request,
                                             hf_dataset_repo_id: Annotated[str, Query(description="The repository ID of the dataset to download from Hugging Face Hub")],
                                             hf_dataset_config: Annotated[Union[str, None], Query(description="The name of the dataset configuration")] = None,
                                             text_column_name: Annotated[str, Query(description="The name of the text column in the dataset")] = "text",
                                             epochs: Annotated[int, Query(description="The number of training epochs", ge=0)] = 1,
                                             lr_override: Annotated[Union[float, None], Query(description="The override of the initial learning rate", gt=0.0)] = None,
                                             test_size: Annotated[Union[float, None], Query(description="The override of the test size in percentage", ge=0.0)] = 0.2,
                                             log_frequency: Annotated[int, Query(description="The number of processed documents after which training metrics will be logged", ge=1)] = 1000,
                                             description: Annotated[Union[str, None], Query(description="The description of the training or change logs")] = None,
                                             model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> JSONResponse:
    """
    Trigger the unsupervised training with a dataset from Hugging Face Hub
    """
    if model_service.info().model_type not in [ModelType.HF_TRANSFORMER]:
        raise ConfigurationException(f"Currently this endpoint is only available for models of type: {ModelType.HF_TRANSFORMER.value}")
    hf_dataset = datasets.load_dataset(hf_dataset_repo_id,
                                       cache_dir=get_settings().TRAINING_CACHE_DIR,
                                       trust_remote_code=True,
                                       name=hf_dataset_config)
    for split in hf_dataset.keys():
        if text_column_name not in hf_dataset[split].column_names:
            raise ValueError(f"The dataset does not contain a '{text_column_name}' column in the split(s)")
        if text_column_name != "text":
            hf_dataset[split] = hf_dataset[split].map(lambda x: {"text": x[text_column_name]}, batched=True)
        hf_dataset[split] = hf_dataset[split].remove_columns([col for col in hf_dataset[split].column_names if col != "text"])
    logger.debug("Training dataset downloaded and transformed")

    data_dir = tempfile.TemporaryDirectory()
    hf_dataset.save_to_disk(data_dir.name)
    training_id = str(uuid.uuid4())
    training_accepted = model_service.train_unsupervised(data_dir,
                                                         epochs,
                                                         log_frequency,
                                                         training_id,
                                                         hf_dataset_repo_id,
                                                         raw_data_files=None,
                                                         synchronised=False,
                                                         lr_override=lr_override,
                                                         test_size=test_size,
                                                         description=description)
    hf_dataset.cleanup_cache_files()
    return _get_training_response(training_accepted, training_id)


def _get_training_response(training_accepted: bool, training_id: str) -> JSONResponse:
    if training_accepted:
        logger.debug("Training accepted with ID: %s", training_id)
        return JSONResponse(content={"message": "Your training started successfully.", "training_id": training_id}, status_code=HTTP_202_ACCEPTED)
    else:
        logger.debug("Training refused due to another active training or evaluation on this model")
        return JSONResponse(content={"message": "Another training or evaluation on this model is still active. Please retry later."}, status_code=HTTP_503_SERVICE_UNAVAILABLE)
