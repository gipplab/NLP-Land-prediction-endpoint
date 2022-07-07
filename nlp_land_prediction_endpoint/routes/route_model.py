"""This module implements the endpoint logic for models."""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel

from nlp_land_prediction_endpoint import __version__
from nlp_land_prediction_endpoint.models.generic_model import (
    GenericInputModel,
    GenericOutputModel,
)
from nlp_land_prediction_endpoint.models.lda_model import LDAModel
from nlp_land_prediction_endpoint.utils.storage_controller import storage

router: APIRouter = APIRouter()


class StorageControllerListReponse(BaseModel):
    """Response model for both
        - GET /
        - GET /implemented
    which returns a list of models
    """

    models: List[str]
    # error: str


class ModelSpecificFunctionCallResponse(BaseModel):
    """Response model for model specific function calls"""

    functionCalls: List[str]


class ModelCreationResponse(BaseModel):
    """Response Model for the successfull creation of a model"""

    modelID: str


class ModelDeletionResponse(BaseModel):
    """Response Model for the successfull deletion of a model"""

    modelID: str


class ModelDeletionRequest(BaseModel):
    """Response model for deleting a model
    This contains the modelID.
    """

    modelID: str


class ModelFunctionRequest(BaseModel):
    """Response model for running a function of a model
    This contains the modelID.
    """

    modelID: str


class ModelUpdateRequest(BaseModel):
    """Response model for updating a model
    This contains the modelID and a dict containing the parameters to be updated
    """

    modelID: str
    modelSpecification: dict


class ModelCreationRequest(BaseModel):
    """Response model for creating a Model
    This contains the modelType (e.g., lda) and the model specification
    which should be parsable to the modelTypes pydentic schema.
    """

    modelType: str
    # XXX-TN For the docker ochestration it will be helpfull to also have an input for
    #        the location of Model initialization (local, local[dockerfile], remote)
    modelSpecification: dict


@router.get(
    "/implemented",
    response_description="Lists all currently available(implemented) models",
    response_model=StorageControllerListReponse,
    status_code=status.HTTP_200_OK,
)
def list_all_implemented_models() -> StorageControllerListReponse:
    """Endpoint for getting a list of all implemented models"""
    return StorageControllerListReponse(models=["lda"])


@router.get(
    "/{current_modelID}",
    response_description="Lists all function calls of the current model",
    response_model=ModelSpecificFunctionCallResponse,
    status_code=status.HTTP_200_OK,
)
def list_all_function_calls(current_modelID: str) -> BaseModel:
    """Endpoint for getting a list of all implemented function calls"""
    # validate id
    currentModel = storage.getModel(current_modelID)
    if currentModel is None:
        # error not found
        raise HTTPException(status_code=404, detail="Model not found")

    # get fun calls
    cMFCalls = currentModel.getFunctionCalls()  # current model functioncalls list
    return ModelSpecificFunctionCallResponse(functionCalls=cMFCalls)


@router.delete(
    "/{current_modelID}",
    response_description="Delete the current model",
    response_model=ModelDeletionResponse,
    status_code=status.HTTP_200_OK,
)
def deleteModel(current_modelID: str) -> ModelDeletionResponse:
    """Endpoint for deleting a model"""
    # validate id
    currentModel = storage.getModel(current_modelID)
    if currentModel is None:
        # error not found
        raise HTTPException(status_code=404, detail="Model not implemented")

    # delete model by id
    storage.delModel(currentModel.getId())
    return ModelDeletionResponse(modelID=current_modelID)


@router.patch(
    "/{current_modelID}",
    response_description="Patch/Update current model",
    response_model=ModelCreationResponse,
    status_code=status.HTTP_200_OK,
)
def patchModel(
    current_modelID: str, modelUpdateRequest: ModelUpdateRequest, response: Response
) -> ModelCreationResponse:
    """Endpoint for updating a model"""
    # Find model
    currentModel = storage.getModel(current_modelID)
    if currentModel is None:
        # error not found
        raise HTTPException(status_code=404, detail="Model not implemented")

    # Run update function of model and return the id, if parameters were updated (return >=0)
    ret = currentModel.update(modelUpdateRequest.dict())
    if ret > -1:
        return ModelCreationResponse(modelID=current_modelID)
    raise HTTPException(status_code=404, detail="Model or parameter not implemented")


@router.get(
    "/",
    response_description="Lists all currently created models",
    response_model=StorageControllerListReponse,
    status_code=status.HTTP_200_OK,
)
def list_all_created_models() -> StorageControllerListReponse:
    """Endpoint for getting a list of all created models"""
    all_models = list([str(i) for i in storage.getAllModels()])
    return StorageControllerListReponse(models=all_models)


@router.post(
    "/",
    response_description="Creates a model",
    response_model=ModelCreationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_model(
    modelCreationRequest: ModelCreationRequest, response: Response
) -> ModelCreationResponse:
    """Endpoint for creating a model

    Arguments:
        modelCreationRequest (ModelCreationRequest): A ModelCreationRequest used for the creation
                                                 of the actual model

    Returns:
        dict: Either an error or the created model id
    """
    model = None
    if modelCreationRequest.modelType == "lda":
        model = LDAModel(**modelCreationRequest.modelSpecification)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not implemented")
    storage.addModel(model)
    response.headers["location"] = f"/api/v{__version__.split('.')[0]}/models/{model.id}"

    return ModelCreationResponse(modelID=model.id)


@router.post(
    "/{current_modelID}",
    response_description="Runs a function",
    response_model=GenericOutputModel,
    status_code=status.HTTP_200_OK,
)
def getInformation(current_modelID: str, genericInput: GenericInputModel) -> BaseModel:
    """Gets info out of post data"""
    return run_function(current_modelID, genericInput.functionCall, genericInput.inputData)


def run_function(current_modelID: str, req_function: str, data_input: Dict[Any, Any]) -> BaseModel:
    """Runs a given function of a given model"""
    # Validate id
    currentModel = storage.getModel(current_modelID)
    if currentModel is None:
        # Error not found
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if the function is actually availabe in the requested model
    # Return an HTTPException if not; Execute the function and return Dict otherwise
    try:
        myFun = getattr(currentModel, req_function)
    except AttributeError:
        raise HTTPException(status_code=404, detail="Function not implemented")

    # Run function and parse output dict into actual response model
    output = myFun(**data_input)
    # XXX-TN we have to ensure that we return a dict on a function call
    #        i dont know if the following is the best way to achive this
    outputDict = {}
    outputDict[req_function] = output
    outModelResp = GenericOutputModel(outputData=outputDict)

    return outModelResp
