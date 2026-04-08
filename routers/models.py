from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from schemas.schemas import ModelCreate, ModelResponse, ModelUpdate, ModelStats
from core.database import get_session
from models.models import MLModel
from sqlalchemy import select
from typing import List
from pathlib import Path

model_router = APIRouter(prefix="/models")

@model_router.post(path="/register")
async def model_register():
    pass

@model_router.get(path="/")
async def list_models(session = Depends(get_session)) -> List[ModelResponse]:
    query = select(MLModel)
    result = await session.execute(query)
    model_list = result.scalars().all()

    return model_list

@model_router.get(path="/{id}")
async def get_model_id(id: str, session = Depends(get_session)) -> ModelResponse:
    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model  = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    return model

@model_router.put(path="/{id}")
async def update_model_id(id: str, model_update: ModelUpdate, session = Depends(get_session)) -> ModelResponse:
    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    
    update_data = model_update.model_dump(exclude_unset=True)

    for k,v in update_data.items():
        setattr(model, k, v)
    
    session.commit()
    return model


@model_router.delete(path="/{id}")
async def delete_model_id(id: str, session = Depends(get_session)) -> Response:
    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    
    #Unlink model weights path in server
    Path(model.weights_path).unlink(missing_ok=True)
    session.delete(model)
    session.commit()

    return Response(status_code = 204)
