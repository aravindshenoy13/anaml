from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from schemas.schemas import ModelCreate, ModelResponse, ModelUpdate, ModelStats
from core.database import get_session
from models.models import MLModel
from sqlalchemy import select
from typing import List

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
    model = await session.execute(query).scalar_one_or_none()
    
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    return model

@model_router.put(path="/{id}")
async def update_model_id():
    pass

@model_router.delete(path="/{id}")
async def delete_model_id():
    pass
