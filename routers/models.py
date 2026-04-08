from fastapi import APIRouter, Depends, HTTPException, UploadFile, Form
from fastapi.responses import Response
from schemas.schemas import ModelCreate, ModelResponse, ModelUpdate
from core.config import MODEL_DIR
from core.database import get_session
from models.models import MLModel, get_uuid
from sqlalchemy import select
from typing import List
from pathlib import Path
import aiofiles

model_router = APIRouter(prefix="/models")

@model_router.post(path="/register", status_code=201)
async def model_register(file: UploadFile,
                        name: str = Form(...),
                        version: str = Form(...),
                        backend_type: str = Form(default="joblib"),
                        description: str | None = Form(default=None),
                        accuracy: float | None = Form(default=None),
                        session = Depends(get_session)) -> ModelResponse:
    model_id = get_uuid()

    file_extension = Path(file.filename).suffix
    file_path = MODEL_DIR + f"/model_{model_id}/model_{model_id}"+ file_extension
    Path(Path(file_path).parent).mkdir(exist_ok=True)
    file_bytes = await file.read()

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    model = MLModel(id = model_id,
            name = name,
            version = version,
            backend = backend_type,
            description = description,
            accuracy = accuracy,
            weights_path = file_path,
            storage_type = "disk",
            inference_url = None
            )
    session.add(model)
    await session.commit()
    return model

@model_router.get(path="/")
async def list_models(session = Depends(get_session)) -> List[ModelResponse]:
    query = select(MLModel)
    result = await session.execute(query)
    model_list = result.scalars().all()

    return model_list

@model_router.get(path="/{id}")
async def get_model(id: str, session = Depends(get_session)) -> ModelResponse:
    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model  = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    return model

@model_router.put(path="/{id}")
async def update_model(id: str, model_update: ModelUpdate, session = Depends(get_session)) -> ModelResponse:
    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    
    update_data = model_update.model_dump(exclude_unset=True)

    for k,v in update_data.items():
        setattr(model, k, v)
    
    await session.commit()
    return model


@model_router.delete(path="/{id}")
async def delete_model(id: str, session = Depends(get_session)) -> Response:
    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    
    #Unlink model weights path in server
    Path(model.weights_path).unlink(missing_ok=True)
    session.delete(model)
    await session.commit()

    return Response(status_code = 204)
