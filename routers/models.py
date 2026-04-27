import shutil
from pathlib import Path
from typing import List, Literal

import aiofiles
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select

from core.config import MODEL_DIR
from core.database import get_session
from core.redis import model_cache, redis_client
from inference.registry import get_model_class
from models.models import MLModel, get_uuid
from schemas.schemas import MetadataResponse, ModelResponse, ModelUpdate

model_router = APIRouter(prefix="/models")

@model_router.post(path="/register", status_code=201)
async def model_register(file: UploadFile,
                        name: str = Form(...),
                        version: str = Form(...),
                        backend_type: Literal["onnx", "joblib"] = Form(default="joblib"),
                        description: str | None = Form(default=None),
                        accuracy: float | None = Form(default=None),
                        session = Depends(get_session)) -> ModelResponse:
    model_id = get_uuid()
    try:
        file_extension = Path(file.filename).suffix
    except TypeError:
        raise HTTPException(status_code=400)
    file_path = Path(MODEL_DIR) / f"model_{model_id}" / f"model_{model_id}{file_extension}"
    Path(file_path).parent.mkdir(exist_ok=True)
    file_bytes = await file.read()

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    try:
        model_class = get_model_class(backend_type)
        model = model_class()
        model.load(file_path)
    except Exception:
        shutil.rmtree(Path(file_path).parent)
        raise HTTPException(status_code=400, detail="Model weights could not be loaded")
    try:
        metadata_dict = model.metadata()
    except Exception as e:
        shutil.rmtree(Path(file_path).parent)
        raise HTTPException(status_code=400, detail=f"Model metadata could not be loaded + {str(e)}")

    db_model = MLModel(id = model_id,
            name = name,
            version = version,
            backend_type = backend_type,
            description = description,
            model_metadata = metadata_dict,
            accuracy = accuracy,
            weights_path = str(file_path),
            storage_type = "disk",
            inference_url = None
            )
    session.add(db_model)
    await session.commit()
    await session.refresh(db_model)
    return db_model

@model_router.get(path="/")
async def list_models(session = Depends(get_session)) -> List[ModelResponse]:
    query = select(MLModel)
    result = await session.execute(query)
    model_list = result.scalars().all()
    return model_list

@model_router.get(path="/{model_id}")
async def get_model(model_id: str, session = Depends(get_session)) -> ModelResponse:
    query = select(MLModel).where(MLModel.id == model_id)
    result = await session.execute(query)
    model  = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} does not exist!")
    return model

@model_router.put(path="/{model_id}")
async def update_model(model_id: str, model_update: ModelUpdate, session = Depends(get_session)) -> ModelResponse:
    query = select(MLModel).where(MLModel.id == model_id)
    result = await session.execute(query)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} does not exist!")
    
    update_data = model_update.model_dump(exclude_unset=True)

    for k,v in update_data.items():
        setattr(model, k, v)
    
    await redis_client.delete(model_id)
    model_cache.pop(model_id, None)

    await session.commit()
    await session.refresh(model)
    return model

@model_router.get(path="/{model_id}/metadata")
async def get_model_metadata(model_id: str, session = Depends(get_session)) -> MetadataResponse:
    query = select(MLModel).where(MLModel.id == model_id)
    result = await session.execute(query)
    model  = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} does not exist!")
    return model


@model_router.delete(path="/{model_id}")
async def delete_model(model_id: str, session = Depends(get_session)) -> Response:
    query = select(MLModel).where(MLModel.id == model_id)
    result = await session.execute(query)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} does not exist!")
    
    if Path(model.weights_path).exists():
        shutil.rmtree(Path(model.weights_path).parent)
    await redis_client.delete(model_id)
    model_cache.pop(model_id, None)
    await session.delete(model)
    await session.commit()

    return Response(status_code = 204)
