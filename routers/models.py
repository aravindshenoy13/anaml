import json
import shutil
from pathlib import Path
from typing import List, Literal

import aiofiles
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select

from core.config import MODEL_DIR
from core.database import get_session
from inference.registry import get_model_class
from models.models import MLModel, get_uuid
from schemas.schemas import ModelResponse, ModelUpdate

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
    except Exception:
        shutil.rmtree(Path(file_path).parent)
        raise HTTPException(status_code=400, detail="Model metadata could not be loaded")

    db_model = MLModel(id = model_id,
            name = name,
            version = version,
            backend_type = backend_type,
            description = description,
            metadata = json.dumps(metadata_dict),
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
    await session.refresh(model)
    return model

@model_router.delete(path="/{id}")
async def delete_model(id: str, session = Depends(get_session)) -> Response:
    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    
    if Path(model.weights_path).exists():
        shutil.rmtree(Path(model.weights_path).parent)
    await session.delete(model)
    await session.commit()

    return Response(status_code = 204)
