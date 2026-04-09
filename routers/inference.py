from fastapi import APIRouter, Depends, HTTPException
from schemas.schemas import PredictRequest, PredictResponse
from models.models import InferenceLog, MLModel
from inference.registry import get_model_class
from core.database import get_session
from sqlalchemy import select
import time
import json

router = APIRouter()
model_cache = {}

@router.post(path="/models/{id}/predict")
async def predict(id: int, predict_req: PredictRequest, session = Depends(get_session)) -> PredictResponse:
    if id not in model_cache:
        query = select(MLModel).where(MLModel.id == id)
        result = await session.execute(query)
        model_db = result.scalar_one_or_none()

        if model_db is None:
            raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
        
        backend = get_model_class(model_db.backend_type)
        model = backend()
        model.load(model_db.weights_path)
        model_cache[id] = model
    else:
        model = model_cache[id]

    before = time.perf_counter()
    model.predict(predict_req.input_data)
    latency = time.perf_counter() - before

    log = InferenceLog()
