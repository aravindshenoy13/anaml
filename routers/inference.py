from fastapi import APIRouter, Depends, HTTPException
from schemas.schemas import PredictRequest, PredictResponse
from models.models import InferenceLog, MLModel, get_uuid
from inference.registry import get_model_class
from core.database import get_session
from sqlalchemy import select
import time
import json

router = APIRouter()
model_cache = {}

@router.post(path="/models/{id}/predict")
async def predict(id: str, predict_req: PredictRequest, session = Depends(get_session)) -> PredictResponse:

    query = select(MLModel).where(MLModel.id == id)
    result = await session.execute(query)
    model_db = result.scalar_one_or_none()

    if model_db is None:
        raise HTTPException(status_code=404, detail=f"Model with id {id} does not exist!")
    
    if id not in model_cache:
        backend = get_model_class(model_db.backend_type)
        model = backend()
        model.load(model_db.weights_path)
        model_cache[id] = model
    else:
        model = model_cache[id]

    inference_start = time.perf_counter()
    try:
        output_data = model.predict(predict_req.input_data)
    except Exception as e:
        log_id  = get_uuid()
        log = InferenceLog(
            id = log_id,
            model_id = id,
            input_data = json.dumps(predict_req.input_data),
            output_data = None,
            latency = None,
            status = "Failed",
            error_message = str(e)
        )
        session.add(log)
        await session.commit()
        raise HTTPException(status_code=500, detail="Model Inference failed")

    latency = time.perf_counter() - inference_start

    log_id  = get_uuid()
    log = InferenceLog(
        id = log_id,
        model_id = id,
        input_data = json.dumps(predict_req.input_data),
        output_data = json.dumps(output_data),
        latency = latency,
        status = "Successful",
        error_message = None
    )
    session.add(log)
    await session.commit()

    response = PredictResponse(
        model_id=id,
        model_name = model_db.name,
        model_version= model_db.version,
        output_data= output_data,
        latency= latency
    )

    return response
