import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from core.database import get_session
from core.redis import redis_client
from inference.registry import get_model_class
from models.models import InferenceLog, MLModel, get_uuid
from schemas.schemas import PredictRequest, PredictResponse

inference_router = APIRouter()
model_cache = {}

@inference_router.post(path="/models/{model_id}/predict")
async def predict(model_id: str, predict_req: PredictRequest, session = Depends(get_session)) -> PredictResponse:
    #Model Cache
    if model_id in model_cache:
        model = model_cache[model_id]["model"]
        model_name = model_cache[model_id]["model_name"]
        model_version = model_cache[model_id]["model_version"]
    else:
        cached = await redis_client.hgetall(model_id)
        #Query Cache
        if cached:
            backend_type = cached["backend_type"]
            weights_path = cached["weights_path"]
            model_name = cached["name"] 
            model_version = cached["version"] 
        else:
            query = select(MLModel).where(MLModel.id == model_id)
            result = await session.execute(query)
            model_db = result.scalar_one_or_none()

            if model_db is None:
                raise HTTPException(status_code=404, detail=f"Model with id {model_id} does not exist!")

            backend_type = model_db.backend_type
            weights_path = model_db.weights_path
            model_name = model_db.name
            model_version = model_db.version

            await redis_client.hset(model_id, mapping={
                "backend_type": backend_type,
                "weights_path": weights_path,
                "name": model_name,
                "version": model_version
            })
            
        backend = get_model_class(backend_type)
        model = backend()
        model.load(weights_path)
        model_cache[model_id] = {
            "model": model,
            "model_name": model_name,
            "model_version": model_version
        } 

    inference_start = time.perf_counter()
    try:
        output_data = model.predict(predict_req.input_data)
    except Exception as e:
        log_id  = get_uuid()
        log = InferenceLog(
            id = log_id,
            model_id = model_id,
            input_data = predict_req.input_data,
            output_data = None,
            latency = None,
            status = "Failed",
            error_message = str(e)
        )
        session.add(log)
        await session.commit()
        raise HTTPException(status_code=500, detail= f"Model Inference failed: {e}")

    latency = time.perf_counter() - inference_start

    log_id  = get_uuid()
    log = InferenceLog(
        id = log_id,
        model_id = model_id,
        input_data = predict_req.input_data,
        output_data = output_data,
        latency = latency,
        status = "Successful",
        error_message = None
    )
    session.add(log)
    await session.commit()

    response = PredictResponse(
        model_id=model_id,
        model_name = model_name,
        model_version= model_version,
        output_data= output_data,
        latency= latency
    )

    return response
