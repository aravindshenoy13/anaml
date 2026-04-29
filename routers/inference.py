import json
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from core.database import get_session
from core.redis import get_redis, model_cache
from core.utils import get_uuid
from inference.registry import get_model_class
from models.models import InferenceLog, MLModel
from schemas.inference import AsyncPredictResponse, JobStatusResponse, PredictRequest, PredictResponse

inference_router = APIRouter()

@inference_router.post(path="/models/{model_id}/predict")
async def predict(model_id: str, predict_req: PredictRequest, 
                session = Depends(get_session), 
                redis_client = Depends(get_redis)) -> PredictResponse:
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
            await redis_client.expire(model_id, 3600)
            
        backend = get_model_class(backend_type)
        model = backend()
        model.load(weights_path)
        model_cache[model_id] = {
            "model": model,
            "model_name": model_name,
            "model_version": model_version,
            "backend_type": backend_type
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

@inference_router.post(path="/models/{model_id}/predict/async")
async def async_predict(model_id: str, predict_req: PredictRequest,
                        session = Depends(get_session),
                        redis_client = Depends(get_redis)) -> AsyncPredictResponse:

    
    if model_id in model_cache:
        backend_type = model_cache[model_id]["backend_type"]
    else:
        cached = await redis_client.hgetall(model_id)
        if cached:
            backend_type = cached["backend_type"]
        else:
            query = select(MLModel).where(MLModel.id == model_id)
            result = await session.execute(query)
            model_db = result.scalar_one_or_none()

            if model_db is None:
                raise HTTPException(status_code=404, detail=f"Model with id {model_id} does not exist!")

            backend_type = model_db.backend_type
    
    job_id = get_uuid()
    job_status = {
        "status": "pending",
        "model_id": model_id,
        "created_at": datetime.now(UTC).isoformat()
    }
    job_data = {
        "job_id": job_id,
        "model_id": model_id,
        "input_data": json.dumps(predict_req.input_data),
        "backend_type": backend_type,
    }

    await redis_client.set(f"job:{job_id}", json.dumps(job_status), ex=3600)
    await redis_client.xadd("inference_jobs", job_data)

    return AsyncPredictResponse(job_id=job_id)

@inference_router.get(path="/jobs/{job_id}")
async def get_job(job_id: str, redis_client = Depends(get_redis)) -> JobStatusResponse:
    job_key = f"job:{job_id}"
    job_data = await redis_client.get(job_key)

    if job_data is None:
        raise HTTPException(status_code=404, detail=f"Job with id {job_id} does not exist/expired!")

    job_status = json.loads(job_data)
    return JobStatusResponse(
        job_id=job_id,
        status=job_status["status"],
        model_id=job_status["model_id"],
        created_at=job_status["created_at"],
        result=job_status.get("result"),
        error_message=job_status.get("error_message")
    )