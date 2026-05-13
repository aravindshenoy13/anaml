import json
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException

from services.api.schemas.inference import AsyncPredictResponse, JobStatusResponse, PredictRequest, PredictResponse
from shared.core.database import get_session
from shared.core.redis import get_redis
from shared.core.resolve import resolve_model
from shared.core.utils import get_uuid
from shared.models.models import InferenceLog

inference_router = APIRouter()

@inference_router.post(path="/models/{model_id}/predict")
async def predict(model_id: str, predict_req: PredictRequest, 
                session = Depends(get_session), 
                redis_client = Depends(get_redis)) -> PredictResponse:
    
    cache = await resolve_model(model_id, session, redis_client, load=True)
    model = cache["model"]
    model_name = cache["model_name"]
    model_version = cache["model_version"]

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
    
    cache = await resolve_model(model_id, session, redis_client, load=False)
    backend_type = cache["backend_type"]
    weights_path = cache["weights_path"]
    
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
        "weights_path": weights_path
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
        output_data=job_status.get("output_data"),
        error_message=job_status.get("error_message")
    )