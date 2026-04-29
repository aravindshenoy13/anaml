import json
import time

import redis

from core.config import REDIS_URL
from inference.registry import get_model_class


async def startup():
    redis_client = redis.asyncio.from_url(REDIS_URL, decode_responses=True)
    redis_client.xreadgroup()
    try:
        await redis_client.xgroup_create("inference_jobs", "worker_group", id="0", mkstream=True)
    except Exception:
        pass #group already exists
    return redis_client

async def process_job(redis_client, message):
    message_id, fields = message
    job_id = fields["job_id"]
    input_data = fields["input_data"]
    backend_type = fields["backend_type"]
    weights_path = fields["weights_path"]

    job_status = json.loads(await redis_client.get(f"job:{job_id}"))
    job_status["status"] = "processing"
    await redis_client.set(f"job:{job_id}", json.dumps(job_status), ex=3600)

    model = get_model_class(backend_type)()
    model.load(weights_path)

    inference_start = time.perf_counter()
    try:
        output_data = model.predict(input_data)
    except Exception as e:
        job_status = json.loads(await redis_client.get(f"job:{job_id}"))
        job_status["status"] = "failed"
        job_status["error_message"] = str(e)
        await redis_client.set(f"job:{job_id}", json.dumps(job_status), ex=3600)
        return None
    
    latency = time.perf_counter() - inference_start
    job_status = json.loads(await redis_client.get(f"job:{job_id}"))
    job_status["status"] = "completed"
    job_status["latency"] = latency
    job_status["output_data"] = output_data
    await redis_client.set(f"job:{job_id}", json.dumps(job_status), ex=3600)
    return message_id

async def main():
    redis_client = startup()

    while(True):
        stream, message_list = redis_client.xreadgroup("worker_group", "worker-1", block=0)

        for message in message_list:
            message_id = process_job(redis_client, message)
            if message_id:
                redis_client.xack(stream, "worker_group", message_id)









