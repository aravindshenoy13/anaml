from typing import Literal

from pydantic import BaseModel


class PredictRequest(BaseModel):
    input_data: dict

class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    output_data: dict
    latency: float

class AsyncPredictResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    model_id: str
    created_at: str
    result: dict | None = None
    error_message: str | None = None