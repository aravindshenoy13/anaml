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