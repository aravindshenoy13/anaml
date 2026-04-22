from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelCreate(BaseModel):
    name: str
    version: str
    description: str | None = None
    accuracy: float | None = None
    backend_type: Literal["onnx", "joblib"] = "joblib"

class ModelResponse(ModelCreate):
    id: str
    status: Literal["active", "archived"] = "active"
    created_at: datetime
    last_updated: datetime
    metadata: dict | None = None

    model_config = ConfigDict(from_attributes=True)

class ModelUpdate(BaseModel):
    name: str | None = None
    version: str | None = None
    description: str | None = None
    accuracy: float | None = None
    backend_type: Literal["onnx", "joblib"] | None = None
    status: Literal["active", "archived"] | None = None

class PredictRequest(BaseModel):
    input_data: dict

class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    output_data: dict
    latency: float

class ModelStats(BaseModel):
    model_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float | None = None
    

