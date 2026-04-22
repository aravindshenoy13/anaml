import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text

from core.database import DBBase


def get_uuid():
    return str(uuid.uuid4())

class MLModel(DBBase):
    __tablename__ = "ml_models"

    id = Column(String, primary_key=True, index=True, default=get_uuid)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    model_metadata = Column(Text, nullable=True)
    accuracy = Column(Float)
    weights_path = Column(String(255), nullable=False)
    backend_type = Column(String(50), nullable=False)
    storage_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default="active")
    inference_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class InferenceLog(DBBase):
    __tablename__ = "inference_logs"

    id = Column(String, primary_key=True, index=True, default=get_uuid)
    model_id = Column(String, ForeignKey("ml_models.id"), nullable=False)
    input_data = Column(Text, nullable=False)
    output_data = Column(Text, nullable=True)
    latency = Column(Float, nullable=True)
    status = Column(String(50), nullable=False)
    error_message = Column(Text, default=None)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)