from fastapi import APIRouter, Depends, HTTPException
from schemas.schemas import PredictRequest, PredictResponse
from models.models import InferenceLog
from inference.registry import get_model_class
from sqlalchemy import select
import time
import json
