from fastapi import FastAPI
from routers.health import health_router
from routers.inference import inference_router
from routers.models import model_router

AnaML = FastAPI()