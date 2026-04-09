from fastapi import FastAPI
from routers.health import health_router
from routers.inference import inference_router
from routers.models import model_router
from core.database import engine, DBBase
from contextlib import asynccontextmanager

@asynccontextmanager
async def run(app):
    async with engine.begin() as conn:
        await conn.run_sync(DBBase.metadata.create_all)
    yield

AnaML = FastAPI(lifespan=run)
AnaML.include_router(health_router)
AnaML.include_router(inference_router)
AnaML.include_router(model_router)
