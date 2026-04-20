from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.database import DBBase, engine
from routers.health import health_router
from routers.inference import inference_router
from routers.models import model_router


@asynccontextmanager
async def run(app):
    async with engine.begin() as conn:
        await conn.run_sync(DBBase.metadata.create_all)
    yield

app = FastAPI(lifespan=run)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(model_router)
