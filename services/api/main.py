from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.api.routers.health import health_router
from services.api.routers.inference import inference_router
from services.api.routers.models import model_router
from shared.core.database import DBBase, engine
from shared.core.redis import close_redis


@asynccontextmanager
async def run(app):
    async with engine.begin() as conn:
        await conn.run_sync(DBBase.metadata.create_all)
    yield
    await close_redis()
    await engine.dispose()

app = FastAPI(lifespan=run)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(model_router)
