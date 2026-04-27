from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text

from core.database import get_session
from core.redis import redis_client

health_router = APIRouter(prefix="/health")

@health_router.get("/live")
async def health_live():
    return JSONResponse(content={"status": "alive"})

@health_router.get("/ready")
async def health_ready(session = Depends(get_session)):
    try:
        await session.execute(text("SELECT 1"))
        await redis_client.ping()
    except Exception as e:
        return JSONResponse(status_code = 503, content={"status": str(e)})
    return JSONResponse(content= {"status": "ready"})
