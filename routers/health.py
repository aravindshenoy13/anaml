from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from core.database import get_session
from sqlalchemy import text

health_router = APIRouter(prefix="/health")

@health_router.get("/live")
async def health_live():
    return JSONResponse(content={"status": "alive"})

@health_router.get("/ready")
async def health_ready(session = Depends(get_session)):
    try:
        await session.execute(text("SELECT 1"))
    except Exception as e:
        return JSONResponse(status_code = 503, content={"status": str(e)})
    return JSONResponse(content= {"status": "ready"})
