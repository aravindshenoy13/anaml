from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=True)
session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

class DBBase(DeclarativeBase):
    pass

async def get_session():
    async with session_maker() as s:
        yield s