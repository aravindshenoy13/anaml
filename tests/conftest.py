from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from core.database import DBBase
from httpx import AsyncClient
import pytest

@pytest.fixture
async def engine():
    engine = create_async_engine(url="sqlite+aiosqlite:///", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(DBBase.metadata.create_all)
        yield engine
    await conn.run_sync(DBBase.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def session(engine):
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session

@pytest.fixture
async def client():
    client = AsyncClient(transport=app)
