import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.database import DBBase, get_session
from main import app


@pytest.fixture
async def engine():
    engine = create_async_engine(url="sqlite+aiosqlite:///")
    async with engine.begin() as conn:
        await conn.run_sync(DBBase.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(DBBase.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def session(engine):
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session

@pytest.fixture
async def client(session):
    async def override():
        yield session
    app.dependency_overrides[get_session] = override
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        yield client
    app.dependency_overrides.clear()
