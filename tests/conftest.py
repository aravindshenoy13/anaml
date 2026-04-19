from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from core.database import DBBase, get_session
from httpx import AsyncClient
import httpx
import pytest
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
