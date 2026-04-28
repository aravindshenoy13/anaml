import redis
import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.database import DBBase, get_session
from core.redis import get_redis
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
async def redis_fixture():
    redis_client = redis.asyncio.from_url("redis://localhost:6379")
    yield redis_client

@pytest.fixture
async def session(engine):
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session

@pytest.fixture
async def client(session, redis_fixture):
    async def override_session():
        yield session
    async def override_redis():
        yield redis_fixture
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_redis] = override_redis
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        yield client
    app.dependency_overrides.clear()
