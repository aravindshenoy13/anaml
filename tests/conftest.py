from io import BytesIO

import httpx
import joblib
import pytest
import redis
import skl2onnx
from httpx import AsyncClient
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from services.api.main import app
from shared.core.database import DBBase, get_session
from shared.core.redis import get_redis


def joblib_test_data():
    data_dict = {
        "name":"test-model",
        "version":"1.0",
        "backend_type":"joblib",
        "description":"cicd test model for joblib",
        "accuracy":"97.79"
    }
    buffer = BytesIO()
    dset = load_iris()
    model = RandomForestClassifier()
    model.fit(dset['data'], dset['target'])
    joblib.dump(model, buffer)

    file_dict = {"file": 
            ("test.pkl",
            buffer.getvalue(),
            "application/octet-stream"
        )}
    return data_dict, file_dict

def onnx_test_data():
    data_dict = {
        "name":"test-model",
        "version":"1.0",
        "backend_type":"onnx",
        "description":"cicd test model for onnx",
        "accuracy":"97.79"
    }
    dset = load_iris()
    model = RandomForestClassifier()
    model.fit(dset['data'], dset['target'])
    model = skl2onnx.convert_sklearn(model,initial_types=[("X",FloatTensorType([None, 4]))])
    model_bytes = model.SerializeToString()

    file_dict = {"file": 
            ("test.onnx",
            model_bytes,
            "application/octet-stream"
        )}
    
    return data_dict, file_dict

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
    await redis_client.flushdb()
    yield redis_client
    await redis_client.aclose()

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
