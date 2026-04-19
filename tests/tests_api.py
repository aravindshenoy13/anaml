import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
from io import BytesIO

async def helper_create_model(client):
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
    return await client.post("/models/register", data = data_dict, files=file_dict)

@pytest.mark.asyncio
async def test_health_live(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"

@pytest.mark.asyncio
async def test_create_model(client):
    response = await helper_create_model(client)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data

@pytest.mark.asyncio
async def test_list_models(client):
    response = await client.get("/models/")
    data = response.json()
    assert isinstance(data, list)
    assert len(response)>0

@pytest.mark.asyncio
async def test_get_model(client, id):
    response = await client.get(f"/models/{id}")
    assert response.status_code == 
    data = response.json()
    assert "" in data

