from tests.conftest import client
import pytest
from typing import List

@pytest.mark.asyncio
async def test_health_live(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert "alive" in data

@pytest.mark.asyncio
async def test_create_model(client):
    response = await client.post("/models/register", )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data

@pytest.mark.asyncio
async def test_list_models(client):
    response = await client.get("/models/")
    assert type(response) is List
    assert len(response)>0

@pytest.mark.asyncio
async def test_get_model(client, id):
    response = await client.get(f"/models/{id}")
    assert response.status_code == 
    data = response.json()
    assert "" in data

