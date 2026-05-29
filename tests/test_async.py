import pytest

from tests.conftest import joblib_test_data


@pytest.mark.asyncio
async def test_async_predict_returns_job_id(client):
    data_dict, file_dict = joblib_test_data()
    create_response = await client.post("/models/register", data = data_dict, files=file_dict)
    data = create_response.json()

    assert create_response.status_code == 201, "Model creation failed: " + str(data)
    model_id = data["id"]

    payload= {
    "input_data":{
    "features": [[5.1, 3.5, 1.4, 0.2]]
    }}
    async_predict_response = await client.post(f"/models/{model_id}/predict/async", json=payload)

    assert async_predict_response.status_code == 202
    async_data = async_predict_response.json()

    assert "job_id" in async_data
    assert async_data["job_id"]

@pytest.mark.asyncio
async def test_job_pending_status(client):
    data_dict, file_dict = joblib_test_data()
    create_response = await client.post("/models/register", data = data_dict, files=file_dict)
    data = create_response.json()

    assert create_response.status_code == 201, "Model creation failed: " + str(data)
    model_id = data["id"]

    payload= {
    "input_data":{
    "features": [[5.1, 3.5, 1.4, 0.2]]
    }}
    async_predict_response = await client.post(f"/models/{model_id}/predict/async", json=payload)

    assert async_predict_response.status_code == 202
    async_data = async_predict_response.json()

    assert "job_id" in async_data
    job_id = async_data["job_id"]
    assert job_id

    job_response = await client.get(f"/jobs/{job_id}")
    assert job_response.status_code == 200
    job_data = job_response.json()
    assert job_data["status"] == "pending"
    assert job_data["model_id"] == model_id
    assert job_data["created_at"]
    assert job_data["output_data"] is None
    assert job_data["error_message"] is None

@pytest.mark.asyncio
async def test_get_nonexistent_job_returns_404(client):
    job_id = "nonexistent-job-id"
    job_response = await client.get(f"/jobs/{job_id}")
    assert job_response.status_code == 404
