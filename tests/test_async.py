import pytest

from tests.test_api import joblib_test_data


@pytest.mark.asyncio
async def test_async_predict_returns_job_id(client):

    data_dict, file_dict = joblib_test_data()
    create_response = await client.post("/models/register", data = data_dict, files=file_dict)
    data = create_response.json()
    model_id = data["id"]

    payload= {
    "input_data":{
    "features": [[5.1, 3.5, 1.4, 0.2]]
    }}
    async_predict_response = await client.post(f"/models/{model_id}/predict/async", data=payload)
    assert async_predict_response.status_code == 200
    data = async_predict_response.json()
    assert "job_id" in data
    assert data["job_id"]
