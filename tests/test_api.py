from io import BytesIO

import joblib
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


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

@pytest.mark.asyncio
async def test_health_live(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"

@pytest.mark.asyncio
async def test_joblib_workflow(client):
    #create_model fixture
    data_dict, file_dict = joblib_test_data()
    create_response = await client.post("/models/register", data = data_dict, files=file_dict)
    assert create_response.status_code == 201, "Model creation failed"
    data = create_response.json()
    id = data["id"]
    assert data["id"], "Model creation id not returned"

    #get_model fixture
    get_response = await client.get(f"/models/{id}")
    assert get_response.status_code == 200, "Model retreival failed"
    data = get_response.json()
    assert data["id"] == id, "Model retreival id not matching/not returned"

    #predict_model fixture
    payload= {
    "input_data":{
    "features": [5.1, 3.5, 1.4, 0.2]
    }}
    predict_response = await client.post(f"/models/{id}/predict", json=payload)
    assert predict_response.status_code == 200, "Model prediction failed"
    data = predict_response.json()
    assert data["output_data"]['predictions'], "Prediction output values not returned"

    #delete_model fixture
    delete_response = await client.delete(f"/models/{id}")
    assert delete_response.status_code == 204, "Model deletion failed"
    
    #verifying delete
    verify_response = await client.get(f"/models/{id}")
    assert verify_response.status_code == 404, "Model exists after deletion"

